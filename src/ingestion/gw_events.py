"""
Gravitational-Wave Transient Catalog (GWTC) ingestion for Project EXODUS.

Downloads and parses the GWTC catalog from the Gravitational-Wave Open
Science Center (GWOSC) event API at ``https://gwosc.org/eventapi/``.
Each confirmed event is parsed into a :class:`GWEvent` dataclass carrying
the event name, GPS time, sky localisation (RA, Dec, credible-region area
in square degrees), component masses, luminosity distance, and network SNR.

When the GWOSC API is unreachable or the ``requests`` library is not
installed, a **simulation fallback** generates 90 realistic GWTC-like
events drawn from the O1, O2, and O3 observing runs with physically
motivated parameter distributions:

    * Sky areas: 10--1000 sq deg (log-uniform, reflecting real GW posteriors)
    * Component masses: 5--80 Msun (power-law + Gaussian peak, mimicking the
      BBH mass function from O3)
    * Luminosity distances: 100--5000 Mpc
    * Network SNR: 8--30

Results are cached locally so that repeated runs avoid redundant network
traffic.

Public API
----------
get_all_events(force_refresh=False, simulate=False)
    Return every confirmed GW event as a list of :class:`GWEvent`.

get_by_skyregion(ra, dec, radius_deg, ...)
    Return events whose best-fit sky position falls within a cone search.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time as _time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project utilities
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import (
    get_logger,
    get_config,
    cache_key,
    load_cache,
    save_cache,
    PROJECT_ROOT,
)

logger = get_logger("ingestion.gw_events")

# ---------------------------------------------------------------------------
# Lazy / optional imports
# ---------------------------------------------------------------------------
try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    _HAS_ASTROPY = True
except ImportError:
    _HAS_ASTROPY = False
    logger.debug("astropy not available -- sky-region filtering disabled")


def _try_import_requests():
    try:
        import requests
        return requests
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GWOSC_API_BASE = "https://gwosc.org/eventapi"
GWOSC_CATALOG_ENDPOINT = "/json/allevents/"
CACHE_SUBFOLDER = "gw_events"
REQUEST_TIMEOUT = 30

# Observing-run GPS time boundaries (approximate) for simulation
_RUN_BOUNDARIES = {
    "O1": (1126051217, 1137254417),   # Sep 2015 -- Jan 2016
    "O2": (1164556817, 1187733618),   # Nov 2016 -- Aug 2017
    "O3a": (1238112018, 1253977218),  # Apr 2019 -- Oct 2019
    "O3b": (1256655618, 1269363618),  # Nov 2019 -- Mar 2020
}

# Number of events per run for simulation (total ~90)
_SIM_EVENTS_PER_RUN = {"O1": 3, "O2": 8, "O3a": 39, "O3b": 40}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GWEvent:
    """A single confirmed gravitational-wave transient event.

    Attributes
    ----------
    name : str
        Event designation (e.g. ``GW150914``, ``GW190521``).
    gps_time : float
        GPS time of the event trigger (seconds).
    ra : float or None
        Right ascension of the best-fit sky position (degrees, ICRS).
    dec : float or None
        Declination of the best-fit sky position (degrees, ICRS).
    sky_area_sq_deg : float or None
        90% credible-region sky area in square degrees.
    mass1 : float or None
        Primary component mass (source-frame solar masses).
    mass2 : float or None
        Secondary component mass (source-frame solar masses).
    luminosity_distance_mpc : float or None
        Luminosity distance in Mpc.
    snr : float or None
        Network matched-filter signal-to-noise ratio.
    observing_run : str or None
        Observing run label (``O1``, ``O2``, ``O3a``, ``O3b``).
    source : str
        Data provenance: ``"gwosc"`` or ``"simulated"``.
    """

    name: str
    gps_time: float
    ra: Optional[float] = None
    dec: Optional[float] = None
    sky_area_sq_deg: Optional[float] = None
    mass1: Optional[float] = None
    mass2: Optional[float] = None
    luminosity_distance_mpc: Optional[float] = None
    snr: Optional[float] = None
    observing_run: Optional[str] = None
    source: str = "gwosc"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GWEvent":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# GWOSC API client
# ---------------------------------------------------------------------------

class GWOSCClient:
    """Thin wrapper around the GWOSC event API."""

    def __init__(
        self,
        base_url: str = GWOSC_API_BASE,
        timeout: int = REQUEST_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._requests = _try_import_requests()

    def is_reachable(self) -> bool:
        """Return True if the GWOSC API responds."""
        if self._requests is None:
            logger.warning("requests library not installed; GWOSC API unreachable")
            return False
        try:
            resp = self._requests.get(
                f"{self.base_url}/",
                timeout=self.timeout,
            )
            return resp.status_code < 500
        except Exception as exc:
            logger.debug("GWOSC API unreachable: %s", exc)
            return False

    def fetch_all_events(self) -> List[Dict[str, Any]]:
        """Fetch the full GWTC event list from GWOSC.

        Returns a list of raw event dicts as provided by the GWOSC JSON API.
        Each dict typically has ``'events'`` containing nested event data
        with parameter estimates.
        """
        if self._requests is None:
            raise RuntimeError("requests library not available")

        url = f"{self.base_url}{GWOSC_CATALOG_ENDPOINT}"
        logger.info("Querying GWOSC: %s", url)

        resp = self._requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        # The top-level JSON may be a dict keyed by event name, OR it
        # may wrap events under a top-level "events" key (newer API).
        if isinstance(data, dict):
            # Unwrap the "events" envelope if present
            if "events" in data and isinstance(data["events"], dict) and len(data) <= 2:
                data = data["events"]
            events = []
            for event_name, event_data in data.items():
                # Flatten: add name into the payload
                if isinstance(event_data, dict):
                    event_data["_name"] = event_name
                    events.append(event_data)
            logger.info("GWOSC returned %d events", len(events))
            return events

        raise RuntimeError(f"Unexpected GWOSC response format: {type(data)}")


# ---------------------------------------------------------------------------
# Event parser
# ---------------------------------------------------------------------------

def _parse_gwosc_event(event_name: str, event_data: dict) -> GWEvent:
    """Parse a single GWOSC event dict into a GWEvent dataclass.

    The GWOSC ``/json/allevents/`` endpoint returns a flat dict with
    parameter fields at the top level (e.g. ``mass_1_source``,
    ``luminosity_distance``).  The per-event JSON URLs may nest under
    ``"events" -> <catalog_name> -> "parameters"``.  We handle both.

    Note: The GWOSC allevents API does **not** provide RA/Dec sky
    positions.  Sky localizations are only available through posterior
    sample FITS files, which are too large for automated download.
    """
    params: Dict[str, Any] = {}

    # Navigate into the nested catalog structure (per-event JSON URLs)
    # Prefer the most recent catalog (highest GWTC version)
    if "events" in event_data and isinstance(event_data["events"], dict):
        catalogs = event_data["events"]
        for cat_key in sorted(catalogs.keys(), reverse=True):
            cat_entry = catalogs[cat_key]
            if isinstance(cat_entry, dict):
                if "parameters" in cat_entry:
                    params = cat_entry["parameters"]
                else:
                    params = cat_entry
                break
    else:
        # Flat structure (allevents endpoint) — parameters at top level
        params = event_data

    gps = _extract_float(params, ["GPS", "gps", "tc", "geocent_time"])
    ra = _extract_float(params, ["ra", "RA", "right_ascension"])
    dec = _extract_float(params, ["dec", "Dec", "declination"])
    sky_area = _extract_float(params, [
        "sky_area_90", "sky90", "area90", "skyarea",
        "sky_area", "90cr_area",
    ])
    m1 = _extract_float(params, ["mass_1_source", "mass1", "m1"])
    m2 = _extract_float(params, ["mass_2_source", "mass2", "m2"])
    dl = _extract_float(params, [
        "luminosity_distance", "distance", "DL",
        "luminosity_distance_Mpc",
    ])
    snr = _extract_float(params, ["network_matched_filter_snr", "snr", "SNR"])

    # Determine observing run from GPS time or catalog name
    obs_run = _gps_to_run(gps) if gps is not None else None
    if obs_run is None:
        catalog = event_data.get("catalog.shortName", "")
        if "O1" in catalog or "O2" in catalog:
            obs_run = "O1/O2"
        elif "O3" in catalog:
            obs_run = "O3"
        elif "O4" in catalog:
            obs_run = "O4"

    return GWEvent(
        name=event_name,
        gps_time=gps or 0.0,
        ra=ra,
        dec=dec,
        sky_area_sq_deg=sky_area,
        mass1=m1,
        mass2=m2,
        luminosity_distance_mpc=dl,
        snr=snr,
        observing_run=obs_run,
        source="gwosc",
    )


def _extract_float(
    params: dict,
    keys: List[str],
) -> Optional[float]:
    """Try multiple key names; return the first that yields a valid float."""
    for key in keys:
        val = params.get(key)
        if val is None:
            continue
        # GWOSC sometimes wraps values in a nested dict with 'best' or
        # 'median' sub-keys
        if isinstance(val, dict):
            for sub in ("best", "median", "value", "pe"):
                if sub in val:
                    try:
                        return float(val[sub])
                    except (TypeError, ValueError):
                        continue
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def _gps_to_run(gps: float) -> Optional[str]:
    """Map a GPS time to an observing-run label."""
    for run, (start, end) in _RUN_BOUNDARIES.items():
        if start <= gps <= end:
            return run
    # O4 or unknown
    if gps > 1269363618:
        return "O4+"
    return None


# ---------------------------------------------------------------------------
# Simulation fallback
# ---------------------------------------------------------------------------

def _generate_simulated_events(
    n_total: int = 90,
    seed: int = 20150914,
) -> List[GWEvent]:
    """Generate a realistic set of GWTC-like simulated events.

    The parameter distributions are designed to approximate the real
    GWTC-3 population:

    * **Sky areas**: log-uniform between 10 and 1000 sq deg, reflecting
      the broad range from well-localised (O3 three-detector) to poorly
      localised (O1 two-detector) events.
    * **Masses**: drawn from a power-law + Gaussian-peak model loosely
      inspired by the LIGO/Virgo population fits: most events have
      m1 in 8--40 Msun with a peak around 35 Msun, and m2 >= 5 Msun.
    * **Distances**: 100--5000 Mpc (roughly uniform in comoving volume
      for a magnitude-limited survey).
    * **SNR**: 8--30, with a steep distribution (most events near
      threshold).

    Parameters
    ----------
    n_total : int
        Total number of simulated events (default 90, matching ~GWTC-3).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[GWEvent]
        Simulated events sorted by GPS time.
    """
    rng = np.random.RandomState(seed)
    events: List[GWEvent] = []

    idx = 0
    for run, n_run in _SIM_EVENTS_PER_RUN.items():
        gps_start, gps_end = _RUN_BOUNDARIES[run]
        gps_times = np.sort(rng.uniform(gps_start, gps_end, n_run))

        for gps in gps_times:
            idx += 1

            # Sky localisation: log-uniform in [10, 1000] sq deg
            sky_area = 10 ** rng.uniform(1.0, 3.0)

            # RA uniform on [0, 360), Dec ~ cos-weighted
            ra = rng.uniform(0.0, 360.0)
            dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0)))

            # Component masses: power-law + Gaussian peak
            # m1 in [8, 80] with peak around 35 Msun
            if rng.random() < 0.3:
                # Gaussian peak component (~35 Msun)
                m1 = rng.normal(35.0, 6.0)
                m1 = np.clip(m1, 10.0, 80.0)
            else:
                # Power-law component
                m1 = _power_law_sample(rng, 5.0, 80.0, alpha=-2.3)

            # m2 <= m1, minimum 5 Msun
            q = rng.uniform(0.2, 1.0)  # mass ratio
            m2 = max(5.0, m1 * q)
            if m2 > m1:
                m1, m2 = m2, m1

            # Luminosity distance: ~ uniform in comoving volume
            # d^3 distribution between 100 and 5000 Mpc
            d_min, d_max = 100.0, 5000.0
            u_vol = rng.random()
            dl = (d_min**3 + u_vol * (d_max**3 - d_min**3)) ** (1.0 / 3.0)

            # SNR: inverse-ish distribution (most events near threshold)
            snr = 8.0 + rng.exponential(3.0)
            snr = min(snr, 30.0)

            # Event name
            name = f"GW-SIM-{idx:04d}"

            events.append(GWEvent(
                name=name,
                gps_time=float(gps),
                ra=float(ra),
                dec=float(dec),
                sky_area_sq_deg=float(sky_area),
                mass1=float(round(m1, 1)),
                mass2=float(round(m2, 1)),
                luminosity_distance_mpc=float(round(dl, 1)),
                snr=float(round(snr, 1)),
                observing_run=run,
                source="simulated",
            ))

    events.sort(key=lambda e: e.gps_time)
    logger.info("Generated %d simulated GW events", len(events))
    return events


def _power_law_sample(
    rng: np.random.RandomState,
    x_min: float,
    x_max: float,
    alpha: float,
) -> float:
    """Sample from a power-law distribution p(x) ~ x^alpha on [x_min, x_max]."""
    if alpha == -1:
        # Log-uniform
        return x_min * (x_max / x_min) ** rng.random()
    ap1 = alpha + 1.0
    u = rng.random()
    return (x_min**ap1 + u * (x_max**ap1 - x_min**ap1)) ** (1.0 / ap1)


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def _deduplicate_gwosc_events(raw_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only the best version of each unique GW event.

    The GWOSC ``/json/allevents/`` endpoint returns multiple catalog
    entries per event (e.g. GW150914 appears in O1_O2-Preliminary,
    GWTC-1-confident, GWTC-2.1-confident).  We keep the version with
    the most non-null parameter fields, preferring ``confident`` catalogs.
    """
    best: Dict[str, Dict[str, Any]] = {}

    for raw in raw_events:
        common = raw.get("commonName", raw.get("_name", "unknown"))
        catalog = raw.get("catalog.shortName", "")

        # Count non-null parameter fields as a quality score
        param_keys = [
            "mass_1_source", "mass_2_source", "luminosity_distance",
            "network_matched_filter_snr", "chi_eff", "redshift",
        ]
        quality = sum(1 for k in param_keys if raw.get(k) is not None)

        # Prefer "confident" catalogs over "marginal" or "preliminary"
        if "confident" in catalog.lower():
            quality += 10

        prev = best.get(common)
        if prev is None or quality > prev.get("_quality", 0):
            raw["_quality"] = quality
            best[common] = raw

    deduped = list(best.values())
    logger.info(
        "Deduplicated %d raw entries -> %d unique events",
        len(raw_events), len(deduped),
    )
    return deduped


def get_all_events(
    *,
    force_refresh: bool = False,
    simulate: bool = False,
) -> List[GWEvent]:
    """Return all confirmed GW transient events from GWTC.

    Tries (in order):
        1. Local cache.
        2. GWOSC event API.
        3. Simulated fallback.

    Parameters
    ----------
    force_refresh : bool
        If True, bypass the cache and re-query GWOSC.
    simulate : bool
        If True, skip the network entirely and return simulated events.

    Returns
    -------
    list[GWEvent]
        All events, sorted by GPS time.
    """
    ck = cache_key("gwtc", "all_events", "v2")

    # -- 1. Cache --
    if not force_refresh and not simulate:
        cached = load_cache(ck, subfolder=CACHE_SUBFOLDER)
        if cached is not None:
            events = [GWEvent.from_dict(d) for d in cached]
            # Validate: require at least some events with mass data
            valid = [e for e in events if e.name != "events"]
            has_params = any(e.mass1 is not None for e in valid)
            if valid and has_params:
                logger.info("Loaded %d GW events from cache (%d with mass data)",
                            len(valid), sum(1 for e in valid if e.mass1 is not None))
                return valid
            logger.warning("GW cache invalid; re-fetching")

    # -- 2. GWOSC API --
    if not simulate:
        try:
            client = GWOSCClient()
            if client.is_reachable():
                raw_events = client.fetch_all_events()

                # Deduplicate: keep best version per commonName
                raw_events = _deduplicate_gwosc_events(raw_events)

                events = []
                for raw in raw_events:
                    name = raw.get("commonName", raw.get("_name", "unknown"))
                    try:
                        ev = _parse_gwosc_event(name, raw)
                        events.append(ev)
                    except Exception as exc:
                        logger.debug("Failed to parse event %s: %s", name, exc)
                        continue

                if events:
                    events.sort(key=lambda e: e.gps_time)
                    # Persist to cache
                    save_cache(
                        ck,
                        [e.to_dict() for e in events],
                        subfolder=CACHE_SUBFOLDER,
                    )
                    n_with_mass = sum(1 for e in events if e.mass1 is not None)
                    n_with_sky = sum(1 for e in events if e.ra is not None)
                    logger.info(
                        "Fetched and cached %d GW events from GWOSC "
                        "(%d with mass, %d with sky position)",
                        len(events), n_with_mass, n_with_sky,
                    )
                    return events
            else:
                logger.warning("GWOSC API unreachable; falling back to simulation")
        except Exception as exc:
            logger.warning("GWOSC query failed: %s -- falling back to simulation", exc)

    # -- 3. Simulated fallback (NOT cached — prevents provenance contamination) --
    logger.warning("Using SIMULATED GWTC catalog (90 events, O1-O3, not cached)")
    events = _generate_simulated_events()
    return events


def get_by_skyregion(
    ra: float,
    dec: float,
    radius_deg: float = 10.0,
    *,
    force_refresh: bool = False,
    simulate: bool = False,
) -> List[GWEvent]:
    """Return GW events whose best-fit sky position falls within a cone.

    Because GW sky localisations are typically tens to hundreds of square
    degrees, this performs a simple cone search on the *best-fit* position
    (RA, Dec) rather than on the full posterior.  A generous default radius
    of 10 degrees is used.

    Parameters
    ----------
    ra, dec : float
        Centre of the search cone (degrees, ICRS).
    radius_deg : float
        Cone radius in degrees.
    force_refresh : bool
        Re-query GWOSC instead of using cached data.
    simulate : bool
        Use simulated data.

    Returns
    -------
    list[GWEvent]
        Events whose best-fit position is within *radius_deg* of
        (ra, dec).
    """
    all_events = get_all_events(force_refresh=force_refresh, simulate=simulate)

    if not _HAS_ASTROPY:
        # Fallback: naive Euclidean approximation (adequate for large radii)
        logger.warning(
            "astropy not available; using approximate Euclidean sky distance"
        )
        results = []
        cos_dec = np.cos(np.radians(dec))
        for ev in all_events:
            if ev.ra is None or ev.dec is None:
                continue
            dra = (ev.ra - ra) * cos_dec
            ddec = ev.dec - dec
            sep = np.sqrt(dra**2 + ddec**2)
            if sep <= radius_deg:
                results.append(ev)
        logger.info(
            "Sky-region filter (approx): %d / %d events within %.1f deg of "
            "(%.2f, %.2f)",
            len(results), len(all_events), radius_deg, ra, dec,
        )
        return results

    # Astropy path
    target = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame="icrs")

    valid_events = [
        ev for ev in all_events if ev.ra is not None and ev.dec is not None
    ]
    if not valid_events:
        logger.warning("No events with valid sky positions")
        return []

    ev_ra = np.array([ev.ra for ev in valid_events])
    ev_dec = np.array([ev.dec for ev in valid_events])
    ev_coords = SkyCoord(ra=ev_ra, dec=ev_dec, unit=(u.deg, u.deg), frame="icrs")

    seps = target.separation(ev_coords)
    mask = seps <= radius_deg * u.deg

    results = [ev for ev, m in zip(valid_events, mask) if m]

    logger.info(
        "Sky-region filter: %d / %d events within %.1f deg of (%.2f, %.2f)",
        len(results), len(all_events), radius_deg, ra, dec,
    )
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("  Project EXODUS -- Gravitational-Wave Transient Catalog Ingestion")
    print("=" * 72)
    print()

    # -- 1. Fetch / generate all events --
    print("[1] Fetching GWTC events ...")
    events = get_all_events()
    print(f"    Total events: {len(events)}")
    source_tag = events[0].source if events else "N/A"
    print(f"    Data source:  {source_tag}")
    print()

    # -- 2. Summary by observing run --
    print("[2] Events by observing run:")
    print("-" * 40)
    runs: Dict[str, List[GWEvent]] = {}
    for ev in events:
        r = ev.observing_run or "unknown"
        runs.setdefault(r, []).append(ev)
    for run_name in sorted(runs.keys()):
        run_events = runs[run_name]
        print(f"    {run_name:>6s}: {len(run_events):>3d} events")
    print()

    # -- 3. Parameter distributions --
    print("[3] Parameter distributions:")
    print("-" * 40)
    sky_areas = [ev.sky_area_sq_deg for ev in events if ev.sky_area_sq_deg]
    masses = [ev.mass1 for ev in events if ev.mass1]
    distances = [ev.luminosity_distance_mpc for ev in events if ev.luminosity_distance_mpc]
    snrs = [ev.snr for ev in events if ev.snr]

    if sky_areas:
        print(f"    Sky area (sq deg): min={min(sky_areas):.1f}, "
              f"median={np.median(sky_areas):.1f}, max={max(sky_areas):.1f}")
    if masses:
        print(f"    Primary mass (Msun): min={min(masses):.1f}, "
              f"median={np.median(masses):.1f}, max={max(masses):.1f}")
    if distances:
        print(f"    Distance (Mpc): min={min(distances):.1f}, "
              f"median={np.median(distances):.1f}, max={max(distances):.1f}")
    if snrs:
        print(f"    SNR: min={min(snrs):.1f}, median={np.median(snrs):.1f}, "
              f"max={max(snrs):.1f}")
    print()

    # -- 4. Sample events --
    print("[4] Sample events (first 10):")
    print("-" * 40)
    for ev in events[:10]:
        ra_str = f"{ev.ra:.2f}" if ev.ra is not None else "N/A"
        dec_str = f"{ev.dec:.2f}" if ev.dec is not None else "N/A"
        area_str = f"{ev.sky_area_sq_deg:.0f}" if ev.sky_area_sq_deg else "N/A"
        m1_str = f"{ev.mass1:.1f}" if ev.mass1 else "N/A"
        m2_str = f"{ev.mass2:.1f}" if ev.mass2 else "N/A"
        dl_str = f"{ev.luminosity_distance_mpc:.0f}" if ev.luminosity_distance_mpc else "N/A"
        snr_str = f"{ev.snr:.1f}" if ev.snr else "N/A"
        print(f"    {ev.name:<20s}  GPS={ev.gps_time:.0f}  "
              f"RA={ra_str}  Dec={dec_str}  area={area_str} sq deg")
        print(f"      m1={m1_str} m2={m2_str} Msun  d={dl_str} Mpc  "
              f"SNR={snr_str}  run={ev.observing_run}")
    print()

    # -- 5. Sky-region search demo --
    print("[5] Sky-region search demo: cone at (RA=180, Dec=0, r=30 deg)")
    print("-" * 40)
    region_events = get_by_skyregion(180.0, 0.0, radius_deg=30.0)
    print(f"    Events in region: {len(region_events)}")
    for ev in region_events[:5]:
        print(f"      {ev.name}  RA={ev.ra:.2f}  Dec={ev.dec:.2f}")
    if len(region_events) > 5:
        print(f"      ... and {len(region_events) - 5} more")
    print()

    print("Done.")
