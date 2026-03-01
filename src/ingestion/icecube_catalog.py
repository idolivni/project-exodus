"""
IceCube neutrino event catalog ingestion module for Project EXODUS.

Accesses IceCube public point-source event catalogs (10-year track-like
events released by the IceCube Collaboration), parses event data, and
provides convenience accessors for energy / positional filtering.

The IceCube 10-year point-source catalog contains ~1.1 million muon-track
neutrino events spanning 2008--2018.  Each event records:
    * time (MJD)
    * reconstructed direction (RA, Dec in degrees, J2000)
    * reconstructed energy proxy (GeV)
    * angular uncertainty (degrees, 90% containment)

When the real catalog is unavailable (network down, file not hosted, etc.)
a simulation fallback generates realistic IceCube-like data:
    * ~100 000 events over 10 years of livetime
    * Power-law energy distribution (E^{-2.5}) from 100 GeV to 10 PeV
    * Isotropic sky distribution (uniform in RA, sinusoidal in Dec)
    * Track angular resolution ~1 degree (energy-dependent)

Results are cached locally for fast repeated access.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field, asdict
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

logger = get_logger("ingestion.icecube_catalog")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# IceCube public data release URLs (10-year point-source tracks)
ICECUBE_PS_DATA_URL = (
    "https://icecube.wisc.edu/data-releases/20210126_PS-IC40-IC86_VII.csv"
)
ICECUBE_PS_DATA_URL_ALT = (
    "https://dataverse.harvard.edu/api/access/datafile/4458603"
)

# HEASARC TAP service (reliable NASA infrastructure — 1,134,450 events)
HEASARC_TAP_URL = "https://heasarc.gsfc.nasa.gov/xamin/vo/tap"
HEASARC_MIN_ENERGY_GEV = 100_000  # 100 TeV: best angular resolution, astrophysical
HEASARC_MAX_ROWS_PER_QUERY = 100_000  # HEASARC sync query limit

CACHE_SUBFOLDER = "icecube"
REQUEST_TIMEOUT = 60

# Simulation parameters
SIM_N_EVENTS = 100_000
SIM_LIVETIME_YEARS = 10.0
SIM_MJD_START = 54682.0    # ~2008-08-01 (IC40 start)
SIM_MJD_END = 58309.0      # ~2018-07-01 (IC86-VII end)
SIM_E_MIN_GEV = 100.0
SIM_E_MAX_GEV = 1e7        # 10 PeV
SIM_SPECTRAL_INDEX = 2.5   # E^{-gamma}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NeutrinoEvent:
    """A single IceCube muon-track neutrino event."""

    mjd: float                           # Modified Julian Date of the event
    ra: float                            # Right ascension (degrees, J2000)
    dec: float                           # Declination (degrees, J2000)
    energy_gev: float                    # Reconstructed energy proxy (GeV)
    angular_err_deg: float               # Angular uncertainty (degrees, ~90%)
    source: str = "icecube"              # "icecube" | "simulated"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NeutrinoEvent":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

def _try_import_requests():
    try:
        import requests
        return requests
    except ImportError:
        return None


def _try_import_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Simulated IceCube event generator
# ---------------------------------------------------------------------------

def _generate_simulated_events(
    n_events: int = SIM_N_EVENTS,
    seed: int = 20210126,
) -> List[NeutrinoEvent]:
    """
    Generate a realistic set of simulated IceCube-like neutrino events.

    Physics model:
        * Isotropic sky distribution (uniform RA, sin(dec) weighting
          biased towards northern sky to mimic IceCube effective area).
        * Power-law energy spectrum: dN/dE ~ E^{-2.5} from 100 GeV to 10 PeV.
        * Angular resolution: sigma ~ 1.0 * (E / 1 TeV)^{-0.4} degrees,
          clipped to [0.2, 5.0] degrees (tracks only).
        * Uniform time distribution between MJD 54682 and 58309.

    Parameters
    ----------
    n_events : int
        Number of events to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[NeutrinoEvent]
    """
    rng = np.random.RandomState(seed)
    logger.info(
        "Generating %d simulated IceCube events (seed=%d)", n_events, seed
    )

    # --- Time: uniform over livetime ---
    mjd = rng.uniform(SIM_MJD_START, SIM_MJD_END, n_events)

    # --- Sky position: isotropic with IceCube effective-area weighting ---
    # IceCube sees the northern sky best (upgoing muons).
    # Approximate effective-area weighting: flat for dec > -5 deg,
    # falling off as cos(dec + 5 deg) below that.
    ra = rng.uniform(0.0, 360.0, n_events)

    # Sample dec from an isotropic distribution (uniform in sin(dec)),
    # then apply acceptance to approximate IceCube geometry.
    sin_dec = rng.uniform(-1.0, 1.0, n_events)
    dec = np.degrees(np.arcsin(sin_dec))

    # Apply IceCube-like acceptance: keep events in northern sky
    # preferentially.  Use accept/reject with an envelope.
    accept_prob = np.where(
        dec > -5.0,
        1.0,
        np.clip(0.5 + 0.5 * np.sin(np.radians(dec + 90)), 0.05, 1.0),
    )
    mask = rng.random(n_events) < accept_prob
    # Re-sample rejected events from the northern hemisphere
    n_rejected = np.sum(~mask)
    if n_rejected > 0:
        sin_dec_resample = rng.uniform(0.0, 1.0, n_rejected)
        dec[~mask] = np.degrees(np.arcsin(sin_dec_resample))
        ra[~mask] = rng.uniform(0.0, 360.0, n_rejected)

    # --- Energy: power-law sampling via inverse CDF ---
    # For dN/dE ~ E^{-gamma}, CDF^{-1}(u) =
    #   [E_min^{1-gamma} + u * (E_max^{1-gamma} - E_min^{1-gamma})]^{1/(1-gamma)}
    gamma = SIM_SPECTRAL_INDEX
    u = rng.random(n_events)
    e_min_term = SIM_E_MIN_GEV ** (1.0 - gamma)
    e_max_term = SIM_E_MAX_GEV ** (1.0 - gamma)
    energy_gev = (e_min_term + u * (e_max_term - e_min_term)) ** (
        1.0 / (1.0 - gamma)
    )

    # --- Angular resolution: energy-dependent ---
    # Better angular resolution at higher energy
    # sigma ~ 1.0 * (E / 1 TeV)^{-0.4} degrees
    angular_err = 1.0 * (energy_gev / 1000.0) ** (-0.4)
    angular_err = np.clip(angular_err, 0.2, 5.0)

    events = []
    for i in range(n_events):
        events.append(NeutrinoEvent(
            mjd=float(mjd[i]),
            ra=float(ra[i]),
            dec=float(dec[i]),
            energy_gev=float(energy_gev[i]),
            angular_err_deg=float(angular_err[i]),
            source="simulated",
        ))

    logger.info(
        "Simulated %d events: E_median=%.0f GeV, Dec_median=%.1f deg",
        len(events),
        float(np.median(energy_gev)),
        float(np.median(dec)),
    )
    return events


# ---------------------------------------------------------------------------
# IceCube catalog parser
# ---------------------------------------------------------------------------

def _parse_icecube_csv(text: str) -> List[NeutrinoEvent]:
    """
    Parse the IceCube point-source CSV data release.

    Expected columns (order may vary):
        MJD, RA[deg], Dec[deg], AngErr[deg], log10(E/GeV)
    or:
        mjd, ra, dec, angErr, logE

    The parser is tolerant of header variations.
    """
    pd = _try_import_pandas()
    if pd is not None:
        return _parse_with_pandas(text, pd)
    return _parse_manual(text)


def _parse_with_pandas(text: str, pd) -> List[NeutrinoEvent]:
    """Parse CSV using pandas (preferred path)."""
    from io import StringIO

    df = pd.read_csv(StringIO(text), comment="#")

    # Normalise column names to lowercase, strip whitespace
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Map known column-name variants
    col_map = {}
    for target, candidates in [
        ("mjd", ["mjd", "time", "t"]),
        ("ra", ["ra", "ra[deg]", "ra_deg", "right_ascension"]),
        ("dec", ["dec", "dec[deg]", "dec_deg", "declination"]),
        ("energy", ["log10(e/gev)", "loge", "log10e", "logenergy", "energy"]),
        ("angerr", ["angerr", "angerr[deg]", "angular_err", "sigma"]),
    ]:
        for cand in candidates:
            if cand in df.columns:
                col_map[target] = cand
                break

    if not all(k in col_map for k in ("mjd", "ra", "dec")):
        raise ValueError(
            f"Cannot identify required columns in IceCube CSV. "
            f"Found columns: {list(df.columns)}"
        )

    events = []
    for _, row in df.iterrows():
        mjd_val = float(row[col_map["mjd"]])
        ra_val = float(row[col_map["ra"]])
        dec_val = float(row[col_map["dec"]])

        # Energy: may be log10(E/GeV) or linear GeV
        if "energy" in col_map:
            e_raw = float(row[col_map["energy"]])
            # Heuristic: if values are < 20, it is probably log10
            energy_gev = 10.0 ** e_raw if e_raw < 20.0 else e_raw
        else:
            energy_gev = 0.0

        # Angular error
        if "angerr" in col_map:
            ang_err = float(row[col_map["angerr"]])
        else:
            ang_err = 1.0  # default ~1 deg for tracks

        events.append(NeutrinoEvent(
            mjd=mjd_val,
            ra=ra_val,
            dec=dec_val,
            energy_gev=energy_gev,
            angular_err_deg=ang_err,
            source="icecube",
        ))

    return events


def _parse_manual(text: str) -> List[NeutrinoEvent]:
    """Fallback CSV parser without pandas."""
    lines = [
        line.strip()
        for line in text.strip().split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        return []

    # First non-comment line is the header
    header = [c.strip().lower().replace(" ", "_") for c in lines[0].split(",")]

    # Find column indices
    def _find_col(names):
        for name in names:
            if name in header:
                return header.index(name)
        return None

    idx_mjd = _find_col(["mjd", "time", "t"])
    idx_ra = _find_col(["ra", "ra[deg]", "ra_deg"])
    idx_dec = _find_col(["dec", "dec[deg]", "dec_deg"])
    idx_energy = _find_col(["log10(e/gev)", "loge", "log10e", "energy"])
    idx_angerr = _find_col(["angerr", "angerr[deg]", "angular_err", "sigma"])

    if idx_mjd is None or idx_ra is None or idx_dec is None:
        raise ValueError(
            f"Cannot identify required columns. Header: {header}"
        )

    events = []
    for line in lines[1:]:
        fields = line.split(",")
        try:
            mjd_val = float(fields[idx_mjd])
            ra_val = float(fields[idx_ra])
            dec_val = float(fields[idx_dec])

            if idx_energy is not None:
                e_raw = float(fields[idx_energy])
                energy_gev = 10.0 ** e_raw if e_raw < 20.0 else e_raw
            else:
                energy_gev = 0.0

            ang_err = float(fields[idx_angerr]) if idx_angerr is not None else 1.0

            events.append(NeutrinoEvent(
                mjd=mjd_val,
                ra=ra_val,
                dec=dec_val,
                energy_gev=energy_gev,
                angular_err_deg=ang_err,
                source="icecube",
            ))
        except (ValueError, IndexError):
            continue

    return events


# ---------------------------------------------------------------------------
# Main public API class
# ---------------------------------------------------------------------------

class IceCubeCatalogIngest:
    """
    Primary interface for ingesting IceCube neutrino event catalogs.

    Provides:
        - get_all_events()                 -> all catalog events
        - get_high_energy(min_gev=100)     -> energy-filtered events
        - get_by_position(ra, dec, r_deg)  -> cone-search events

    Behaviour:
        1. Attempts to download the IceCube 10-year point-source catalog.
        2. Parses and caches the events locally.
        3. Falls back to simulated data when the catalog is unreachable.
    """

    def __init__(self, force_simulated: bool = False):
        """
        Parameters
        ----------
        force_simulated : bool
            If True, skip network access and generate simulated data.
        """
        self._cfg = get_config()
        self._force_simulated = force_simulated
        self._simulated_mode = force_simulated
        self._events: Optional[List[NeutrinoEvent]] = None

        self._data_dir = (
            PROJECT_ROOT
            / self._cfg["project"]["data_dir"]
            / "icecube"
        )
        self._data_dir.mkdir(parents=True, exist_ok=True)

        if self._simulated_mode:
            logger.info("IceCube ingestion: forced simulated mode")

    # -- public API ----------------------------------------------------------

    def get_all_events(
        self, force_refresh: bool = False,
    ) -> List[NeutrinoEvent]:
        """
        Return all neutrino events from the catalog (or simulation).

        Events are loaded once and cached in memory for subsequent calls.

        Parameters
        ----------
        force_refresh : bool
            If True, bypass both memory and disk cache and re-download.
        """
        if self._events is not None and not force_refresh:
            return self._events

        if force_refresh:
            # Clear disk cache to force re-download
            ck = cache_key("icecube", "heasarc_events", "v2")
            save_cache(ck, None, subfolder=CACHE_SUBFOLDER)
            logger.info("Force refresh: cleared IceCube disk cache")

        self._events = self._load_events()
        return self._events

    def get_arrays(self) -> Optional[Dict[str, np.ndarray]]:
        """Return neutrino event data as raw numpy arrays (fast path).

        Bypasses dataclass instantiation — ideal for crossmatch code
        that immediately converts to arrays anyway.

        Returns
        -------
        dict with keys: 'mjd', 'ra', 'dec', 'energy_gev', 'angular_err_deg'
            Each value is a float64 numpy array.
        None if no data is available.
        """
        npz_path = self._data_dir / "icecube_events.npz"
        if npz_path.exists():
            try:
                data = np.load(str(npz_path))
                arrays = {
                    "mjd": data["mjd"],
                    "ra": data["ra"],
                    "dec": data["dec"],
                    "energy_gev": data["energy_gev"],
                    "angular_err_deg": data["angular_err_deg"],
                }
                logger.info(
                    "Loaded %d IceCube events as arrays from npz",
                    len(arrays["ra"]),
                )
                return arrays
            except Exception as exc:
                logger.warning("npz array load failed: %s", exc)

        # Fallback: load events, convert to arrays
        events = self.get_all_events()
        if not events:
            return None
        return {
            "mjd": np.array([e.mjd for e in events], dtype=np.float64),
            "ra": np.array([e.ra for e in events], dtype=np.float64),
            "dec": np.array([e.dec for e in events], dtype=np.float64),
            "energy_gev": np.array([e.energy_gev for e in events], dtype=np.float64),
            "angular_err_deg": np.array([e.angular_err_deg for e in events], dtype=np.float64),
        }

    def get_high_energy(self, min_gev: float = 100.0) -> List[NeutrinoEvent]:
        """
        Return neutrino events with reconstructed energy above a threshold.

        Parameters
        ----------
        min_gev : float
            Minimum energy in GeV (default 100).

        Returns
        -------
        list[NeutrinoEvent]
            Filtered events sorted by energy descending.
        """
        events = self.get_all_events()
        filtered = [e for e in events if e.energy_gev >= min_gev]
        filtered.sort(key=lambda e: e.energy_gev, reverse=True)
        logger.info(
            "High-energy filter (>%.0f GeV): %d / %d events",
            min_gev, len(filtered), len(events),
        )
        return filtered

    def get_by_position(
        self,
        ra: float,
        dec: float,
        radius_deg: float = 2.0,
    ) -> List[NeutrinoEvent]:
        """
        Return neutrino events within an angular radius of a sky position.

        Uses astropy SkyCoord for accurate angular separation calculation.

        Parameters
        ----------
        ra : float
            Right ascension of the search centre (degrees).
        dec : float
            Declination of the search centre (degrees).
        radius_deg : float
            Search cone radius in degrees (default 2.0).

        Returns
        -------
        list[NeutrinoEvent]
            Events within the search cone, sorted by separation.
        """
        try:
            import astropy.units as u
            from astropy.coordinates import SkyCoord
        except ImportError:
            logger.warning(
                "astropy not available; falling back to approximate "
                "angular distance calculation"
            )
            return self._get_by_position_approx(ra, dec, radius_deg)

        events = self.get_all_events()
        if not events:
            return []

        target = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame="icrs")

        evt_ra = np.array([e.ra for e in events])
        evt_dec = np.array([e.dec for e in events])
        evt_coords = SkyCoord(
            ra=evt_ra, dec=evt_dec, unit=(u.deg, u.deg), frame="icrs"
        )

        sep = target.separation(evt_coords).deg
        mask = sep <= radius_deg

        matched = [
            (events[i], sep[i])
            for i in range(len(events))
            if mask[i]
        ]
        matched.sort(key=lambda pair: pair[1])
        result = [pair[0] for pair in matched]

        logger.info(
            "Cone search (%.4f, %.4f) r=%.2f deg: %d events found",
            ra, dec, radius_deg, len(result),
        )
        return result

    def _get_by_position_approx(
        self,
        ra: float,
        dec: float,
        radius_deg: float,
    ) -> List[NeutrinoEvent]:
        """Approximate cone search without astropy."""
        events = self.get_all_events()
        cos_dec = np.cos(np.radians(dec))
        result = []
        for e in events:
            dra = (e.ra - ra) * cos_dec
            ddec = e.dec - dec
            dist = np.sqrt(dra ** 2 + ddec ** 2)
            if dist <= radius_deg:
                result.append(e)
        result.sort(key=lambda e: np.sqrt(
            ((e.ra - ra) * cos_dec) ** 2 + (e.dec - dec) ** 2
        ))
        logger.info(
            "Approximate cone search (%.4f, %.4f) r=%.2f deg: %d events",
            ra, dec, radius_deg, len(result),
        )
        return result

    # -- internal helpers ----------------------------------------------------

    def _load_events(self) -> List[NeutrinoEvent]:
        """Load events from cache, network, or simulation.

        Priority:
            1. Local numpy .npz cache (fastest — ~0.1s for 195k events)
            2. Local JSON cache (slower — ~5s for 195k events)
            3. HEASARC TAP (NASA infrastructure, most reliable)
            4. Direct IceCube / Harvard CSV download (original release)
            5. Simulation fallback (NOT cached)
        """
        ck = cache_key("icecube", "heasarc_events", "v2")
        npz_path = self._data_dir / "icecube_events.npz"

        # 1. Try numpy .npz cache (fast binary arrays)
        if npz_path.exists():
            try:
                data = np.load(str(npz_path))
                n = len(data["ra"])
                source_val = str(data["source"]) if "source" in data else "icecube_heasarc"
                events = []
                for i in range(n):
                    events.append(NeutrinoEvent(
                        mjd=float(data["mjd"][i]),
                        ra=float(data["ra"][i]),
                        dec=float(data["dec"][i]),
                        energy_gev=float(data["energy_gev"][i]),
                        angular_err_deg=float(data["angular_err_deg"][i]),
                        source=source_val,
                    ))
                logger.info("Loaded %d IceCube events from npz cache", n)
                return events
            except Exception as exc:
                logger.warning("npz cache load failed: %s", exc)

        # 2. Try JSON cache
        cached = load_cache(ck, subfolder=CACHE_SUBFOLDER)
        if cached is not None and isinstance(cached, list):
            logger.info("Loaded %d IceCube events from JSON cache", len(cached))
            events = [NeutrinoEvent.from_dict(d) for d in cached]
            # Save as npz for next time
            self._save_npz(events, npz_path)
            return events

        # 2. Try HEASARC TAP (preferred — real data, reliable NASA endpoint)
        if not self._force_simulated:
            events = self._download_heasarc_tap()
            if events:
                logger.info(
                    "Downloaded %d real IceCube events from HEASARC TAP",
                    len(events),
                )
                self._save_events_to_cache(events, ck)
                return events

        # 3. Try direct CSV download (original release URLs)
        if not self._force_simulated:
            events = self._download_catalog_csv()
            if events:
                logger.info(
                    "Downloaded %d events from IceCube CSV", len(events)
                )
                self._save_events_to_cache(events, ck)
                return events

        # 4. Simulation fallback (NOT cached — prevents provenance contamination)
        logger.warning("Using SIMULATED IceCube events (not cached)")
        self._simulated_mode = True
        events = _generate_simulated_events()
        return events

    def _download_heasarc_tap(self) -> Optional[List[NeutrinoEvent]]:
        """Download IceCube events from HEASARC ICECUBEPSC table via TAP.

        The HEASARC ICECUBEPSC table contains 1,134,450 muon-track
        neutrino events from 2008–2018 (IceCube 10-year point source).

        We download events above 100 TeV (best angular resolution,
        astrophysical-dominated).  HEASARC limits sync queries to
        100,000 rows, so we split by energy range if needed.

        Returns
        -------
        list[NeutrinoEvent] or None
            Real neutrino events, or None if download fails.
        """
        try:
            import pyvo
        except ImportError:
            logger.debug("pyvo not installed; skipping HEASARC TAP")
            return None

        logger.info(
            "Querying HEASARC TAP for IceCube events (E > %d GeV) ...",
            HEASARC_MIN_ENERGY_GEV,
        )

        try:
            tap = pyvo.dal.TAPService(HEASARC_TAP_URL)

            # Split downloads into energy bands to stay under the 100k row limit.
            # Band boundaries chosen so each band has < 100k events.
            energy_bands = [
                (1_000_000, None),           # > 1 PeV  (~4k events)
                (500_000, 1_000_000),         # 500 TeV–1 PeV (~28k)
                (200_000, 500_000),           # 200–500 TeV (~84k)
                (HEASARC_MIN_ENERGY_GEV, 200_000),  # 100–200 TeV (~76k)
            ]

            all_events: List[NeutrinoEvent] = []

            for e_lo, e_hi in energy_bands:
                if e_hi is not None:
                    adql = (
                        f"SELECT time, ra, dec, error_radius, event_energy "
                        f"FROM icecubepsc "
                        f"WHERE event_energy >= {e_lo} AND event_energy < {e_hi}"
                    )
                    band_label = f"{e_lo/1000:.0f}–{e_hi/1000:.0f} TeV"
                else:
                    adql = (
                        f"SELECT time, ra, dec, error_radius, event_energy "
                        f"FROM icecubepsc "
                        f"WHERE event_energy >= {e_lo}"
                    )
                    band_label = f">{e_lo/1000:.0f} TeV"

                logger.info("  HEASARC TAP band %s ...", band_label)
                result = tap.search(adql, maxrec=HEASARC_MAX_ROWS_PER_QUERY)

                batch_events = []
                for row in result:
                    batch_events.append(NeutrinoEvent(
                        mjd=float(row["time"]),
                        ra=float(row["ra"]),
                        dec=float(row["dec"]),
                        energy_gev=float(row["event_energy"]),
                        angular_err_deg=float(row["error_radius"]),
                        source="icecube_heasarc",
                    ))

                logger.info(
                    "    Band %s: %d events", band_label, len(batch_events)
                )
                all_events.extend(batch_events)

            if all_events:
                # Sort by energy descending (highest energy first)
                all_events.sort(key=lambda e: e.energy_gev, reverse=True)
                energies = np.array([e.energy_gev for e in all_events])
                logger.info(
                    "HEASARC TAP complete: %d events, "
                    "E_median=%.0f GeV, E_max=%.0f GeV",
                    len(all_events),
                    float(np.median(energies)),
                    float(np.max(energies)),
                )
                return all_events

        except Exception as exc:
            logger.warning("HEASARC TAP download failed: %s", exc)

        return None

    def _download_catalog_csv(self) -> Optional[List[NeutrinoEvent]]:
        """Attempt to download the IceCube point-source CSV catalog."""
        requests = _try_import_requests()
        if requests is None:
            logger.warning("requests library not installed; cannot download")
            return None

        for url in (ICECUBE_PS_DATA_URL, ICECUBE_PS_DATA_URL_ALT):
            try:
                logger.info("Downloading IceCube catalog from %s ...", url)
                resp = requests.get(url, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    text = resp.text
                    events = _parse_icecube_csv(text)
                    if events:
                        logger.info(
                            "Parsed %d events from %s", len(events), url
                        )
                        return events
                    else:
                        logger.warning(
                            "Downloaded data from %s but parsed 0 events", url
                        )
                else:
                    logger.debug(
                        "HTTP %d from %s", resp.status_code, url
                    )
            except Exception as exc:
                logger.debug("Download failed from %s: %s", url, exc)
                continue

        logger.warning("All IceCube CSV download URLs failed")
        return None

    def _save_npz(self, events: List[NeutrinoEvent], npz_path: Path) -> None:
        """Save events as numpy .npz for fast binary loading."""
        try:
            n = len(events)
            mjd = np.array([e.mjd for e in events], dtype=np.float64)
            ra = np.array([e.ra for e in events], dtype=np.float64)
            dec = np.array([e.dec for e in events], dtype=np.float64)
            energy = np.array([e.energy_gev for e in events], dtype=np.float64)
            ang_err = np.array([e.angular_err_deg for e in events], dtype=np.float64)
            source = events[0].source if events else "unknown"
            np.savez_compressed(
                str(npz_path),
                mjd=mjd, ra=ra, dec=dec,
                energy_gev=energy, angular_err_deg=ang_err,
                source=np.array(source),
            )
            logger.info("Saved %d events to npz cache: %s", n, npz_path)
        except Exception as exc:
            logger.warning("Failed to save npz cache: %s", exc)

    def _save_events_to_cache(
        self, events: List[NeutrinoEvent], ck: str
    ) -> None:
        """Persist events list to the local cache (JSON + npz)."""
        try:
            data = [e.to_dict() for e in events]
            save_cache(ck, data, subfolder=CACHE_SUBFOLDER)
            logger.info("Cached %d IceCube events (JSON)", len(events))
        except Exception as exc:
            logger.warning("Failed to cache IceCube events: %s", exc)
        # Also save as npz for fast future loading
        npz_path = self._data_dir / "icecube_events.npz"
        self._save_npz(events, npz_path)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_default_ingest: Optional[IceCubeCatalogIngest] = None


def _get_ingest() -> IceCubeCatalogIngest:
    """Lazily initialise and return the default ingest instance."""
    global _default_ingest
    if _default_ingest is None:
        _default_ingest = IceCubeCatalogIngest()
    return _default_ingest


def get_all_events(force_refresh: bool = False) -> List[NeutrinoEvent]:
    """Return all IceCube neutrino events."""
    return _get_ingest().get_all_events(force_refresh=force_refresh)


def get_arrays() -> Optional[Dict[str, np.ndarray]]:
    """Return IceCube events as raw numpy arrays (fast path for crossmatch)."""
    return _get_ingest().get_arrays()


def get_high_energy(min_gev: float = 100.0) -> List[NeutrinoEvent]:
    """Return neutrino events above a minimum energy threshold."""
    return _get_ingest().get_high_energy(min_gev=min_gev)


def get_by_position(
    ra: float, dec: float, radius_deg: float = 2.0
) -> List[NeutrinoEvent]:
    """Return neutrino events within an angular cone on the sky."""
    return _get_ingest().get_by_position(ra, dec, radius_deg)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("  Project EXODUS -- IceCube Neutrino Catalog Ingestion")
    print("=" * 72)
    print()

    ingest = IceCubeCatalogIngest()

    # 1. Load all events
    print("[1] Loading all IceCube events ...")
    all_events = ingest.get_all_events()
    print(f"    Total events: {len(all_events):,}")
    if all_events:
        src = all_events[0].source
        print(f"    Source:        {src}")
        energies = np.array([e.energy_gev for e in all_events])
        mjds = np.array([e.mjd for e in all_events])
        decs = np.array([e.dec for e in all_events])
        print(f"    Energy range:  {energies.min():.0f} - {energies.max():.0f} GeV")
        print(f"    MJD range:     {mjds.min():.1f} - {mjds.max():.1f}")
        print(f"    Dec range:     {decs.min():.1f} - {decs.max():.1f} deg")
        print(f"    Median energy: {np.median(energies):.0f} GeV")
    print()

    # 2. High-energy events
    print("[2] High-energy events (>1 TeV):")
    print("-" * 40)
    he_events = ingest.get_high_energy(min_gev=1000.0)
    print(f"    Events above 1 TeV: {len(he_events):,}")
    if he_events:
        print(f"    Highest energy:     {he_events[0].energy_gev:.0f} GeV")
        print()
        print("    Top 10 highest-energy events:")
        for i, evt in enumerate(he_events[:10], 1):
            print(
                f"      {i:>3}. E={evt.energy_gev:>12,.0f} GeV  "
                f"RA={evt.ra:7.3f}  Dec={evt.dec:+7.3f}  "
                f"MJD={evt.mjd:.2f}  angErr={evt.angular_err_deg:.2f} deg"
            )
    print()

    # 3. Positional search
    print("[3] Cone search near Cygnus X-3 (RA=308.1, Dec=+40.96):")
    print("-" * 40)
    cone_events = ingest.get_by_position(ra=308.1, dec=40.96, radius_deg=2.0)
    print(f"    Events within 2 deg: {len(cone_events)}")
    if cone_events:
        for i, evt in enumerate(cone_events[:5], 1):
            print(
                f"      {i}. E={evt.energy_gev:>10,.0f} GeV  "
                f"RA={evt.ra:7.3f}  Dec={evt.dec:+7.3f}  "
                f"angErr={evt.angular_err_deg:.2f} deg"
            )
        if len(cone_events) > 5:
            print(f"      ... and {len(cone_events) - 5} more")
    print()

    # 4. Statistics summary
    print("[4] Catalog statistics:")
    print("-" * 40)
    all_e = np.array([e.energy_gev for e in all_events])
    all_err = np.array([e.angular_err_deg for e in all_events])
    print(f"    Total events:          {len(all_events):,}")
    print(f"    Events > 100 GeV:      {np.sum(all_e > 100):,}")
    print(f"    Events > 1 TeV:        {np.sum(all_e > 1e3):,}")
    print(f"    Events > 100 TeV:      {np.sum(all_e > 1e5):,}")
    print(f"    Events > 1 PeV:        {np.sum(all_e > 1e6):,}")
    print(f"    Median ang. error:     {np.median(all_err):.2f} deg")
    print(f"    Mean ang. error:       {np.mean(all_err):.2f} deg")
    print()
    print("Done.")
