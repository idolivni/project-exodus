"""
GW Event -- Exoplanet Host Sky Cross-Match for Project EXODUS.

**EXPLORATORY / SPECULATIVE** -- This is the most speculative expansion in
the EXODUS pipeline.  Gravitational-wave sky localisations are typically
tens to hundreds of square degrees, meaning that *chance coincidence* with
known exoplanet host stars is very common.  This module quantifies exactly
that: for every spatial coincidence between a GW event credible region and
an exoplanet host star, it computes the probability of a chance match
given the sky area and the local exoplanet-host surface density.

The key output is a comparison of *observed coincidences* versus the
*expected number from chance alone*.  Only matches whose individual
false-alarm probability is below a configurable threshold (default 1%)
are flagged -- though even those should be treated with extreme caution.

Statistical framework
---------------------
For a GW event with 90%-credible sky area ``A`` (sq deg) and a catalog of
``N_host`` exoplanet host stars distributed over the full sky (41253 sq deg),
the probability of at least one host falling inside the credible region
purely by chance is:

    P_chance = 1 - (1 - A / 41253)^N_host

For typical values (A ~ 100 sq deg, N_host ~ 4000), P_chance ~ 1 -- i.e.,
it is virtually *certain* that at least one host star lies inside the GW
error box by coincidence.  The expected number of chance coincidences is:

    N_expected = N_host * A / 41253

This module computes both quantities and reports them alongside the actual
observed coincidences, so the user can judge statistical significance.

Public API
----------
crossmatch_gw_exoplanets(gw_events, exoplanet_hosts, ...)
    Cross-match GW event sky regions against exoplanet host positions.

Returns a :class:`GWExoplanetCrossmatchResult` with detailed per-event
and aggregate statistics.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project utilities
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger, get_config, save_result

log = get_logger("detection.gw_exoplanet_crossmatch")

# ---------------------------------------------------------------------------
# Lazy / optional imports
# ---------------------------------------------------------------------------
try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    _HAS_ASTROPY = True
except ImportError:
    _HAS_ASTROPY = False
    log.debug("astropy not available -- positional cross-matching disabled")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FULL_SKY_SQ_DEG = 41_253.0   # area of the full celestial sphere in sq deg

# Default threshold for flagging a coincidence as "low false-alarm"
DEFAULT_FA_THRESHOLD = 0.01   # 1% per-event false-alarm probability


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GWHostCoincidence:
    """A single spatial coincidence between a GW event and an exoplanet host.

    Attributes
    ----------
    gw_event_name : str
        Name of the gravitational-wave event.
    host_name : str
        Name of the exoplanet host star.
    host_ra : float
        RA of the host star (degrees).
    host_dec : float
        Dec of the host star (degrees).
    angular_separation_deg : float
        Angular distance between the GW best-fit position and the host.
    gw_sky_area_sq_deg : float
        90% credible sky area of the GW event (sq deg).
    p_chance_single : float
        Probability that *this specific host* falls inside the GW error
        region by chance (A / full_sky).
    p_chance_any : float
        Probability that *at least one host* falls inside the GW error
        region by chance: 1 - (1 - A/full_sky)^N_hosts.
    n_expected : float
        Expected number of host coincidences from chance alone.
    is_low_fa : bool
        True if ``p_chance_any`` is below the false-alarm threshold.
    """

    gw_event_name: str
    host_name: str
    host_ra: float
    host_dec: float
    angular_separation_deg: float
    gw_sky_area_sq_deg: float
    p_chance_single: float
    p_chance_any: float
    n_expected: float
    is_low_fa: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GWExoplanetCrossmatchResult:
    """Aggregate result of cross-matching GW events against exoplanet hosts.

    Attributes
    ----------
    n_gw_events : int
        Number of GW events analysed.
    n_hosts : int
        Number of exoplanet host stars in the catalog.
    n_coincidences : int
        Total observed spatial coincidences.
    n_expected_total : float
        Sum of per-event expected coincidences from chance.
    excess_ratio : float
        Observed / expected ratio (values >> 1 would indicate a signal,
        but values near 1.0 are anticipated).
    coincidences : list[GWHostCoincidence]
        Individual coincidences with detailed statistics.
    low_fa_coincidences : list[GWHostCoincidence]
        Subset of coincidences below the false-alarm threshold (the most
        "interesting" matches, though likely still chance).
    fa_threshold : float
        The false-alarm probability threshold used.
    note : str
        Cautionary note about the exploratory nature of this analysis.
    """

    n_gw_events: int = 0
    n_hosts: int = 0
    n_coincidences: int = 0
    n_expected_total: float = 0.0
    excess_ratio: float = 0.0
    coincidences: List[GWHostCoincidence] = field(default_factory=list)
    low_fa_coincidences: List[GWHostCoincidence] = field(default_factory=list)
    fa_threshold: float = DEFAULT_FA_THRESHOLD
    note: str = (
        "EXPLORATORY: GW sky localisations are very large (tens to hundreds "
        "of sq deg). Most coincidences with exoplanet hosts are expected "
        "from chance alone. An excess_ratio near 1.0 is normal."
    )

    def to_dict(self) -> dict:
        return {
            "n_gw_events": self.n_gw_events,
            "n_hosts": self.n_hosts,
            "n_coincidences": self.n_coincidences,
            "n_expected_total": round(self.n_expected_total, 2),
            "excess_ratio": round(self.excess_ratio, 4),
            "fa_threshold": self.fa_threshold,
            "n_low_fa": len(self.low_fa_coincidences),
            "note": self.note,
            "coincidences": [c.to_dict() for c in self.coincidences],
            "low_fa_coincidences": [c.to_dict() for c in self.low_fa_coincidences],
        }


# ---------------------------------------------------------------------------
# Core cross-match logic
# ---------------------------------------------------------------------------

def crossmatch_gw_exoplanets(
    gw_events: List[Any],
    exoplanet_hosts: List[Dict[str, Any]],
    *,
    fa_threshold: float = DEFAULT_FA_THRESHOLD,
    use_sky_area: bool = True,
) -> GWExoplanetCrossmatchResult:
    """Cross-match GW event sky localisations against exoplanet host positions.

    For each GW event, identifies all exoplanet hosts that fall within the
    event's 90% credible sky region (approximated as a circular cap with the
    same solid angle, centred on the best-fit position).  Computes chance-
    coincidence probabilities for each match.

    Parameters
    ----------
    gw_events : list
        GW events, each having attributes or dict keys: ``name``,
        ``ra``, ``dec``, ``sky_area_sq_deg``.  Accepts :class:`GWEvent`
        dataclass instances or plain dicts.
    exoplanet_hosts : list[dict]
        Host star catalog.  Each dict must have ``ra`` and ``dec``
        (degrees), and optionally ``host_star`` or ``hostname``.
    fa_threshold : float
        Per-event false-alarm probability threshold for flagging
        "low-FA" coincidences (default 0.01 = 1%).
    use_sky_area : bool
        If True (default), use the GW event sky area to define the
        search radius.  If False, use a fixed 10-degree cone.

    Returns
    -------
    GWExoplanetCrossmatchResult
        Detailed cross-match results including chance-coincidence stats.
    """
    n_hosts = len(exoplanet_hosts)
    n_events = len(gw_events)

    log.info(
        "GW-exoplanet cross-match: %d GW events x %d exoplanet hosts "
        "(FA threshold=%.4f)",
        n_events, n_hosts, fa_threshold,
    )

    if n_events == 0 or n_hosts == 0:
        log.warning("Empty input: %d events, %d hosts", n_events, n_hosts)
        return GWExoplanetCrossmatchResult(
            n_gw_events=n_events, n_hosts=n_hosts, fa_threshold=fa_threshold,
        )

    # Pre-compute host coordinates
    host_ra = np.array([h["ra"] for h in exoplanet_hosts], dtype=np.float64)
    host_dec = np.array([h["dec"] for h in exoplanet_hosts], dtype=np.float64)

    if _HAS_ASTROPY:
        host_coords = SkyCoord(
            ra=host_ra, dec=host_dec, unit=(u.deg, u.deg), frame="icrs",
        )
    else:
        host_coords = None

    all_coincidences: List[GWHostCoincidence] = []
    n_expected_total = 0.0

    for ev in gw_events:
        # Extract event attributes (support dataclass or dict)
        ev_name = _get_attr(ev, "name", "unknown")
        ev_ra = _get_attr(ev, "ra")
        ev_dec = _get_attr(ev, "dec")
        ev_area = _get_attr(ev, "sky_area_sq_deg")

        if ev_ra is None or ev_dec is None:
            log.debug("Skipping %s: no sky position", ev_name)
            continue

        # Determine search radius from sky area
        # Approximate the credible region as a circular cap:
        #   A = pi * r^2  =>  r = sqrt(A / pi)
        if use_sky_area and ev_area is not None and ev_area > 0:
            search_radius_deg = np.sqrt(ev_area / np.pi)
        else:
            search_radius_deg = 10.0
            ev_area = np.pi * search_radius_deg**2

        # Expected coincidences for this event
        n_expected_event = n_hosts * ev_area / FULL_SKY_SQ_DEG
        n_expected_total += n_expected_event

        # Probability of at least one chance coincidence
        p_chance_any = 1.0 - (1.0 - ev_area / FULL_SKY_SQ_DEG) ** n_hosts

        # Find hosts within the search cone
        matches = _find_hosts_in_cone(
            ev_ra, ev_dec, search_radius_deg,
            host_ra, host_dec, host_coords,
        )

        for i_host, sep_deg in matches:
            host = exoplanet_hosts[i_host]
            host_name = (
                host.get("host_star")
                or host.get("hostname")
                or host.get("name")
                or f"host@({host['ra']:.4f},{host['dec']:.4f})"
            )

            p_chance_single = ev_area / FULL_SKY_SQ_DEG

            coincidence = GWHostCoincidence(
                gw_event_name=ev_name,
                host_name=host_name,
                host_ra=host["ra"],
                host_dec=host["dec"],
                angular_separation_deg=float(sep_deg),
                gw_sky_area_sq_deg=float(ev_area),
                p_chance_single=float(p_chance_single),
                p_chance_any=float(p_chance_any),
                n_expected=float(n_expected_event),
                is_low_fa=(p_chance_any < fa_threshold),
            )
            all_coincidences.append(coincidence)

    # Sort by false-alarm probability (ascending = most significant first)
    all_coincidences.sort(key=lambda c: c.p_chance_any)

    low_fa = [c for c in all_coincidences if c.is_low_fa]
    n_obs = len(all_coincidences)
    excess = n_obs / n_expected_total if n_expected_total > 0 else 0.0

    log.info(
        "Cross-match complete: %d coincidences observed, %.1f expected "
        "from chance (ratio=%.3f)",
        n_obs, n_expected_total, excess,
    )
    if low_fa:
        log.info(
            "  %d coincidences below FA threshold %.4f (still likely chance)",
            len(low_fa), fa_threshold,
        )

    result = GWExoplanetCrossmatchResult(
        n_gw_events=n_events,
        n_hosts=n_hosts,
        n_coincidences=n_obs,
        n_expected_total=n_expected_total,
        excess_ratio=excess,
        coincidences=all_coincidences,
        low_fa_coincidences=low_fa,
        fa_threshold=fa_threshold,
    )

    # Persist result
    try:
        save_result("gw_exoplanet_crossmatch", result.to_dict())
    except Exception as exc:
        log.debug("Could not save result: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Extract an attribute from a dataclass or dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _find_hosts_in_cone(
    center_ra: float,
    center_dec: float,
    radius_deg: float,
    host_ra: np.ndarray,
    host_dec: np.ndarray,
    host_coords: Optional[Any],
) -> List[Tuple[int, float]]:
    """Find host-star indices within a cone search.

    Returns a list of (index, separation_deg) tuples.
    """
    if _HAS_ASTROPY and host_coords is not None:
        center = SkyCoord(
            ra=center_ra, dec=center_dec, unit=(u.deg, u.deg), frame="icrs",
        )
        seps = center.separation(host_coords)
        mask = seps <= radius_deg * u.deg
        indices = np.where(mask)[0]
        return [(int(i), float(seps[i].deg)) for i in indices]

    # Fallback: approximate Euclidean (adequate for the large radii here)
    cos_dec = np.cos(np.radians(center_dec))
    dra = (host_ra - center_ra) * cos_dec
    ddec = host_dec - center_dec
    sep = np.sqrt(dra**2 + ddec**2)
    mask = sep <= radius_deg
    indices = np.where(mask)[0]
    return [(int(i), float(sep[i])) for i in indices]


# ---------------------------------------------------------------------------
# Simulation mode
# ---------------------------------------------------------------------------

def _generate_simulated_hosts(
    n_hosts: int = 4000,
    seed: int = 12345,
) -> List[Dict[str, Any]]:
    """Generate a realistic set of simulated exoplanet host positions.

    The hosts are distributed roughly isotropically over the sky with a
    mild concentration toward the Galactic plane (reflecting the Kepler
    field bias in the real catalog).

    Parameters
    ----------
    n_hosts : int
        Number of host stars to generate (default 4000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        Each dict has ``ra``, ``dec``, and ``host_star`` keys.
    """
    rng = np.random.RandomState(seed)
    hosts = []

    for i in range(n_hosts):
        if rng.random() < 0.4:
            # Kepler-field-like concentration: RA ~ 290 deg, Dec ~ +44 deg
            ra = rng.normal(290.0, 10.0) % 360.0
            dec = np.clip(rng.normal(44.0, 8.0), -90.0, 90.0)
        else:
            # Isotropic
            ra = rng.uniform(0.0, 360.0)
            dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0)))

        hosts.append({
            "ra": float(ra),
            "dec": float(dec),
            "host_star": f"SIM-Host-{i:05d}",
        })

    return hosts


def _generate_simulated_gw_events(
    n_events: int = 90,
    seed: int = 20150914,
) -> List[Dict[str, Any]]:
    """Generate simulated GW events as plain dicts for demo use.

    Uses the same distribution logic as the gw_events module simulation.

    Parameters
    ----------
    n_events : int
        Number of events (default 90).
    seed : int
        Random seed.

    Returns
    -------
    list[dict]
        Each dict has ``name``, ``ra``, ``dec``, ``sky_area_sq_deg``.
    """
    rng = np.random.RandomState(seed)
    events = []

    for i in range(n_events):
        ra = rng.uniform(0.0, 360.0)
        dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0)))
        sky_area = 10.0 ** rng.uniform(1.0, 3.0)  # 10 -- 1000 sq deg

        events.append({
            "name": f"GW-SIM-{i:04d}",
            "ra": float(ra),
            "dec": float(dec),
            "sky_area_sq_deg": float(sky_area),
        })

    return events


def run_simulation(
    n_events: int = 90,
    n_hosts: int = 4000,
    seed: int = 42,
    fa_threshold: float = DEFAULT_FA_THRESHOLD,
) -> GWExoplanetCrossmatchResult:
    """Run a full simulated cross-match for demonstration and testing.

    Generates simulated GW events and exoplanet hosts, cross-matches them,
    and returns the result with full chance-coincidence statistics.

    Parameters
    ----------
    n_events : int
        Number of simulated GW events.
    n_hosts : int
        Number of simulated exoplanet host stars.
    seed : int
        Random seed for reproducibility.
    fa_threshold : float
        False-alarm probability threshold.

    Returns
    -------
    GWExoplanetCrossmatchResult
        Cross-match result with detailed statistics.
    """
    log.info(
        "Running GW-exoplanet cross-match simulation: "
        "%d events x %d hosts (seed=%d)",
        n_events, n_hosts, seed,
    )

    gw_events = _generate_simulated_gw_events(n_events=n_events, seed=seed)
    hosts = _generate_simulated_hosts(n_hosts=n_hosts, seed=seed + 1)

    return crossmatch_gw_exoplanets(
        gw_events, hosts, fa_threshold=fa_threshold,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("  Project EXODUS -- GW Event / Exoplanet Host Cross-Match")
    print("  ** EXPLORATORY MODULE -- Most Speculative Expansion **")
    print("=" * 72)
    print()
    print("GW sky localisations are typically 10--1000 sq deg, so chance")
    print("coincidence with exoplanet hosts is very common.  This module")
    print("quantifies the expected vs observed coincidence rate.")
    print()

    # ---- Try to load real data first ------------------------------------
    use_simulation = False

    try:
        from src.ingestion.gw_events import get_all_events, GWEvent
        gw_events = get_all_events()
        log.info("Loaded %d GW events", len(gw_events))
    except Exception as exc:
        log.warning("Could not load GW events: %s -- using simulation", exc)
        gw_events = None
        use_simulation = True

    if not use_simulation:
        try:
            from src.ingestion.exoplanet_archive import get_all_hosts
            hosts_df = get_all_hosts()
            hosts = []
            for _, row in hosts_df.iterrows():
                hosts.append({
                    "ra": float(row.get("ra_deg", row.get("ra", 0))),
                    "dec": float(row.get("dec_deg", row.get("dec", 0))),
                    "host_star": str(row.get("host_star", row.get("hostname", "unknown"))),
                })
            log.info("Loaded %d exoplanet hosts", len(hosts))
        except Exception as exc:
            log.warning("Could not load exoplanet hosts: %s -- using simulation", exc)
            use_simulation = True

    # ---- Run cross-match ------------------------------------------------
    if use_simulation:
        print("[SIM] Using simulated data (real catalogs not available)")
        print()
        result = run_simulation(n_events=90, n_hosts=4000, seed=42)
    else:
        print(f"[REAL] Cross-matching {len(gw_events)} GW events against "
              f"{len(hosts)} exoplanet hosts")
        print()
        result = crossmatch_gw_exoplanets(gw_events, hosts)

    # ---- Display results ------------------------------------------------
    print()
    print("-" * 72)
    print("  RESULTS")
    print("-" * 72)
    print(f"  GW events analysed:          {result.n_gw_events}")
    print(f"  Exoplanet hosts:             {result.n_hosts}")
    print(f"  Observed coincidences:       {result.n_coincidences}")
    print(f"  Expected from chance:        {result.n_expected_total:.1f}")
    print(f"  Excess ratio (obs/exp):      {result.excess_ratio:.3f}")
    print(f"  FA threshold:                {result.fa_threshold}")
    print(f"  Low-FA coincidences:         {len(result.low_fa_coincidences)}")
    print()

    # Interpretation
    if result.excess_ratio > 0:
        if 0.5 <= result.excess_ratio <= 2.0:
            print("  Interpretation: Observed coincidence rate is consistent with")
            print("  chance expectation.  No evidence for a physical association")
            print("  between GW events and exoplanet systems.")
        elif result.excess_ratio > 2.0:
            print("  Interpretation: Observed coincidence rate is ABOVE expectation.")
            print("  This is likely due to non-uniform sky coverage or catalog biases,")
            print("  but warrants further investigation.")
        else:
            print("  Interpretation: Observed coincidence rate is BELOW expectation.")
            print("  This may indicate incomplete sky coverage in one of the catalogs.")
    print()

    # Show some example coincidences
    if result.coincidences:
        print("  Example coincidences (first 10):")
        print("  " + "-" * 68)
        for c in result.coincidences[:10]:
            print(f"    {c.gw_event_name:<20s}  <->  {c.host_name:<20s}")
            print(f"      sep={c.angular_separation_deg:.2f} deg  "
                  f"area={c.gw_sky_area_sq_deg:.0f} sq deg  "
                  f"P(chance)={c.p_chance_any:.4f}  "
                  f"N_exp={c.n_expected:.1f}")
        if len(result.coincidences) > 10:
            print(f"    ... and {len(result.coincidences) - 10} more")
        print()

    # Low-FA coincidences
    if result.low_fa_coincidences:
        print("  LOW FALSE-ALARM coincidences (P_chance < {:.2%}):".format(
            result.fa_threshold))
        print("  " + "-" * 68)
        for c in result.low_fa_coincidences:
            print(f"    {c.gw_event_name} <-> {c.host_name}")
            print(f"      area={c.gw_sky_area_sq_deg:.1f} sq deg  "
                  f"P(chance)={c.p_chance_any:.6f}")
        print()
        print("  NOTE: Even low-FA coincidences are expected to occur in a")
        print("  large enough sample.  These are flagged for completeness,")
        print("  not as detections.")
    else:
        print("  No coincidences below the FA threshold (which is expected")
        print("  given the large GW sky areas).")

    print()
    print("  " + "=" * 68)
    print("  DISCLAIMER: This is an EXPLORATORY analysis.  Gravitational-wave")
    print("  events have no known physical connection to exoplanet systems.")
    print("  This module exists to quantify the chance-coincidence baseline")
    print("  and to flag any statistically unusual overlaps for curiosity.")
    print("  " + "=" * 68)
    print()
    print("Done.")
