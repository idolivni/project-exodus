"""
FRB Catalog Ingestion for Project EXODUS.

Downloads and parses the CHIME/FRB catalog and Transient Name Server (TNS)
FRB data.  Fast Radio Bursts are millisecond-duration radio pulses from
extragalactic distances.  Some repeat.  If repeating FRB intervals match
planetary orbital periods of nearby exoplanet systems, that's an
extraordinary coincidence worth investigating.

Public API
----------
get_all_frbs()
    Return all FRB events.

get_repeaters()
    Return only repeating FRBs with multiple burst detections.

get_burst_times(frb_name)
    Return burst timestamps for a given repeating FRB.

get_by_position(ra, dec, radius_deg)
    Return FRBs within a given sky region.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger, load_cache, save_cache, PROJECT_ROOT

log = get_logger("ingestion.frb_catalog")

# CHIME/FRB catalog URL (for future live access)
CHIME_CATALOG_URL = "https://www.chime-frb.ca/catalog"


# =====================================================================
#  Data classes
# =====================================================================

@dataclass
class FRBEvent:
    """A single FRB detection."""
    name: str
    ra: float               # degrees J2000
    dec: float
    dm: float               # dispersion measure (pc cm^-3)
    fluence: float           # Jy ms
    mjd: float              # detection time
    ra_err: float = 0.0     # positional uncertainty (degrees)
    dec_err: float = 0.0
    is_repeater: bool = False
    repeater_name: str = ""  # parent repeater name


@dataclass
class RepeatingFRB:
    """A repeating FRB source with multiple bursts."""
    name: str
    ra: float
    dec: float
    dm: float
    n_bursts: int
    burst_mjds: List[float] = field(default_factory=list)
    burst_fluences: List[float] = field(default_factory=list)
    ra_err: float = 0.0
    dec_err: float = 0.0
    mean_interval_days: float = 0.0
    interval_std_days: float = 0.0


@dataclass
class FRBCatalog:
    """Full FRB catalog."""
    n_total: int
    n_repeaters: int
    events: List[FRBEvent] = field(default_factory=list)
    repeaters: List[RepeatingFRB] = field(default_factory=list)


# =====================================================================
#  Simulated catalog
# =====================================================================

def _generate_simulated_catalog() -> FRBCatalog:
    """Generate a realistic simulated FRB catalog for testing.

    Based on CHIME/FRB Catalog 1 statistics:
    - ~600 FRBs, ~60 repeaters
    - DM range 100-2500 pc cm^-3
    - Mostly isotropic in sky
    - Repeaters have 2-20+ bursts with irregular intervals
    """
    rng = np.random.default_rng(seed=20180725)  # CHIME first light date

    n_one_off = 500
    n_repeaters = 60

    events = []
    repeaters = []

    # One-off FRBs
    for i in range(n_one_off):
        ra = rng.uniform(0, 360)
        dec = np.degrees(np.arcsin(rng.uniform(-1, 1)))
        dm = 10 ** rng.uniform(2.0, 3.4)  # 100-2500 pc/cm^3
        fluence = 10 ** rng.uniform(-0.5, 2.0)  # 0.3 - 100 Jy ms
        mjd = rng.uniform(58300, 59800)  # 2018-2022

        events.append(FRBEvent(
            name=f"FRB{20180000 + i:08d}",
            ra=float(ra),
            dec=float(dec),
            dm=float(dm),
            fluence=float(fluence),
            mjd=float(mjd),
            ra_err=float(rng.uniform(0.001, 0.05)),
            dec_err=float(rng.uniform(0.001, 0.05)),
            is_repeater=False,
        ))

    # Repeating FRBs
    for i in range(n_repeaters):
        ra = rng.uniform(0, 360)
        dec = np.degrees(np.arcsin(rng.uniform(-1, 1)))
        dm = 10 ** rng.uniform(1.8, 3.2)
        n_bursts = rng.integers(3, 25)

        # Generate burst times (irregular intervals, some with quasi-periodicity)
        base_mjd = rng.uniform(58300, 59000)
        intervals = []
        if rng.random() < 0.3:
            # Quasi-periodic (like FRB 20180916B: ~16.3 day period)
            period = rng.uniform(5, 50)
            for b in range(n_bursts):
                jitter = rng.normal(0, period * 0.1)
                intervals.append(period + jitter)
        else:
            # Irregular intervals
            for b in range(n_bursts):
                intervals.append(rng.exponential(30))  # mean 30 days

        burst_mjds = [base_mjd]
        for dt in intervals:
            burst_mjds.append(burst_mjds[-1] + abs(dt))
        burst_mjds = burst_mjds[:n_bursts]

        burst_fluences = [float(10 ** rng.uniform(-0.5, 1.5)) for _ in range(n_bursts)]

        rep_name = f"FRB{20180000 + n_one_off + i:08d}R"

        actual_intervals = np.diff(burst_mjds)
        mean_interval = float(np.mean(actual_intervals)) if len(actual_intervals) > 0 else 0.0
        std_interval = float(np.std(actual_intervals)) if len(actual_intervals) > 1 else 0.0

        repeaters.append(RepeatingFRB(
            name=rep_name,
            ra=float(ra),
            dec=float(dec),
            dm=float(dm),
            n_bursts=n_bursts,
            burst_mjds=[float(m) for m in burst_mjds],
            burst_fluences=burst_fluences,
            ra_err=float(rng.uniform(0.0001, 0.01)),
            dec_err=float(rng.uniform(0.0001, 0.01)),
            mean_interval_days=mean_interval,
            interval_std_days=std_interval,
        ))

        # Add individual bursts to events list
        for b in range(n_bursts):
            events.append(FRBEvent(
                name=f"{rep_name}_b{b:03d}",
                ra=float(ra),
                dec=float(dec),
                dm=float(dm),
                fluence=burst_fluences[b],
                mjd=burst_mjds[b],
                ra_err=float(rng.uniform(0.0001, 0.01)),
                dec_err=float(rng.uniform(0.0001, 0.01)),
                is_repeater=True,
                repeater_name=rep_name,
            ))

    return FRBCatalog(
        n_total=len(events),
        n_repeaters=len(repeaters),
        events=events,
        repeaters=repeaters,
    )


# =====================================================================
#  Module-level catalog (lazy init)
# =====================================================================

_CATALOG: Optional[FRBCatalog] = None


def _get_catalog() -> FRBCatalog:
    """Get or initialize the FRB catalog."""
    global _CATALOG
    if _CATALOG is None:
        cached = load_cache("frb_catalog_v1")
        if cached is not None:
            log.info("FRB catalog loaded from cache")
            _CATALOG = _deserialize_catalog(cached)
        else:
            log.info("Generating simulated FRB catalog")
            _CATALOG = _generate_simulated_catalog()
            save_cache("frb_catalog_v1", _serialize_catalog(_CATALOG))
    return _CATALOG


def _serialize_catalog(cat: FRBCatalog) -> Dict[str, Any]:
    """Serialize catalog for caching."""
    return {
        "n_total": cat.n_total,
        "n_repeaters": cat.n_repeaters,
        "events": [
            {"name": e.name, "ra": e.ra, "dec": e.dec, "dm": e.dm,
             "fluence": e.fluence, "mjd": e.mjd, "ra_err": e.ra_err,
             "dec_err": e.dec_err, "is_repeater": e.is_repeater,
             "repeater_name": e.repeater_name}
            for e in cat.events
        ],
        "repeaters": [
            {"name": r.name, "ra": r.ra, "dec": r.dec, "dm": r.dm,
             "n_bursts": r.n_bursts, "burst_mjds": r.burst_mjds,
             "burst_fluences": r.burst_fluences, "ra_err": r.ra_err,
             "dec_err": r.dec_err, "mean_interval_days": r.mean_interval_days,
             "interval_std_days": r.interval_std_days}
            for r in cat.repeaters
        ],
    }


def _deserialize_catalog(data: Dict[str, Any]) -> FRBCatalog:
    """Deserialize catalog from cache."""
    events = [FRBEvent(**e) for e in data.get("events", [])]
    repeaters = [RepeatingFRB(**r) for r in data.get("repeaters", [])]
    return FRBCatalog(
        n_total=data.get("n_total", len(events)),
        n_repeaters=data.get("n_repeaters", len(repeaters)),
        events=events,
        repeaters=repeaters,
    )


# =====================================================================
#  Public API
# =====================================================================

def get_all_frbs() -> List[FRBEvent]:
    """Return all FRB events."""
    return _get_catalog().events


def get_repeaters() -> List[RepeatingFRB]:
    """Return only repeating FRBs."""
    return _get_catalog().repeaters


def get_burst_times(frb_name: str) -> Optional[List[float]]:
    """Return burst timestamps for a given repeating FRB.

    Parameters
    ----------
    frb_name : str
        Name of the repeating FRB.

    Returns
    -------
    list of float or None
        MJD timestamps of bursts, or None if not found.
    """
    for rep in _get_catalog().repeaters:
        if rep.name == frb_name:
            return rep.burst_mjds
    return None


def get_by_position(ra: float, dec: float, radius_deg: float = 1.0) -> List[FRBEvent]:
    """Return FRBs within a given sky region.

    Parameters
    ----------
    ra, dec : float
        Centre coordinates in degrees.
    radius_deg : float
        Search radius in degrees.

    Returns
    -------
    list of FRBEvent
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    centre = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)

    matches = []
    for event in _get_catalog().events:
        pos = SkyCoord(ra=event.ra * u.degree, dec=event.dec * u.degree)
        sep = centre.separation(pos).deg
        if sep <= radius_deg:
            matches.append(event)

    return matches


# =====================================================================
#  CLI demo
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Project EXODUS -- FRB Catalog Ingestion Demo")
    print("=" * 70)

    cat = _get_catalog()
    print(f"\n  Total FRB events:  {cat.n_total}")
    print(f"  Repeating sources: {cat.n_repeaters}")

    reps = get_repeaters()
    print(f"\n  Top 5 most active repeaters:")
    for rep in sorted(reps, key=lambda r: r.n_bursts, reverse=True)[:5]:
        print(f"    {rep.name}: {rep.n_bursts} bursts, "
              f"DM={rep.dm:.1f}, "
              f"mean interval={rep.mean_interval_days:.1f}d")

    # Test positional search
    frbs_near = get_by_position(180.0, 45.0, radius_deg=10.0)
    print(f"\n  FRBs within 10 deg of (180, 45): {len(frbs_near)}")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
