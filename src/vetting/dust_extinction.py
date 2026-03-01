"""
3D Dust Extinction Queries for Project EXODUS.

Uses the Bayestar2019 3D dust map (Green et al. 2019) to estimate
interstellar reddening E(B-V) along target sightlines.  This helps
distinguish genuine circumstellar IR excess from interstellar reddening:

- For most EXODUS targets (d < 50 pc), interstellar extinction is
  negligible (E(B-V) < 0.01).  The local bubble is mostly dust-free.
- For targets beyond ~100 pc or in the Galactic plane, extinction
  can be significant and must be accounted for.
- Mid-IR (W3/W4) extinction is tiny even at high E(B-V):
      A_W3 ≈ 0.06 * E(B-V)   A_W4 ≈ 0.03 * E(B-V)
  So mid-IR excess is robust against interstellar dust.
- Optical (G, BP, RP) extinction IS significant:
      A_G  ≈ 2.74 * E(B-V)   A_BP ≈ 3.37 * E(B-V)   A_RP ≈ 2.07 * E(B-V)
  This can bias the blackbody Teff fit downward (star appears redder).

Primary use: Red-Team vetting context — flag high-extinction sightlines
where IR excess interpretation requires care.

References
----------
- Green et al. 2019, ApJ 887, 93 (Bayestar2019)
- Fitzpatrick 1999, PASP 111, 63 (R_V=3.1 extinction law)
- Wang & Chen 2019, ApJ 877, 116 (WISE extinction coefficients)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger, PROJECT_ROOT

log = get_logger("vetting.dust_extinction")

# Extinction coefficients A_lambda / E(B-V) for R_V = 3.1
# Sources: Fitzpatrick 1999 (optical/NIR), Wang & Chen 2019 (WISE)
EXTINCTION_COEFFICIENTS: Dict[str, float] = {
    # Gaia
    "G":   2.740,
    "BP":  3.374,
    "RP":  2.035,
    # 2MASS
    "J":   0.723,
    "H":   0.460,
    "Ks":  0.306,
    # WISE
    "W1":  0.180,
    "W2":  0.120,
    "W3":  0.058,
    "W4":  0.030,
}

# Data directory for dust maps
_DUSTMAPS_DATA_DIR = str(PROJECT_ROOT / "data" / "cache" / "dustmaps")

# Module-level Bayestar query object (lazy init)
_BAYESTAR: Any = None
_BAYESTAR_AVAILABLE: Optional[bool] = None


def _get_bayestar():
    """Lazy-init the Bayestar2019 query object."""
    global _BAYESTAR, _BAYESTAR_AVAILABLE

    if _BAYESTAR_AVAILABLE is False:
        return None
    if _BAYESTAR is not None:
        return _BAYESTAR

    try:
        from dustmaps.config import config
        config["data_dir"] = _DUSTMAPS_DATA_DIR

        from dustmaps.bayestar import BayestarQuery
        _BAYESTAR = BayestarQuery(version="bayestar2019")
        _BAYESTAR_AVAILABLE = True
        log.info("Bayestar2019 dust map loaded")
        return _BAYESTAR
    except Exception as exc:
        log.warning("Bayestar2019 not available: %s", exc)
        _BAYESTAR_AVAILABLE = False
        return None


def query_ebv(
    ra: float,
    dec: float,
    distance_pc: float,
) -> Optional[float]:
    """Query E(B-V) reddening at a given position and distance.

    Parameters
    ----------
    ra, dec : float
        Equatorial coordinates in degrees (J2000).
    distance_pc : float
        Distance in parsecs.

    Returns
    -------
    float or None
        E(B-V) reddening in magnitudes, or None if unavailable.
    """
    bq = _get_bayestar()
    if bq is None:
        return None

    if distance_pc <= 0 or not np.isfinite(distance_pc):
        return None

    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(
            ra=ra * u.deg,
            dec=dec * u.deg,
            distance=distance_pc * u.pc,
            frame="icrs",
        )
        ebv = float(bq(coord, mode="median"))
        if np.isnan(ebv):
            return None
        return max(0.0, ebv)
    except Exception as exc:
        log.debug("E(B-V) query failed for (%.4f, %.4f, %.1f pc): %s",
                  ra, dec, distance_pc, exc)
        return None


def compute_extinction(
    ebv: float,
    bands: Optional[list] = None,
) -> Dict[str, float]:
    """Compute extinction A_lambda for given E(B-V).

    Parameters
    ----------
    ebv : float
        E(B-V) reddening.
    bands : list of str, optional
        Band names to compute. Default: all known bands.

    Returns
    -------
    dict
        Maps band name → A_lambda (extinction in magnitudes).
    """
    if bands is None:
        bands = list(EXTINCTION_COEFFICIENTS.keys())

    return {
        band: EXTINCTION_COEFFICIENTS.get(band, 0) * ebv
        for band in bands
    }


def get_extinction_context(
    ra: float,
    dec: float,
    distance_pc: float,
) -> Dict[str, Any]:
    """Get full extinction context for a target sightline.

    Returns a dict with E(B-V), per-band extinctions, and a
    qualitative assessment of whether extinction is a concern.

    Parameters
    ----------
    ra, dec : float
        Equatorial coordinates (degrees, J2000).
    distance_pc : float
        Distance in parsecs.

    Returns
    -------
    dict
        Keys: ebv, extinctions (per-band), concern_level, note.
        If dust map is unavailable, returns minimal dict with note.
    """
    ebv = query_ebv(ra, dec, distance_pc)

    if ebv is None:
        return {
            "ebv": None,
            "source": "bayestar2019",
            "available": False,
            "note": "3D dust map query failed or unavailable",
        }

    extinctions = compute_extinction(ebv)

    # Qualitative assessment
    if ebv < 0.01:
        concern = "NEGLIGIBLE"
        note = f"E(B-V)={ebv:.4f} — negligible extinction, no correction needed"
    elif ebv < 0.05:
        concern = "LOW"
        note = (
            f"E(B-V)={ebv:.4f} — low extinction. "
            f"A_G={extinctions['G']:.3f}, A_W3={extinctions['W3']:.4f} mag"
        )
    elif ebv < 0.3:
        concern = "MODERATE"
        note = (
            f"E(B-V)={ebv:.3f} — moderate extinction. "
            f"Optical photometry affected (A_G={extinctions['G']:.2f} mag), "
            f"mid-IR still reliable (A_W3={extinctions['W3']:.3f} mag). "
            f"Blackbody Teff may be underestimated"
        )
    else:
        concern = "HIGH"
        note = (
            f"E(B-V)={ebv:.2f} — high extinction sightline. "
            f"Optical photometry severely affected (A_G={extinctions['G']:.1f} mag). "
            f"IR excess interpretation requires care: interstellar reddening "
            f"may mimic or mask circumstellar excess"
        )

    return {
        "ebv": round(ebv, 5),
        "source": "bayestar2019",
        "available": True,
        "concern_level": concern,
        "extinctions": {k: round(v, 5) for k, v in extinctions.items()},
        "note": note,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  EXODUS Dust Extinction Module — Demo")
    print("=" * 70)

    # Test a few sightlines
    tests = [
        ("Proxima Cen", 217.39, -62.68, 1.3),
        ("Barnard's Star", 269.45, 4.69, 1.8),
        ("Sun-like at 50pc toward Gal center", 266.40, -28.94, 50.0),
        ("Sun-like at 50pc high lat", 0.0, 60.0, 50.0),
        ("Distant star in plane", 80.0, 0.5, 200.0),
    ]

    for name, ra, dec, dist in tests:
        ctx = get_extinction_context(ra, dec, dist)
        ebv = ctx.get("ebv")
        print(f"\n  {name} ({dist} pc):")
        if ebv is not None:
            print(f"    E(B-V) = {ebv:.4f}  [{ctx.get('concern_level')}]")
            print(f"    {ctx.get('note')}")
        else:
            print(f"    {ctx.get('note')}")

    print("\n" + "=" * 70)
