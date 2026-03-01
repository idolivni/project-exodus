"""
Project EXODUS — HR Diagram Outlier Detection Channel
======================================================

Flags stars that don't belong where they are on the HR diagram.
Uses Gaia DR3 effective temperature (teff_gspphot) and absolute G
magnitude to identify statistical outliers from the main sequence,
giant branch, and white dwarf cooling sequence.

Why SETI: stellar engineering (mass extraction, luminosity alteration)
would shift a star's position on the HR diagram. Combined with the
binary veto (RUWE < 1.4), an HR outlier becomes genuinely unexplained.

Natural false positives:
  - Unresolved binaries (shifts above MS) → mitigated by RUWE check
  - Metal-poor subdwarfs (shifts below MS) → identifiable by [M/H]
  - Young pre-MS stars (shifts above MS) → identifiable by kinematics
  - Interstellar reddening (shifts right) → mitigated by dust map

This channel uses ONLY pre-existing Gaia data (no new queries needed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


# ── Main sequence ridge line (empirical, Gaia DR3) ────────────────
# Points are (bp_rp, M_G) from the Gaia DR3 CMD
# This is a simplified piecewise-linear approximation
_MS_RIDGE = [
    (-0.3, -1.0),   # O/B stars
    (0.0, 0.5),     # A stars
    (0.3, 2.0),     # F stars
    (0.6, 3.5),     # early G
    (0.8, 4.5),     # late G (Sun)
    (1.0, 5.5),     # early K
    (1.3, 7.0),     # mid K
    (1.5, 8.0),     # late K
    (1.8, 9.5),     # early M
    (2.2, 11.0),    # mid M
    (2.8, 13.0),    # late M
    (3.5, 15.0),    # very late M
    (4.5, 18.0),    # L dwarfs
]

# Main sequence scatter (typical 1σ in M_G at each bp_rp)
_MS_SCATTER = 0.75  # mag


@dataclass
class HRAnomalyResult:
    """Result of HR diagram outlier detection."""

    has_data: bool = False
    data_source: str = "none"

    # Input measurements
    teff: Optional[float] = None
    bp_rp: Optional[float] = None
    abs_g: Optional[float] = None
    logg: Optional[float] = None
    metallicity: Optional[float] = None

    # HR diagram position
    ms_expected_abs_g: Optional[float] = None  # Expected M_G from MS ridge
    ms_residual: float = 0.0                   # M_G - expected (neg = brighter)
    ms_sigma: float = 0.0                      # Residual in units of scatter

    # Classification
    is_above_ms: bool = False     # Brighter than expected (binary, evolved, engineered?)
    is_below_ms: bool = False     # Fainter than expected (subdwarf, stripped?)
    is_white_dwarf_region: bool = False  # In WD locus
    is_giant_region: bool = False        # In giant branch

    # Scoring
    anomaly_score: float = 0.0   # [0, 1]
    is_anomalous: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_data": self.has_data,
            "data_source": self.data_source,
            "teff": self.teff,
            "bp_rp": self.bp_rp,
            "abs_g": round(self.abs_g, 3) if self.abs_g is not None else None,
            "logg": self.logg,
            "metallicity": self.metallicity,
            "ms_expected_abs_g": (
                round(self.ms_expected_abs_g, 3)
                if self.ms_expected_abs_g is not None else None
            ),
            "ms_residual": round(self.ms_residual, 3),
            "ms_sigma": round(self.ms_sigma, 2),
            "is_above_ms": self.is_above_ms,
            "is_below_ms": self.is_below_ms,
            "is_white_dwarf_region": self.is_white_dwarf_region,
            "is_giant_region": self.is_giant_region,
            "anomaly_score": round(self.anomaly_score, 4),
            "is_anomalous": self.is_anomalous,
        }


def _interpolate_ms_ridge(bp_rp: float) -> Optional[float]:
    """Interpolate expected absolute G magnitude from main sequence ridge."""
    if bp_rp < _MS_RIDGE[0][0] or bp_rp > _MS_RIDGE[-1][0]:
        return None

    for i in range(len(_MS_RIDGE) - 1):
        x0, y0 = _MS_RIDGE[i]
        x1, y1 = _MS_RIDGE[i + 1]
        if x0 <= bp_rp <= x1:
            frac = (bp_rp - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)

    return None


def compute_hr_anomaly(
    gaia_params: Optional[Dict[str, Any]],
    astrometry: Optional[Dict[str, Any]] = None,
    distance_pc: Optional[float] = None,
) -> HRAnomalyResult:
    """Compute HR diagram outlier score.

    Parameters
    ----------
    gaia_params : dict or None
        Gaia stellar parameters. Expected keys:
        teff_gspphot, logg_gspphot, mh_gspphot, bp_rp, phot_g_mean_mag
    astrometry : dict or None
        Gaia astrometry. Expected keys: parallax, ruwe
    distance_pc : float or None
        Pre-computed distance. If None, computed from parallax.

    Returns
    -------
    HRAnomalyResult
    """
    result = HRAnomalyResult()

    if not gaia_params:
        return result

    bp_rp = gaia_params.get("bp_rp")
    g_mag = gaia_params.get("phot_g_mean_mag")

    # Need parallax for absolute magnitude
    parallax = None
    if astrometry:
        parallax = astrometry.get("parallax")
    if parallax is None and distance_pc and distance_pc > 0:
        parallax = 1000.0 / distance_pc

    if bp_rp is None or g_mag is None or parallax is None or parallax <= 0:
        return result

    result.has_data = True
    result.data_source = "gaia_dr3"
    result.bp_rp = float(bp_rp)
    result.teff = _safe_float(gaia_params.get("teff_gspphot"))
    result.logg = _safe_float(gaia_params.get("logg_gspphot"))
    result.metallicity = _safe_float(gaia_params.get("mh_gspphot"))

    # Absolute magnitude
    abs_g = float(g_mag) + 5 * np.log10(float(parallax)) - 10
    result.abs_g = abs_g

    # Expected MS position
    expected = _interpolate_ms_ridge(float(bp_rp))
    if expected is None:
        # Outside colour range of our ridge line
        return result

    result.ms_expected_abs_g = expected
    residual = abs_g - expected  # negative = brighter than expected
    result.ms_residual = residual
    result.ms_sigma = residual / _MS_SCATTER

    # Classification
    result.is_above_ms = residual < -1.5 * _MS_SCATTER  # >1.5σ brighter
    result.is_below_ms = residual > 1.5 * _MS_SCATTER    # >1.5σ fainter
    result.is_white_dwarf_region = abs_g > 10 and float(bp_rp) < 1.5
    result.is_giant_region = abs_g < 3.0 and float(bp_rp) > 0.8

    # Scoring
    # An HR outlier is interesting if it deviates significantly from
    # the main sequence AND is not in a known evolutionary region
    # (giant branch, WD cooling sequence).
    sigma = abs(result.ms_sigma)

    if result.is_white_dwarf_region or result.is_giant_region:
        # Stars in expected evolutionary regions are less interesting
        # as HR outliers (though they may be interesting for other reasons)
        score = 0.0
    elif sigma < 1.5:
        # Within normal MS scatter
        score = 0.0
    else:
        # Outlier score: saturating function of sigma beyond 1.5
        excess_sigma = sigma - 1.5
        score = 1.0 - np.exp(-excess_sigma / 2.0)

    # Metallicity correction: metal-poor stars naturally sit below MS
    # If [M/H] < -0.5, reduce below-MS anomaly score
    if result.is_below_ms and result.metallicity is not None and result.metallicity < -0.5:
        score *= 0.5  # Metal-poor subdwarf explanation available

    result.anomaly_score = float(np.clip(score, 0.0, 1.0))
    result.is_anomalous = result.anomaly_score > 0.3

    return result


def _safe_float(val) -> Optional[float]:
    """Convert to float, return None if invalid."""
    if val is None:
        return None
    try:
        import math
        f = float(val)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None
