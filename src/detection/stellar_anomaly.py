"""
Stellar Anomaly Detector for Project EXODUS.

A non-anthropocentric approach: instead of looking for radio or IR signals,
we look for STARS THAT VIOLATE STELLAR PHYSICS.  A civilization that modifies
its star (adding/removing mass, altering fusion, partially occluding it) would
place that star in the "wrong" position on the Hertzsprung-Russell (HR) diagram.

Detection modes:
  a) Too luminous for temperature  → mass added?
  b) Too dim for temperature       → mass removed or partially occluded?
  c) Wrong color for luminosity    → unusual chemical composition?
  d) Unexpected kinematics         → stellar engine / acceleration?

Uses Gaia DR3 photometry (G, BP, RP) and astrometry.  The HR diagram
position is compared against standard isochrone models (simplified here
as polynomial fits to the main sequence, giant branch, etc.).

Public API
----------
detect_hr_anomaly(target)
    Detect anomalous HR diagram position for a single target.

batch_detect(targets)
    Run HR anomaly detection on a list of targets.

compute_hr_position(bp_rp, abs_g)
    Compute expected position and deviation from main sequence.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger

log = get_logger("detection.stellar_anomaly")


# =====================================================================
#  Main-Sequence Polynomial Model
# =====================================================================
# Simplified main-sequence relation: M_G as a function of BP-RP color.
# Coefficients fit to Gaia DR3 main-sequence locus for solar-neighbourhood
# stars (Gaia Collaboration, 2022).  This is a 4th-order polynomial that
# captures the main sequence from O-type (BP-RP ~ -0.3) to late M-type
# (BP-RP ~ 4.0).
#
#   M_G(BP-RP) = c0 + c1*(BP-RP) + c2*(BP-RP)^2 + c3*(BP-RP)^3 + c4*(BP-RP)^4
#
# The scatter about this relation is ~0.3-0.5 mag for single main-sequence
# stars.  Anything deviating by >3 sigma is anomalous.

_MS_COEFFS = np.array([
    1.0,      # c0: intercept (approx M_G at BP-RP=0)
    4.2,      # c1: slope
    -0.6,     # c2: curvature
    0.15,     # c3
    -0.01,    # c4
])

# Intrinsic scatter of the main sequence (magnitudes)
# Varies with color; wider at red end due to metallicity spread
_MS_SCATTER_BASE = 0.4  # base scatter in mag
_MS_SCATTER_SLOPE = 0.1  # additional scatter per unit BP-RP


def _main_sequence_model(bp_rp: np.ndarray) -> np.ndarray:
    """Predicted absolute G magnitude for a main-sequence star at given color."""
    bp_rp = np.asarray(bp_rp, dtype=np.float64)
    return np.polyval(_MS_COEFFS[::-1], bp_rp)


def _main_sequence_scatter(bp_rp: np.ndarray) -> np.ndarray:
    """Expected 1-sigma scatter of the main sequence at given color."""
    bp_rp = np.asarray(bp_rp, dtype=np.float64)
    return _MS_SCATTER_BASE + _MS_SCATTER_SLOPE * np.abs(bp_rp)


# Giant branch locus (rough): M_G ~ -1 to +2 for BP-RP > 1.0
def _is_giant_branch(bp_rp: float, abs_g: float) -> bool:
    """Check if a star sits on the giant branch."""
    if bp_rp < 0.8:
        return False
    expected_giant = -2.0 + 2.5 * bp_rp  # rough giant branch locus
    return abs_g < expected_giant + 1.0


# =====================================================================
#  Data classes
# =====================================================================

@dataclass
class StellarAnomalyResult:
    """Result from the stellar anomaly detector."""
    source_id: str
    bp_rp: float               # Gaia BP-RP color index
    abs_g: float               # Absolute G magnitude
    expected_abs_g: float      # Expected M_G from MS model
    deviation_mag: float       # abs_g - expected (negative = too bright)
    deviation_sigma: float     # deviation / scatter
    is_anomalous: bool
    anomaly_type: str          # "too_bright", "too_dim", "wrong_color", "normal"
    is_giant: bool
    ruwe: float = 0.0
    proper_motion_anomaly: float = 0.0
    interpretation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "bp_rp": self.bp_rp,
            "abs_g": self.abs_g,
            "expected_abs_g": self.expected_abs_g,
            "deviation_mag": self.deviation_mag,
            "deviation_sigma": self.deviation_sigma,
            "is_anomalous": self.is_anomalous,
            "anomaly_type": self.anomaly_type,
            "is_giant": self.is_giant,
            "ruwe": self.ruwe,
            "interpretation": self.interpretation,
        }


# =====================================================================
#  Public API
# =====================================================================

def detect_hr_anomaly(
    target: Dict[str, Any],
    sigma_threshold: float = 3.0,
) -> StellarAnomalyResult:
    """Detect anomalous HR diagram position for a single target.

    Parameters
    ----------
    target : dict
        Must contain at minimum:
          - source_id : str
          - bp_rp : float (Gaia BP - RP color)
          - abs_g : float (absolute G magnitude, or apparent_g + distance_modulus)
        Optional:
          - apparent_g, parallax_mas (to compute abs_g if not provided)
          - ruwe : float (Gaia RUWE astrometric quality)
          - pmra, pmdec : float (proper motion in mas/yr)
    sigma_threshold : float
        Deviation threshold in sigma units.

    Returns
    -------
    StellarAnomalyResult
    """
    source_id = str(target.get("source_id", "unknown"))

    # Get BP-RP color
    bp_rp = target.get("bp_rp")
    if bp_rp is None:
        bp = target.get("BP") or target.get("phot_bp_mean_mag")
        rp = target.get("RP") or target.get("phot_rp_mean_mag")
        if bp is not None and rp is not None:
            bp_rp = float(bp) - float(rp)
        else:
            log.warning("No BP-RP color for %s; cannot compute HR anomaly", source_id)
            return _empty_result(source_id)

    bp_rp = float(bp_rp)

    # Get absolute G magnitude
    abs_g = target.get("abs_g")
    if abs_g is None:
        apparent_g = target.get("G") or target.get("phot_g_mean_mag")
        parallax = target.get("parallax_mas") or target.get("parallax")
        if apparent_g is not None and parallax is not None:
            parallax = float(parallax)
            if parallax > 0:
                dist_pc = 1000.0 / parallax
                abs_g = float(apparent_g) - 5.0 * np.log10(dist_pc) + 5.0
            else:
                log.warning("Non-positive parallax for %s", source_id)
                return _empty_result(source_id)
        else:
            log.warning("Cannot compute absolute magnitude for %s", source_id)
            return _empty_result(source_id)

    abs_g = float(abs_g)

    # Validate color range
    if bp_rp < -0.5 or bp_rp > 5.0:
        log.info("BP-RP=%.2f outside model range for %s", bp_rp, source_id)
        return _empty_result(source_id)

    # Compute expected MS position
    expected_g = float(_main_sequence_model(bp_rp))
    scatter = float(_main_sequence_scatter(bp_rp))

    deviation_mag = abs_g - expected_g  # negative = brighter than expected
    deviation_sigma = abs(deviation_mag) / scatter if scatter > 0 else 0.0

    is_giant = _is_giant_branch(bp_rp, abs_g)
    ruwe = float(target.get("ruwe") or 0.0)

    # Determine anomaly type
    is_anomalous = deviation_sigma > sigma_threshold and not is_giant
    anomaly_type = "normal"

    if is_anomalous:
        if deviation_mag < 0:
            anomaly_type = "too_bright"
        else:
            anomaly_type = "too_dim"

        # Special case: "wrong color" — if the star has large RUWE and
        # large deviation, it might be an unresolved binary or have
        # unusual composition
        if ruwe > 1.4 and deviation_sigma > 5.0:
            anomaly_type = "wrong_color"

    # Build interpretation
    if is_giant:
        interpretation = (
            f"Star lies on giant branch (BP-RP={bp_rp:.2f}, M_G={abs_g:.2f}). "
            f"Deviation from MS model not meaningful."
        )
    elif not is_anomalous:
        interpretation = (
            f"Normal main-sequence star. Deviation {deviation_sigma:.1f}sigma "
            f"from MS model (threshold={sigma_threshold}sigma)."
        )
    elif anomaly_type == "too_bright":
        interpretation = (
            f"ANOMALY: Star is {abs(deviation_mag):.2f} mag BRIGHTER than "
            f"expected for its color ({deviation_sigma:.1f}sigma). "
            f"Possible: blue straggler, mass accretion, unresolved binary, "
            f"or stellar engineering (mass addition)."
        )
    elif anomaly_type == "too_dim":
        interpretation = (
            f"ANOMALY: Star is {deviation_mag:.2f} mag DIMMER than expected "
            f"for its color ({deviation_sigma:.1f}sigma). "
            f"Possible: subdwarf, partial occultation by circumstellar structure, "
            f"or stellar engineering (mass removal)."
        )
    elif anomaly_type == "wrong_color":
        interpretation = (
            f"ANOMALY: Star has WRONG COLOR for its luminosity "
            f"({deviation_sigma:.1f}sigma deviation, RUWE={ruwe:.1f}). "
            f"Possible: unusual chemical composition, unresolved companion, "
            f"or modified stellar atmosphere."
        )
    else:
        interpretation = f"Deviation: {deviation_sigma:.1f}sigma, type: {anomaly_type}"

    result = StellarAnomalyResult(
        source_id=source_id,
        bp_rp=bp_rp,
        abs_g=abs_g,
        expected_abs_g=expected_g,
        deviation_mag=deviation_mag,
        deviation_sigma=deviation_sigma,
        is_anomalous=is_anomalous,
        anomaly_type=anomaly_type,
        is_giant=is_giant,
        ruwe=ruwe,
        interpretation=interpretation,
    )

    if is_anomalous:
        log.info(
            "HR ANOMALY: %s  BP-RP=%.2f  M_G=%.2f  expected=%.2f  "
            "dev=%.1fsigma  type=%s",
            source_id, bp_rp, abs_g, expected_g, deviation_sigma, anomaly_type,
        )

    return result


def batch_detect(
    targets: List[Dict[str, Any]],
    sigma_threshold: float = 3.0,
) -> List[StellarAnomalyResult]:
    """Run HR anomaly detection on a list of targets.

    Parameters
    ----------
    targets : list of dict
        Each dict follows the same format as for detect_hr_anomaly.
    sigma_threshold : float
        Deviation threshold.

    Returns
    -------
    list of StellarAnomalyResult
        Sorted by deviation_sigma (most anomalous first).
    """
    results = []
    for t in targets:
        result = detect_hr_anomaly(t, sigma_threshold)
        results.append(result)

    results.sort(key=lambda r: r.deviation_sigma, reverse=True)

    n_anomalous = sum(1 for r in results if r.is_anomalous)
    log.info(
        "Batch HR anomaly detection: %d targets, %d anomalous (%.1f%%)",
        len(results), n_anomalous,
        100.0 * n_anomalous / max(len(results), 1),
    )
    return results


def compute_hr_position(
    bp_rp: float,
    abs_g: float,
) -> Dict[str, float]:
    """Compute expected HR position and deviation from main sequence.

    Parameters
    ----------
    bp_rp : float
        Gaia BP-RP color.
    abs_g : float
        Absolute G magnitude.

    Returns
    -------
    dict
        expected_abs_g, deviation_mag, deviation_sigma, is_giant
    """
    expected = float(_main_sequence_model(bp_rp))
    scatter = float(_main_sequence_scatter(bp_rp))
    deviation = abs_g - expected
    sigma = abs(deviation) / scatter if scatter > 0 else 0.0

    return {
        "expected_abs_g": expected,
        "deviation_mag": deviation,
        "deviation_sigma": sigma,
        "is_giant": _is_giant_branch(bp_rp, abs_g),
    }


def _empty_result(source_id: str) -> StellarAnomalyResult:
    """Return an empty/default result when analysis fails."""
    return StellarAnomalyResult(
        source_id=source_id,
        bp_rp=0.0,
        abs_g=0.0,
        expected_abs_g=0.0,
        deviation_mag=0.0,
        deviation_sigma=0.0,
        is_anomalous=False,
        anomaly_type="insufficient_data",
        is_giant=False,
        interpretation="Insufficient data for HR anomaly analysis",
    )


# =====================================================================
#  WISE-Gaia Proper Motion Consistency Check
# =====================================================================

def _catwise_systematic_floor(
    pm_total_gaia: float,
    phot_g_mean_mag: Optional[float] = None,
) -> float:
    """Compute magnitude- and PM-dependent CatWISE systematic floor.

    CatWISE2020 PM errors grow significantly for:
    - Faint sources (low SNR in individual WISE frames)
    - High-PM sources (source moves between epochs, causing confusion
      in the CatWISE pipeline which assumes near-stationary sources)

    Based on empirical analysis of EXODUS-500 results (2026-02-26) and
    CatWISE2020 characterization (Eisenhardt+ 2020, Marocco+ 2021).

    Parameters
    ----------
    pm_total_gaia : float
        Total Gaia proper motion (mas/yr).
    phot_g_mean_mag : float or None
        Gaia G-band magnitude. If None, only PM-dependent floor is used.

    Returns
    -------
    float
        CatWISE systematic floor in mas/yr.
    """
    # Audit fix F3: raised from 2.0 to 3.0 per Marocco et al. (2021)
    # CatWISE systematic floor for bright stars is 3-5 mas/yr, not 2.
    base = 3.0  # mas/yr floor for bright (G<15), slow (PM<40) stars

    # Magnitude-dependent component: CatWISE errors grow exponentially
    # for faint sources. Factor of ~2.5x per magnitude fainter than G=15.
    mag_floor = base
    if phot_g_mean_mag is not None and phot_g_mean_mag > 15.0:
        mag_floor = base * 10 ** (0.3 * (phot_g_mean_mag - 15.0))
        # G=16: 4.0, G=17: 8.0, G=18: 15.8, G=19: 31.6

    # PM-dependent component: 5% of total Gaia PM as systematic floor.
    # High-PM stars move significantly between WISE epochs (6" PSF),
    # causing source confusion and PM measurement errors in CatWISE.
    pm_floor = 0.05 * pm_total_gaia
    # PM=40: 2.0, PM=100: 5.0, PM=200: 10.0, PM=500: 25.0

    return max(base, mag_floor, pm_floor)


def compute_pm_consistency(
    pmra_gaia: float,
    pmdec_gaia: float,
    pmra_err_gaia: float,
    pmdec_err_gaia: float,
    pmra_wise: float,
    pmdec_wise: float,
    pmra_err_wise: float,
    pmdec_err_wise: float,
    phot_g_mean_mag: Optional[float] = None,
) -> Dict[str, Any]:
    """Compare Gaia and CatWISE2020 proper motion measurements.

    Gaia DR3 (2014-2017 baseline) and CatWISE2020 (2010-2020 WISE+NEOWISE)
    measure proper motions independently.  For isolated stars these should
    agree within uncertainties.  A significant discrepancy can indicate:

    - Unresolved companion causing photocentre wobble (different wavelength
      → different photocentre)
    - Non-linear motion (acceleration from an unseen body)
    - Confusion/blending in the WISE PSF (6" vs Gaia's 0.1")

    Parameters
    ----------
    pmra_gaia, pmdec_gaia : float
        Gaia proper motion (mas/yr).  pmra includes cos(dec) factor.
    pmra_err_gaia, pmdec_err_gaia : float
        Gaia formal errors (mas/yr).
    pmra_wise, pmdec_wise : float
        CatWISE2020 proper motion (mas/yr).
    pmra_err_wise, pmdec_err_wise : float
        CatWISE2020 formal errors (mas/yr).
    phot_g_mean_mag : float or None
        Gaia G-band apparent magnitude. Used for magnitude-dependent
        CatWISE systematic floor. If None, only PM-dependent floor is used.

    Returns
    -------
    dict with keys:
        delta_pmra, delta_pmdec : float
            PM difference (Gaia − WISE) in mas/yr.
        delta_pmra_sigma, delta_pmdec_sigma : float
            Significance of each component.
        chi2 : float
            Combined chi-squared (2 dof).
        pm_discrepancy_sigma : float
            Equivalent sigma for the combined discrepancy.
        is_discrepant : bool
            True if chi2 > 13.82 (p < 0.001 for 2 dof).
        catwise_systematic_flag : bool
            True if discrepancy is likely a CatWISE systematic artifact
            (PM offset > 30% of total PM for fast-moving stars).
        wise_sys_floor : float
            The CatWISE systematic floor used (mas/yr).
        interpretation : str
    """
    # Gaia systematic floor (constant — Gaia PM is exquisite)
    _GAIA_SYS = 0.02  # mas/yr

    # CatWISE systematic floor: magnitude- and PM-dependent
    pm_total_gaia = np.sqrt(pmra_gaia ** 2 + pmdec_gaia ** 2)
    _WISE_SYS = _catwise_systematic_floor(pm_total_gaia, phot_g_mean_mag)

    sigma_ra = np.sqrt(
        max(pmra_err_gaia, _GAIA_SYS) ** 2
        + max(pmra_err_wise, _WISE_SYS) ** 2
    )
    sigma_dec = np.sqrt(
        max(pmdec_err_gaia, _GAIA_SYS) ** 2
        + max(pmdec_err_wise, _WISE_SYS) ** 2
    )

    delta_ra = pmra_gaia - pmra_wise
    delta_dec = pmdec_gaia - pmdec_wise

    sig_ra = abs(delta_ra) / sigma_ra if sigma_ra > 0 else 0.0
    sig_dec = abs(delta_dec) / sigma_dec if sigma_dec > 0 else 0.0

    chi2 = sig_ra ** 2 + sig_dec ** 2

    # Convert chi2 (2 dof) to equivalent sigma
    from scipy import stats as sp_stats
    p_value = float(sp_stats.chi2.sf(chi2, df=2))
    if p_value > 1e-300:
        # Use isf (inverse survival function) — avoids 1-p precision loss
        equiv_sigma = float(sp_stats.norm.isf(p_value / 2.0))
        equiv_sigma = max(equiv_sigma, 0.0)
    else:
        # p underflows — use direct approximation for chi2(2 dof):
        # p = exp(-chi2/2), two-tailed sigma ~ sqrt(chi2 + 2*ln(2))
        equiv_sigma = float(np.sqrt(chi2 + 2.0 * np.log(2.0)))
    equiv_sigma = min(equiv_sigma, 50.0)  # cap at 50 sigma

    # Threshold: chi2 > 13.82 corresponds to p < 0.001 (2 dof)
    is_discrepant = chi2 > 13.82

    # Detect CatWISE systematic artifacts:
    # If PM offset is >30% of total Gaia PM for fast-moving stars (>50 mas/yr),
    # this is almost certainly a CatWISE measurement artifact, not a real
    # astrophysical signal. CatWISE struggles with high-PM stars because the
    # 6" PSF blends source positions across the multi-year baseline.
    delta_total = np.sqrt(delta_ra ** 2 + delta_dec ** 2)
    catwise_systematic_flag = False
    if pm_total_gaia > 50.0 and delta_total > 0.30 * pm_total_gaia:
        catwise_systematic_flag = True
        is_discrepant = False  # Override: this is a systematic, not a real signal
        chi2 = 0.0
        equiv_sigma = 0.0
        p_value = 1.0

    if catwise_systematic_flag:
        interpretation = (
            f"CATWISE SYSTEMATIC: PM offset "
            f"({delta_ra:+.1f}, {delta_dec:+.1f}) mas/yr is "
            f"{delta_total:.1f}/{pm_total_gaia:.1f} mas/yr "
            f"({100*delta_total/pm_total_gaia:.0f}% of total PM). "
            f"CatWISE unreliable for high-PM stars. Discrepancy suppressed."
        )
    elif is_discrepant:
        interpretation = (
            f"PM DISCREPANCY: Gaia-WISE offset "
            f"({delta_ra:+.1f}, {delta_dec:+.1f}) mas/yr "
            f"({equiv_sigma:.1f}sigma, floor={_WISE_SYS:.1f}). "
            f"Possible wavelength-dependent photocentre shift "
            f"(unresolved companion) or non-linear motion."
        )
    else:
        interpretation = (
            f"PM consistent: Gaia-WISE offset "
            f"({delta_ra:+.1f}, {delta_dec:+.1f}) mas/yr "
            f"({equiv_sigma:.1f}sigma, floor={_WISE_SYS:.1f})."
        )

    return {
        "delta_pmra": float(delta_ra),
        "delta_pmdec": float(delta_dec),
        "delta_pmra_sigma": float(sig_ra),
        "delta_pmdec_sigma": float(sig_dec),
        "chi2": float(chi2),
        "p_value": float(p_value),
        "pm_discrepancy_sigma": float(equiv_sigma),
        "is_discrepant": is_discrepant,
        "catwise_systematic_flag": catwise_systematic_flag,
        "wise_sys_floor": float(_WISE_SYS),
        "interpretation": interpretation,
    }


# =====================================================================
#  CLI demo / self-test
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Project EXODUS -- Stellar Anomaly Detector Demo")
    print("=" * 70)

    rng = np.random.default_rng(seed=42)

    # ── Test 1: Normal main-sequence stars ────────────────────────────
    print("\n[1] Normal main-sequence stars (should NOT be anomalous)")
    print("-" * 50)

    normal_stars = []
    bp_rps = [0.5, 0.8, 1.0, 1.3, 1.8, 2.5, 3.0]
    for i, bprp in enumerate(bp_rps):
        expected = float(_main_sequence_model(bprp))
        scatter = float(_main_sequence_scatter(bprp))
        abs_g = expected + rng.normal(0, scatter * 0.5)
        normal_stars.append({
            "source_id": f"NORMAL_{i:03d}",
            "bp_rp": bprp,
            "abs_g": float(abs_g),
            "ruwe": float(rng.uniform(0.9, 1.2)),
        })

    normal_results = batch_detect(normal_stars)
    n_false_pos = sum(1 for r in normal_results if r.is_anomalous)
    print(f"  Normal stars: {len(normal_stars)}")
    print(f"  False positives: {n_false_pos}")
    for r in normal_results[:3]:
        print(f"    {r.source_id}: BP-RP={r.bp_rp:.2f} M_G={r.abs_g:.2f} "
              f"dev={r.deviation_sigma:.1f}sigma {r.anomaly_type}")

    # ── Test 2: Blue stragglers (too bright for their color) ──────────
    print("\n[2] Blue stragglers (too bright for their color)")
    print("-" * 50)

    blue_stragglers = []
    for i in range(5):
        bprp = 0.5 + rng.uniform(-0.1, 0.3)
        expected = float(_main_sequence_model(bprp))
        # 2-3 magnitudes brighter than MS
        abs_g = expected - 2.5 - rng.uniform(0, 1)
        blue_stragglers.append({
            "source_id": f"BLUE_STRAGGLER_{i:03d}",
            "bp_rp": float(bprp),
            "abs_g": float(abs_g),
            "ruwe": float(rng.uniform(1.0, 1.5)),
        })

    bs_results = batch_detect(blue_stragglers)
    n_detected = sum(1 for r in bs_results if r.is_anomalous)
    print(f"  Blue stragglers: {len(blue_stragglers)}")
    print(f"  Detected as anomalous: {n_detected}")
    for r in bs_results[:3]:
        print(f"    {r.source_id}: BP-RP={r.bp_rp:.2f} M_G={r.abs_g:.2f} "
              f"dev={r.deviation_sigma:.1f}sigma {r.anomaly_type}")
        print(f"      {r.interpretation[:80]}...")

    # ── Test 3: Subdwarfs (too dim for their color) ───────────────────
    print("\n[3] Subdwarfs (too dim for their color)")
    print("-" * 50)

    subdwarfs = []
    for i in range(5):
        bprp = 1.0 + rng.uniform(0, 1.5)
        expected = float(_main_sequence_model(bprp))
        # 2-4 mag dimmer than MS
        abs_g = expected + 3.0 + rng.uniform(0, 1)
        subdwarfs.append({
            "source_id": f"SUBDWARF_{i:03d}",
            "bp_rp": float(bprp),
            "abs_g": float(abs_g),
            "ruwe": float(rng.uniform(0.9, 1.3)),
        })

    sd_results = batch_detect(subdwarfs)
    n_detected_sd = sum(1 for r in sd_results if r.is_anomalous)
    print(f"  Subdwarfs: {len(subdwarfs)}")
    print(f"  Detected as anomalous: {n_detected_sd}")
    for r in sd_results[:3]:
        print(f"    {r.source_id}: BP-RP={r.bp_rp:.2f} M_G={r.abs_g:.2f} "
              f"dev={r.deviation_sigma:.1f}sigma {r.anomaly_type}")

    # ── Test 4: Red giants (should NOT be flagged) ────────────────────
    print("\n[4] Red giants (on giant branch, should NOT be anomalous)")
    print("-" * 50)

    giants = []
    for i in range(5):
        bprp = 1.2 + rng.uniform(0, 1.5)
        abs_g = -1.0 + rng.uniform(-1, 2)  # giant branch luminosities
        giants.append({
            "source_id": f"GIANT_{i:03d}",
            "bp_rp": float(bprp),
            "abs_g": float(abs_g),
        })

    giant_results = batch_detect(giants)
    n_giant_fp = sum(1 for r in giant_results if r.is_anomalous)
    print(f"  Giants: {len(giants)}")
    print(f"  False positives: {n_giant_fp}")
    for r in giant_results[:3]:
        print(f"    {r.source_id}: BP-RP={r.bp_rp:.2f} M_G={r.abs_g:.2f} "
              f"giant={r.is_giant} anom={r.is_anomalous}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  Normal MS false pos:    {n_false_pos}/{len(normal_stars)} "
          f"({'PASS' if n_false_pos == 0 else 'CHECK'})")
    print(f"  Blue stragglers found:  {n_detected}/{len(blue_stragglers)} "
          f"({'PASS' if n_detected == len(blue_stragglers) else 'CHECK'})")
    print(f"  Subdwarfs found:        {n_detected_sd}/{len(subdwarfs)} "
          f"({'PASS' if n_detected_sd == len(subdwarfs) else 'CHECK'})")
    print(f"  Giant false pos:        {n_giant_fp}/{len(giants)} "
          f"({'PASS' if n_giant_fp == 0 else 'CHECK'})")
    print("=" * 70)
    print("  Done.")
    print("=" * 70)
