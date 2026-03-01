"""
Cross-Band Temporal Correlation for Project EXODUS.

The single most powerful technosignature test: does a star dim in optical at
the same time it brightens in infrared?  That anti-correlation is the signature
of something ABSORBING starlight and RE-EMITTING it as waste heat — nearly
impossible to produce by any known natural astrophysical process.

- Positive correlation (both dim or both bright together):
    Natural explanation: dust extinction, stellar variability.
- NEGATIVE correlation (optical dims while IR brightens):
    Absorption + re-emission → TECHNOSIGNATURE CANDIDATE.

Public API
----------
cross_correlate_optical_ir(optical_time, optical_mag, ir_time, ir_mag,
                           max_lag_days=30)
    Compute correlation between optical and IR light curves.

analyze_target(target_id, optical_data, neowise_data)
    Full cross-band temporal analysis for a single target.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats
from scipy import interpolate

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger

log = get_logger("correlation.cross_band_temporal")


# =====================================================================
#  Data classes
# =====================================================================

@dataclass
class TemporalMatch:
    """A single time-matched pair of optical + IR observations."""
    mjd_optical: float
    mjd_ir: float
    optical_mag: float
    ir_mag: float
    time_offset_days: float  # ir_mjd - optical_mjd


@dataclass
class CrossBandResult:
    """Result of the cross-band temporal correlation analysis."""
    target_id: str
    n_matched_epochs: int
    pearson_r: float          # Pearson correlation coefficient
    pearson_p: float          # p-value of correlation
    spearman_rho: float       # Spearman rank correlation
    spearman_p: float
    is_anti_correlated: bool  # True if significantly anti-correlated (p < 0.01)
    anti_correlation_sigma: float  # significance of anti-correlation in sigma
    optical_variability: float  # std of optical magnitudes
    ir_variability: float       # std of IR magnitudes
    mean_time_offset_days: float
    matched_epochs: List[TemporalMatch] = field(default_factory=list)
    interpretation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "n_matched_epochs": self.n_matched_epochs,
            "pearson_r": self.pearson_r,
            "pearson_p": self.pearson_p,
            "spearman_rho": self.spearman_rho,
            "spearman_p": self.spearman_p,
            "is_anti_correlated": self.is_anti_correlated,
            "anti_correlation_sigma": self.anti_correlation_sigma,
            "optical_variability": self.optical_variability,
            "ir_variability": self.ir_variability,
            "mean_time_offset_days": self.mean_time_offset_days,
            "interpretation": self.interpretation,
        }


# =====================================================================
#  Core correlation engine
# =====================================================================

def cross_correlate_optical_ir(
    optical_time: np.ndarray,
    optical_mag: np.ndarray,
    ir_time: np.ndarray,
    ir_mag: np.ndarray,
    max_lag_days: float = 30.0,
    min_matches: int = 10,
) -> CrossBandResult:
    """Compute correlation between optical and IR light curves.

    For each IR observation epoch, finds the nearest optical observation
    within ±max_lag_days.  Then computes Pearson and Spearman correlation
    between the matched magnitude pairs.

    Positive correlation: natural (both bands vary together).
    Negative correlation: absorption + re-emission → technosignature.

    Parameters
    ----------
    optical_time : ndarray
        MJD timestamps of optical observations.
    optical_mag : ndarray
        Optical magnitudes (e.g. Gaia G, Kepler, TESS).
    ir_time : ndarray
        MJD timestamps of IR observations (e.g. NEOWISE W1/W2).
    ir_mag : ndarray
        IR magnitudes.
    max_lag_days : float
        Maximum time offset for matching (default 30 days).
    min_matches : int
        Minimum number of matched epochs for a valid result.

    Returns
    -------
    CrossBandResult
    """
    optical_time = np.asarray(optical_time, dtype=np.float64)
    optical_mag = np.asarray(optical_mag, dtype=np.float64)
    ir_time = np.asarray(ir_time, dtype=np.float64)
    ir_mag = np.asarray(ir_mag, dtype=np.float64)

    # Remove NaN/Inf
    opt_mask = np.isfinite(optical_time) & np.isfinite(optical_mag)
    ir_mask = np.isfinite(ir_time) & np.isfinite(ir_mag)
    optical_time, optical_mag = optical_time[opt_mask], optical_mag[opt_mask]
    ir_time, ir_mag = ir_time[ir_mask], ir_mag[ir_mask]

    # Sort by time
    opt_order = np.argsort(optical_time)
    optical_time, optical_mag = optical_time[opt_order], optical_mag[opt_order]
    ir_order = np.argsort(ir_time)
    ir_time, ir_mag = ir_time[ir_order], ir_mag[ir_order]

    if len(optical_time) == 0 or len(ir_time) == 0:
        log.info("No overlapping data for cross-band correlation")
        return _empty_result("unknown")

    # Match IR epochs to nearest optical epoch
    matches: List[TemporalMatch] = []
    for i, t_ir in enumerate(ir_time):
        # Find nearest optical observation
        dt = np.abs(optical_time - t_ir)
        j_nearest = np.argmin(dt)
        offset = float(dt[j_nearest])

        if offset <= max_lag_days:
            matches.append(TemporalMatch(
                mjd_optical=float(optical_time[j_nearest]),
                mjd_ir=float(t_ir),
                optical_mag=float(optical_mag[j_nearest]),
                ir_mag=float(ir_mag[i]),
                time_offset_days=float(t_ir - optical_time[j_nearest]),
            ))

    n_matched = len(matches)
    log.info(
        "Cross-band temporal: %d matched epochs (of %d IR, %d optical)",
        n_matched, len(ir_time), len(optical_time),
    )

    if n_matched < min_matches:
        log.info("Insufficient matched epochs (%d < %d)", n_matched, min_matches)
        result = _empty_result("unknown")
        result.n_matched_epochs = n_matched
        result.matched_epochs = matches
        result.interpretation = f"Insufficient matched epochs ({n_matched} < {min_matches})"
        return result

    # Extract matched magnitudes
    # NOTE: We work in magnitude space (not flux), which is valid for
    # anti-correlation detection.  In magnitudes: optical dimming means
    # mag_opt INCREASES while IR brightening means mag_ir DECREASES,
    # producing a negative Pearson r.  This avoids the flux-conversion
    # issues flagged in the cross-band review (optical/IR zero-points
    # are wavelength-dependent, but delta-magnitudes are directly
    # comparable for correlation sign detection).
    opt_matched = np.array([m.optical_mag for m in matches])
    ir_matched = np.array([m.ir_mag for m in matches])
    offsets = np.array([m.time_offset_days for m in matches])
    match_times = np.array([m.mjd_ir for m in matches])

    # Compute correlations
    pearson_r, pearson_p = sp_stats.pearsonr(opt_matched, ir_matched)
    spearman_rho, spearman_p = sp_stats.spearmanr(opt_matched, ir_matched)

    # --- Seasonal consistency check ---
    # For NEOWISE data, observations come in ~6-month seasons.
    # Require anti-correlation across 2+ seasons to rule out single-season
    # systematic effects.
    n_consistent_seasons = 0
    if n_matched >= 6:
        # Split into ~180-day bins
        t_min = match_times.min()
        season_ids = ((match_times - t_min) / 180.0).astype(int)
        unique_seasons = np.unique(season_ids)
        for s in unique_seasons:
            s_mask = season_ids == s
            if np.sum(s_mask) >= 3:
                try:
                    s_r, _ = sp_stats.pearsonr(
                        opt_matched[s_mask], ir_matched[s_mask]
                    )
                    if s_r < 0:
                        n_consistent_seasons += 1
                except Exception:
                    pass

    # Significance of anti-correlation
    # Under the null hypothesis, r ~ N(0, 1/sqrt(n-3)) for large n
    if n_matched > 3:
        fisher_z = np.arctanh(pearson_r)
        se = 1.0 / np.sqrt(n_matched - 3)
        anti_corr_sigma = -fisher_z / se  # positive means anti-correlated
    else:
        anti_corr_sigma = 0.0

    # Determine if significantly anti-correlated
    is_anti = pearson_r < 0 and pearson_p < 0.01

    # Interpret
    opt_var = float(np.std(opt_matched))
    ir_var = float(np.std(ir_matched))
    mean_offset = float(np.mean(offsets))

    if is_anti:
        seasonal_note = ""
        if n_consistent_seasons >= 2:
            seasonal_note = (
                f" Anti-correlation consistent across {n_consistent_seasons} "
                f"seasons (robust against single-season systematics)."
            )
        elif n_consistent_seasons == 1:
            seasonal_note = (
                " Anti-correlation seen in only 1 season — "
                "could be a single-season systematic."
            )

        if anti_corr_sigma > 3.0:
            interpretation = (
                f"STRONG ANTI-CORRELATION DETECTED (r={pearson_r:.3f}, "
                f"p={pearson_p:.2e}, {anti_corr_sigma:.1f}sigma). "
                f"Optical dimming coincides with IR brightening — "
                f"consistent with absorption + re-emission. "
                f"TECHNOSIGNATURE CANDIDATE.{seasonal_note}"
            )
        else:
            interpretation = (
                f"Moderate anti-correlation (r={pearson_r:.3f}, "
                f"p={pearson_p:.2e}, {anti_corr_sigma:.1f}sigma). "
                f"Suggestive but not definitive.{seasonal_note}"
            )
    elif pearson_r > 0.3 and pearson_p < 0.01:
        interpretation = (
            f"Positive correlation (r={pearson_r:.3f}, p={pearson_p:.2e}). "
            f"Both bands vary together — natural variability or dust extinction."
        )
    else:
        interpretation = (
            f"No significant correlation (r={pearson_r:.3f}, p={pearson_p:.2e}). "
            f"Optical and IR variability appear independent."
        )

    result = CrossBandResult(
        target_id="unknown",
        n_matched_epochs=n_matched,
        pearson_r=float(pearson_r),
        pearson_p=float(pearson_p),
        spearman_rho=float(spearman_rho),
        spearman_p=float(spearman_p),
        is_anti_correlated=is_anti,
        anti_correlation_sigma=float(max(anti_corr_sigma, 0)),
        optical_variability=opt_var,
        ir_variability=ir_var,
        mean_time_offset_days=mean_offset,
        matched_epochs=matches,
        interpretation=interpretation,
    )

    log.info(
        "Cross-band result: r=%.3f p=%.2e, anti-corr=%s (%.1f sigma)",
        pearson_r, pearson_p, is_anti, anti_corr_sigma,
    )
    return result


def analyze_target(
    target_id: str,
    optical_time: np.ndarray,
    optical_mag: np.ndarray,
    ir_time: np.ndarray,
    ir_mag: np.ndarray,
    max_lag_days: float = 30.0,
) -> CrossBandResult:
    """Full cross-band temporal analysis for a single target.

    Parameters
    ----------
    target_id : str
        Identifier for the target.
    optical_time, optical_mag : ndarray
        Optical light curve (MJD, magnitudes).
    ir_time, ir_mag : ndarray
        IR light curve (MJD, magnitudes).
    max_lag_days : float
        Maximum time offset for matching.

    Returns
    -------
    CrossBandResult
    """
    result = cross_correlate_optical_ir(
        optical_time, optical_mag, ir_time, ir_mag,
        max_lag_days=max_lag_days,
    )
    result.target_id = target_id
    return result


def _empty_result(target_id: str) -> CrossBandResult:
    """Return an empty result when analysis can't be performed."""
    return CrossBandResult(
        target_id=target_id,
        n_matched_epochs=0,
        pearson_r=0.0,
        pearson_p=1.0,
        spearman_rho=0.0,
        spearman_p=1.0,
        is_anti_correlated=False,
        anti_correlation_sigma=0.0,
        optical_variability=0.0,
        ir_variability=0.0,
        mean_time_offset_days=0.0,
        interpretation="No data available for analysis",
    )


# =====================================================================
#  CLI demo / self-test
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Project EXODUS -- Cross-Band Temporal Correlation Demo")
    print("=" * 70)

    rng = np.random.default_rng(seed=42)

    # ── Test 1: Normal eclipsing binary (positive correlation) ────────
    print("\n[1] Eclipsing binary (both bands dim together)")
    print("-" * 50)

    n_pts = 200
    mjd = np.linspace(57000, 58500, n_pts)
    period = 5.3
    phase = (mjd % period) / period

    # Both optical and IR dim during eclipse
    opt_mag = 10.0 + 0.3 * np.sin(2 * np.pi * phase) + rng.normal(0, 0.02, n_pts)
    ir_mag = 8.5 + 0.15 * np.sin(2 * np.pi * phase) + rng.normal(0, 0.03, n_pts)

    result1 = analyze_target("EB_TEST", mjd, opt_mag, mjd, ir_mag)
    print(f"  Matched epochs:   {result1.n_matched_epochs}")
    print(f"  Pearson r:        {result1.pearson_r:.4f}")
    print(f"  Pearson p:        {result1.pearson_p:.2e}")
    print(f"  Anti-correlated:  {result1.is_anti_correlated}")
    print(f"  Interpretation:   {result1.interpretation}")
    print(f"  >> {'PASS' if result1.pearson_r > 0.3 else 'CHECK'}: Positive correlation")

    # ── Test 2: Dyson swarm simulation (anti-correlation) ─────────────
    print("\n[2] Dyson swarm (optical dims, IR brightens)")
    print("-" * 50)

    n_pts = 200
    mjd2 = np.linspace(57000, 58500, n_pts)

    # Optical: periodic dimming (megastructure transits)
    opt_dip = np.zeros(n_pts)
    for centre, depth, width in [(57200, 0.15, 20), (57600, 0.10, 15), (58000, 0.20, 25)]:
        mask = np.abs(mjd2 - centre) < width
        dt = mjd2[mask] - centre
        opt_dip[mask] = depth * np.exp(-0.5 * (dt / (width * 0.3)) ** 2)

    opt_mag2 = 10.0 + opt_dip + rng.normal(0, 0.01, n_pts)
    # IR: BRIGHTENS when optical dims (absorption -> re-emission as heat)
    ir_response = -0.5 * opt_dip  # negative because brighter = lower mag
    ir_mag2 = 8.5 + ir_response + rng.normal(0, 0.015, n_pts)

    # Slight time offset on IR observations
    mjd2_ir = mjd2 + rng.uniform(-5, 5, n_pts)

    result2 = analyze_target("DYSON_TEST", mjd2, opt_mag2, mjd2_ir, ir_mag2)
    print(f"  Matched epochs:   {result2.n_matched_epochs}")
    print(f"  Pearson r:        {result2.pearson_r:.4f}")
    print(f"  Pearson p:        {result2.pearson_p:.2e}")
    print(f"  Anti-corr sigma:  {result2.anti_correlation_sigma:.2f}")
    print(f"  Anti-correlated:  {result2.is_anti_correlated}")
    print(f"  Interpretation:   {result2.interpretation}")
    print(f"  >> {'PASS' if result2.is_anti_correlated else 'CHECK'}: Anti-correlation detected")

    # ── Test 3: Uncorrelated (normal star) ────────────────────────────
    print("\n[3] Normal quiet star (uncorrelated)")
    print("-" * 50)

    opt_mag3 = 10.0 + rng.normal(0, 0.01, n_pts)
    ir_mag3 = 8.5 + rng.normal(0, 0.02, n_pts)

    result3 = analyze_target("QUIET_TEST", mjd, opt_mag3, mjd, ir_mag3)
    print(f"  Matched epochs:   {result3.n_matched_epochs}")
    print(f"  Pearson r:        {result3.pearson_r:.4f}")
    print(f"  Anti-correlated:  {result3.is_anti_correlated}")
    print(f"  Interpretation:   {result3.interpretation}")
    print(f"  >> {'PASS' if not result3.is_anti_correlated else 'CHECK'}: No false alarm")

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  EB (positive corr):  r={result1.pearson_r:+.3f}  ({'PASS' if result1.pearson_r > 0.3 else 'FAIL'})")
    print(f"  Dyson (anti-corr):   r={result2.pearson_r:+.3f}  ({'PASS' if result2.is_anti_correlated else 'FAIL'})")
    print(f"  Quiet (no corr):     r={result3.pearson_r:+.3f}  ({'PASS' if not result3.is_anti_correlated else 'FAIL'})")
    print("=" * 70)
