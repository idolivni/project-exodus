"""
NEOWISE IR Variability Detection for Project EXODUS.

Analyzes NEOWISE multi-epoch W1/W2 photometry to detect:
  1. Secular IR brightening/dimming (linear trend over 10+ years)
  2. Anomalous variability amplitude (compared to photometric noise)
  3. Epoch-to-epoch scatter inconsistent with constant source

This provides a genuinely INDEPENDENT detection channel from static IR excess:
  - IR excess: "is this star brighter in IR than expected RIGHT NOW?"
  - IR variability: "has this star's IR brightness CHANGED over time?"

A Dyson sphere under construction would show secular IR brightening as more
surface area is enclosed. A transiting megastructure would show periodic
IR dips/brightening. Variable dust emission shows stochastic IR changes.

Public API
----------
compute_ir_variability(ra, dec, **kwargs)
    Analyze NEOWISE time-series for a single target.
    Returns IRVariabilityResult with score, trend, variability metrics.

batch_compute(targets)
    Compute for multiple targets.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger

log = get_logger("detection.ir_variability")


# ── Result dataclass ─────────────────────────────────────────────────

@dataclass
class IRVariabilityResult:
    """Result of NEOWISE IR variability analysis for a single target."""
    # Data quality
    n_epochs: int = 0
    time_baseline_years: float = 0.0
    data_source: str = "none"  # "real", "simulated", "none"

    # Variability metrics
    w1_std: float = 0.0           # Raw standard deviation of W1
    w2_std: float = 0.0           # Raw standard deviation of W2
    w1_excess_scatter: float = 0.0  # Scatter / median_error - 1 (0 = expected noise)
    w2_excess_scatter: float = 0.0

    # Secular trend (linear fit: mag = a + b * years)
    w1_trend_mag_per_year: float = 0.0  # Positive = dimming, negative = brightening
    w1_trend_sigma: float = 0.0          # Significance of trend in sigma
    w2_trend_mag_per_year: float = 0.0
    w2_trend_sigma: float = 0.0

    # Construction-in-progress indicators
    w1_monotonic_frac: float = 0.0  # Fraction of bin transitions in same direction
    w2_monotonic_frac: float = 0.0
    cross_band_consistent: bool = False  # W1 and W2 trending same direction
    is_brightening: bool = False         # Negative mag trend (IR brightening)
    secular_trend_score: float = 0.0     # Combined construction-in-progress score

    # Combined score (0-1, higher = more anomalous)
    variability_score: float = 0.0

    # Anomaly flags
    is_variable: bool = False      # Excess scatter > 3σ above expected
    has_secular_trend: bool = False  # Trend significance > 3σ
    is_anomalous: bool = False      # Combined score > threshold

    # For channel integration
    p_value: float = 1.0  # Statistical p-value of variability

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_epochs": self.n_epochs,
            "time_baseline_years": round(self.time_baseline_years, 2),
            "data_source": self.data_source,
            "w1_std": round(self.w1_std, 5),
            "w2_std": round(self.w2_std, 5),
            "w1_excess_scatter": round(self.w1_excess_scatter, 3),
            "w2_excess_scatter": round(self.w2_excess_scatter, 3),
            "w1_trend_mag_per_year": round(self.w1_trend_mag_per_year, 6),
            "w1_trend_sigma": round(self.w1_trend_sigma, 2),
            "w2_trend_mag_per_year": round(self.w2_trend_mag_per_year, 6),
            "w2_trend_sigma": round(self.w2_trend_sigma, 2),
            "w1_monotonic_frac": round(self.w1_monotonic_frac, 3),
            "w2_monotonic_frac": round(self.w2_monotonic_frac, 3),
            "cross_band_consistent": self.cross_band_consistent,
            "is_brightening": self.is_brightening,
            "secular_trend_score": round(self.secular_trend_score, 4),
            "variability_score": round(self.variability_score, 4),
            "is_variable": self.is_variable,
            "has_secular_trend": self.has_secular_trend,
            "is_anomalous": self.is_anomalous,
            "p_value": self.p_value,
        }


# ── Thresholds ───────────────────────────────────────────────────────

# Minimum data requirements
MIN_EPOCHS = 15        # Need at least 15 good epochs for meaningful analysis
MIN_BASELINE_YR = 3.0  # At least 3 years of data

# Variability detection thresholds
EXCESS_SCATTER_THRESH = 3.0   # Scatter must be 3× above photometric noise
TREND_SIGMA_THRESH = 5.0      # Trend must be > 5σ significant (raised from 3σ
                               # per research: keeps survey-wide FPR < 1 for
                               # secular trend claims over 10+ yr baseline)

# Score weights
WEIGHT_EXCESS_SCATTER = 0.4   # How much weight for anomalous scatter
WEIGHT_SECULAR_TREND = 0.6    # How much weight for secular trend

# ── NEOWISE calibration constants ────────────────────────────────────
# From Perplexity research brief: photometric noise floors per exposure
W1_NOISE_FLOOR_MMAG = 2.6    # W1 is 2.4× more stable than W2
W2_NOISE_FLOOR_MMAG = 6.1    # W2 also has 34 mmag seasonal systematics
W2_SEASONAL_AMPLITUDE_MMAG = 34.0  # Peak-to-peak W2 seasonal variation

# W2 bad epoch range — zero-point correction was incorrect
W2_BAD_MJD_RANGE = (57000, 57071)

# NEOWISE stable start — pre-Feb 2014 data has flat-field residuals
NEOWISE_STABLE_START_MJD = 56700

# W1-W2 color consistency: real dust reddening increases W1-W2,
# calibration drift is gray (W1-W2 unchanged). Use this to discriminate.
COLOR_CONSISTENCY_THRESH = 0.02  # mag — W1-W2 change must exceed this

# Epoch-averaged trend: minimum sky passes for trend detection
MIN_EPOCH_AVERAGES = 6  # Need at least 6 sky passes (~3 years)


# ── Main computation ─────────────────────────────────────────────────

def compute_ir_variability(
    ra: float,
    dec: float,
    neowise_data: Optional[Any] = None,
    use_cache: bool = True,
) -> IRVariabilityResult:
    """Analyze NEOWISE time-series for IR variability.

    Parameters
    ----------
    ra, dec : float
        Target coordinates in degrees.
    neowise_data : NEOWISETimeSeries, optional
        Pre-fetched NEOWISE data. If None, queries IRSA.
    use_cache : bool
        Whether to use cache for NEOWISE queries.

    Returns
    -------
    IRVariabilityResult
        Variability metrics and anomaly score.
    """
    # Get NEOWISE time-series
    if neowise_data is None:
        try:
            from src.ingestion.neowise_timeseries import query_neowise_timeseries
            neowise_data = query_neowise_timeseries(ra, dec, use_cache=use_cache)
        except Exception as exc:
            log.warning("NEOWISE query failed for (%.4f, %.4f): %s", ra, dec, exc)
            return IRVariabilityResult(data_source="none")

    # Check data quality
    if neowise_data.n_epochs < MIN_EPOCHS:
        log.debug(
            "Insufficient NEOWISE data for (%.4f, %.4f): %d epochs (need %d)",
            ra, dec, neowise_data.n_epochs, MIN_EPOCHS,
        )
        return IRVariabilityResult(
            n_epochs=neowise_data.n_epochs,
            time_baseline_years=neowise_data.time_baseline_years,
            data_source="insufficient",
        )

    if neowise_data.time_baseline_years < MIN_BASELINE_YR:
        return IRVariabilityResult(
            n_epochs=neowise_data.n_epochs,
            time_baseline_years=neowise_data.time_baseline_years,
            data_source="insufficient",
        )

    # Determine data source from the timeseries object (authoritative)
    # NOTE: dtype=float64 is guaranteed by _dict_to_timeseries (N1b fix).
    # The np.asarray calls below are defensive — no-ops when already float64.
    mjd = np.asarray(neowise_data.mjd, dtype=np.float64)
    w1 = np.asarray(neowise_data.w1_mag, dtype=np.float64)
    w1_err = np.asarray(neowise_data.w1_err, dtype=np.float64)
    w2 = np.asarray(neowise_data.w2_mag, dtype=np.float64)
    w2_err = np.asarray(neowise_data.w2_err, dtype=np.float64)

    data_source = getattr(neowise_data, "data_source", "unknown")
    if data_source not in ("real", "simulated"):
        # Fallback heuristic for old cached data without data_source tag
        data_source = "real" if mjd[0] > 56500 else "unknown"

    # ── 1. Excess scatter analysis ───────────────────────────────────
    # Compare observed scatter to expected from photometric errors
    w1_obs_scatter = np.nanstd(w1)
    w1_expected_scatter = np.nanmedian(w1_err)
    w1_excess = (w1_obs_scatter / max(w1_expected_scatter, 1e-6)) - 1.0

    w2_valid = np.isfinite(w2) & np.isfinite(w2_err)
    if np.sum(w2_valid) > MIN_EPOCHS:
        w2_obs_scatter = np.nanstd(w2[w2_valid])
        w2_expected_scatter = np.nanmedian(w2_err[w2_valid])
        w2_excess = (w2_obs_scatter / max(w2_expected_scatter, 1e-6)) - 1.0
    else:
        w2_obs_scatter = 0.0
        w2_excess = 0.0

    is_variable = max(w1_excess, w2_excess) > EXCESS_SCATTER_THRESH

    # ── 2. Secular trend analysis (weighted linear fit) ──────────────
    # Convert MJD to years from start
    t_years = (mjd - mjd[0]) / 365.25

    # CALIBRATION FIX: W1 is the PRIMARY band for secular trends.
    # W1 noise floor = 2.6 mmag vs W2 = 6.1 mmag (2.4× worse).
    # W2 also has 34 mmag seasonal systematics from spacecraft thermal cycling.
    # W1 trend is authoritative; W2 trend is corroborative only.

    # Weighted linear regression on W1 (primary)
    w1_trend, w1_trend_sigma = _weighted_linear_trend(t_years, w1, w1_err)

    # Weighted linear regression on W2 (secondary/corroborative)
    if np.sum(w2_valid) > MIN_EPOCHS:
        w2_trend, w2_trend_sigma = _weighted_linear_trend(
            t_years[w2_valid], w2[w2_valid], w2_err[w2_valid]
        )
    else:
        w2_trend, w2_trend_sigma = 0.0, 0.0

    # ── 2a. Epoch-averaged trend analysis ─────────────────────────
    # Single-exposure trends are dominated by per-frame noise.
    # Epoch-averaging (~12 exposures per sky pass) reduces scatter by ~√12
    # and gives ~10-20 high-S/N points for trend detection.
    epoch_w1_trend_sigma = 0.0
    epoch_w2_trend_sigma = 0.0
    epoch_w1_trend = 0.0
    epoch_w2_trend = 0.0

    try:
        from src.ingestion.neowise_timeseries import compute_epoch_averages
        epoch_avgs = compute_epoch_averages(neowise_data)
        if epoch_avgs["n_epochs"] >= MIN_EPOCH_AVERAGES:
            ep_t = (epoch_avgs["epoch_mjd"] - epoch_avgs["epoch_mjd"][0]) / 365.25
            epoch_w1_trend, epoch_w1_trend_sigma = _weighted_linear_trend(
                ep_t, epoch_avgs["epoch_w1"], epoch_avgs["epoch_w1_err"]
            )
            # W2 epoch-averaged (if enough valid data)
            w2_ep_valid = np.isfinite(epoch_avgs["epoch_w2"])
            if np.sum(w2_ep_valid) >= MIN_EPOCH_AVERAGES:
                epoch_w2_trend, epoch_w2_trend_sigma = _weighted_linear_trend(
                    ep_t[w2_ep_valid],
                    epoch_avgs["epoch_w2"][w2_ep_valid],
                    epoch_avgs["epoch_w2_err"][w2_ep_valid],
                )
            log.debug(
                "Epoch-averaged trend (%.4f, %.4f): W1 %.1fσ, W2 %.1fσ (%d passes)",
                ra, dec, epoch_w1_trend_sigma, epoch_w2_trend_sigma,
                epoch_avgs["n_epochs"],
            )
    except Exception as exc:
        log.debug("Epoch averaging unavailable: %s", exc)

    # Use the MORE significant of single-exposure vs epoch-averaged trend
    # (epoch-averaged is typically more reliable but may have fewer points)
    effective_w1_sigma = max(abs(w1_trend_sigma), abs(epoch_w1_trend_sigma))
    effective_w2_sigma = max(abs(w2_trend_sigma), abs(epoch_w2_trend_sigma))

    # CALIBRATION FIX: 5σ threshold (raised from 3σ)
    # At 3σ with ~500 stars, expect ~1 false positive. At 5σ, expect << 1.
    has_secular_trend = (effective_w1_sigma > TREND_SIGMA_THRESH or
                         effective_w2_sigma > TREND_SIGMA_THRESH)

    # ── 2b. Construction-in-progress characterisation ──────────────
    # A Dyson sphere under construction shows MONOTONIC IR brightening
    # (negative mag trend = brightening) over years. Natural variables
    # oscillate. This specifically targets the construction signature.
    N_BINS = 5
    w1_monotonic_frac = 0.0
    w2_monotonic_frac = 0.0

    if len(t_years) >= MIN_EPOCHS:
        _, w1_monotonic_frac = _check_monotonicity(t_years, w1, N_BINS)
        if np.sum(w2_valid) >= MIN_EPOCHS:
            _, w2_monotonic_frac = _check_monotonicity(
                t_years[w2_valid], w2[w2_valid], N_BINS
            )

    # Cross-band consistency: both W1 and W2 trending in same direction?
    cross_band_consistent = False
    if abs(w1_trend_sigma) > 2.0 and abs(w2_trend_sigma) > 2.0:
        cross_band_consistent = bool(np.sign(w1_trend) == np.sign(w2_trend))

    # ── 2c. W1-W2 color consistency check ─────────────────────────
    # Real astrophysical dust causes REDDENING (W1-W2 increases as IR
    # brightens). Calibration drift is GRAY (W1-W2 stays constant).
    # A real Dyson sphere should redden because the thermal emission
    # is strongest in W2 (the longer wavelength).
    color_change_real = False
    if np.sum(w2_valid) >= MIN_EPOCHS:
        # Audit fix N1: use common valid mask for BOTH W1 and W2 to avoid
        # shape mismatch when NaN positions differ between bands.
        _both_valid = np.isfinite(w1) & np.isfinite(w1_err) & w2_valid
        color = w1[_both_valid] - w2[_both_valid]
        if len(color) >= MIN_EPOCHS:
            # Split into first half and second half
            half = len(color) // 2
            color_early = np.nanmedian(color[:half])
            color_late = np.nanmedian(color[half:])
            color_delta = abs(color_late - color_early)
            color_change_real = color_delta > COLOR_CONSISTENCY_THRESH
            if color_change_real:
                log.debug(
                    "W1-W2 color change at (%.4f, %.4f): Δ=%.4f mag (real astrophysics likely)",
                    ra, dec, color_delta,
                )

    # Is it brightening (negative mag trend)?
    # CALIBRATION FIX: require W1 (primary band) to show brightening
    is_brightening = bool(
        (w1_trend < 0 and effective_w1_sigma > 3.0) or
        (w2_trend < 0 and effective_w2_sigma > 3.0 and cross_band_consistent)
    )

    # Secular trend score: specifically for construction-in-progress
    # CALIBRATION FIX: W1 is primary, use effective (max of raw/epoch-avg) sigma,
    # require 5σ for full score, color consistency gives bonus
    secular_trend_score = 0.0
    best_trend_sigma = max(effective_w1_sigma, effective_w2_sigma)
    best_monotonic_frac = max(w1_monotonic_frac, w2_monotonic_frac)

    if best_trend_sigma > 3.0:  # Preliminary threshold (full score needs 5σ)
        # Scale: 3σ → 0.3, 5σ → 0.5, 10σ → 1.0
        trend_base = min(1.0, best_trend_sigma / 10.0)
        mono_bonus = 1.5 if best_monotonic_frac > 0.8 else 1.0
        xband_bonus = 1.3 if cross_band_consistent else 1.0
        bright_bonus = 1.2 if is_brightening else 1.0
        # NEW: color consistency bonus — real reddening vs calibration drift
        color_bonus = 1.2 if color_change_real else 1.0
        secular_trend_score = min(1.0,
            trend_base * mono_bonus * xband_bonus * bright_bonus * color_bonus
        )

    # ── 3. Combined variability score ────────────────────────────────
    # Normalize excess scatter to [0, 1]: 0 = expected noise, 1 = extreme
    scatter_component = min(1.0, max(0.0, max(w1_excess, w2_excess) / 10.0))

    # Normalize trend significance to [0, 1]: 0 = no trend, 1 = extreme
    # Use effective sigma (max of raw and epoch-averaged)
    trend_component = min(1.0, max(0.0,
        max(effective_w1_sigma, effective_w2_sigma) / 10.0
    ))

    variability_score = (
        WEIGHT_EXCESS_SCATTER * scatter_component +
        WEIGHT_SECULAR_TREND * trend_component
    )
    # Use secular_trend_score if higher (rewards monotonic trends)
    variability_score = max(variability_score, secular_trend_score)

    # ── 4. P-value from chi-squared test ─────────────────────────────
    # Chi-squared: sum((w1 - mean)^2 / err^2) / (N-1) should be ~1 for constant
    if w1_expected_scatter > 0:
        chi2_w1 = np.nansum(((w1 - np.nanmean(w1)) / w1_err) ** 2)
        dof = max(1, np.sum(np.isfinite(w1)) - 1)
        # Use survival function of chi2 distribution
        try:
            from scipy.stats import chi2 as chi2_dist
            p_value = float(chi2_dist.sf(chi2_w1, dof))
        except ImportError:
            # Approximate: if chi2/dof >> 1, p is very small
            p_value = 1.0 if chi2_w1 / dof < 2 else 0.01
    else:
        p_value = 1.0

    is_anomalous = variability_score > 0.3 and (is_variable or has_secular_trend)

    result = IRVariabilityResult(
        n_epochs=neowise_data.n_epochs,
        time_baseline_years=neowise_data.time_baseline_years,
        data_source=data_source,
        w1_std=float(w1_obs_scatter),
        w2_std=float(w2_obs_scatter),
        w1_excess_scatter=float(w1_excess),
        w2_excess_scatter=float(w2_excess),
        w1_trend_mag_per_year=float(w1_trend),
        w1_trend_sigma=float(w1_trend_sigma),
        w2_trend_mag_per_year=float(w2_trend),
        w2_trend_sigma=float(w2_trend_sigma),
        w1_monotonic_frac=float(w1_monotonic_frac),
        w2_monotonic_frac=float(w2_monotonic_frac),
        cross_band_consistent=cross_band_consistent,
        is_brightening=is_brightening,
        secular_trend_score=float(secular_trend_score),
        variability_score=float(variability_score),
        is_variable=bool(is_variable),
        has_secular_trend=bool(has_secular_trend),
        is_anomalous=bool(is_anomalous),
        p_value=float(p_value),
    )

    if is_anomalous:
        log.info(
            "IR variability ANOMALY at (%.4f, %.4f): score=%.3f, "
            "secular=%.3f, excess_scatter=%.1f, trend_sigma=%.1f, "
            "monotonic=%.0f%%, %d epochs over %.1f yr",
            ra, dec, variability_score, secular_trend_score,
            max(w1_excess, w2_excess),
            max(abs(w1_trend_sigma), abs(w2_trend_sigma)),
            best_monotonic_frac * 100,
            neowise_data.n_epochs, neowise_data.time_baseline_years,
        )

    return result


def batch_compute(
    targets: List[Dict[str, Any]],
) -> Dict[str, IRVariabilityResult]:
    """Compute IR variability for multiple targets.

    Parameters
    ----------
    targets : list of dict
        Each dict must have 'ra', 'dec', and 'target_id'.

    Returns
    -------
    dict
        target_id -> IRVariabilityResult
    """
    results = {}
    for i, tgt in enumerate(targets):
        ra = tgt["ra"]
        dec = tgt["dec"]
        tid = tgt.get("target_id", f"target_{i:04d}")

        if (i + 1) % 50 == 0:
            log.info("IR variability: %d/%d targets processed", i + 1, len(targets))

        results[tid] = compute_ir_variability(ra, dec)

    return results


# ── Internal helpers ─────────────────────────────────────────────────

def _check_monotonicity(
    t: np.ndarray, mag: np.ndarray, n_bins: int = 5
) -> tuple[bool, float]:
    """Check if a time-series is monotonically trending.

    Splits data into n_bins equal-time bins, computes the median magnitude
    in each bin, then checks how many consecutive bin transitions go in the
    same direction.

    Returns
    -------
    is_monotonic : bool
        True if >80% of bin transitions go in the same direction.
    monotonic_frac : float
        Fraction of transitions in the dominant direction (0-1).
    """
    valid = np.isfinite(t) & np.isfinite(mag)
    if np.sum(valid) < n_bins * 2:
        return False, 0.0

    t_v = t[valid]
    m_v = mag[valid]

    # Create equal-time bins
    t_min, t_max = t_v.min(), t_v.max()
    if t_max - t_min < 0.5:  # less than half a year span
        return False, 0.0

    bin_edges = np.linspace(t_min, t_max + 1e-10, n_bins + 1)
    bin_medians = []
    for i in range(n_bins):
        mask = (t_v >= bin_edges[i]) & (t_v < bin_edges[i + 1])
        if np.sum(mask) >= 2:
            bin_medians.append(float(np.nanmedian(m_v[mask])))
        else:
            bin_medians.append(None)

    # Count transitions
    diffs = []
    prev = None
    for bm in bin_medians:
        if bm is None:
            continue
        if prev is not None:
            diffs.append(bm - prev)
        prev = bm

    if len(diffs) < 2:
        return False, 0.0

    n_pos = sum(1 for d in diffs if d > 0)
    n_neg = sum(1 for d in diffs if d < 0)
    dominant = max(n_pos, n_neg)
    frac = dominant / len(diffs)

    return frac > 0.8, frac


def _weighted_linear_trend(
    t: np.ndarray, mag: np.ndarray, err: np.ndarray
) -> tuple[float, float]:
    """Compute weighted linear trend: mag = a + b * t.

    Returns (slope, slope_significance_sigma).
    """
    valid = np.isfinite(t) & np.isfinite(mag) & np.isfinite(err) & (err > 0)
    if np.sum(valid) < 5:
        return 0.0, 0.0

    t_v = t[valid]
    m_v = mag[valid]
    w = 1.0 / (err[valid] ** 2)

    # Weighted linear regression
    S = np.sum(w)
    Sx = np.sum(w * t_v)
    Sy = np.sum(w * m_v)
    Sxx = np.sum(w * t_v ** 2)
    Sxy = np.sum(w * t_v * m_v)

    denom = S * Sxx - Sx ** 2
    if abs(denom) < 1e-30:
        return 0.0, 0.0

    # Slope and its error
    slope = (S * Sxy - Sx * Sy) / denom
    slope_err = np.sqrt(S / denom)

    sigma = slope / max(slope_err, 1e-15)

    return float(slope), float(sigma)


# ── CLI demo ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  Project EXODUS -- NEOWISE IR Variability Detection")
    print("=" * 70)

    # Tabby's Star (KIC 8462852) — known IR variable
    ra, dec = 301.5643, 44.4568
    print(f"\n  Analyzing Tabby's Star (RA={ra}, Dec={dec})")
    result = compute_ir_variability(ra, dec)
    print(f"  Epochs: {result.n_epochs}")
    print(f"  Baseline: {result.time_baseline_years:.1f} years")
    print(f"  Data source: {result.data_source}")
    print(f"  W1 excess scatter: {result.w1_excess_scatter:.2f}")
    print(f"  W1 trend: {result.w1_trend_mag_per_year:.5f} mag/yr ({result.w1_trend_sigma:.1f}σ)")
    print(f"  W1 monotonic: {result.w1_monotonic_frac:.0%}")
    print(f"  Cross-band consistent: {result.cross_band_consistent}")
    print(f"  Is brightening: {result.is_brightening}")
    print(f"  Secular trend score: {result.secular_trend_score:.4f}")
    print(f"  Variability score: {result.variability_score:.4f}")
    print(f"  Is anomalous: {result.is_anomalous}")
    print(f"  P-value: {result.p_value:.4e}")

    # Vega — stable star
    ra2, dec2 = 279.2347, 38.7837
    print(f"\n  Analyzing Vega (RA={ra2}, Dec={dec2})")
    result2 = compute_ir_variability(ra2, dec2)
    print(f"  Epochs: {result2.n_epochs}")
    print(f"  Variability score: {result2.variability_score:.4f}")
    print(f"  Secular trend score: {result2.secular_trend_score:.4f}")
    print(f"  Is anomalous: {result2.is_anomalous}")

    print("\n" + "=" * 70)
