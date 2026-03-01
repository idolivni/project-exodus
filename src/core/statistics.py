"""
Statistical rigor infrastructure for Project EXODUS.

Provides False Discovery Rate control, p-value calibration from empirical
nulls, and p-value combination methods (Fisher, Stouffer) for multi-channel
anomaly ranking.

Public API
----------
benjamini_hochberg(p_values, alpha=0.05)
    FDR control: returns boolean mask of surviving hypotheses.

fisher_combine(p_values)
    Fisher's method: combine independent p-values into one.

stouffer_combine(p_values, weights=None)
    Stouffer's weighted Z-method for combining p-values.

calibrate_score_to_pvalue(score, control_scores)
    Map a heuristic [0,1] detector score to a calibrated p-value
    using the empirical distribution of control (null) scores.

empirical_null_pvalue(observed, null_distribution)
    Compute p-value of an observed statistic against a null sample.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger

log = get_logger("core.statistics")


# =====================================================================
#  False Discovery Rate (Benjamini-Hochberg)
# =====================================================================

def benjamini_hochberg(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the Benjamini-Hochberg procedure for FDR control.

    Parameters
    ----------
    p_values : array-like of float
        Raw p-values, one per hypothesis.
    alpha : float
        Desired false discovery rate (default 0.05).

    Returns
    -------
    rejected : np.ndarray of bool
        True where the corresponding hypothesis is significant after
        FDR correction.
    adjusted_p : np.ndarray of float
        Adjusted p-values (q-values).  A hypothesis is rejected if its
        q-value <= alpha.
    """
    pv = np.asarray(p_values, dtype=np.float64)
    m = len(pv)

    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=np.float64)

    # Sort indices
    order = np.argsort(pv)
    sorted_p = pv[order]

    # BH threshold: p_(k) <= k/m * alpha
    ranks = np.arange(1, m + 1)
    thresholds = ranks / m * alpha

    # Adjusted p-values (step-up)
    adjusted = np.minimum(1.0, sorted_p * m / ranks)

    # Enforce monotonicity from the right
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    # Map back to original order
    adjusted_original = np.empty(m, dtype=np.float64)
    adjusted_original[order] = adjusted

    rejected = adjusted_original <= alpha

    n_rejected = int(np.sum(rejected))
    log.debug(
        "BH FDR control: %d/%d hypotheses rejected at alpha=%.3f",
        n_rejected, m, alpha,
    )
    return rejected, adjusted_original


# =====================================================================
#  P-value combination methods
# =====================================================================

def fisher_combine(p_values: Sequence[float]) -> float:
    """Combine independent p-values using Fisher's method.

    The test statistic is -2 * sum(log(p_i)), which follows a
    chi-squared distribution with 2k degrees of freedom under H0.

    Parameters
    ----------
    p_values : array-like of float
        Independent p-values to combine.

    Returns
    -------
    combined_p : float
        Combined p-value.  Lower means stronger overall evidence.
    """
    pv = np.asarray(p_values, dtype=np.float64)
    pv = pv[(pv > 0) & (pv <= 1.0) & np.isfinite(pv)]

    if len(pv) == 0:
        return 1.0

    if len(pv) == 1:
        return float(pv[0])

    # Fisher's statistic
    chi2_stat = -2.0 * np.sum(np.log(pv))
    dof = 2 * len(pv)
    combined_p = float(sp_stats.chi2.sf(chi2_stat, dof))

    return max(combined_p, 1e-300)


def stouffer_combine(
    p_values: Sequence[float],
    weights: Optional[Sequence[float]] = None,
) -> float:
    """Combine p-values using Stouffer's weighted Z-method.

    Each p-value is converted to a z-score, weighted (optionally),
    summed, and normalised.

    Parameters
    ----------
    p_values : array-like of float
        Independent p-values to combine.
    weights : array-like of float, optional
        Positive weights for each p-value.  Default: equal weights.

    Returns
    -------
    combined_p : float
        Combined p-value (one-sided, upper tail).
    """
    pv = np.asarray(p_values, dtype=np.float64)

    # Filter valid p-values
    valid = (pv > 0) & (pv <= 1.0) & np.isfinite(pv)
    pv = pv[valid]

    if len(pv) == 0:
        return 1.0

    if len(pv) == 1:
        return float(pv[0])

    z_scores = sp_stats.norm.isf(pv)  # inverse survival function

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = w[valid]
        if len(w) != len(z_scores):
            w = np.ones(len(z_scores))
    else:
        w = np.ones(len(z_scores))

    w = np.maximum(w, 0)
    w_sum = np.sum(w**2)
    if w_sum <= 0:
        return 1.0

    z_combined = np.sum(w * z_scores) / np.sqrt(w_sum)
    combined_p = float(sp_stats.norm.sf(z_combined))

    return max(combined_p, 1e-300)


# =====================================================================
#  P-value calibration from empirical nulls
# =====================================================================

def calibrate_score_to_pvalue(
    score: float,
    control_scores: Sequence[float],
) -> float:
    """Convert a heuristic detector score to a calibrated p-value.

    Uses the empirical distribution of control (null) scores to estimate
    the probability of seeing a score >= ``score`` under the null hypothesis.

    Parameters
    ----------
    score : float
        The observed detector score for the target.
    control_scores : array-like of float
        Detector scores for matched control sources (the null distribution).

    Returns
    -------
    p_value : float
        Empirical p-value: fraction of controls with score >= observed.
        Includes a continuity correction for small samples.
    """
    ctrl = np.asarray(control_scores, dtype=np.float64)
    ctrl = ctrl[np.isfinite(ctrl)]

    if len(ctrl) == 0:
        log.warning("No valid control scores; returning p=1.0")
        return 1.0

    # Fraction of controls >= the observed score
    n_exceed = int(np.sum(ctrl >= score))
    n_total = len(ctrl)

    # Continuity correction (Laplace smoothing): prevents p=0
    p_value = (n_exceed + 1) / (n_total + 2)

    return float(np.clip(p_value, 1e-300, 1.0))


def empirical_null_pvalue(
    observed: float,
    null_distribution: Sequence[float],
) -> float:
    """Compute p-value of an observation against a null sample.

    Identical to ``calibrate_score_to_pvalue`` but with different
    semantics: higher observed values are more extreme.

    Parameters
    ----------
    observed : float
        Observed test statistic.
    null_distribution : array-like of float
        Null distribution samples.

    Returns
    -------
    p_value : float
    """
    return calibrate_score_to_pvalue(observed, null_distribution)


def calibrate_channel_scores(
    target_scores: Sequence[float],
    control_scores: Sequence[float],
) -> np.ndarray:
    """Calibrate a batch of target scores against the control distribution.

    Parameters
    ----------
    target_scores : array-like of float
        Detector scores for each target.
    control_scores : array-like of float
        Detector scores for the control sample.

    Returns
    -------
    p_values : np.ndarray of float
        Calibrated p-value for each target.
    """
    ctrl = np.sort(np.asarray(control_scores, dtype=np.float64))[::-1]
    n_ctrl = len(ctrl)

    if n_ctrl == 0:
        return np.ones(len(target_scores), dtype=np.float64)

    p_vals = []
    for score in target_scores:
        n_exceed = int(np.searchsorted(-ctrl, -score))
        p = (n_exceed + 1) / (n_ctrl + 2)
        p_vals.append(float(np.clip(p, 1e-300, 1.0)))

    return np.array(p_vals, dtype=np.float64)


# =====================================================================
#  CLI demo / self-test
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Project EXODUS -- Statistical Rigor Module Demo")
    print("=" * 70)

    rng = np.random.default_rng(seed=42)

    # ------------------------------------------------------------------
    # Test 1: Benjamini-Hochberg on mixed real/null p-values
    # ------------------------------------------------------------------
    print("\n[1] Benjamini-Hochberg FDR control")
    print("-" * 50)

    # 5 real signals (very small p-values) + 95 null (uniform p-values)
    real_p = np.array([1e-8, 5e-7, 1e-5, 3e-4, 1e-3])
    null_p = rng.uniform(0.01, 1, 95)
    all_p = np.concatenate([real_p, null_p])

    rejected, q_values = benjamini_hochberg(all_p, alpha=0.05)
    n_rejected = int(np.sum(rejected))
    n_true_positive = int(np.sum(rejected[:5]))
    n_false_positive = int(np.sum(rejected[5:]))

    print(f"  Total hypotheses: {len(all_p)}")
    print(f"  Rejected:         {n_rejected}")
    print(f"  True positives:   {n_true_positive}/5")
    print(f"  False positives:  {n_false_positive}/95")
    assert n_true_positive >= 3, f"Expected >= 3 true positives, got {n_true_positive}"
    assert n_false_positive <= 5, f"Expected <= 5 false positives, got {n_false_positive}"
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 2: Fisher combination
    # ------------------------------------------------------------------
    print("\n[2] Fisher's combined p-value")
    print("-" * 50)

    # Combine several marginally significant p-values
    marginal_p = [0.05, 0.08, 0.03, 0.10]
    combined = fisher_combine(marginal_p)
    print(f"  Input p-values:  {marginal_p}")
    print(f"  Combined p:      {combined:.6f}")
    assert combined < min(marginal_p), "Combined should be smaller than individual"
    print("  PASS: Combined p < min(individual)")

    # Null p-values should not combine to significance
    null_pvals = [0.5, 0.7, 0.3, 0.8]
    combined_null = fisher_combine(null_pvals)
    print(f"  Null p-values:   {null_pvals}")
    print(f"  Combined p:      {combined_null:.6f}")
    assert combined_null > 0.05, "Null combination should stay insignificant"
    print("  PASS: Null combination stays > 0.05")

    # ------------------------------------------------------------------
    # Test 3: Stouffer combination
    # ------------------------------------------------------------------
    print("\n[3] Stouffer's weighted Z combination")
    print("-" * 50)

    combined_s = stouffer_combine(marginal_p)
    print(f"  Input p-values:  {marginal_p}")
    print(f"  Combined p:      {combined_s:.6f}")
    assert combined_s < 0.05, "Combined marginal should be significant"
    print("  PASS")

    # Weighted version -- up-weight the most significant
    weighted = stouffer_combine(marginal_p, weights=[1, 1, 3, 1])
    print(f"  Weighted (3x on p=0.03): {weighted:.6f}")
    assert weighted < combined_s, "Weighting the best p should improve combination"
    print("  PASS: Weighted < unweighted")

    # ------------------------------------------------------------------
    # Test 4: Empirical calibration
    # ------------------------------------------------------------------
    print("\n[4] Score-to-p-value calibration")
    print("-" * 50)

    control = rng.uniform(0, 0.3, 1000)  # controls score 0-0.3
    p_low = calibrate_score_to_pvalue(0.1, control)
    p_high = calibrate_score_to_pvalue(0.8, control)
    p_extreme = calibrate_score_to_pvalue(0.99, control)

    print(f"  Score=0.10  ->  p={p_low:.4f}  (should be large)")
    print(f"  Score=0.80  ->  p={p_high:.6f}  (should be very small)")
    print(f"  Score=0.99  ->  p={p_extreme:.6f}  (should be tiny)")
    assert p_low > p_high > p_extreme, "Higher scores should have lower p-values"
    print("  PASS: Monotonic ordering correct")

    # ------------------------------------------------------------------
    # Test 5: Batch calibration
    # ------------------------------------------------------------------
    print("\n[5] Batch channel calibration")
    print("-" * 50)

    target_sc = [0.1, 0.3, 0.5, 0.8, 0.95]
    batch_p = calibrate_channel_scores(target_sc, control)
    for sc, pv in zip(target_sc, batch_p):
        print(f"  Score={sc:.2f}  ->  p={pv:.6f}")
    assert all(batch_p[i] >= batch_p[i + 1] for i in range(len(batch_p) - 1)), \
        "Batch calibration should be monotonic"
    print("  PASS")

    print("\n" + "=" * 70)
    print("  All tests passed.")
    print("=" * 70)
