"""
Transit anomaly detection for Project EXODUS.

Analyses light curves from Kepler, TESS, and Gaia to identify transits that
deviate from natural planetary transit signatures.  Natural transits are
symmetric, constant-depth, and obey Kepler's third law.  Megastructures
(Dyson swarms, stellar engines, etc.) would produce asymmetric, variable-depth
dimming that violates these properties.

Two complementary detection modes:

1. **Periodic transit analysis** via Box Least Squares (BLS) periodogram:
   finds periodic dips and scores each one on symmetry, depth variability,
   duration consistency, and shape residuals.

2. **Irregular dimming detection** via Isolation Forest on windowed light-curve
   features: catches non-periodic, aperiodic events like those seen in
   Boyajian's Star (KIC 8462852).

Public API
----------
detect_transit_anomaly(time, flux, flux_err=None)
    Run BLS-based periodic transit analysis.  Returns TransitAnomalyResult.

detect_irregular_dimming(time, flux)
    Run unsupervised anomaly detection on the full light curve.
    Returns IrregularDimmingResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import interpolate

try:
    from astropy.timeseries import BoxLeastSquares
except ImportError as exc:
    raise ImportError(
        "astropy is required for transit anomaly detection. "
        "Install it with:  pip install astropy"
    ) from exc

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required for anomaly detection. "
        "Install it with:  pip install scikit-learn"
    ) from exc

# -- Project utilities -------------------------------------------------------
from src.utils import get_config, get_logger

logger = get_logger("processing.transit_anomaly")

# -- Configuration -----------------------------------------------------------
_DEFAULT_ANOMALY_SIGMA: float = 3.0


def _configured_anomaly_sigma() -> float:
    """Read the anomaly sigma threshold from the project config, falling back
    to the hard-coded default if the config key is absent."""
    try:
        cfg = get_config()
        return float(cfg["search"]["anomaly_sigma"])
    except Exception:
        logger.debug(
            "Could not read search.anomaly_sigma from config; "
            "using default %.1f",
            _DEFAULT_ANOMALY_SIGMA,
        )
        return _DEFAULT_ANOMALY_SIGMA


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class DetectedDip:
    """A single transit-like dip extracted from the folded light curve."""

    epoch: float
    depth: float
    duration: float
    symmetry_score: float
    shape_residual: float


@dataclass
class TransitAnomalyResult:
    """Full result from the periodic transit anomaly detector."""

    period: float
    depth: float
    symmetry_score: float
    depth_variability: float
    duration_consistency: float
    shape_residual: float
    anomaly_score: float          # composite 0-1
    is_anomalous: bool
    detected_dips: List[DetectedDip] = field(default_factory=list)


@dataclass
class DimmingEvent:
    """A single irregular dimming event found in the light curve."""

    start_time: float
    end_time: float
    depth: float
    duration: float
    asymmetry: float


@dataclass
class IrregularDimmingResult:
    """Result from the unsupervised irregular dimming detector."""

    n_events: int
    events: List[DimmingEvent] = field(default_factory=list)
    max_depth: float = 0.0
    anomaly_score: float = 0.0


# ============================================================================
# Internal helpers
# ============================================================================

def _validate_inputs(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Convert to float64 numpy arrays, strip NaNs, and sort by time."""
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=np.float64)

    # Remove NaN / Inf entries
    mask = np.isfinite(time) & np.isfinite(flux)
    if flux_err is not None:
        mask &= np.isfinite(flux_err)
    time, flux = time[mask], flux[mask]
    if flux_err is not None:
        flux_err = flux_err[mask]

    # Sort by time
    order = np.argsort(time)
    time, flux = time[order], flux[order]
    if flux_err is not None:
        flux_err = flux_err[order]

    return time, flux, flux_err


def _compute_symmetry(phase: np.ndarray, folded_flux: np.ndarray) -> float:
    """Compute the symmetry score of a folded transit.

    Folds the transit around its centre, interpolates the left and right
    halves onto a common grid, and returns the normalised RMS difference.
    A perfectly symmetric transit scores 0.0; higher values indicate
    greater asymmetry.

    The comparison uses the ingress/egress shoulders (the gradient regions)
    rather than the flat bottom, which makes it robust to noise in the
    flat-bottomed box portion of natural transits.

    Returns
    -------
    float
        Symmetry score in [0, 1].  Values > ~0.3 suggest non-natural
        asymmetry.
    """
    if len(phase) < 10:
        return 0.0

    # Find the flux-weighted centroid of the dip as the centre, which is
    # more robust than just argmin for noisy data.
    weights = np.clip(1.0 - folded_flux, 0, None)
    weight_sum = np.sum(weights)
    if weight_sum == 0:
        return 0.0
    centre_phase = float(np.average(phase, weights=weights))

    left_mask = phase < centre_phase
    right_mask = phase > centre_phase

    n_left = int(np.sum(left_mask))
    n_right = int(np.sum(right_mask))

    if n_left < 3 or n_right < 3:
        return 0.0

    # Mirror the left half so both sides share a positive-offset axis
    left_phase = centre_phase - phase[left_mask]
    left_flux = folded_flux[left_mask]
    right_phase = phase[right_mask] - centre_phase
    right_flux = folded_flux[right_mask]

    # Sort for interpolation
    left_order = np.argsort(left_phase)
    left_phase, left_flux = left_phase[left_order], left_flux[left_order]
    right_order = np.argsort(right_phase)
    right_phase, right_flux = right_phase[right_order], right_flux[right_order]

    # Common grid spanning the overlap region
    max_offset = min(left_phase[-1], right_phase[-1])
    if max_offset <= 0:
        return 0.0
    grid = np.linspace(0, max_offset, 80)

    try:
        interp_left = interpolate.interp1d(
            left_phase, left_flux, kind="linear", fill_value="extrapolate"
        )
        interp_right = interpolate.interp1d(
            right_phase, right_flux, kind="linear", fill_value="extrapolate"
        )
    except Exception:
        return 0.0

    diff = interp_left(grid) - interp_right(grid)
    flux_range = np.ptp(folded_flux)
    if flux_range == 0:
        return 0.0

    rms = np.sqrt(np.mean(diff ** 2))
    score = float(np.clip(rms / flux_range, 0.0, 1.0))
    return score


def _compute_depth_variability(depths: np.ndarray) -> float:
    """Return the coefficient of variation (std / mean) of individual
    transit depths.  Natural transits have near-zero CV; variable-depth
    events are technosignature candidates.

    Returns a value in [0, 1] clipped.
    """
    if len(depths) < 2:
        return 0.0
    mean_depth = np.mean(depths)
    if mean_depth == 0:
        return 0.0
    cv = float(np.std(depths) / abs(mean_depth))
    return float(np.clip(cv, 0.0, 1.0))


def _compute_duration_consistency(durations: np.ndarray) -> float:
    """Score how consistent transit durations are.

    Natural transits governed by Kepler's laws have nearly identical
    durations.  Returns 0 for perfect consistency, 1 for maximally
    inconsistent durations.
    """
    if len(durations) < 2:
        return 0.0
    mean_dur = np.mean(durations)
    if mean_dur == 0:
        return 0.0
    cv = float(np.std(durations) / abs(mean_dur))
    return float(np.clip(cv, 0.0, 1.0))


def _box_model(phase: np.ndarray, depth: float, duration_phase: float) -> np.ndarray:
    """Simple box-shaped transit model centred at phase = 0."""
    model = np.ones_like(phase)
    in_transit = np.abs(phase) <= duration_phase / 2.0
    model[in_transit] = 1.0 - depth
    return model


def _compute_shape_residual(
    phase: np.ndarray, folded_flux: np.ndarray, depth: float, duration_phase: float,
) -> float:
    """RMS residual after subtracting the best-fit box model, normalised
    by the transit depth so the result is in [0, 1]."""
    if len(phase) < 4 or depth == 0:
        return 0.0
    model = _box_model(phase, depth, duration_phase)
    residual = folded_flux - model
    rms = np.sqrt(np.mean(residual ** 2))
    score = float(np.clip(rms / abs(depth), 0.0, 1.0))
    return score


def _extract_individual_dips(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the depths and durations of every individual transit event
    in the unfolded light curve.

    Returns
    -------
    depths : ndarray
        Depth (1 - min_flux_in_event) for each transit.
    durations : ndarray
        Measured duration of each transit (time span of in-transit points).
    """
    # Identify which transit epoch each in-transit point belongs to
    phase_from_t0 = (time - t0) / period
    epoch_numbers = np.round(phase_from_t0).astype(int)
    unique_epochs = np.unique(epoch_numbers)

    depths_list: List[float] = []
    durations_list: List[float] = []

    half_dur = duration / 2.0

    for epoch in unique_epochs:
        t_centre = t0 + epoch * period
        in_transit = np.abs(time - t_centre) <= half_dur * 1.5
        if np.sum(in_transit) < 3:
            continue

        transit_flux = flux[in_transit]
        transit_time = time[in_transit]

        dip_depth = 1.0 - float(np.min(transit_flux))
        dip_duration = float(transit_time[-1] - transit_time[0])

        if dip_depth > 0:
            depths_list.append(dip_depth)
            durations_list.append(dip_duration)

    return np.array(depths_list), np.array(durations_list)


# ============================================================================
# Windowed feature extraction for Isolation Forest
# ============================================================================

def _extract_window_features(
    time: np.ndarray,
    flux: np.ndarray,
    window_size: int = 50,
    step: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide a window across the light curve and compute statistical
    features for each window.

    Returns
    -------
    features : ndarray, shape (n_windows, n_features)
        Feature matrix for anomaly detection.
    window_centres : ndarray
        Time at the centre of each window.
    window_indices : ndarray of int pairs, shape (n_windows, 2)
        Start and end indices for each window.
    """
    n = len(flux)
    if n < window_size:
        window_size = max(n // 2, 5)
        step = max(1, window_size // 5)

    centres = []
    feats = []
    indices = []

    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        seg_flux = flux[start:end]
        seg_time = time[start:end]

        mean_f = np.mean(seg_flux)
        std_f = np.std(seg_flux)
        min_f = np.min(seg_flux)
        max_f = np.max(seg_flux)
        depth = 1.0 - min_f  # depth of deepest dip in window
        skew = float(
            np.mean(((seg_flux - mean_f) / std_f) ** 3) if std_f > 0 else 0.0
        )
        # Slope: overall gradient across window
        dt = seg_time[-1] - seg_time[0]
        slope = (seg_flux[-1] - seg_flux[0]) / dt if dt > 0 else 0.0
        # Number of points below median - sigma
        med = np.median(seg_flux)
        n_deep = np.sum(seg_flux < med - 2.0 * std_f) / window_size

        feats.append([mean_f, std_f, min_f, max_f, depth, skew, slope, n_deep])
        centres.append(0.5 * (seg_time[0] + seg_time[-1]))
        indices.append([start, end])

    return (
        np.array(feats, dtype=np.float64),
        np.array(centres, dtype=np.float64),
        np.array(indices, dtype=np.int64),
    )


def _events_from_anomaly_mask(
    time: np.ndarray,
    flux: np.ndarray,
    anomaly_mask: np.ndarray,
) -> List[DimmingEvent]:
    """Convert a boolean anomaly mask over indices into contiguous
    DimmingEvent objects."""
    events: List[DimmingEvent] = []
    if not np.any(anomaly_mask):
        return events

    # Find contiguous runs of True
    diff = np.diff(anomaly_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    # Handle edge cases
    if anomaly_mask[0]:
        starts = np.concatenate([[0], starts])
    if anomaly_mask[-1]:
        ends = np.concatenate([ends, [len(anomaly_mask)]])

    for s, e in zip(starts, ends):
        if e <= s:
            continue
        seg_time = time[s:e]
        seg_flux = flux[s:e]

        start_time = float(seg_time[0])
        end_time = float(seg_time[-1])
        depth = float(1.0 - np.min(seg_flux))
        duration = end_time - start_time

        # Asymmetry: compare first half vs second half of dip
        mid = len(seg_flux) // 2
        if mid > 0 and mid < len(seg_flux):
            left_mean = np.mean(seg_flux[:mid])
            right_mean = np.mean(seg_flux[mid:])
            flux_range = max(np.ptp(seg_flux), 1e-10)
            asymmetry = float(np.clip(abs(left_mean - right_mean) / flux_range, 0, 1))
        else:
            asymmetry = 0.0

        events.append(
            DimmingEvent(
                start_time=start_time,
                end_time=end_time,
                depth=depth,
                duration=duration,
                asymmetry=asymmetry,
            )
        )

    return events


# ============================================================================
# Public API
# ============================================================================

def detect_transit_anomaly(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
) -> TransitAnomalyResult:
    """Run BLS-based periodic transit anomaly analysis on a light curve.

    Parameters
    ----------
    time : array-like
        Time stamps (e.g. BJD - 2457000).
    flux : array-like
        Normalised flux values (baseline ~ 1.0).
    flux_err : array-like or None
        Per-point flux uncertainties.  If *None*, uniform weights are used.

    Returns
    -------
    TransitAnomalyResult
        Composite anomaly assessment including symmetry, depth variability,
        duration consistency, shape residuals, and a final 0-1 anomaly score.
    """
    sigma_threshold = _configured_anomaly_sigma()

    time, flux, flux_err = _validate_inputs(time, flux, flux_err)
    n_pts = len(time)
    logger.info(
        "Running periodic transit anomaly detection on %d data points", n_pts,
    )

    if n_pts < 20:
        logger.warning("Too few data points (%d) for BLS analysis", n_pts)
        return TransitAnomalyResult(
            period=0.0, depth=0.0, symmetry_score=0.0,
            depth_variability=0.0, duration_consistency=0.0,
            shape_residual=0.0, anomaly_score=0.0, is_anomalous=False,
        )

    # -- BLS periodogram -----------------------------------------------------
    baseline = time[-1] - time[0]
    median_cadence = float(np.median(np.diff(time)))

    # Period search range: minimum period must allow at least ~2 transits
    # in the baseline, with a floor at 10x the cadence.
    min_period = max(10.0 * median_cadence, 0.5)
    max_period = baseline / 2.0

    if min_period >= max_period:
        min_period = baseline * 0.05
        max_period = baseline * 0.5

    # Duration grid: transit durations are typically 1-5% of the orbital
    # period.  The maximum trial duration MUST be shorter than min_period
    # (astropy BLS requirement).
    min_duration = max(2.0 * median_cadence, 0.005 * min_period)
    max_duration = min(0.15 * min_period, 0.05 * baseline)
    max_duration = max(max_duration, min_duration * 2.0)
    # Safety clamp: BLS requires max(duration) < min_period
    max_duration = min(max_duration, 0.9 * min_period)

    durations = np.linspace(min_duration, max_duration, 10)

    bls = BoxLeastSquares(time, flux, dy=flux_err)

    # Autoperiod: let BLS choose a sensible frequency grid
    try:
        periodogram = bls.autopower(
            duration=durations,
            minimum_period=min_period,
            maximum_period=max_period,
        )
    except Exception as exc:
        logger.error("BLS autopower failed: %s", exc)
        return TransitAnomalyResult(
            period=0.0, depth=0.0, symmetry_score=0.0,
            depth_variability=0.0, duration_consistency=0.0,
            shape_residual=0.0, anomaly_score=0.0, is_anomalous=False,
        )

    # Best period from BLS
    best_idx = np.argmax(periodogram.power)
    best_period = float(periodogram.period[best_idx])
    best_duration = float(periodogram.duration[best_idx])
    best_t0 = float(periodogram.transit_time[best_idx])

    # Best-fit depth from BLS stats
    try:
        stats = bls.compute_stats(best_period, best_duration, best_t0)
        best_depth = float(stats["depth"][0]) if hasattr(stats["depth"], "__len__") else float(stats["depth"])
    except Exception:
        best_depth = float(1.0 - np.min(flux))

    logger.info(
        "BLS best period=%.6f d, depth=%.6f, duration=%.6f d",
        best_period, best_depth, best_duration,
    )

    # -- Significance check: is the BLS depth above the noise floor? ---------
    # Estimate photometric noise from the MAD of point-to-point differences,
    # which is robust to long-term trends and transit dips.
    if len(flux) > 2:
        diff_flux = np.diff(flux)
        mad_diff = float(np.median(np.abs(diff_flux - np.median(diff_flux))))
        noise_est = 1.4826 * mad_diff / np.sqrt(2.0) if mad_diff > 0 else float(np.std(flux))
    else:
        noise_est = float(np.std(flux))
    if noise_est <= 0:
        noise_est = 1e-10

    depth_snr = best_depth / noise_est
    logger.info(
        "BLS depth SNR=%.2f (depth=%.6f, noise=%.6f)",
        depth_snr, best_depth, noise_est,
    )

    # If the BLS "detection" is not significantly above noise, the periodic
    # transit analysis is fitting noise -- return a zero anomaly score.
    if depth_snr < sigma_threshold:
        logger.info(
            "BLS depth SNR (%.2f) below threshold (%.1f); "
            "no significant periodic transit detected",
            depth_snr, sigma_threshold,
        )
        return TransitAnomalyResult(
            period=best_period,
            depth=best_depth,
            symmetry_score=0.0,
            depth_variability=0.0,
            duration_consistency=0.0,
            shape_residual=0.0,
            anomaly_score=0.0,
            is_anomalous=False,
        )

    # -- Phase-fold the light curve ------------------------------------------
    phase = ((time - best_t0) % best_period) / best_period
    # Centre phase on 0.0 (transit at phase 0)
    phase[phase > 0.5] -= 1.0
    sort_ph = np.argsort(phase)
    phase = phase[sort_ph]
    folded_flux = flux[sort_ph]

    duration_phase = best_duration / best_period

    # -- Symmetry score ------------------------------------------------------
    # Isolate in-transit points plus ingress/egress shoulders (3x the BLS
    # duration in phase) to give the symmetry comparison enough signal.
    symmetry_half_width = max(duration_phase * 3.0, 0.05)
    in_transit_mask = np.abs(phase) <= symmetry_half_width
    transit_phase = phase[in_transit_mask]
    transit_flux = folded_flux[in_transit_mask]
    symmetry = _compute_symmetry(transit_phase, transit_flux)

    # -- Individual dip extraction -------------------------------------------
    ind_depths, ind_durations = _extract_individual_dips(
        time, flux, best_period, best_t0, best_duration,
    )

    depth_var = _compute_depth_variability(ind_depths)
    dur_consistency = _compute_duration_consistency(ind_durations)

    # -- Shape residual ------------------------------------------------------
    shape_res = _compute_shape_residual(phase, folded_flux, best_depth, duration_phase)

    # -- Build detected dips list -------------------------------------------
    detected_dips: List[DetectedDip] = []
    n_dips = len(ind_depths)
    unique_epochs = np.round(((time - best_t0) / best_period)).astype(int)
    epoch_set = sorted(set(unique_epochs))

    for i, d in enumerate(ind_depths):
        epoch_val = float(epoch_set[i]) if i < len(epoch_set) else float(i)
        dur_val = float(ind_durations[i]) if i < len(ind_durations) else best_duration
        detected_dips.append(
            DetectedDip(
                epoch=epoch_val,
                depth=float(d),
                duration=dur_val,
                symmetry_score=symmetry,
                shape_residual=shape_res,
            )
        )

    # -- Composite anomaly score (0=normal, 1=highly anomalous) --------------
    # Combine metrics using a weighted-mean plus a max-boost term.
    # The max-boost ensures that a single very strong anomaly signal
    # (e.g. large depth variability alone) can still flag the target,
    # while multi-metric anomalies score even higher.
    #
    # Weights for the mean component:
    #   symmetry:           0.30  (most diagnostic for megastructures)
    #   depth_variability:  0.30  (constant depth = natural)
    #   duration_incons:    0.15  (Kepler's 3rd law)
    #   shape_residual:     0.25  (poor box fit = unusual shape)
    w_sym, w_dep, w_dur, w_shp = 0.30, 0.30, 0.15, 0.25
    weighted_mean = (
        w_sym * symmetry
        + w_dep * depth_var
        + w_dur * dur_consistency
        + w_shp * shape_res
    )
    peak_signal = max(symmetry, depth_var, dur_consistency, shape_res)

    # 60% weighted mean + 40% peak signal: a single strong metric
    # can push the score up significantly.
    composite = 0.6 * weighted_mean + 0.4 * peak_signal

    # Natural transit guard: periodic transits with very consistent
    # depths (CV < 0.10) across 3+ events are almost certainly natural
    # planets.  Symmetry and shape_residual are sensitive to limb
    # darkening, Rossiter-McLaughlin effect, and photometric noise,
    # which inflate the composite for normal transits.  Discount the
    # score proportionally to depth consistency.
    if depth_var < 0.10 and n_dips >= 3:
        # Scale: CV=0 → factor=0.4, CV=0.10 → factor=1.0
        natural_factor = 0.4 + 6.0 * depth_var
        composite *= natural_factor
        logger.debug(
            "Natural transit guard: depth_var=%.3f n_dips=%d "
            "factor=%.3f composite %.4f -> %.4f",
            depth_var, n_dips, natural_factor,
            composite / natural_factor, composite,
        )

    anomaly_score = float(np.clip(composite, 0.0, 1.0))

    # Flag as anomalous if composite exceeds a sigma-derived threshold.
    # Map sigma_threshold into a 0-1 cutoff: lower sigma = more sensitive.
    # Default sigma=3 -> threshold ~0.20
    anomaly_threshold = 0.6 / sigma_threshold
    is_anomalous = anomaly_score > anomaly_threshold

    logger.info(
        "Transit anomaly analysis complete: score=%.4f (threshold=%.4f) "
        "anomalous=%s  [sym=%.3f dep=%.3f dur=%.3f shp=%.3f]",
        anomaly_score, anomaly_threshold, is_anomalous,
        symmetry, depth_var, dur_consistency, shape_res,
    )

    return TransitAnomalyResult(
        period=best_period,
        depth=best_depth,
        symmetry_score=symmetry,
        depth_variability=depth_var,
        duration_consistency=dur_consistency,
        shape_residual=shape_res,
        anomaly_score=anomaly_score,
        is_anomalous=is_anomalous,
        detected_dips=detected_dips,
    )


def detect_irregular_dimming(
    time: np.ndarray,
    flux: np.ndarray,
    contamination: float = 0.05,
    window_size: int = 50,
    step: int = 10,
) -> IrregularDimmingResult:
    """Detect non-periodic, irregular dimming events using Isolation Forest.

    Extracts windowed statistical features from the light curve and
    identifies anomalous windows that deviate from the bulk distribution.
    Designed to find Tabby's-Star-type aperiodic dimming.

    Parameters
    ----------
    time : array-like
        Time stamps.
    flux : array-like
        Normalised flux values.
    contamination : float
        Expected fraction of anomalous windows (Isolation Forest parameter).
    window_size : int
        Number of data points per sliding window.
    step : int
        Stride between consecutive windows.

    Returns
    -------
    IrregularDimmingResult
        Summary of detected dimming events with anomaly scores.
    """
    time, flux, _ = _validate_inputs(time, flux)
    n_pts = len(time)
    logger.info(
        "Running irregular dimming detection on %d data points "
        "(window=%d, step=%d)",
        n_pts, window_size, step,
    )

    if n_pts < window_size:
        logger.warning("Too few data points for windowed analysis")
        return IrregularDimmingResult(n_events=0, max_depth=0.0, anomaly_score=0.0)

    # -- Feature extraction --------------------------------------------------
    features, centres, indices = _extract_window_features(
        time, flux, window_size=window_size, step=step,
    )

    if len(features) < 10:
        logger.warning("Too few windows (%d) for Isolation Forest", len(features))
        return IrregularDimmingResult(n_events=0, max_depth=0.0, anomaly_score=0.0)

    # -- Scale features and run Isolation Forest -----------------------------
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    iso_forest = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=42,
    )
    predictions = iso_forest.fit_predict(features_scaled)
    # decision_function: lower = more anomalous
    scores = iso_forest.decision_function(features_scaled)

    anomalous_windows = predictions == -1
    n_anomalous = int(np.sum(anomalous_windows))
    logger.info("Isolation Forest flagged %d / %d windows", n_anomalous, len(predictions))

    # -- Map anomalous windows back to data-point-level mask -----------------
    point_anomaly = np.zeros(n_pts, dtype=bool)
    for i, is_anom in enumerate(anomalous_windows):
        if is_anom:
            s, e = indices[i]
            point_anomaly[s:e] = True

    # -- Build event list from contiguous anomalous regions ------------------
    events = _events_from_anomaly_mask(time, flux, point_anomaly)

    max_depth = float(max((e.depth for e in events), default=0.0))

    # Overall anomaly score: fraction of anomalous windows weighted by
    # how extreme their isolation scores are.
    if n_anomalous > 0 and len(scores) > 0:
        anomalous_scores = scores[anomalous_windows]
        # Normalise: scores are typically in [-0.5, 0.5]; map extreme
        # negative values to higher anomaly scores.
        raw = float(np.mean(-anomalous_scores))
        overall_score = float(np.clip(raw, 0.0, 1.0))
    else:
        overall_score = 0.0

    logger.info(
        "Irregular dimming detection complete: %d events, "
        "max_depth=%.6f, anomaly_score=%.4f",
        len(events), max_depth, overall_score,
    )

    return IrregularDimmingResult(
        n_events=len(events),
        events=events,
        max_depth=max_depth,
        anomaly_score=overall_score,
    )


# ============================================================================
# CLI demo / self-test
# ============================================================================

def _make_clean_transit(
    n_points: int = 5000,
    period: float = 3.5,
    depth: float = 0.01,
    duration: float = 0.15,
    noise: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic clean, symmetric planetary transit light curve.

    Uses a smooth limb-darkened transit profile (raised-cosine) rather
    than a hard box, which is more physically realistic and produces
    genuinely symmetric ingress/egress.
    """
    rng = np.random.default_rng(seed=42)
    time = np.linspace(0, 30, n_points)
    flux = np.ones(n_points) + rng.normal(0, noise, n_points)

    # Phase relative to mid-transit (transit centred at phase = 0.5)
    phase = ((time % period) / period)
    phase_offset = np.abs(phase - 0.5)
    half_dur = (duration / period) / 2.0

    # Smooth symmetric profile: raised cosine inside transit window
    in_transit = phase_offset < half_dur
    transit_phase = phase_offset[in_transit] / half_dur  # 0 at centre, 1 at edge
    profile = 0.5 * (1.0 + np.cos(np.pi * transit_phase))  # 1 at centre, 0 at edge
    flux[in_transit] -= depth * profile

    return time, flux


def _make_anomalous_transit(
    n_points: int = 5000,
    period: float = 3.5,
    noise: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic asymmetric, variable-depth transit (megastructure
    candidate).

    Each transit event has:
      - A randomly varying depth (large coefficient of variation)
      - A strongly asymmetric profile: very steep ingress, long slow egress
      - Variable duration from event to event
    """
    rng = np.random.default_rng(seed=99)
    time = np.linspace(0, 30, n_points)
    flux = np.ones(n_points) + rng.normal(0, noise, n_points)

    n_transits = int(30.0 / period) + 1
    for k in range(n_transits):
        t_centre = 0.5 * period + k * period
        # Strongly variable depth: factor of ~3x variation between transits
        depth = 0.005 + 0.020 * rng.random()
        # Extremely asymmetric: very short ingress, very long egress
        ingress_dur = 0.02 + 0.01 * rng.random()
        egress_dur = 0.20 + 0.10 * rng.random()

        for i, t in enumerate(time):
            dt = t - t_centre
            if -ingress_dur <= dt < 0:
                # Very steep ingress (near-vertical drop)
                frac = (dt + ingress_dur) / ingress_dur
                flux[i] -= depth * frac ** 0.3  # concave profile
            elif 0 <= dt < egress_dur:
                # Very slow, gradual egress (long tail)
                frac = 1.0 - (dt / egress_dur) ** 2  # quadratic decay
                flux[i] -= depth * frac

    return time, flux


def _make_irregular_dimming(
    n_points: int = 8000,
    noise: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic Tabby's-Star-like light curve with irregular,
    non-periodic dimming events of varying depth and duration."""
    rng = np.random.default_rng(seed=77)
    time = np.linspace(0, 100, n_points)
    flux = np.ones(n_points) + rng.normal(0, noise, n_points)

    # Insert several irregular dimming events at random times
    event_params = [
        (15.0, 0.5, 0.005),    # (centre, half-width, depth) -- mild
        (35.0, 1.0, 0.015),    # moderate
        (52.0, 0.3, 0.020),    # deep, short
        (73.0, 2.0, 0.008),    # shallow, long
        (88.0, 0.8, 0.025),    # very deep
    ]

    for centre, half_width, depth in event_params:
        mask = np.abs(time - centre) < half_width
        # Asymmetric Gaussian-ish shape
        dt = time[mask] - centre
        profile = depth * np.exp(-0.5 * ((dt - 0.1 * half_width) / (0.4 * half_width)) ** 2)
        flux[mask] -= profile

    return time, flux


if __name__ == "__main__":
    print("=" * 70)
    print("  Project EXODUS  --  Transit Anomaly Detection Demo")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Clean periodic transit (should be LOW anomaly)
    # ------------------------------------------------------------------
    print("\n[1] Clean symmetric transit (planetary model)")
    print("-" * 50)
    time_clean, flux_clean = _make_clean_transit()
    result_clean = detect_transit_anomaly(time_clean, flux_clean)

    print(f"    Detected period     : {result_clean.period:.4f} d")
    print(f"    Depth               : {result_clean.depth:.6f}")
    print(f"    Symmetry score      : {result_clean.symmetry_score:.4f}")
    print(f"    Depth variability   : {result_clean.depth_variability:.4f}")
    print(f"    Duration consistency: {result_clean.duration_consistency:.4f}")
    print(f"    Shape residual      : {result_clean.shape_residual:.4f}")
    print(f"    ANOMALY SCORE       : {result_clean.anomaly_score:.4f}")
    print(f"    Is anomalous?       : {result_clean.is_anomalous}")
    print(f"    Dips detected       : {len(result_clean.detected_dips)}")

    # ------------------------------------------------------------------
    # 2. Anomalous transit (should be HIGH anomaly)
    # ------------------------------------------------------------------
    print("\n[2] Asymmetric + variable-depth transit (megastructure candidate)")
    print("-" * 50)
    time_anom, flux_anom = _make_anomalous_transit()
    result_anom = detect_transit_anomaly(time_anom, flux_anom)

    print(f"    Detected period     : {result_anom.period:.4f} d")
    print(f"    Depth               : {result_anom.depth:.6f}")
    print(f"    Symmetry score      : {result_anom.symmetry_score:.4f}")
    print(f"    Depth variability   : {result_anom.depth_variability:.4f}")
    print(f"    Duration consistency: {result_anom.duration_consistency:.4f}")
    print(f"    Shape residual      : {result_anom.shape_residual:.4f}")
    print(f"    ANOMALY SCORE       : {result_anom.anomaly_score:.4f}")
    print(f"    Is anomalous?       : {result_anom.is_anomalous}")
    print(f"    Dips detected       : {len(result_anom.detected_dips)}")

    # ------------------------------------------------------------------
    # 3. Irregular dimming (Tabby's Star type)
    # ------------------------------------------------------------------
    print("\n[3] Irregular dimming events (Tabby's Star type)")
    print("-" * 50)
    time_irr, flux_irr = _make_irregular_dimming()
    result_irr = detect_irregular_dimming(time_irr, flux_irr)

    print(f"    Events detected     : {result_irr.n_events}")
    print(f"    Max depth           : {result_irr.max_depth:.6f}")
    print(f"    Anomaly score       : {result_irr.anomaly_score:.4f}")
    if result_irr.events:
        print("    Event details:")
        for i, evt in enumerate(result_irr.events):
            print(
                f"      [{i+1}] t={evt.start_time:.2f}..{evt.end_time:.2f}  "
                f"depth={evt.depth:.6f}  dur={evt.duration:.2f} d  "
                f"asym={evt.asymmetry:.4f}"
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"    Clean transit anomaly score  : {result_clean.anomaly_score:.4f}"
          f"  ({'PASS' if not result_clean.is_anomalous else 'UNEXPECTED'})")
    print(f"    Anomalous transit score      : {result_anom.anomaly_score:.4f}"
          f"  ({'PASS' if result_anom.is_anomalous else 'CHECK'})")
    print(f"    Irregular dimming events     : {result_irr.n_events}"
          f"  ({'PASS' if result_irr.n_events > 0 else 'CHECK'})")
    print("=" * 70)
    print("  Demo complete.")
    print("=" * 70)
