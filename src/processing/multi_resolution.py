"""
Multi-Resolution Temporal Analysis for Project EXODUS.

Ensures we search at multiple time resolutions so we don't accidentally
filter out the very signals we're looking for.  Standard time-averaging
(integration) improves SNR for persistent signals but KILLS brief
transient signals.  This module runs anomaly detection at every available
time resolution and flags signals that appear at one resolution but vanish
at another.

Resolution tiers:
  Tier 1: Full integration (hours/days) — standard approach
  Tier 2: 1-minute bins
  Tier 3: 10-second bins
  Tier 4: 1-second bins (if data supports it)

A signal found at short timescales but NOT at long timescales is exactly
the kind of transient that standard pipelines miss — and exactly the kind
a technological beacon or pulsed communication would produce.

Public API
----------
multi_resolution_analysis(time, data, data_type="radio")
    Run anomaly detection at all available resolutions.

analyze_radio_multiresolution(spectrogram, freqs, times)
    Multi-resolution analysis specifically for radio spectrograms.

analyze_lightcurve_multiresolution(time, flux)
    Multi-resolution analysis for optical/IR light curves.
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

log = get_logger("processing.multi_resolution")


# =====================================================================
#  Data classes
# =====================================================================

@dataclass
class ResolutionResult:
    """Result from a single resolution tier."""
    resolution_name: str        # e.g. "1-second", "10-second", "1-minute", "full"
    bin_size_seconds: float
    n_bins: int
    n_anomalies: int
    peak_snr: float
    anomaly_times: List[float] = field(default_factory=list)
    anomaly_snrs: List[float] = field(default_factory=list)
    rms_noise: float = 0.0


@dataclass
class MultiResolutionResult:
    """Full multi-resolution analysis result."""
    data_type: str              # "radio", "lightcurve", "ir_timeseries"
    n_resolutions: int
    resolution_results: List[ResolutionResult] = field(default_factory=list)
    transient_only: List[Dict[str, Any]] = field(default_factory=list)
    persistent_only: List[Dict[str, Any]] = field(default_factory=list)
    resolution_discrepant: bool = False  # True if signals differ between resolutions
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_type": self.data_type,
            "n_resolutions": self.n_resolutions,
            "resolution_discrepant": self.resolution_discrepant,
            "n_transient_only": len(self.transient_only),
            "n_persistent_only": len(self.persistent_only),
            "summary": self.summary,
            "resolutions": [
                {
                    "name": r.resolution_name,
                    "bin_size_s": r.bin_size_seconds,
                    "n_bins": r.n_bins,
                    "n_anomalies": r.n_anomalies,
                    "peak_snr": r.peak_snr,
                }
                for r in self.resolution_results
            ],
        }


# =====================================================================
#  Core multi-resolution engine
# =====================================================================

def _compute_resolution_tiers(
    total_duration_sec: float,
    n_samples: int,
) -> List[Tuple[str, float]]:
    """Determine which resolution tiers are viable for the data.

    Parameters
    ----------
    total_duration_sec : float
        Total time span of the data.
    n_samples : int
        Number of data points.

    Returns
    -------
    list of (name, bin_size_seconds)
        Viable resolution tiers, from finest to coarsest.
    """
    cadence = total_duration_sec / max(n_samples - 1, 1)

    tiers = []

    # Tier 4: 1-second bins (need cadence < 1s and at least 30 bins)
    if cadence < 1.0 and total_duration_sec >= 30:
        tiers.append(("1-second", 1.0))

    # Tier 3: 10-second bins
    if cadence < 10.0 and total_duration_sec >= 300:
        tiers.append(("10-second", 10.0))

    # Tier 2: 1-minute bins
    if cadence < 60.0 and total_duration_sec >= 600:
        tiers.append(("1-minute", 60.0))

    # Tier 1: Full integration — always available
    # Use total duration / 10 as bin size (at least 10 bins)
    full_bin = max(total_duration_sec / 10.0, cadence * 10)
    tiers.append(("full-integration", full_bin))

    return tiers


def _bin_timeseries(
    time: np.ndarray,
    values: np.ndarray,
    bin_size_sec: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin a time series into fixed-size bins.

    Returns
    -------
    bin_centres, bin_means, bin_stds : ndarray
    """
    t0 = time[0]
    t_end = time[-1]
    n_bins = max(int((t_end - t0) / bin_size_sec), 1)

    bin_edges = np.linspace(t0, t_end, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_means = np.zeros(n_bins)
    bin_stds = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (time >= bin_edges[i]) & (time < bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_means[i] = np.mean(values[mask])
            bin_stds[i] = np.std(values[mask]) if np.sum(mask) > 1 else 0.0
        else:
            bin_means[i] = np.nan
            bin_stds[i] = np.nan

    # Remove empty bins
    valid = np.isfinite(bin_means)
    return bin_centres[valid], bin_means[valid], bin_stds[valid]


def _detect_anomalies_in_tier(
    bin_centres: np.ndarray,
    bin_values: np.ndarray,
    sigma_threshold: float = 5.0,
) -> ResolutionResult:
    """Detect anomalies in a binned time series using MAD-based z-scores.

    Parameters
    ----------
    bin_centres : ndarray
        Time centres of each bin.
    bin_values : ndarray
        Mean value in each bin.
    sigma_threshold : float
        Detection threshold in sigma.

    Returns
    -------
    ResolutionResult (partially filled — resolution_name and bin_size_seconds
    must be set by the caller).
    """
    if len(bin_values) < 5:
        return ResolutionResult(
            resolution_name="", bin_size_seconds=0,
            n_bins=len(bin_values), n_anomalies=0, peak_snr=0.0,
        )

    # Robust noise estimate via MAD
    median_val = np.median(bin_values)
    mad = np.median(np.abs(bin_values - median_val))
    sigma = 1.4826 * mad if mad > 0 else np.std(bin_values)

    if sigma <= 0:
        return ResolutionResult(
            resolution_name="", bin_size_seconds=0,
            n_bins=len(bin_values), n_anomalies=0, peak_snr=0.0,
            rms_noise=0.0,
        )

    z_scores = np.abs(bin_values - median_val) / sigma
    anomaly_mask = z_scores > sigma_threshold

    n_anomalies = int(np.sum(anomaly_mask))
    peak_snr = float(np.max(z_scores)) if len(z_scores) > 0 else 0.0

    anomaly_times = bin_centres[anomaly_mask].tolist() if n_anomalies > 0 else []
    anomaly_snrs = z_scores[anomaly_mask].tolist() if n_anomalies > 0 else []

    return ResolutionResult(
        resolution_name="",
        bin_size_seconds=0,
        n_bins=len(bin_values),
        n_anomalies=n_anomalies,
        peak_snr=peak_snr,
        anomaly_times=anomaly_times,
        anomaly_snrs=anomaly_snrs,
        rms_noise=float(sigma),
    )


# =====================================================================
#  Public API
# =====================================================================

def multi_resolution_analysis(
    time: np.ndarray,
    data: np.ndarray,
    data_type: str = "generic",
    sigma_threshold: float = 5.0,
) -> MultiResolutionResult:
    """Run anomaly detection at all available time resolutions.

    Parameters
    ----------
    time : ndarray
        Time stamps in seconds (or any consistent unit).
    data : ndarray
        Data values (flux, power, magnitude, etc.).
    data_type : str
        Label for the data type.
    sigma_threshold : float
        Detection threshold in sigma for each resolution tier.

    Returns
    -------
    MultiResolutionResult
    """
    time = np.asarray(time, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)

    # Clean
    mask = np.isfinite(time) & np.isfinite(data)
    time, data = time[mask], data[mask]

    if len(time) < 10:
        return MultiResolutionResult(
            data_type=data_type, n_resolutions=0,
            summary="Insufficient data points for multi-resolution analysis",
        )

    # Sort
    order = np.argsort(time)
    time, data = time[order], data[order]

    total_duration = float(time[-1] - time[0])
    tiers = _compute_resolution_tiers(total_duration, len(time))

    log.info(
        "Multi-resolution analysis: %d points, %.1f sec duration, %d tiers",
        len(time), total_duration, len(tiers),
    )

    resolution_results = []
    for tier_name, bin_size in tiers:
        bin_c, bin_v, bin_s = _bin_timeseries(time, data, bin_size)
        result = _detect_anomalies_in_tier(bin_c, bin_v, sigma_threshold)
        result.resolution_name = tier_name
        result.bin_size_seconds = bin_size
        resolution_results.append(result)

        log.info(
            "  [%s] bin=%.1fs, n_bins=%d, anomalies=%d, peak_snr=%.1f",
            tier_name, bin_size, result.n_bins, result.n_anomalies, result.peak_snr,
        )

    # ── Identify resolution-discrepant signals ──────────────────────
    # Transient-only: found at fine resolution but NOT at coarse
    # Persistent-only: found at coarse resolution but NOT at fine
    transient_only = []
    persistent_only = []

    if len(resolution_results) >= 2:
        finest = resolution_results[0]
        coarsest = resolution_results[-1]

        # Signals in finest but not in coarsest
        if finest.n_anomalies > 0 and coarsest.n_anomalies == 0:
            transient_only.append({
                "resolution": finest.resolution_name,
                "n_anomalies": finest.n_anomalies,
                "peak_snr": finest.peak_snr,
                "times": finest.anomaly_times[:10],
                "note": (
                    f"Signal detected at {finest.resolution_name} resolution "
                    f"but VANISHES at {coarsest.resolution_name} — "
                    f"this is a transient being killed by time-averaging!"
                ),
            })

        # Signals in coarsest but not in finest
        if coarsest.n_anomalies > 0 and finest.n_anomalies == 0:
            persistent_only.append({
                "resolution": coarsest.resolution_name,
                "n_anomalies": coarsest.n_anomalies,
                "peak_snr": coarsest.peak_snr,
                "note": (
                    f"Signal detected at {coarsest.resolution_name} resolution "
                    f"but NOT at {finest.resolution_name} — "
                    f"low-level persistent signal below single-epoch noise."
                ),
            })

    resolution_discrepant = len(transient_only) > 0 or len(persistent_only) > 0

    # Build summary
    anomaly_counts = [r.n_anomalies for r in resolution_results]
    tier_names = [r.resolution_name for r in resolution_results]
    summary_parts = [
        f"Analyzed {len(resolution_results)} resolution tiers: "
        f"{', '.join(tier_names)}.",
    ]
    for r in resolution_results:
        summary_parts.append(
            f"  {r.resolution_name}: {r.n_anomalies} anomalies (peak SNR={r.peak_snr:.1f})"
        )
    if transient_only:
        summary_parts.append(
            f"WARNING: {len(transient_only)} transient-only signal(s) found "
            f"(visible at fine resolution, killed by integration)!"
        )
    if persistent_only:
        summary_parts.append(
            f"NOTE: {len(persistent_only)} persistent-only signal(s) found "
            f"(visible only at coarse resolution)."
        )

    return MultiResolutionResult(
        data_type=data_type,
        n_resolutions=len(resolution_results),
        resolution_results=resolution_results,
        transient_only=transient_only,
        persistent_only=persistent_only,
        resolution_discrepant=resolution_discrepant,
        summary="\n".join(summary_parts),
    )


def analyze_radio_multiresolution(
    spectrogram: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    sigma_threshold: float = 5.0,
) -> MultiResolutionResult:
    """Multi-resolution analysis for radio spectrograms.

    Collapses the spectrogram along the frequency axis (total power vs time)
    and runs multi-resolution anomaly detection on the resulting time series.

    Parameters
    ----------
    spectrogram : ndarray, shape (n_freq, n_time)
        Power spectrogram.
    freqs : ndarray
        Frequency axis in MHz.
    times : ndarray
        Time axis in seconds.
    sigma_threshold : float
        Detection threshold.

    Returns
    -------
    MultiResolutionResult
    """
    spectrogram = np.asarray(spectrogram)
    times = np.asarray(times)

    # Total power vs time (collapse frequency axis)
    total_power = np.nanmean(spectrogram, axis=0)

    return multi_resolution_analysis(
        times, total_power, data_type="radio_spectrogram",
        sigma_threshold=sigma_threshold,
    )


def analyze_lightcurve_multiresolution(
    time: np.ndarray,
    flux: np.ndarray,
    sigma_threshold: float = 5.0,
) -> MultiResolutionResult:
    """Multi-resolution analysis for optical/IR light curves.

    Parameters
    ----------
    time : ndarray
        Time stamps (days, typically BJD).
    flux : ndarray
        Normalized flux values.
    sigma_threshold : float
        Detection threshold.

    Returns
    -------
    MultiResolutionResult
    """
    # Convert days to seconds for consistent resolution tiers
    time_sec = np.asarray(time, dtype=np.float64) * 86400.0
    flux = np.asarray(flux, dtype=np.float64)

    return multi_resolution_analysis(
        time_sec, flux, data_type="lightcurve",
        sigma_threshold=sigma_threshold,
    )


# =====================================================================
#  CLI demo / self-test
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Project EXODUS -- Multi-Resolution Temporal Analysis Demo")
    print("=" * 70)

    rng = np.random.default_rng(seed=42)

    # ── Test 1: Inject a 5-second pulse into a 1-hour spectrogram ────
    print("\n[1] 5-second radio pulse in 1-hour observation")
    print("-" * 50)

    n_freq = 256
    n_time = 3600  # 1 point per second for 1 hour
    times = np.arange(n_time, dtype=np.float64)
    freqs = np.linspace(1400, 1500, n_freq)

    # Noise spectrogram
    spec = np.abs(rng.normal(10, 1, (n_freq, n_time)))

    # Inject a 5-second bright pulse at t=1800s
    pulse_start = 1800
    pulse_end = 1805
    spec[:, pulse_start:pulse_end] += 30.0  # very bright pulse

    result1 = analyze_radio_multiresolution(spec, freqs, times)
    print(f"  Resolutions analyzed: {result1.n_resolutions}")
    for r in result1.resolution_results:
        print(f"    {r.resolution_name:<20s}: {r.n_anomalies} anomalies, peak SNR={r.peak_snr:.1f}")
    print(f"  Resolution discrepant: {result1.resolution_discrepant}")
    if result1.transient_only:
        for t in result1.transient_only:
            print(f"  TRANSIENT: {t['note']}")
    print(f"  >> {'PASS' if result1.resolution_discrepant else 'CHECK'}: "
          f"Pulse visible at fine but not coarse resolution")

    # ── Test 2: Persistent weak signal in a radio observation ────────
    print("\n[2] Persistent weak signal over full observation")
    print("-" * 50)

    spec2 = np.abs(rng.normal(10, 1, (n_freq, n_time)))
    # Persistent signal: slightly elevated across ALL time
    spec2[128, :] += 3.0  # one frequency channel slightly elevated

    # Collapse to total power
    total_power = np.nanmean(spec2, axis=0)
    result2 = multi_resolution_analysis(times, total_power, "radio_persistent")
    print(f"  Resolutions analyzed: {result2.n_resolutions}")
    for r in result2.resolution_results:
        print(f"    {r.resolution_name:<20s}: {r.n_anomalies} anomalies, peak SNR={r.peak_snr:.1f}")

    # ── Test 3: Light curve with brief dip (hours in day-cadence data)
    print("\n[3] Brief transit dip in day-cadence light curve")
    print("-" * 50)

    n_pts = 5000
    lc_time = np.linspace(0, 100, n_pts)  # 100 days, ~30-min cadence in seconds
    lc_time_sec = lc_time * 86400.0
    lc_flux = np.ones(n_pts) + rng.normal(0, 0.001, n_pts)

    # Inject a brief (2-hour = 0.083 day) dip at day 50
    dip_mask = (lc_time > 49.96) & (lc_time < 50.04)
    lc_flux[dip_mask] -= 0.015

    result3 = analyze_lightcurve_multiresolution(lc_time, lc_flux)
    print(f"  Resolutions analyzed: {result3.n_resolutions}")
    for r in result3.resolution_results:
        print(f"    {r.resolution_name:<20s}: {r.n_anomalies} anomalies, peak SNR={r.peak_snr:.1f}")
    print(f"  Resolution discrepant: {result3.resolution_discrepant}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  Test 1 (transient pulse): discrepant={result1.resolution_discrepant} "
          f"({'PASS' if result1.resolution_discrepant else 'CHECK'})")
    print(f"  Test 2 (persistent):      discrepant={result2.resolution_discrepant}")
    print(f"  Test 3 (brief dip):       discrepant={result3.resolution_discrepant}")
    print("=" * 70)
    print("  Done.")
    print("=" * 70)
