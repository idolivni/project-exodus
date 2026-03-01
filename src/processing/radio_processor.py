"""
Breakthrough Listen radio spectrogram processor for Project EXODUS.

Processes frequency x time spectrograms to detect narrowband Doppler-drifting
signals that could indicate extraterrestrial technosignatures.  The pipeline:

    1.  RFI flagging  -- mask known interference, broadband sweeps, and
        persistent constant-frequency lines.
    2.  Dedoppler search -- brute-force Taylor-tree-style integration over
        a grid of drift rates, summing power along each hypothesised
        Doppler track.
    3.  SNR estimation -- compare each candidate's integrated power to the
        local noise floor.
    4.  Candidate extraction -- return every detection above the configured
        sigma threshold.

The module also provides `inject_signal` for testing with synthetic drifting
narrowband signals.
"""

from __future__ import annotations

import sys
import time as _time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------------------
# Project utilities
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger, get_config

logger = get_logger("processing.radio_processor")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _anomaly_sigma() -> float:
    """Return the anomaly sigma threshold from project config, with fallback."""
    try:
        cfg = get_config()
        return float(cfg.get("search", {}).get("anomaly_sigma", 3.0))
    except Exception:
        return 3.0


# ---------------------------------------------------------------------------
# Known terrestrial RFI frequencies (MHz) -- common offenders
# ---------------------------------------------------------------------------
# These are approximate centre frequencies of known persistent terrestrial
# transmitters (mobile, Wi-Fi, GPS, aviation, satellite downlinks, etc.)
# that contaminate radio telescope observations.
KNOWN_RFI_FREQUENCIES_MHZ: List[Tuple[float, float]] = [
    (700.0, 770.0),      # LTE Band 12/13/17
    (824.0, 849.0),      # Cell-phone uplink (US)
    (869.0, 894.0),      # Cell-phone downlink (US)
    (928.0, 960.0),      # ISM / paging
    (1164.0, 1189.0),    # GPS L5 / Galileo E5a
    (1200.0, 1210.0),    # Radar (L-band)
    (1217.0, 1237.0),    # GPS L2
    (1525.0, 1559.0),    # Inmarsat / Globalstar downlink
    (1559.0, 1591.0),    # GPS L1 / Galileo E1
    (1610.0, 1618.5),    # Iridium
    (1930.0, 1990.0),    # PCS downlink
    (2300.0, 2400.0),    # WiMAX / S-band radar
    (2400.0, 2500.0),    # Wi-Fi / Bluetooth 2.4 GHz
    (5150.0, 5350.0),    # Wi-Fi 5 GHz (UNII-1/2)
    (5470.0, 5725.0),    # Wi-Fi 5 GHz (UNII-2e)
    (5725.0, 5875.0),    # Wi-Fi 5 GHz (UNII-3)
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SignalCandidate:
    """A single narrowband candidate detection."""
    frequency_hz: float           # Centre frequency of the detection (Hz)
    drift_rate_hz_per_s: float    # Best-fit Doppler drift rate (Hz/s)
    snr: float                    # Signal-to-noise ratio of the detection
    start_time: float             # Start timestamp (seconds from obs start)
    end_time: float               # End timestamp (seconds from obs start)
    channel_idx: int              # Frequency channel index in the spectrogram
    is_rfi: bool = False          # Whether this hit overlaps flagged RFI

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RadioProcessorResult:
    """Aggregate result from processing one spectrogram."""
    n_candidates: int                           # Total candidates found
    candidates: List[SignalCandidate] = field(default_factory=list)
    n_rfi_flagged: int = 0                      # Channels flagged as RFI
    noise_floor: float = 0.0                    # Estimated noise floor (power)
    processing_time: float = 0.0                # Wall-clock seconds

    def to_dict(self) -> dict:
        d = asdict(self)
        d["candidates"] = [c.to_dict() for c in self.candidates]
        return d


# ---------------------------------------------------------------------------
# RFI flagging
# ---------------------------------------------------------------------------

def flag_rfi(
    spectrogram: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    known_rfi_mhz: Optional[List[Tuple[float, float]]] = None,
    broadband_sigma: float = 3.0,
    persistence_threshold: float = 0.90,
) -> np.ndarray:
    """
    Build a boolean RFI mask for the spectrogram.

    The mask combines three layers:
        1. Known terrestrial RFI frequency bands (hard mask).
        2. Broadband time-domain sweeps (hard mask).
        3. Persistent constant-frequency lines -- channels whose median
           power is significantly above the global median (soft mask).

    The returned mask is the union of all three layers.  Callers that
    need only the "hard" mask (layers 1+2) for the dedoppler search
    can use `_flag_rfi_hard` directly.

    Parameters
    ----------
    spectrogram : ndarray, shape (n_freq, n_time)
        Power spectrogram.
    freqs : ndarray, shape (n_freq,)
        Frequency axis in MHz.
    times : ndarray, shape (n_time,)
        Time axis in seconds.
    known_rfi_mhz : list of (lo, hi) tuples, optional
        Known RFI frequency bands in MHz.  Defaults to the module-level table.
    broadband_sigma : float
        Sigma threshold for flagging broadband time-domain spikes.
    persistence_threshold : float
        Fraction of time steps a channel must be elevated to be flagged as
        persistent RFI (0.0 - 1.0).

    Returns
    -------
    mask : ndarray of bool, same shape as *spectrogram*
        True where data is considered RFI and should be excluded.
    """
    sigma = broadband_sigma if broadband_sigma != 3.0 else _anomaly_sigma()
    hard_mask = _flag_rfi_hard(spectrogram, freqs, times, known_rfi_mhz,
                               sigma)
    soft_mask = _flag_rfi_persistent(spectrogram, freqs, hard_mask,
                                     sigma, persistence_threshold)
    mask = hard_mask | soft_mask

    n_flagged = int(np.sum(mask))
    total = spectrogram.size
    logger.info(
        "RFI flagging: %d / %d samples masked (%.1f%%), "
        "%d persistent-narrowband channels flagged",
        n_flagged,
        total,
        100.0 * n_flagged / total if total > 0 else 0.0,
        int(np.sum(np.all(soft_mask, axis=1))),
    )
    return mask


def _flag_rfi_hard(
    spectrogram: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    known_rfi_mhz: Optional[List[Tuple[float, float]]] = None,
    broadband_sigma: float = 3.0,
) -> np.ndarray:
    """
    Return a mask for known-frequency RFI and broadband sweeps only.

    This "hard" mask is safe to apply before the dedoppler search,
    because it does not flag narrowband drifting signals.
    """
    if known_rfi_mhz is None:
        known_rfi_mhz = KNOWN_RFI_FREQUENCIES_MHZ

    n_freq, n_time = spectrogram.shape
    mask = np.zeros_like(spectrogram, dtype=bool)

    # --- 1. Known-frequency RFI bands ---
    for lo_mhz, hi_mhz in known_rfi_mhz:
        band_mask = (freqs >= lo_mhz) & (freqs <= hi_mhz)
        if np.any(band_mask):
            mask[band_mask, :] = True

    # --- 2. Broadband (time-domain) sweeps ---
    time_profile = np.nanmedian(spectrogram, axis=0)
    time_med = np.median(time_profile)
    time_mad = np.median(np.abs(time_profile - time_med))
    time_std = 1.4826 * time_mad if time_mad > 0 else np.std(time_profile)
    broadband_threshold = time_med + broadband_sigma * time_std
    for t_idx in range(n_time):
        if time_profile[t_idx] > broadband_threshold:
            mask[:, t_idx] = True

    return mask


def _flag_rfi_persistent(
    spectrogram: np.ndarray,
    freqs: np.ndarray,
    hard_mask: np.ndarray,
    broadband_sigma: float = 3.0,
    persistence_threshold: float = 0.90,
) -> np.ndarray:
    """
    Flag persistent constant-frequency lines (narrowband RFI).

    A channel whose *median* power is significantly elevated above the
    global median is considered persistent RFI (e.g. a local oscillator
    leak).  This mask is used to *label* candidates as potential RFI,
    but is NOT applied before the dedoppler search to avoid masking
    real narrowband signals.
    """
    n_freq, n_time = spectrogram.shape
    mask = np.zeros_like(spectrogram, dtype=bool)

    unmasked_channels = [
        f_idx for f_idx in range(n_freq) if not hard_mask[f_idx, :].all()
    ]
    if not unmasked_channels:
        return mask

    # Compute median power for each un-masked channel
    chan_meds = np.array([
        np.median(spectrogram[f, ~hard_mask[f, :]]) for f in unmasked_channels
    ])
    global_med = np.median(chan_meds)
    global_mad = np.median(np.abs(chan_meds - global_med))
    global_std = 1.4826 * global_mad if global_mad > 0 else np.std(chan_meds)

    # Threshold for a channel median to be "persistently elevated"
    chan_threshold = global_med + broadband_sigma * global_std

    for i, f_idx in enumerate(unmasked_channels):
        if chan_meds[i] > chan_threshold:
            mask[f_idx, :] = True

    return mask


# ---------------------------------------------------------------------------
# Dedoppler search (brute-force Taylor-tree style)
# ---------------------------------------------------------------------------

def dedoppler_search(
    spectrogram: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    min_snr: float = 10.0,
    max_drift: float = 10.0,
    n_drift_steps: int = 201,
    rfi_mask: Optional[np.ndarray] = None,
) -> List[SignalCandidate]:
    """
    Brute-force dedoppler search over a grid of drift rates.

    For each frequency channel and each trial drift rate, the power is
    integrated along the corresponding slanted track across the
    time-frequency plane.  The mean power along each track is compared to
    the local noise statistics to produce an SNR estimate.  Channels
    whose best-drift SNR exceeds *min_snr* are reported as candidates.

    Parameters
    ----------
    spectrogram : ndarray, shape (n_freq, n_time)
    freqs : ndarray, shape (n_freq,)
        Frequency axis in MHz.
    times : ndarray, shape (n_time,)
        Time axis in seconds.
    min_snr : float
        Minimum signal-to-noise ratio for a candidate.
    max_drift : float
        Maximum absolute drift rate to search (Hz/s).
    n_drift_steps : int
        Number of trial drift rates in [-max_drift, +max_drift].
    rfi_mask : ndarray of bool, optional
        If provided, True marks samples to exclude.

    Returns
    -------
    candidates : list of SignalCandidate
    """
    n_freq, n_time = spectrogram.shape
    if n_time < 2:
        logger.warning("Spectrogram has < 2 time steps; skipping dedoppler search")
        return []

    # Work on a copy; apply RFI mask by setting masked values to NaN
    data = spectrogram.astype(np.float64, copy=True)
    valid_mask = np.ones_like(data, dtype=bool)
    if rfi_mask is not None:
        data[rfi_mask] = np.nan
        valid_mask[rfi_mask] = False

    # Frequency resolution (Hz per channel) and time span
    freq_resolution_hz = np.abs(freqs[1] - freqs[0]) * 1e6 if n_freq > 1 else 1.0
    dt = times[-1] - times[0]  # total observation duration in seconds
    if dt <= 0:
        logger.warning("Zero or negative time span; skipping dedoppler search")
        return []

    # Build drift rate grid (Hz/s)
    drift_rates = np.linspace(-max_drift, max_drift, n_drift_steps)

    # ---- Noise estimation ----
    # Use a *global* baseline (median of all valid samples) rather than
    # per-channel medians, because per-channel subtraction would erase
    # any narrowband signal that persists in a single channel.
    all_valid = data[np.isfinite(data)]
    if len(all_valid) == 0:
        logger.warning("All data masked; no candidates")
        return []

    global_median = float(np.median(all_valid))
    global_mad = float(np.median(np.abs(all_valid - global_median)))
    global_noise = 1.4826 * global_mad if global_mad > 0 else float(np.std(all_valid))

    # Subtract the global baseline so that "excess" power is what remains
    data_excess = data - global_median
    # Replace NaN with 0 for summation
    data_sum = np.where(np.isfinite(data_excess), data_excess, 0.0)
    valid_float = valid_mask.astype(np.float64)

    # ---- Integrate along each trial drift track ----
    # integrated[f, d] = sum of excess power along the track starting at
    #     channel f with drift rate drift_rates[d]
    # counts[f, d] = number of valid samples in that track
    integrated = np.zeros((n_freq, len(drift_rates)), dtype=np.float64)
    counts = np.zeros((n_freq, len(drift_rates)), dtype=np.float64)

    for d_idx, drift_rate in enumerate(drift_rates):
        for t_idx in range(n_time):
            dt_step = times[t_idx] - times[0]
            channel_shift = int(round(drift_rate * dt_step / freq_resolution_hz))

            if channel_shift == 0:
                integrated[:, d_idx] += data_sum[:, t_idx]
                counts[:, d_idx] += valid_float[:, t_idx]
            elif channel_shift > 0:
                integrated[:n_freq - channel_shift, d_idx] += data_sum[channel_shift:, t_idx]
                counts[:n_freq - channel_shift, d_idx] += valid_float[channel_shift:, t_idx]
            else:
                shift = -channel_shift
                integrated[shift:, d_idx] += data_sum[:n_freq - shift, t_idx]
                counts[shift:, d_idx] += valid_float[:n_freq - shift, t_idx]

    # ---- SNR computation ----
    # For N valid samples each with noise sigma = global_noise, the sum
    # of pure noise has std = global_noise * sqrt(N).
    #   SNR = integrated_excess / (global_noise * sqrt(N))
    counts = np.maximum(counts, 1.0)
    noise_denom = global_noise * np.sqrt(counts)
    snr_array = integrated / noise_denom

    # Extract candidates above threshold
    candidates: List[SignalCandidate] = []
    for f_idx in range(n_freq):
        best_d_idx = int(np.argmax(snr_array[f_idx, :]))
        best_snr = snr_array[f_idx, best_d_idx]
        if best_snr >= min_snr:
            is_rfi_hit = False
            if rfi_mask is not None:
                is_rfi_hit = bool(np.mean(rfi_mask[f_idx, :]) > 0.5)

            candidates.append(SignalCandidate(
                frequency_hz=float(freqs[f_idx]) * 1e6,
                drift_rate_hz_per_s=float(drift_rates[best_d_idx]),
                snr=float(best_snr),
                start_time=float(times[0]),
                end_time=float(times[-1]),
                channel_idx=int(f_idx),
                is_rfi=is_rfi_hit,
            ))

    # De-duplicate: if adjacent channels found the same signal at the same
    # drift rate, keep only the one with the highest SNR.
    candidates = _deduplicate_candidates(candidates, freq_resolution_hz)

    logger.info(
        "Dedoppler search complete: %d candidates (min_snr=%.1f, max_drift=%.1f Hz/s)",
        len(candidates),
        min_snr,
        max_drift,
    )
    return candidates


def _deduplicate_candidates(
    candidates: List[SignalCandidate],
    freq_resolution_hz: float,
    merge_channels: int = 3,
) -> List[SignalCandidate]:
    """
    Merge candidates that are within *merge_channels* of each other and
    share a similar drift rate.  Keeps the highest-SNR detection.
    """
    if not candidates:
        return candidates

    # Sort by channel index
    candidates.sort(key=lambda c: c.channel_idx)
    merged: List[SignalCandidate] = []
    current = candidates[0]

    for cand in candidates[1:]:
        close_in_freq = abs(cand.channel_idx - current.channel_idx) <= merge_channels
        close_in_drift = abs(cand.drift_rate_hz_per_s - current.drift_rate_hz_per_s) < 1.0
        if close_in_freq and close_in_drift:
            # Keep the higher SNR candidate
            if cand.snr > current.snr:
                current = cand
        else:
            merged.append(current)
            current = cand
    merged.append(current)
    return merged


# ---------------------------------------------------------------------------
# Signal injection (for testing / validation)
# ---------------------------------------------------------------------------

def inject_signal(
    spectrogram: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    freq_hz: float,
    drift_rate: float,
    snr: float,
) -> np.ndarray:
    """
    Inject a synthetic narrowband drifting signal into a spectrogram.

    Parameters
    ----------
    spectrogram : ndarray, shape (n_freq, n_time)
        Input spectrogram (modified in-place and also returned).
    freqs : ndarray, shape (n_freq,)
        Frequency axis in MHz.
    times : ndarray, shape (n_time,)
        Time axis in seconds.
    freq_hz : float
        Centre frequency of the injected signal in Hz.
    drift_rate : float
        Drift rate in Hz/s.
    snr : float
        Signal-to-noise ratio of the injected signal.  The amplitude is
        set so that the per-time-step contribution, when integrated over
        all time steps, yields approximately this SNR.

    Returns
    -------
    spectrogram : ndarray
        The modified spectrogram (same object as input).
    """
    n_freq, n_time = spectrogram.shape
    freq_resolution_hz = np.abs(freqs[1] - freqs[0]) * 1e6 if n_freq > 1 else 1.0
    freqs_hz = freqs * 1e6

    # Estimate noise level from the spectrogram
    med = np.median(spectrogram)
    mad = np.median(np.abs(spectrogram - med))
    noise_std = 1.4826 * mad if mad > 0 else np.std(spectrogram)

    # Per-time-step amplitude: to achieve total SNR after summing n_time
    # steps, each step contributes snr * noise_std / sqrt(n_time).
    amplitude = snr * noise_std / np.sqrt(n_time) if n_time > 0 else snr * noise_std

    for t_idx in range(n_time):
        # Current frequency of the signal at this time step
        current_freq_hz = freq_hz + drift_rate * (times[t_idx] - times[0])

        # Find the nearest channel
        f_idx = int(np.argmin(np.abs(freqs_hz - current_freq_hz)))
        if 0 <= f_idx < n_freq:
            spectrogram[f_idx, t_idx] += amplitude
            # Spectral leakage into adjacent channels (Gaussian profile)
            if f_idx > 0:
                spectrogram[f_idx - 1, t_idx] += amplitude * 0.25
            if f_idx < n_freq - 1:
                spectrogram[f_idx + 1, t_idx] += amplitude * 0.25

    logger.info(
        "Injected signal: freq=%.3f MHz, drift=%.3f Hz/s, target SNR=%.1f, "
        "per-step amplitude=%.4f",
        freq_hz / 1e6,
        drift_rate,
        snr,
        amplitude,
    )
    return spectrogram


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def process_spectrogram(
    spectrogram: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    min_snr: float = 10.0,
    max_drift: float = 10.0,
) -> RadioProcessorResult:
    """
    Full radio spectrogram processing pipeline.

    Steps:
        1. Flag RFI (known frequencies, broadband sweeps, persistent lines).
        2. Run dedoppler search over the cleaned spectrogram.
        3. Compute SNR for each candidate.
        4. Package results.

    Parameters
    ----------
    spectrogram : ndarray, shape (n_freq, n_time)
        Power spectrogram (frequency x time).
    freqs : ndarray, shape (n_freq,)
        Frequency axis in MHz.
    times : ndarray, shape (n_time,)
        Time axis in seconds.
    min_snr : float
        Minimum SNR for a candidate to be reported.
    max_drift : float
        Maximum absolute drift rate to search (Hz/s).

    Returns
    -------
    RadioProcessorResult
    """
    t_start = _time.time()
    n_freq, n_time = spectrogram.shape
    logger.info(
        "Processing spectrogram: %d freq channels x %d time steps, "
        "freq range %.2f - %.2f MHz, duration %.2f s",
        n_freq,
        n_time,
        freqs[0],
        freqs[-1],
        times[-1] - times[0],
    )

    # Step 1: RFI flagging
    # Build the "hard" mask (known-frequency + broadband) for the dedoppler
    # search, and the full mask (including persistent narrowband) for
    # candidate labeling only.
    hard_mask = _flag_rfi_hard(spectrogram, freqs, times)
    full_mask = flag_rfi(spectrogram, freqs, times)
    # Count channels that are *fully* masked (all time steps flagged)
    n_rfi_flagged = int(np.sum(np.all(full_mask, axis=1)))

    # Estimate noise floor from data not flagged by the hard mask
    clean_data = spectrogram[~hard_mask] if np.any(~hard_mask) else spectrogram.ravel()
    noise_med = float(np.median(clean_data))
    noise_mad = float(np.median(np.abs(clean_data - noise_med)))
    noise_floor = 1.4826 * noise_mad if noise_mad > 0 else float(np.std(clean_data))

    # Step 2: Dedoppler search (use only the hard mask so we do not
    # accidentally mask real narrowband drifting signals)
    candidates = dedoppler_search(
        spectrogram,
        freqs,
        times,
        min_snr=min_snr,
        max_drift=max_drift,
        rfi_mask=hard_mask,
    )

    # Step 3: Label candidates that overlap with the full RFI mask
    # (including persistent narrowband), which helps downstream filters
    # distinguish likely-RFI from genuine technosignature candidates.
    for cand in candidates:
        f_idx = cand.channel_idx
        if 0 <= f_idx < n_freq:
            rfi_frac = float(np.mean(full_mask[f_idx, :]))
            cand.is_rfi = rfi_frac > 0.5

    processing_time = _time.time() - t_start
    result = RadioProcessorResult(
        n_candidates=len(candidates),
        candidates=candidates,
        n_rfi_flagged=n_rfi_flagged,
        noise_floor=noise_floor,
        processing_time=processing_time,
    )

    logger.info(
        "Processing complete in %.2f s: %d candidates, %d channels RFI-flagged, "
        "noise floor %.4f",
        processing_time,
        result.n_candidates,
        result.n_rfi_flagged,
        result.noise_floor,
    )
    return result


# ---------------------------------------------------------------------------
# CLI entry point -- synthetic demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("  Project EXODUS -- Radio Spectrogram Processor Demo")
    print("=" * 72)
    print()

    # ---- Parameters ----
    # Use a narrow frequency span (0.5 MHz = 500 kHz across 1024 channels)
    # giving ~488 Hz per channel.  With a drift rate of 2.0 Hz/s over
    # 300 s the signal drifts ~600 Hz (~1.2 channels) -- enough to be
    # distinguishable from zero-drift by the dedoppler search.
    n_freq = 1024
    n_time = 128
    freq_start_mhz = 1420.0     # L-band (hydrogen line neighbourhood)
    freq_end_mhz = 1420.5       # 500 kHz span
    duration_sec = 300.0         # 5-minute observation

    freqs = np.linspace(freq_start_mhz, freq_end_mhz, n_freq)
    times = np.linspace(0.0, duration_sec, n_time)

    freq_res_hz = (freq_end_mhz - freq_start_mhz) / n_freq * 1e6
    print(f"    Channel resolution: {freq_res_hz:.1f} Hz")

    # ---- Step 1: Create synthetic noise spectrogram ----
    print("[1] Generating synthetic noise spectrogram ...")
    rng = np.random.RandomState(42)
    spectrogram = rng.normal(loc=10.0, scale=1.0, size=(n_freq, n_time)).astype(np.float64)
    # Ensure positive power values
    spectrogram = np.abs(spectrogram)
    print(f"    Shape: {spectrogram.shape}")
    print(f"    Freq range: {freqs[0]:.4f} - {freqs[-1]:.4f} MHz")
    print(f"    Time range: {times[0]:.2f} - {times[-1]:.2f} s")
    print(f"    Noise mean: {spectrogram.mean():.4f}")
    print(f"    Noise std:  {spectrogram.std():.4f}")
    print()

    # ---- Step 2: Inject a synthetic drifting signal ----
    inject_freq_hz = 1420.25e6   # mid-band
    inject_drift = 2.0           # Hz/s (drifts ~600 Hz = ~1.2 channels over 300 s)
    inject_snr = 25.0

    print("[2] Injecting synthetic drifting signal ...")
    print(f"    Frequency:  {inject_freq_hz / 1e6:.6f} MHz")
    print(f"    Drift rate: {inject_drift:.3f} Hz/s")
    print(f"    Target SNR: {inject_snr:.1f}")
    total_drift_hz = inject_drift * duration_sec
    print(f"    Total drift: {total_drift_hz:.1f} Hz ({total_drift_hz / freq_res_hz:.1f} channels)")
    inject_signal(spectrogram, freqs, times, inject_freq_hz, inject_drift, inject_snr)
    print()

    # ---- Step 3: Run the processor ----
    print("[3] Running radio spectrogram processor ...")
    result = process_spectrogram(
        spectrogram,
        freqs,
        times,
        min_snr=8.0,
        max_drift=10.0,
    )
    print(f"    Processing time: {result.processing_time:.2f} s")
    print(f"    Noise floor:     {result.noise_floor:.4f}")
    print(f"    RFI-flagged ch:  {result.n_rfi_flagged}")
    print(f"    Candidates:      {result.n_candidates}")
    print()

    # ---- Step 4: Print candidate details ----
    if result.candidates:
        print("[4] Detected candidates:")
        print("-" * 72)
        for i, cand in enumerate(result.candidates, 1):
            rfi_tag = " [RFI]" if cand.is_rfi else ""
            print(
                f"  #{i:>3}  freq={cand.frequency_hz / 1e6:12.6f} MHz  "
                f"drift={cand.drift_rate_hz_per_s:+7.3f} Hz/s  "
                f"SNR={cand.snr:7.2f}  "
                f"ch={cand.channel_idx:>5}{rfi_tag}"
            )
        print()
    else:
        print("[4] No candidates detected.")
        print()

    # ---- Step 5: Verify the injected signal was found ----
    print("[5] Verification:")
    inject_freq_mhz = inject_freq_hz / 1e6
    # Drift rate tolerance: determined by channel resolution and observation
    # duration.  A drift rate difference that shifts the track by less than
    # one channel over the observation is unresolvable, so:
    #   tol ~= freq_resolution_hz / duration_sec
    drift_tol = max(freq_res_hz / duration_sec, 1.0)
    print(f"    Drift rate tolerance: {drift_tol:.2f} Hz/s "
          f"(channel resolution / duration)")
    found = False
    for cand in result.candidates:
        cand_freq_mhz = cand.frequency_hz / 1e6
        freq_match = abs(cand_freq_mhz - inject_freq_mhz) < 0.01  # within 10 kHz
        drift_match = abs(cand.drift_rate_hz_per_s - inject_drift) < drift_tol
        if freq_match and drift_match:
            found = True
            print(
                f"    FOUND injected signal: freq={cand_freq_mhz:.6f} MHz "
                f"(expected {inject_freq_mhz:.6f}), "
                f"drift={cand.drift_rate_hz_per_s:+.3f} Hz/s "
                f"(expected {inject_drift:+.3f}), "
                f"SNR={cand.snr:.2f} (target {inject_snr:.1f})"
            )
            break

    if found:
        print("    >> PASS: Injected signal successfully recovered.")
    else:
        print("    >> FAIL: Injected signal NOT recovered!")
        print("    Candidates near expected frequency:")
        for cand in result.candidates:
            cand_freq_mhz = cand.frequency_hz / 1e6
            if abs(cand_freq_mhz - inject_freq_mhz) < 0.1:
                print(
                    f"      freq={cand_freq_mhz:.6f} MHz, "
                    f"drift={cand.drift_rate_hz_per_s:+.3f} Hz/s, "
                    f"SNR={cand.snr:.2f}"
                )

    print()
    print("Done.")
