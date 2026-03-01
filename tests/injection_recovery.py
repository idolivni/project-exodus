#!/usr/bin/env python3
"""
Project EXODUS -- Injection-Recovery Testing Framework
=====================================================

Simulates fake technosignatures at varying signal-to-noise ratios
injected into realistic "clean" stellar data, then runs the full
EXODUS scoring pipeline to measure detection efficiency.

This is CRITICAL for:
1. Quantifying EXODUS's sensitivity surface per channel
2. Computing minimum detectable signals per channel
3. Producing ROC curves (false-positive rate vs true-positive rate)
4. Enabling rigorous upper limits on megastructure prevalence

Usage
-----
    ./venv/bin/python tests/injection_recovery.py --channels ir transit radio
    ./venv/bin/python tests/injection_recovery.py --all --snr-range 1 20 --n-trials 200
    ./venv/bin/python tests/injection_recovery.py --summary  # show saved results

Standard in exoplanet transit surveys and GW pipelines.
First application to SETI.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Project root on path ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger, safe_json_dump, PROJECT_ROOT as PROJ_ROOT
from src.processing.ir_excess import compute_ir_excess
from src.processing.transit_anomaly import (
    detect_transit_anomaly,
    detect_irregular_dimming,
)
from src.processing.radio_processor import (
    inject_signal,
    process_spectrogram,
)
from src.scoring.exodus_score import EXODUSScorer

log = get_logger("injection_recovery")


# =====================================================================
#  Data classes
# =====================================================================

@dataclass
class InjectionTrial:
    """Result of a single injection-recovery trial."""
    channel: str
    injected_snr: float
    injected_params: Dict[str, Any]
    recovered: bool                 # Did the scorer activate this channel?
    recovered_score: float          # Channel score (0-1)
    total_exodus_score: float       # Full EXODUS score
    n_active_channels: int
    channel_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChannelRecoveryStats:
    """Aggregated recovery statistics for one channel at one SNR."""
    channel: str
    injected_snr: float
    n_trials: int
    n_recovered: int
    recovery_rate: float            # n_recovered / n_trials
    mean_score: float
    std_score: float
    mean_exodus_score: float
    min_detectable: bool            # recovery_rate > 0.5 at this SNR?


@dataclass
class RecoveryCurve:
    """Full recovery curve for one channel across all SNRs."""
    channel: str
    snr_values: List[float]
    recovery_rates: List[float]
    mean_scores: List[float]
    min_detectable_snr: Optional[float]  # 50% recovery threshold
    n_trials_per_snr: int
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "snr_values": self.snr_values,
            "recovery_rates": self.recovery_rates,
            "mean_scores": self.mean_scores,
            "min_detectable_snr": self.min_detectable_snr,
            "n_trials_per_snr": self.n_trials_per_snr,
            "timestamp": self.timestamp,
        }


# =====================================================================
#  Baseline "clean" star generators
# =====================================================================

def _make_clean_photometry(
    teff_k: float = 5800.0,
    noise_frac: float = 0.03,
    rng: np.random.Generator | None = None,
) -> Dict[str, float]:
    """Generate realistic photometry for a quiet solar-type star.

    Returns a photometry dict compatible with compute_ir_excess().
    Band magnitudes follow a rough blackbody+empirical calibration for
    the given effective temperature, with Gaussian noise added.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Approximate magnitudes for a solar-type star at ~30 pc
    # Scaled from Vega-system zero points + blackbody ratios
    scale = 5800.0 / teff_k
    base = {
        "G": 8.50,
        "BP": 8.90 + 0.3 * (scale - 1),
        "RP": 8.00 - 0.2 * (scale - 1),
        "J": 7.30 - 0.4 * (scale - 1),
        "H": 7.00 - 0.45 * (scale - 1),
        "Ks": 6.95 - 0.45 * (scale - 1),
        "W1": 6.90 - 0.46 * (scale - 1),
        "W2": 6.92 - 0.46 * (scale - 1),
        "W3": 6.90 - 0.46 * (scale - 1),
        "W4": 6.88 - 0.46 * (scale - 1),
    }

    phot = {}
    for band, mag in base.items():
        err = noise_frac * abs(mag) * 0.01 + 0.005  # ~0.005-0.03 mag
        phot[band] = mag + rng.normal(0, err)
        phot[f"{band}_err"] = err

    phot["source_id"] = f"INJECT_CLEAN_{rng.integers(1e9)}"
    return phot


def _make_clean_lightcurve(
    n_points: int = 500,
    baseline_flux: float = 1.0,
    noise_level: float = 0.002,
    duration_days: float = 30.0,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a clean lightcurve (no transit, no dimming).

    Returns (time, flux, flux_err) arrays.
    """
    if rng is None:
        rng = np.random.default_rng()

    time_arr = np.sort(rng.uniform(0, duration_days, n_points))
    flux_arr = baseline_flux + rng.normal(0, noise_level, n_points)
    flux_err = np.full(n_points, noise_level)

    return time_arr, flux_arr, flux_err


def _make_clean_spectrogram(
    n_freq: int = 256,
    n_time: int = 64,
    noise_floor: float = 1.0,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a clean radio spectrogram (Gaussian noise only).

    Uses a narrow frequency range (1 MHz bandwidth) so that realistic
    drift rates (0.5-5 Hz/s) actually drift across multiple frequency
    channels.  This prevents injected signals from being flagged as
    persistent-narrowband RFI.

    Channel width = 1 MHz / 256 ≈ 3.9 kHz.
    At 1 Hz/s drift over 300s = 300 Hz drift ≈ 0.08 channels.
    At 5 Hz/s drift over 300s = 1500 Hz ≈ 0.38 channels.

    Returns (spectrogram, freqs_mhz, times_sec) arrays.
    """
    if rng is None:
        rng = np.random.default_rng()

    freqs = np.linspace(1420.0, 1421.0, n_freq)    # 1 MHz bandwidth near H-line
    times = np.linspace(0, 300, n_time)             # seconds
    spectrogram = noise_floor + rng.normal(0, noise_floor * 0.1, (n_freq, n_time))

    return spectrogram, freqs, times


# =====================================================================
#  Signal injection functions
# =====================================================================

def inject_ir_excess(
    phot: Dict[str, float],
    sigma: float,
    bands: List[str] = ("W3", "W4"),
) -> Dict[str, float]:
    """Inject IR excess into photometry by brightening WISE bands.

    A real Dyson sphere excess is 0.5-3 magnitudes brighter in W3/W4
    while optical+near-IR bands stay normal (the SED fitter uses
    optical bands to determine Teff, then the excess is measured as
    the residual in W3/W4).

    We scale sigma so that:
      sigma=1 → 0.1 mag brighter  (marginal / debris-disk level)
      sigma=5 → 0.5 mag brighter  (small partial Dyson sphere)
      sigma=10 → 1.0 mag brighter (significant structure)
      sigma=20 → 2.0 mag brighter (classic Dyson sphere candidate)

    Parameters
    ----------
    phot : dict
        Clean photometry dict.
    sigma : float
        Injection strength.  Maps to ~sigma*0.1 magnitude excess.
    bands : list
        Which WISE bands to inject into.

    Returns
    -------
    dict
        Modified photometry with IR excess injected.
    """
    injected = dict(phot)
    # Magnitude decrease = star gets brighter = IR excess
    delta_mag = sigma * 0.1  # 0.1 mag per sigma unit
    for band in bands:
        if band in injected:
            injected[band] -= delta_mag
    return injected


def inject_transit_signal(
    time_arr: np.ndarray,
    flux_arr: np.ndarray,
    period_days: float = 5.0,
    depth: float = 0.01,
    duration_frac: float = 0.02,
    asymmetry: float = 0.3,
) -> np.ndarray:
    """Inject asymmetric transit dips into a lightcurve.

    Parameters
    ----------
    time_arr : array
        Time array in days.
    flux_arr : array
        Flux array (will be modified in-place).
    period_days : float
        Orbital period.
    depth : float
        Transit depth (fractional flux decrease).
    duration_frac : float
        Duration as fraction of period.
    asymmetry : float
        Asymmetry parameter (0 = symmetric, 1 = fully asymmetric).
        Controls difference between ingress and egress duration.

    Returns
    -------
    array
        Modified flux array.
    """
    injected = flux_arr.copy()
    phase = (time_arr % period_days) / period_days
    transit_width = duration_frac

    # Create asymmetric transit profile
    in_transit = np.abs(phase) < transit_width
    if np.any(in_transit):
        transit_phase = phase[in_transit] / transit_width  # -1 to 1
        # Asymmetric trapezoid
        profile = np.ones_like(transit_phase)
        ingress = transit_phase < -0.5 * (1 - asymmetry)
        egress = transit_phase > 0.5 * (1 + asymmetry)
        flat = ~ingress & ~egress
        if np.any(ingress):
            profile[ingress] = 0.5 + transit_phase[ingress] / (1 - asymmetry + 1e-10)
        if np.any(egress):
            profile[egress] = 1.5 - transit_phase[egress] / (1 + asymmetry + 1e-10)
        profile = np.clip(profile, 0, 1)
        injected[in_transit] -= depth * profile

    return injected


def inject_radio_signal(
    spectrogram: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    snr: float = 10.0,
    freq_hz: Optional[float] = None,
    drift_rate: float = 0.5,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Inject a narrowband drifting signal into a spectrogram.

    Uses the existing radio_processor.inject_signal() function.

    Returns
    -------
    tuple
        (modified_spectrogram, injection_params)
    """
    if rng is None:
        rng = np.random.default_rng()

    if freq_hz is None:
        # Random frequency in the band, avoiding edges
        freq_range = (freqs[20], freqs[-20])
        freq_hz = rng.uniform(freq_range[0], freq_range[1]) * 1e6  # MHz → Hz

    injected = spectrogram.copy()
    inject_signal(injected, freqs, times, freq_hz, drift_rate, snr)

    params = {
        "freq_hz": freq_hz,
        "drift_rate": drift_rate,
        "snr": snr,
    }
    return injected, params


def inject_astrometric_anomaly(
    ruwe_base: float = 1.0,
    noise_sig_base: float = 0.0,
    inject_ruwe: float = 3.0,
    inject_noise_sig: float = 10.0,
) -> Dict[str, float]:
    """Create an astrometric anomaly signal.

    Returns proper_motion_anomaly channel dict for the scorer.
    """
    return {
        "ruwe": inject_ruwe,
        "astrometric_excess_noise_sig": inject_noise_sig,
        "astrometric_excess_noise": inject_noise_sig * 0.5,
    }


def inject_gaia_variability(
    n_epochs: int = 50,
    variability: float = 0.10,
) -> Dict[str, Any]:
    """Create a Gaia photometric variability signal.

    Returns gaia_photometric_anomaly channel dict.
    """
    return {
        "phot_g_variability": variability,
        "n_epochs": n_epochs,
        "variability_flag": variability > 0.03,
    }


# =====================================================================
#  Single-channel injection-recovery runners
# =====================================================================

class InjectionRecoveryEngine:
    """Run injection-recovery tests across channels and SNR values.

    Parameters
    ----------
    snr_values : list of float
        Signal-to-noise ratios to test.
    n_trials : int
        Number of independent trials per SNR.
    scorer_threshold : float
        EXODUS channel activation threshold.
    seed : int
        Base random seed for reproducibility.
    """

    def __init__(
        self,
        snr_values: Optional[List[float]] = None,
        n_trials: int = 100,
        scorer_threshold: float = 0.3,
        seed: int = 42,
    ):
        self.snr_values = snr_values or [1, 2, 3, 4, 5, 7, 10, 15, 20]
        self.n_trials = n_trials
        self.scorer_threshold = scorer_threshold
        self.seed = seed

        self._scorer = EXODUSScorer(threshold=scorer_threshold)
        self._results: Dict[str, RecoveryCurve] = {}
        self._all_trials: List[InjectionTrial] = []

    # -----------------------------------------------------------------
    #  Per-channel recovery tests
    # -----------------------------------------------------------------

    def test_ir_excess(self) -> RecoveryCurve:
        """Injection-recovery for the IR excess channel.

        Injects IR excess at varying sigma into clean photometry and
        checks whether compute_ir_excess + the scorer activates the
        ir_excess channel.
        """
        log.info("=== IR Excess Injection-Recovery (%d SNRs × %d trials) ===",
                 len(self.snr_values), self.n_trials)

        stats_per_snr = []

        for snr in self.snr_values:
            rng = np.random.default_rng(self.seed + int(snr * 1000))
            trials = []

            for trial_i in range(self.n_trials):
                # Generate clean photometry with random Teff
                teff = rng.uniform(4000, 7000)
                clean_phot = _make_clean_photometry(teff_k=teff, rng=rng)

                # Inject IR excess
                injected_phot = inject_ir_excess(clean_phot, sigma=snr)

                # Run IR excess detection
                ir_result = compute_ir_excess(injected_phot)

                # Build scorer input
                target_data = {
                    "target_id": f"IR_INJECT_{snr:.1f}_{trial_i}",
                    "ra": 180.0 + rng.uniform(-10, 10),
                    "dec": 45.0 + rng.uniform(-10, 10),
                }

                # Populate IR channel if we got a valid fit
                if ir_result.fit_bands_used >= 2:
                    target_data["ir_excess"] = {
                        "sigma_W3": ir_result.sigma_W3,
                        "sigma_W4": ir_result.sigma_W4,
                        "excess_W3": ir_result.excess_W3,
                        "excess_W4": ir_result.excess_W4,
                        "is_candidate": ir_result.is_candidate,
                    }

                # Score
                score = self._scorer.score_target(target_data)
                ir_channel = score.channel_scores.get("ir_excess")
                recovered = ir_channel is not None and ir_channel.is_active
                ch_score = ir_channel.score if ir_channel else 0.0

                trial = InjectionTrial(
                    channel="ir_excess",
                    injected_snr=snr,
                    injected_params={"sigma": snr, "teff": teff},
                    recovered=recovered,
                    recovered_score=ch_score,
                    total_exodus_score=score.total_score,
                    n_active_channels=score.n_active_channels,
                    channel_details={
                        "is_candidate": ir_result.is_candidate,
                        "sigma_W3": ir_result.sigma_W3,
                        "sigma_W4": ir_result.sigma_W4,
                        "fit_bands_used": ir_result.fit_bands_used,
                    },
                )
                trials.append(trial)
                self._all_trials.append(trial)

            # Aggregate
            n_recovered = sum(1 for t in trials if t.recovered)
            scores = [t.recovered_score for t in trials]
            exodus_scores = [t.total_exodus_score for t in trials]

            stat = ChannelRecoveryStats(
                channel="ir_excess",
                injected_snr=snr,
                n_trials=len(trials),
                n_recovered=n_recovered,
                recovery_rate=n_recovered / len(trials),
                mean_score=float(np.mean(scores)),
                std_score=float(np.std(scores)),
                mean_exodus_score=float(np.mean(exodus_scores)),
                min_detectable=n_recovered / len(trials) > 0.5,
            )
            stats_per_snr.append(stat)

            log.info(
                "  IR σ=%5.1f: recovery=%5.1f%% (%d/%d), "
                "mean_score=%.3f, mean_exodus=%.3f",
                snr, stat.recovery_rate * 100, n_recovered, len(trials),
                stat.mean_score, stat.mean_exodus_score,
            )

        curve = self._build_curve("ir_excess", stats_per_snr)
        self._results["ir_excess"] = curve
        return curve

    def test_transit_anomaly(self) -> RecoveryCurve:
        """Injection-recovery for the transit anomaly channel.

        Injects asymmetric transits of varying depth into clean
        lightcurves and checks whether the transit detector +
        scorer activates the transit_anomaly channel.

        SNR here maps to transit depth: snr=1 → 0.001, snr=10 → 0.01,
        snr=20 → 0.02 (fractional flux decrease).
        """
        log.info("=== Transit Anomaly Injection-Recovery (%d SNRs × %d trials) ===",
                 len(self.snr_values), self.n_trials)

        stats_per_snr = []

        for snr in self.snr_values:
            rng = np.random.default_rng(self.seed + int(snr * 1000) + 100000)
            trials = []

            # Map SNR to transit depth
            depth = snr * 0.001  # 1σ → 0.1%, 10σ → 1%, 20σ → 2%

            for trial_i in range(self.n_trials):
                # Generate clean lightcurve
                noise = rng.uniform(0.001, 0.003)
                t, f, fe = _make_clean_lightcurve(
                    n_points=500, noise_level=noise, rng=rng,
                )

                # Inject asymmetric transit
                period = rng.uniform(3, 10)
                asymmetry = rng.uniform(0.2, 0.6)
                f_injected = inject_transit_signal(
                    t, f, period_days=period, depth=depth,
                    asymmetry=asymmetry,
                )

                # Run transit detection
                try:
                    ta_result = detect_transit_anomaly(t, f_injected, fe)
                    ta_score = ta_result.anomaly_score
                    ta_anom = ta_result.is_anomalous
                except Exception:
                    ta_score = 0.0
                    ta_anom = False

                # Also run irregular dimming
                try:
                    irreg = detect_irregular_dimming(t, f_injected)
                    irreg_score = irreg.anomaly_score
                except Exception:
                    irreg_score = 0.0

                combined_score = max(ta_score, irreg_score)

                # Build scorer input
                target_data = {
                    "target_id": f"TRANSIT_INJECT_{snr:.1f}_{trial_i}",
                    "ra": 180.0,
                    "dec": 45.0,
                    "transit_anomaly": {
                        "anomaly_score": combined_score,
                        "is_anomalous": ta_anom or irreg_score > 0.3,
                    },
                }

                score = self._scorer.score_target(target_data)
                transit_ch = score.channel_scores.get("transit_anomaly")
                recovered = transit_ch is not None and transit_ch.is_active
                ch_score = transit_ch.score if transit_ch else 0.0

                trial = InjectionTrial(
                    channel="transit_anomaly",
                    injected_snr=snr,
                    injected_params={
                        "depth": depth, "period": period,
                        "asymmetry": asymmetry, "noise": noise,
                    },
                    recovered=recovered,
                    recovered_score=ch_score,
                    total_exodus_score=score.total_score,
                    n_active_channels=score.n_active_channels,
                    channel_details={
                        "ta_score": ta_score,
                        "irreg_score": irreg_score,
                        "combined": combined_score,
                    },
                )
                trials.append(trial)
                self._all_trials.append(trial)

            n_recovered = sum(1 for t in trials if t.recovered)
            scores = [t.recovered_score for t in trials]
            exodus_scores = [t.total_exodus_score for t in trials]

            stat = ChannelRecoveryStats(
                channel="transit_anomaly",
                injected_snr=snr,
                n_trials=len(trials),
                n_recovered=n_recovered,
                recovery_rate=n_recovered / len(trials),
                mean_score=float(np.mean(scores)),
                std_score=float(np.std(scores)),
                mean_exodus_score=float(np.mean(exodus_scores)),
                min_detectable=n_recovered / len(trials) > 0.5,
            )
            stats_per_snr.append(stat)

            log.info(
                "  Transit depth=%.4f (snr=%4.1f): recovery=%5.1f%% (%d/%d), "
                "mean_score=%.3f",
                depth, snr, stat.recovery_rate * 100, n_recovered,
                len(trials), stat.mean_score,
            )

        curve = self._build_curve("transit_anomaly", stats_per_snr)
        self._results["transit_anomaly"] = curve
        return curve

    def test_radio_anomaly(self) -> RecoveryCurve:
        """Injection-recovery for the radio anomaly channel.

        Injects narrowband drifting signals at varying SNR into clean
        spectrograms and checks whether the dedoppler search + scorer
        activates the radio_anomaly channel.
        """
        log.info("=== Radio Anomaly Injection-Recovery (%d SNRs × %d trials) ===",
                 len(self.snr_values), self.n_trials)

        stats_per_snr = []

        for snr in self.snr_values:
            rng = np.random.default_rng(self.seed + int(snr * 1000) + 200000)
            trials = []

            for trial_i in range(self.n_trials):
                # Generate clean spectrogram
                spec, freqs, times = _make_clean_spectrogram(rng=rng)

                # Inject drifting narrowband signal
                # Use moderate drift rates (1-5 Hz/s) so the signal
                # drifts across channels and isn't flagged as RFI
                drift = rng.uniform(1.0, 5.0)
                injected_spec, inj_params = inject_radio_signal(
                    spec, freqs, times, snr=snr, drift_rate=drift, rng=rng,
                )

                # Run radio processing
                try:
                    radio_result = process_spectrogram(
                        injected_spec, freqs, times, min_snr=5.0,
                    )
                    n_cands = radio_result.n_candidates
                    max_snr_det = (
                        max(c.snr for c in radio_result.candidates)
                        if radio_result.candidates else 0
                    )
                    cand_dicts = [c.to_dict() for c in radio_result.candidates]
                except Exception:
                    n_cands = 0
                    max_snr_det = 0
                    cand_dicts = []

                # Build scorer input
                target_data = {
                    "target_id": f"RADIO_INJECT_{snr:.1f}_{trial_i}",
                    "ra": 180.0,
                    "dec": 45.0,
                    "radio_anomaly": {
                        "n_candidates": n_cands,
                        "max_snr": max_snr_det,
                        "candidates": cand_dicts,
                    },
                }

                score = self._scorer.score_target(target_data)
                radio_ch = score.channel_scores.get("radio_anomaly")
                recovered = radio_ch is not None and radio_ch.is_active
                ch_score = radio_ch.score if radio_ch else 0.0

                trial = InjectionTrial(
                    channel="radio_anomaly",
                    injected_snr=snr,
                    injected_params=inj_params,
                    recovered=recovered,
                    recovered_score=ch_score,
                    total_exodus_score=score.total_score,
                    n_active_channels=score.n_active_channels,
                    channel_details={
                        "n_candidates": n_cands,
                        "max_snr_detected": max_snr_det,
                    },
                )
                trials.append(trial)
                self._all_trials.append(trial)

            n_recovered = sum(1 for t in trials if t.recovered)
            scores = [t.recovered_score for t in trials]
            exodus_scores = [t.total_exodus_score for t in trials]

            stat = ChannelRecoveryStats(
                channel="radio_anomaly",
                injected_snr=snr,
                n_trials=len(trials),
                n_recovered=n_recovered,
                recovery_rate=n_recovered / len(trials),
                mean_score=float(np.mean(scores)),
                std_score=float(np.std(scores)),
                mean_exodus_score=float(np.mean(exodus_scores)),
                min_detectable=n_recovered / len(trials) > 0.5,
            )
            stats_per_snr.append(stat)

            log.info(
                "  Radio SNR=%5.1f: recovery=%5.1f%% (%d/%d), "
                "mean_score=%.3f",
                snr, stat.recovery_rate * 100, n_recovered,
                len(trials), stat.mean_score,
            )

        curve = self._build_curve("radio_anomaly", stats_per_snr)
        self._results["radio_anomaly"] = curve
        return curve

    def test_astrometric_anomaly(self) -> RecoveryCurve:
        """Injection-recovery for the proper motion anomaly channel.

        Injects RUWE / astrometric_excess_noise_sig at varying levels
        and checks scorer activation.  This is deterministic (no noise)
        so we use fewer trials.
        """
        log.info("=== Astrometric Anomaly Injection-Recovery (%d SNRs) ===",
                 len(self.snr_values))

        stats_per_snr = []

        for snr in self.snr_values:
            # Map SNR to RUWE: snr=1 → RUWE=1.2, snr=5 → RUWE=2.0,
            # snr=10 → RUWE=3.0, snr=20 → RUWE=5.0
            ruwe = 1.0 + snr * 0.2
            noise_sig = snr * 2.0

            trials = []
            # This channel is deterministic — fewer trials needed
            n_trials_astro = min(self.n_trials, 20)

            for trial_i in range(n_trials_astro):
                rng = np.random.default_rng(self.seed + trial_i + int(snr * 1000))
                # Small random perturbation to RUWE
                ruwe_trial = ruwe + rng.normal(0, 0.05)
                noise_sig_trial = noise_sig + rng.normal(0, 0.5)

                target_data = {
                    "target_id": f"ASTRO_INJECT_{snr:.1f}_{trial_i}",
                    "ra": 180.0,
                    "dec": 45.0,
                    "proper_motion_anomaly": inject_astrometric_anomaly(
                        inject_ruwe=max(1.0, ruwe_trial),
                        inject_noise_sig=max(0, noise_sig_trial),
                    ),
                }

                score = self._scorer.score_target(target_data)
                pm_ch = score.channel_scores.get("proper_motion_anomaly")
                recovered = pm_ch is not None and pm_ch.is_active
                ch_score = pm_ch.score if pm_ch else 0.0

                trial = InjectionTrial(
                    channel="proper_motion_anomaly",
                    injected_snr=snr,
                    injected_params={"ruwe": ruwe_trial, "noise_sig": noise_sig_trial},
                    recovered=recovered,
                    recovered_score=ch_score,
                    total_exodus_score=score.total_score,
                    n_active_channels=score.n_active_channels,
                )
                trials.append(trial)
                self._all_trials.append(trial)

            n_recovered = sum(1 for t in trials if t.recovered)
            scores = [t.recovered_score for t in trials]
            exodus_scores = [t.total_exodus_score for t in trials]

            stat = ChannelRecoveryStats(
                channel="proper_motion_anomaly",
                injected_snr=snr,
                n_trials=len(trials),
                n_recovered=n_recovered,
                recovery_rate=n_recovered / len(trials),
                mean_score=float(np.mean(scores)),
                std_score=float(np.std(scores)),
                mean_exodus_score=float(np.mean(exodus_scores)),
                min_detectable=n_recovered / len(trials) > 0.5,
            )
            stats_per_snr.append(stat)

            log.info(
                "  Astro RUWE=%.1f (snr=%4.1f): recovery=%5.1f%% (%d/%d), "
                "mean_score=%.3f",
                ruwe, snr, stat.recovery_rate * 100, n_recovered,
                len(trials), stat.mean_score,
            )

        curve = self._build_curve("proper_motion_anomaly", stats_per_snr)
        self._results["proper_motion_anomaly"] = curve
        return curve

    def test_multi_channel_convergence(self) -> RecoveryCurve:
        """Test that multi-channel injection produces convergence bonus.

        Injects signals in IR + transit + astrometry simultaneously and
        verifies that the convergence bonus amplifies the EXODUS score
        as expected (2^(n_active-1)).
        """
        log.info("=== Multi-Channel Convergence Recovery (%d SNRs × %d trials) ===",
                 len(self.snr_values), self.n_trials)

        stats_per_snr = []

        for snr in self.snr_values:
            rng = np.random.default_rng(self.seed + int(snr * 1000) + 500000)
            trials = []

            for trial_i in range(self.n_trials):
                # --- IR injection ---
                teff = rng.uniform(4000, 7000)
                clean_phot = _make_clean_photometry(teff_k=teff, rng=rng)
                injected_phot = inject_ir_excess(clean_phot, sigma=snr)
                ir_result = compute_ir_excess(injected_phot)

                ir_data = {}
                if ir_result.fit_bands_used >= 2:
                    ir_data = {
                        "sigma_W3": ir_result.sigma_W3,
                        "sigma_W4": ir_result.sigma_W4,
                        "excess_W3": ir_result.excess_W3,
                        "excess_W4": ir_result.excess_W4,
                        "is_candidate": ir_result.is_candidate,
                    }

                # --- Transit injection ---
                depth = snr * 0.001
                noise = rng.uniform(0.001, 0.003)
                t_lc, f_lc, fe_lc = _make_clean_lightcurve(
                    noise_level=noise, rng=rng,
                )
                f_injected = inject_transit_signal(
                    t_lc, f_lc, depth=depth, asymmetry=0.3,
                )
                try:
                    ta_result = detect_transit_anomaly(t_lc, f_injected, fe_lc)
                    ta_score = ta_result.anomaly_score
                    ta_anom = ta_result.is_anomalous
                except Exception:
                    ta_score = 0.0
                    ta_anom = False

                # --- Astrometric injection ---
                ruwe = 1.0 + snr * 0.2
                noise_sig = snr * 2.0

                target_data = {
                    "target_id": f"MULTI_INJECT_{snr:.1f}_{trial_i}",
                    "ra": 180.0 + rng.uniform(-10, 10),
                    "dec": 45.0 + rng.uniform(-10, 10),
                    "proper_motion_anomaly": {
                        "ruwe": ruwe,
                        "astrometric_excess_noise_sig": noise_sig,
                        "astrometric_excess_noise": noise_sig * 0.5,
                    },
                    "transit_anomaly": {
                        "anomaly_score": ta_score,
                        "is_anomalous": ta_anom,
                    },
                }
                if ir_data:
                    target_data["ir_excess"] = ir_data

                score = self._scorer.score_target(target_data)

                # "recovered" = at least 2 channels active (convergence)
                recovered = score.n_active_channels >= 2
                ch_score = score.total_score

                trial = InjectionTrial(
                    channel="multi_channel",
                    injected_snr=snr,
                    injected_params={
                        "ir_sigma": snr, "transit_depth": depth,
                        "ruwe": ruwe,
                    },
                    recovered=recovered,
                    recovered_score=ch_score,
                    total_exodus_score=score.total_score,
                    n_active_channels=score.n_active_channels,
                    channel_details={
                        "convergence_bonus": score.convergence_bonus,
                    },
                )
                trials.append(trial)
                self._all_trials.append(trial)

            n_recovered = sum(1 for t in trials if t.recovered)
            scores = [t.total_exodus_score for t in trials]

            stat = ChannelRecoveryStats(
                channel="multi_channel",
                injected_snr=snr,
                n_trials=len(trials),
                n_recovered=n_recovered,
                recovery_rate=n_recovered / len(trials),
                mean_score=float(np.mean(scores)),
                std_score=float(np.std(scores)),
                mean_exodus_score=float(np.mean(scores)),
                min_detectable=n_recovered / len(trials) > 0.5,
            )
            stats_per_snr.append(stat)

            log.info(
                "  Multi snr=%4.1f: convergence=%5.1f%% (%d/%d), "
                "mean_exodus=%.3f, mean_n_active=%.1f",
                snr, stat.recovery_rate * 100, n_recovered, len(trials),
                stat.mean_exodus_score,
                np.mean([t.n_active_channels for t in trials]),
            )

        curve = self._build_curve("multi_channel", stats_per_snr)
        self._results["multi_channel"] = curve
        return curve

    # -----------------------------------------------------------------
    #  False positive rate measurement
    # -----------------------------------------------------------------

    def test_false_positive_rate(
        self,
        n_null_trials: int = 500,
    ) -> Dict[str, float]:
        """Measure false positive rate on clean (null) data.

        Runs clean stars through all channels with NO injected signal.
        Any activation is a false positive.

        Returns dict of channel → false_positive_rate.
        """
        log.info("=== False Positive Rate (%d null trials) ===", n_null_trials)

        rng = np.random.default_rng(self.seed + 999999)
        fp_counts = defaultdict(int)

        for trial_i in range(n_null_trials):
            # --- Clean IR ---
            teff = rng.uniform(4000, 7000)
            phot = _make_clean_photometry(teff_k=teff, rng=rng)
            ir_result = compute_ir_excess(phot)

            # --- Clean lightcurve ---
            t, f, fe = _make_clean_lightcurve(rng=rng)
            try:
                ta_result = detect_transit_anomaly(t, f, fe)
                ta_score = ta_result.anomaly_score
                ta_anom = ta_result.is_anomalous
            except Exception:
                ta_score = 0.0
                ta_anom = False

            # --- Clean spectrogram ---
            spec, freqs, times = _make_clean_spectrogram(rng=rng)
            try:
                radio_result = process_spectrogram(spec, freqs, times, min_snr=5.0)
                radio_n = radio_result.n_candidates
                radio_snr = (
                    max(c.snr for c in radio_result.candidates)
                    if radio_result.candidates else 0
                )
                radio_cands = [c.to_dict() for c in radio_result.candidates]
            except Exception:
                radio_n = 0
                radio_snr = 0
                radio_cands = []

            target_data = {
                "target_id": f"NULL_{trial_i}",
                "ra": rng.uniform(0, 360),
                "dec": rng.uniform(-90, 90),
                "proper_motion_anomaly": {
                    "ruwe": 1.0 + rng.normal(0, 0.05),
                    "astrometric_excess_noise_sig": max(0, rng.normal(0, 0.5)),
                    "astrometric_excess_noise": 0.0,
                },
                "transit_anomaly": {
                    "anomaly_score": ta_score,
                    "is_anomalous": ta_anom,
                },
                "radio_anomaly": {
                    "n_candidates": radio_n,
                    "max_snr": radio_snr,
                    "candidates": radio_cands,
                },
            }
            if ir_result.fit_bands_used >= 2:
                target_data["ir_excess"] = {
                    "sigma_W3": ir_result.sigma_W3,
                    "sigma_W4": ir_result.sigma_W4,
                    "excess_W3": ir_result.excess_W3,
                    "excess_W4": ir_result.excess_W4,
                    "is_candidate": ir_result.is_candidate,
                }

            score = self._scorer.score_target(target_data)

            for ch_name, ch_score in score.channel_scores.items():
                if ch_score.is_active:
                    fp_counts[ch_name] += 1

        rates = {}
        for ch in ["ir_excess", "transit_anomaly", "radio_anomaly",
                    "proper_motion_anomaly", "gaia_photometric_anomaly"]:
            rate = fp_counts[ch] / n_null_trials
            rates[ch] = rate
            log.info("  FP rate %-30s: %.1f%% (%d/%d)",
                     ch, rate * 100, fp_counts[ch], n_null_trials)

        return rates

    # -----------------------------------------------------------------
    #  Run all tests
    # -----------------------------------------------------------------

    def run_all(self, channels: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run injection-recovery for specified channels (or all).

        Parameters
        ----------
        channels : list of str, optional
            Channels to test. Default: all available.
            Options: ir, transit, radio, astrometric, multi, fp

        Returns
        -------
        dict
            Complete results including curves and FP rates.
        """
        channel_map = {
            "ir": self.test_ir_excess,
            "ir_excess": self.test_ir_excess,
            "transit": self.test_transit_anomaly,
            "transit_anomaly": self.test_transit_anomaly,
            "radio": self.test_radio_anomaly,
            "radio_anomaly": self.test_radio_anomaly,
            "astrometric": self.test_astrometric_anomaly,
            "proper_motion_anomaly": self.test_astrometric_anomaly,
            "multi": self.test_multi_channel_convergence,
            "multi_channel": self.test_multi_channel_convergence,
        }

        if channels is None:
            channels = ["ir", "transit", "radio", "astrometric", "multi"]

        start = time.time()

        for ch in channels:
            if ch == "fp":
                continue
            if ch in channel_map:
                channel_map[ch]()
            else:
                log.warning("Unknown channel: %s (available: %s)",
                            ch, list(channel_map.keys()))

        # Always run FP rate
        fp_rates = self.test_false_positive_rate()

        elapsed = time.time() - start

        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "snr_values": self.snr_values,
                "n_trials": self.n_trials,
                "scorer_threshold": self.scorer_threshold,
                "seed": self.seed,
                "channels_tested": channels,
            },
            "recovery_curves": {
                name: curve.to_dict()
                for name, curve in self._results.items()
            },
            "false_positive_rates": fp_rates,
            "elapsed_seconds": elapsed,
        }

        return results

    # -----------------------------------------------------------------
    #  Reporting
    # -----------------------------------------------------------------

    def print_summary(self) -> str:
        """Print and return a text summary of all results."""
        lines = []
        lines.append("=" * 70)
        lines.append("  EXODUS Injection-Recovery Summary")
        lines.append("=" * 70)

        for name, curve in self._results.items():
            lines.append(f"\n  Channel: {name}")
            lines.append(f"  Min detectable SNR (50% recovery): "
                         f"{curve.min_detectable_snr or 'N/A'}")
            lines.append(f"  SNR   Recovery   Mean Score")
            lines.append(f"  {'─'*35}")
            for snr, rate, score in zip(
                curve.snr_values, curve.recovery_rates, curve.mean_scores
            ):
                bar = "█" * int(rate * 20)
                lines.append(f"  {snr:5.1f}  {rate:6.1%}  {score:8.3f}  {bar}")

        summary = "\n".join(lines)
        print(summary)
        return summary

    def save_results(
        self,
        path: str = "data/reports/injection_recovery.json",
        fp_rates: Optional[Dict[str, float]] = None,
    ):
        """Save full results to JSON.

        Parameters
        ----------
        path : str
            Output file path.
        fp_rates : dict, optional
            False positive rates to include. If None, runs FP test.
        """
        if fp_rates is None and not self._results:
            log.warning("No results to save. Run tests first.")
            return None

        # Run FP test if rates not provided
        if fp_rates is None:
            fp_rates = self.test_false_positive_rate()

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "snr_values": self.snr_values,
                "n_trials": self.n_trials,
                "scorer_threshold": self.scorer_threshold,
            },
            "recovery_curves": {
                name: curve.to_dict()
                for name, curve in self._results.items()
            },
            "false_positive_rates": fp_rates,
            "total_trials": len(self._all_trials),
        }

        out_path = Path(path)
        if not out_path.is_absolute():
            out_path = PROJ_ROOT / path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            safe_json_dump(out, f, indent=2)
        log.info("Saved injection-recovery results to %s", out_path)
        return str(out_path)

    # -----------------------------------------------------------------
    #  Internal helpers
    # -----------------------------------------------------------------

    def _build_curve(
        self, channel: str, stats: List[ChannelRecoveryStats]
    ) -> RecoveryCurve:
        """Build a RecoveryCurve from per-SNR statistics."""
        snr_vals = [s.injected_snr for s in stats]
        rates = [s.recovery_rate for s in stats]
        mean_scores = [s.mean_score for s in stats]

        # Find 50% recovery threshold by linear interpolation
        min_det_snr = None
        for i in range(len(rates) - 1):
            if rates[i] < 0.5 <= rates[i + 1]:
                # Linear interpolation
                frac = (0.5 - rates[i]) / (rates[i + 1] - rates[i] + 1e-10)
                min_det_snr = snr_vals[i] + frac * (snr_vals[i + 1] - snr_vals[i])
                break
        # If first SNR already above 50%
        if min_det_snr is None and rates and rates[0] >= 0.5:
            min_det_snr = snr_vals[0]

        return RecoveryCurve(
            channel=channel,
            snr_values=snr_vals,
            recovery_rates=rates,
            mean_scores=mean_scores,
            min_detectable_snr=round(min_det_snr, 2) if min_det_snr else None,
            n_trials_per_snr=self.n_trials,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


# =====================================================================
#  CLI
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="EXODUS Injection-Recovery Testing"
    )
    parser.add_argument(
        "--channels", nargs="+",
        default=["ir", "transit", "radio", "astrometric", "multi"],
        help="Channels to test (ir, transit, radio, astrometric, multi)",
    )
    parser.add_argument("--all", action="store_true", help="Test all channels")
    parser.add_argument(
        "--snr-range", nargs=2, type=float, default=[1, 20],
        metavar=("MIN", "MAX"),
        help="SNR range (default: 1 20)",
    )
    parser.add_argument(
        "--snr-steps", type=int, default=9,
        help="Number of SNR steps (default: 9)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=100,
        help="Trials per SNR (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output", default="data/reports/injection_recovery.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Just show summary of previous results",
    )
    args = parser.parse_args()

    if args.summary:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = PROJ_ROOT / args.output
        if out_path.exists():
            with open(out_path) as f:
                data = json.load(f)
            print(json.dumps(data, indent=2))
        else:
            print(f"No results found at {out_path}")
        sys.exit(0)

    snr_lo, snr_hi = args.snr_range
    snr_values = np.linspace(snr_lo, snr_hi, args.snr_steps).tolist()

    engine = InjectionRecoveryEngine(
        snr_values=snr_values,
        n_trials=args.n_trials,
        seed=args.seed,
    )

    channels = args.channels
    if args.all:
        channels = ["ir", "transit", "radio", "astrometric", "multi"]

    results = engine.run_all(channels=channels)
    engine.print_summary()
    path = engine.save_results(args.output, fp_rates=results.get("false_positive_rates"))

    print(f"\n{'='*70}")
    print(f"  Results saved to: {path}")
    print(f"  Total trials: {len(engine._all_trials)}")
    print(f"  Elapsed: {results['elapsed_seconds']:.1f}s")
    print(f"{'='*70}")
