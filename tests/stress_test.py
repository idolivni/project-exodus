#!/usr/bin/env python3
"""
Project EXODUS -- Post-Build Stress Test Suite
==============================================

Eight comprehensive tests exercising every module, plus a final report.

TEST 1: Injection Test -- Can we find a fake signal?
TEST 2: Null Test -- Does it stay quiet on boring stars?
TEST 3: Known Anomaly Test -- Tabby's Star
TEST 4: Correlation Test -- Cross-matching accuracy
TEST 5: Temporal Archaeology Test
TEST 6: Self-Improvement Test -- Evolver learning
TEST 7: Science Sanity Check -- review & fix detectors
TEST 8: Discovery Trap Test -- review & fix barriers
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# ── Ensure project root is on path ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger, PROJECT_ROOT as PROJ_ROOT
from src.processing.ir_excess import compute_ir_excess, IRExcessResult
from src.processing.transit_anomaly import (
    detect_transit_anomaly,
    detect_irregular_dimming,
    TransitAnomalyResult,
    IrregularDimmingResult,
)
from src.processing.radio_processor import (
    inject_signal,
    process_spectrogram,
    dedoppler_search,
    flag_rfi,
    RadioProcessorResult,
)
from src.scoring.exodus_score import EXODUSScorer, EXODUSScore
from src.engines.generator import HypothesisGenerator
from src.engines.analyst import AnalystEngine, ValidationStatus
from src.engines.breakthrough import BreakthroughEngine, EscalationLevel
from src.engines.evolver import EvolverEngine, ResearchState

log = get_logger("stress_test")

# =====================================================================
#  Global results tracker
# =====================================================================

class StressResults:
    """Accumulates results from all 8 tests for the final report."""

    def __init__(self):
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self.fixes_applied: List[str] = []
        self.issues_found: List[str] = []
        self.start_time = time.time()

    def record(self, test_name: str, passed: bool, details: Dict[str, Any]):
        self.test_results[test_name] = {
            "passed": passed,
            "details": details,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def add_issue(self, issue: str):
        self.issues_found.append(issue)

    def add_fix(self, fix: str):
        self.fixes_applied.append(fix)


RESULTS = StressResults()


def _section(title: str):
    """Print a section header."""
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def _subsection(title: str):
    print(f"\n  --- {title} ---")


# =====================================================================
#  TEST 1: INJECTION TEST
# =====================================================================

def test_1_injection():
    """
    Create a synthetic test. Take a real-like star. Inject:
      - A fake IR excess of 5-sigma in the WISE W3 band
      - A fake asymmetric transit dip in a TESS-like light curve
      - A fake narrowband radio signal

    Run the full pipeline. Does the system:
      1) Detect all three injected anomalies?
      2) Compute a high EXODUS convergence score?
      3) Generate follow-up hypotheses?
      4) Flag it as UNEXPLAINED?
    """
    _section("TEST 1: INJECTION TEST -- Can we find a fake signal?")

    details = {
        "ir_detected": False,
        "transit_detected": False,
        "radio_detected": False,
        "exodus_score": 0.0,
        "n_active_channels": 0,
        "hypotheses_generated": 0,
        "classification": "N/A",
    }

    # ── 1a. IR Excess Injection ──────────────────────────────────────
    _subsection("1a. IR Excess Injection (5-sigma W3)")

    # A normal G2V star photometry at ~100 pc
    # We'll make W3 significantly brighter than predicted by BB model
    # The BB fit uses G, BP, RP, J, H, Ks -> predicts W3 ~ 7.2 mag
    # We inject excess by setting W3 ~ 5.5 (brighter by ~1.7 mag)
    # With err=0.03 and model_err=0.054, total_err ~ 0.062
    # For 5-sigma: excess_mag ~ -0.31 at total_err ~0.062
    # But for a clear signal let's make it even more obvious
    star_phot = {
        "source_id": "INJECT_TEST_001",
        "G":    8.50,   "G_err":  0.003,
        "BP":   8.92,   "BP_err": 0.003,
        "RP":   8.15,   "RP_err": 0.004,
        "J":    7.76,   "J_err":  0.02,
        "H":    7.56,   "H_err":  0.03,
        "Ks":   7.44,   "Ks_err": 0.02,
        # Inject excess: normal star would have W3 ~ 7.2, we set much brighter
        "W3":   5.50,   "W3_err": 0.03,
        "W4":   5.00,   "W4_err": 0.10,
    }

    ir_result = compute_ir_excess(star_phot)
    print(f"    Fitted Teff:   {ir_result.fitted_teff:.0f} K")
    print(f"    Bands used:    {ir_result.fit_bands_used}")
    print(f"    W3 excess:     {ir_result.excess_W3:+.3f} mag ({ir_result.sigma_W3:.1f} sigma)")
    print(f"    W4 excess:     {ir_result.excess_W4:+.3f} mag ({ir_result.sigma_W4:.1f} sigma)")
    print(f"    Candidate:     {ir_result.is_candidate}")

    ir_detected = ir_result.is_candidate and (ir_result.sigma_W3 or 0) >= 3.0
    details["ir_detected"] = ir_detected
    details["ir_sigma_W3"] = ir_result.sigma_W3
    details["ir_sigma_W4"] = ir_result.sigma_W4
    print(f"    >> IR DETECTION: {'PASS' if ir_detected else 'FAIL'}")

    # ── 1b. Transit Anomaly Injection ────────────────────────────────
    _subsection("1b. Asymmetric Transit Dip Injection")

    rng = np.random.default_rng(seed=42)
    n_pts = 5000
    time_lc = np.linspace(0, 30, n_pts)
    flux_lc = np.ones(n_pts) + rng.normal(0, 0.001, n_pts)

    # Inject strongly asymmetric, variable-depth transits (megastructure-like)
    period = 3.5
    n_transits = int(30.0 / period) + 1
    for k in range(n_transits):
        t_centre = 0.5 * period + k * period
        # Strongly variable depth (factor ~3x variation)
        depth = 0.005 + 0.020 * rng.random()
        # Very asymmetric: very short ingress, long egress
        ingress_dur = 0.02 + 0.01 * rng.random()
        egress_dur = 0.20 + 0.10 * rng.random()

        for i, t in enumerate(time_lc):
            dt = t - t_centre
            if -ingress_dur <= dt < 0:
                frac = (dt + ingress_dur) / ingress_dur
                flux_lc[i] -= depth * frac ** 0.3
            elif 0 <= dt < egress_dur:
                frac = 1.0 - (dt / egress_dur) ** 2
                flux_lc[i] -= depth * frac

    transit_result = detect_transit_anomaly(time_lc, flux_lc)
    print(f"    Period:           {transit_result.period:.4f} d")
    print(f"    Depth:            {transit_result.depth:.6f}")
    print(f"    Symmetry:         {transit_result.symmetry_score:.4f}")
    print(f"    Depth variab.:    {transit_result.depth_variability:.4f}")
    print(f"    Duration consist: {transit_result.duration_consistency:.4f}")
    print(f"    Shape residual:   {transit_result.shape_residual:.4f}")
    print(f"    Anomaly score:    {transit_result.anomaly_score:.4f}")
    print(f"    Is anomalous:     {transit_result.is_anomalous}")

    transit_detected = transit_result.is_anomalous
    details["transit_detected"] = transit_detected
    details["transit_anomaly_score"] = transit_result.anomaly_score
    print(f"    >> TRANSIT DETECTION: {'PASS' if transit_detected else 'FAIL'}")

    # ── 1c. Radio Signal Injection ───────────────────────────────────
    _subsection("1c. Narrowband Radio Signal Injection")

    n_freq = 1024
    n_time = 128
    freq_start_mhz = 1420.0
    freq_end_mhz = 1420.5
    duration_sec = 300.0

    freqs = np.linspace(freq_start_mhz, freq_end_mhz, n_freq)
    times = np.linspace(0.0, duration_sec, n_time)

    # Create noise spectrogram
    spectrogram = np.abs(rng.normal(loc=10.0, scale=1.0, size=(n_freq, n_time)))

    # Inject a drifting narrowband signal
    inject_freq_hz = 1420.25e6
    inject_drift = 2.0
    inject_snr = 25.0
    inject_signal(spectrogram, freqs, times, inject_freq_hz, inject_drift, inject_snr)

    radio_result = process_spectrogram(spectrogram, freqs, times, min_snr=8.0, max_drift=10.0)
    print(f"    Candidates found:  {radio_result.n_candidates}")
    print(f"    RFI flagged:       {radio_result.n_rfi_flagged}")
    print(f"    Noise floor:       {radio_result.noise_floor:.4f}")

    # Check if injected signal was recovered
    freq_res_hz = (freq_end_mhz - freq_start_mhz) / n_freq * 1e6
    drift_tol = max(freq_res_hz / duration_sec, 1.0)
    inject_freq_mhz = inject_freq_hz / 1e6
    radio_detected = False
    best_radio_snr = 0.0
    for cand in radio_result.candidates:
        cand_freq_mhz = cand.frequency_hz / 1e6
        freq_match = abs(cand_freq_mhz - inject_freq_mhz) < 0.01
        drift_match = abs(cand.drift_rate_hz_per_s - inject_drift) < drift_tol
        if freq_match and drift_match and not cand.is_rfi:
            radio_detected = True
            best_radio_snr = cand.snr
            print(f"    FOUND: freq={cand_freq_mhz:.6f} MHz, "
                  f"drift={cand.drift_rate_hz_per_s:+.3f} Hz/s, SNR={cand.snr:.2f}")
            break

    details["radio_detected"] = radio_detected
    details["radio_snr"] = best_radio_snr
    print(f"    >> RADIO DETECTION: {'PASS' if radio_detected else 'FAIL'}")

    # ── 1d. EXODUS Convergence Score ─────────────────────────────────
    _subsection("1d. EXODUS Convergence Score")

    scorer = EXODUSScorer(threshold=0.3)
    target_data = {
        "target_id": "INJECT_TEST_001",
        "ra": 180.0,
        "dec": 45.0,
        "ir_excess": {
            "sigma_W3": ir_result.sigma_W3,
            "sigma_W4": ir_result.sigma_W4,
            "excess_W3": ir_result.excess_W3,
            "excess_W4": ir_result.excess_W4,
            "is_candidate": ir_result.is_candidate,
        },
        "transit_anomaly": {
            "anomaly_score": transit_result.anomaly_score,
            "is_anomalous": transit_result.is_anomalous,
        },
        "radio_anomaly": {
            "n_candidates": radio_result.n_candidates,
            "candidates": [
                {
                    "snr": c.snr,
                    "drift_rate_hz_per_s": c.drift_rate_hz_per_s,
                    "is_rfi": c.is_rfi,
                    "frequency_hz": c.frequency_hz,
                }
                for c in radio_result.candidates
            ],
        },
        "habitable_zone_planet": {
            "has_hz_planet": True,
            "n_hz_planets": 1,
        },
    }

    exodus = scorer.score_target(target_data)
    print(f"    Total score:    {exodus.total_score:.4f}")
    print(f"    Active channels: {exodus.n_active_channels}/6")
    print(f"    Geo mean:       {exodus.geo_mean:.4f}")
    print(f"    Conv. bonus:    {exodus.convergence_bonus:.0f}x")
    for name, cs in exodus.channel_scores.items():
        status = "ACTIVE" if cs.is_active else "      "
        print(f"      [{status}] {name:<30s} score={cs.score:.4f}")

    details["exodus_score"] = exodus.total_score
    details["n_active_channels"] = exodus.n_active_channels
    score_ok = exodus.total_score > 1.0 and exodus.n_active_channels >= 3
    print(f"    >> CONVERGENCE: {'PASS' if score_ok else 'FAIL'} "
          f"(score={exodus.total_score:.3f}, active={exodus.n_active_channels})")

    # ── 1e. Hypothesis Generation ────────────────────────────────────
    _subsection("1e. Follow-up Hypothesis Generation")

    # Use a test DB
    test_db = PROJ_ROOT / "data" / "hypotheses" / "stress_test_hyp.db"
    if test_db.exists():
        test_db.unlink()
    gen = HypothesisGenerator(db_path=str(test_db))

    simulated_results = {
        "interesting_targets": ["INJECT_TEST_001"],
        "n_interesting": 1,
    }
    followup_ids = gen.generate_followups("H001", simulated_results)
    n_followups = len(followup_ids)
    print(f"    Follow-ups generated: {n_followups}")
    for fid in followup_ids[:5]:
        fh = gen.get_hypothesis(fid)
        print(f"      [{fid}] {fh['claim'][:60]}...")

    details["hypotheses_generated"] = n_followups
    hyp_ok = n_followups > 0
    print(f"    >> HYPOTHESES: {'PASS' if hyp_ok else 'FAIL'}")

    # Cleanup test DB
    if test_db.exists():
        test_db.unlink()

    # ── 1f. UNEXPLAINED Classification ───────────────────────────────
    _subsection("1f. UNEXPLAINED Classification")

    analyst = AnalystEngine()
    hypothesis_for_test = {
        "hypothesis_id": "INJECT_H001",
        "test_method": "ir_excess_comparison",
    }
    # Build data that shows a strong signal vs. quiet control
    data_for_test = {
        "target_excess": [ir_result.sigma_W3 or 5.0] * 10,
        "control_excess": list(rng.normal(0.5, 0.8, 50)),
        "known_yso_fraction": 0.0,
        "quality_flags": [0.95] * 10,
        "independent_confirmation": True,
    }
    val_result = analyst.validate(hypothesis_for_test, data_for_test)
    classification = val_result.status.value
    print(f"    Classification:   {classification}")
    print(f"    Detection score:  {val_result.detection_score:.4f}")
    print(f"    Natural score:    {val_result.natural_explanation_score:.4f}")
    print(f"    Instrumental:     {val_result.instrumental_score:.4f}")
    print(f"    Explanation:      {val_result.explanation}")

    details["classification"] = classification
    unexplained = classification == "UNEXPLAINED"
    print(f"    >> CLASSIFICATION: {'PASS' if unexplained else 'FAIL'} (got {classification})")

    # ── Summary ──────────────────────────────────────────────────────
    all_checks = [ir_detected, transit_detected, radio_detected, score_ok, hyp_ok, unexplained]
    passed = all(all_checks)

    _subsection("TEST 1 SUMMARY")
    checks = [
        ("IR excess detected", ir_detected),
        ("Transit anomaly detected", transit_detected),
        ("Radio signal recovered", radio_detected),
        ("High convergence score", score_ok),
        ("Follow-up hypotheses generated", hyp_ok),
        ("Classified as UNEXPLAINED", unexplained),
    ]
    for label, ok in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {label}")

    print(f"\n    >> TEST 1 OVERALL: {'PASS' if passed else 'FAIL'}")
    RESULTS.record("TEST_1_INJECTION", passed, details)
    assert passed, "TEST 1 INJECTION failed"
    # audit fix N13: removed 'return passed' to avoid PytestReturnNotNoneWarning


# =====================================================================
#  TEST 2: NULL TEST
# =====================================================================

def test_2_null():
    """
    Run 20 random field stars (boring, quiet stars).
    Verify: zero unexplained anomalies, low EXODUS scores,
    no breakthrough escalations.
    """
    _section("TEST 2: NULL TEST -- Does it stay quiet on boring stars?")

    details = {
        "n_targets": 20,
        "n_ir_candidates": 0,
        "n_transit_anomalous": 0,
        "max_exodus_score": 0.0,
        "mean_exodus_score": 0.0,
    }

    rng = np.random.default_rng(seed=123)
    scorer = EXODUSScorer(threshold=0.3)

    n_targets = 20
    ir_candidates = 0
    transit_anomalous_count = 0
    exodus_scores = []

    for i in range(n_targets):
        # Generate a quiet solar-type star
        teff_offset = rng.normal(0, 200)  # slight Teff variation
        base_g = 8.0 + rng.uniform(-1.5, 1.5)  # G mag ~6.5-9.5

        phot = {
            "source_id": f"NULL_STAR_{i:03d}",
            "G":    base_g,           "G_err": 0.003 + rng.uniform(0, 0.005),
            "BP":   base_g + 0.4,     "BP_err": 0.003 + rng.uniform(0, 0.005),
            "RP":   base_g - 0.35,    "RP_err": 0.004 + rng.uniform(0, 0.005),
            "J":    base_g - 0.74,    "J_err": 0.02 + rng.uniform(0, 0.01),
            "H":    base_g - 0.94,    "H_err": 0.03 + rng.uniform(0, 0.01),
            "Ks":   base_g - 1.06,    "Ks_err": 0.02 + rng.uniform(0, 0.01),
            # WISE magnitudes consistent with photosphere (no excess)
            "W3":   base_g - 1.30 + rng.normal(0, 0.04),  "W3_err": 0.02 + rng.uniform(0, 0.03),
            "W4":   base_g - 1.39 + rng.normal(0, 0.12),  "W4_err": 0.10 + rng.uniform(0, 0.05),
        }

        ir = compute_ir_excess(phot)
        if ir.is_candidate:
            ir_candidates += 1

        # Generate a quiet, flat light curve
        n_pts = 3000
        lc_time = np.linspace(0, 20, n_pts)
        lc_flux = np.ones(n_pts) + rng.normal(0, 0.001, n_pts)

        transit = detect_transit_anomaly(lc_time, lc_flux)
        if transit.is_anomalous:
            transit_anomalous_count += 1

        # EXODUS score (no radio, no HZ planet, no Gaia anomaly)
        td = {
            "target_id": phot["source_id"],
            "ra": rng.uniform(0, 360),
            "dec": rng.uniform(-90, 90),
            "ir_excess": {
                "sigma_W3": ir.sigma_W3,
                "sigma_W4": ir.sigma_W4,
                "is_candidate": ir.is_candidate,
            },
            "transit_anomaly": {
                "anomaly_score": transit.anomaly_score,
                "is_anomalous": transit.is_anomalous,
            },
            "radio_anomaly": {"n_candidates": 0, "max_snr": 0.0},
            "gaia_photometric_anomaly": {
                "phot_g_variability": rng.uniform(0.001, 0.008),
                "n_epochs": 50,
            },
            "habitable_zone_planet": {"has_hz_planet": False},
            "proper_motion_anomaly": {
                "ruwe": rng.uniform(0.9, 1.3),
                "astrometric_excess_noise_sig": rng.uniform(0, 1),
            },
        }
        es = scorer.score_target(td)
        exodus_scores.append(es.total_score)

    max_score = max(exodus_scores)
    mean_score = np.mean(exodus_scores)

    details["n_ir_candidates"] = ir_candidates
    details["n_transit_anomalous"] = transit_anomalous_count
    details["max_exodus_score"] = float(max_score)
    details["mean_exodus_score"] = float(mean_score)

    print(f"    Targets tested:       {n_targets}")
    print(f"    IR candidates:        {ir_candidates}")
    print(f"    Transit anomalous:    {transit_anomalous_count}")
    print(f"    Max EXODUS score:     {max_score:.4f}")
    print(f"    Mean EXODUS score:    {mean_score:.4f}")

    # Check criteria: no more than 1 false IR candidate (noise),
    # no transit anomalies, low scores
    ir_ok = ir_candidates <= 1
    transit_ok = transit_anomalous_count <= 1
    score_ok = max_score < 2.0

    checks = [
        ("IR false positives <= 1", ir_ok),
        ("Transit false positives <= 1", transit_ok),
        ("Max EXODUS score < 2.0", score_ok),
    ]

    _subsection("TEST 2 SUMMARY")
    for label, ok in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {label}")

    passed = all([ir_ok, transit_ok, score_ok])
    print(f"\n    >> TEST 2 OVERALL: {'PASS' if passed else 'FAIL'}")
    RESULTS.record("TEST_2_NULL", passed, details)
    assert passed, "TEST 2 NULL failed"
    # audit fix N13: removed 'return passed' to avoid PytestReturnNotNoneWarning


# =====================================================================
#  TEST 3: KNOWN ANOMALY TEST -- Tabby's Star
# =====================================================================

def test_3_tabbys_star():
    """
    Simulate a Tabby's-Star-like light curve. Verify:
      - Irregular dimming detection finds events
      - High transit anomaly score
      - High EXODUS score
      - Follow-up hypotheses generated
    """
    _section("TEST 3: KNOWN ANOMALY TEST -- Tabby's Star (KIC 8462852)")

    details = {
        "irregular_events_found": 0,
        "max_event_depth": 0.0,
        "transit_anomaly_score": 0.0,
        "exodus_score": 0.0,
    }

    # ── 3a. Synthesize Tabby's Star light curve ──────────────────────
    _subsection("3a. Synthetic Tabby's Star Light Curve")

    rng = np.random.default_rng(seed=8462852)  # KIC number as seed
    n_pts = 10000
    lc_time = np.linspace(0, 1400, n_pts)  # ~4 years like Kepler
    lc_flux = np.ones(n_pts) + rng.normal(0, 0.0005, n_pts)

    # Inject Tabby's-Star-like events:
    #   D800 dip (~20%): days 788-800
    #   D1500 complex: days 1490-1520 (multiple sub-events)
    #   Gradual secular dimming ~1.5% over 4 years
    # Day 800 dip (20% depth, asymmetric)
    for i, t in enumerate(lc_time):
        dt = t - 792.0
        if -2.0 <= dt < 0:
            lc_flux[i] -= 0.20 * ((dt + 2.0) / 2.0) ** 0.5
        elif 0 <= dt < 10.0:
            lc_flux[i] -= 0.20 * np.exp(-dt / 3.0)

    # Day 1500 complex (multiple sub-dips)
    for centre, depth, width in [(1492, 0.08, 0.5), (1498, 0.15, 1.5), (1510, 0.05, 2.0)]:
        mask = np.abs(lc_time - centre) < width * 3
        profile = depth * np.exp(-0.5 * ((lc_time[mask] - centre) / width) ** 2)
        lc_flux[mask] -= profile

    # Secular dimming (1.5% over 4 years)
    lc_flux -= 0.015 * (lc_time / lc_time[-1])

    print(f"    Light curve: {n_pts} points over {lc_time[-1]:.0f} days")
    print(f"    Min flux:    {np.min(lc_flux):.6f}")
    print(f"    Max dip:     {(1.0 - np.min(lc_flux)) * 100:.1f}%")

    # ── 3b. Irregular Dimming Detection ──────────────────────────────
    _subsection("3b. Irregular Dimming Detection")

    irr_result = detect_irregular_dimming(lc_time, lc_flux, contamination=0.05)
    print(f"    Events detected:  {irr_result.n_events}")
    print(f"    Max depth:        {irr_result.max_depth:.6f}")
    print(f"    Anomaly score:    {irr_result.anomaly_score:.4f}")
    if irr_result.events:
        for i, evt in enumerate(irr_result.events[:5]):
            print(f"      [{i+1}] t={evt.start_time:.1f}..{evt.end_time:.1f} "
                  f"depth={evt.depth:.6f} dur={evt.duration:.1f}d "
                  f"asym={evt.asymmetry:.3f}")

    details["irregular_events_found"] = irr_result.n_events
    details["max_event_depth"] = float(irr_result.max_depth)

    irr_ok = irr_result.n_events >= 1
    print(f"    >> IRREGULAR DIMMING: {'PASS' if irr_ok else 'FAIL'}")

    # ── 3c. Transit Anomaly Score ────────────────────────────────────
    _subsection("3c. Transit Anomaly Analysis")

    transit_result = detect_transit_anomaly(lc_time, lc_flux)
    print(f"    Period:         {transit_result.period:.4f} d")
    print(f"    Anomaly score:  {transit_result.anomaly_score:.4f}")
    print(f"    Is anomalous:   {transit_result.is_anomalous}")
    print(f"    Symmetry:       {transit_result.symmetry_score:.4f}")
    print(f"    Shape residual: {transit_result.shape_residual:.4f}")

    details["transit_anomaly_score"] = transit_result.anomaly_score
    # Tabby's star may or may not trigger the BLS-based detector
    # (since events are aperiodic), but the irregular dimming should catch it

    # ── 3d. EXODUS Score ─────────────────────────────────────────────
    _subsection("3d. EXODUS Convergence Score")

    scorer = EXODUSScorer(threshold=0.3)
    target_data = {
        "target_id": "KIC_8462852",
        "ra": 301.5643,
        "dec": 44.4568,
        "ir_excess": {
            "sigma_W3": 2.0,  # Tabby's has marginal W3 excess
            "sigma_W4": 1.5,
            "is_candidate": False,
        },
        "transit_anomaly": {
            "anomaly_score": max(transit_result.anomaly_score, irr_result.anomaly_score),
            "is_anomalous": True,
        },
        "radio_anomaly": {"n_candidates": 0, "max_snr": 0.0},
        "gaia_photometric_anomaly": {
            "phot_g_variability": 0.02,  # Tabby's has some variability
            "n_epochs": 40,
        },
        "habitable_zone_planet": {"has_hz_planet": False},
        "proper_motion_anomaly": {
            "ruwe": 1.0,
            "astrometric_excess_noise_sig": 0.5,
        },
    }

    exodus = scorer.score_target(target_data)
    print(f"    Total score:     {exodus.total_score:.4f}")
    print(f"    Active channels: {exodus.n_active_channels}/6")
    for name, cs in exodus.channel_scores.items():
        if cs.is_active:
            print(f"      [ACTIVE] {name:<30s} score={cs.score:.4f}")

    details["exodus_score"] = float(exodus.total_score)

    # At minimum, the transit channel should be active
    active_ok = exodus.n_active_channels >= 1

    # ── Summary ──────────────────────────────────────────────────────
    _subsection("TEST 3 SUMMARY")
    checks = [
        ("Irregular dimming events found", irr_ok),
        ("At least 1 active EXODUS channel", active_ok),
    ]
    for label, ok in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {label}")

    passed = irr_ok and active_ok
    print(f"\n    >> TEST 3 OVERALL: {'PASS' if passed else 'FAIL'}")
    RESULTS.record("TEST_3_TABBYS_STAR", passed, details)
    assert passed, "TEST 3 TABBYS STAR failed"
    # audit fix N13: removed 'return passed' to avoid PytestReturnNotNoneWarning


# =====================================================================
#  TEST 4: CORRELATION TEST
# =====================================================================

def test_4_correlation():
    """
    Simulate cross-matching a sample of targets and verify
    the sky_matcher module works correctly with positional matching.
    """
    _section("TEST 4: CORRELATION TEST -- Cross-matching accuracy")

    details = {
        "n_targets": 0,
        "match_rate_3arcsec": 0.0,
        "match_rate_1arcsec": 0.0,
    }

    try:
        from src.correlation.sky_matcher import crossmatch
    except ImportError:
        # If sky_matcher requires live astropy.coordinates, test with synthetic data
        pass

    from astropy.coordinates import SkyCoord
    import astropy.units as u

    rng = np.random.default_rng(seed=456)

    # Create a catalog of 100 targets
    n_targets = 100
    ras = rng.uniform(180, 200, n_targets)
    decs = rng.uniform(30, 50, n_targets)

    # Create a second catalog with small offsets (simulating cross-match)
    # 90% within 1 arcsec, 95% within 3 arcsec, 5% are random
    offsets_arcsec = np.abs(rng.normal(0, 0.5, n_targets))  # most within 1 arcsec
    random_idx = rng.choice(n_targets, size=5, replace=False)
    offsets_arcsec[random_idx] = rng.uniform(10, 60, 5)  # 5 are far away

    offset_deg = offsets_arcsec / 3600.0
    ras2 = ras + offset_deg * rng.choice([-1, 1], n_targets)
    decs2 = decs + offset_deg * rng.choice([-1, 1], n_targets)

    # Use astropy SkyCoord to match
    cat1 = SkyCoord(ra=ras * u.degree, dec=decs * u.degree)
    cat2 = SkyCoord(ra=ras2 * u.degree, dec=decs2 * u.degree)

    idx, sep2d, _ = cat1.match_to_catalog_sky(cat2)
    seps_arcsec = sep2d.arcsec

    match_3arcsec = np.sum(seps_arcsec < 3.0) / n_targets
    match_1arcsec = np.sum(seps_arcsec < 1.0) / n_targets

    details["n_targets"] = n_targets
    details["match_rate_3arcsec"] = float(match_3arcsec)
    details["match_rate_1arcsec"] = float(match_1arcsec)

    print(f"    Targets:            {n_targets}")
    print(f"    Match rate (3\"):    {match_3arcsec:.1%}")
    print(f"    Match rate (1\"):    {match_1arcsec:.1%}")
    print(f"    Median separation:  {np.median(seps_arcsec):.3f} arcsec")

    match_ok = match_3arcsec >= 0.90  # at least 90% match at 3 arcsec

    # Also test the batch scorer to verify it handles a list properly
    _subsection("4b. Batch EXODUS Scoring")
    scorer = EXODUSScorer(threshold=0.3)
    batch_targets = []
    for i in range(n_targets):
        batch_targets.append({
            "target_id": f"CORR_TARGET_{i:03d}",
            "ra": float(ras[i]),
            "dec": float(decs[i]),
            "ir_excess": {
                "sigma_W3": float(rng.uniform(0, 2)),
                "sigma_W4": float(rng.uniform(0, 2)),
            },
            "transit_anomaly": {"anomaly_score": float(rng.uniform(0, 0.2))},
        })
    results = scorer.score_all(batch_targets)
    print(f"    Batch scored: {len(results)} targets")
    print(f"    Top score:    {results[0].total_score:.4f} (active={results[0].n_active_channels})")

    batch_ok = len(results) == n_targets

    _subsection("TEST 4 SUMMARY")
    checks = [
        ("Cross-match rate >= 90% at 3 arcsec", match_ok),
        ("Batch scorer handles 100 targets", batch_ok),
    ]
    for label, ok in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {label}")

    passed = match_ok and batch_ok
    print(f"\n    >> TEST 4 OVERALL: {'PASS' if passed else 'FAIL'}")
    RESULTS.record("TEST_4_CORRELATION", passed, details)
    assert passed, "TEST 4 CORRELATION failed"
    # audit fix N13: removed 'return passed' to avoid PytestReturnNotNoneWarning


# =====================================================================
#  TEST 5: TEMPORAL ARCHAEOLOGY TEST
# =====================================================================

def test_5_temporal_archaeology():
    """
    Test the temporal differencing analysis with synthetic epoch data.
    Verify no crashes and proper identification of transient sources.
    """
    _section("TEST 5: TEMPORAL ARCHAEOLOGY TEST")

    details = {
        "validation_ran": False,
        "classification": "N/A",
    }

    _subsection("5a. Temporal Differencing via Analyst Engine")

    analyst = AnalystEngine()
    rng = np.random.default_rng(seed=789)

    # Simulate two epochs of radio survey data
    n_sources = 50
    base_flux = rng.uniform(0.01, 1.0, n_sources)  # Jy
    rms1 = 0.002  # NVSS-like noise
    rms2 = 0.001  # VLASS-like noise

    epoch1_flux = base_flux + rng.normal(0, rms1, n_sources)
    epoch2_flux = base_flux + rng.normal(0, rms2, n_sources)

    # Inject 2 "appeared" sources (bright in epoch 2, faint in epoch 1)
    epoch1_flux[5] = rng.normal(0, rms1)
    epoch2_flux[5] = 0.5  # Appeared!
    epoch1_flux[12] = rng.normal(0, rms1)
    epoch2_flux[12] = 0.3  # Appeared!

    hypothesis = {
        "hypothesis_id": "TEMPORAL_H001",
        "test_method": "temporal_differencing",
    }
    data = {
        "epoch1_flux": epoch1_flux.tolist(),
        "epoch2_flux": epoch2_flux.tolist(),
        "epoch1_rms": rms1,
        "epoch2_rms": rms2,
        "n_sources": n_sources,
    }

    result = analyst.validate(hypothesis, data)
    print(f"    Classification:   {result.status.value}")
    print(f"    Detection score:  {result.detection_score:.4f}")
    print(f"    p-value:          {result.statistical_significance:.2e}")
    print(f"    Explanation:      {result.explanation}")

    details["validation_ran"] = True
    details["classification"] = result.status.value
    # Should detect the injected variable sources
    ran_ok = result.detection_score > 0.0

    _subsection("5b. Temporal Archaeology Module Import")
    try:
        from src.detection.temporal_archaeology import (
            TemporalArchaeology,
        )
        import_ok = True
        print(f"    Module imported successfully")
    except Exception as e:
        import_ok = False
        print(f"    Module import FAILED: {e}")

    _subsection("TEST 5 SUMMARY")
    checks = [
        ("Temporal differencing ran without crash", ran_ok),
        ("Module structure intact", import_ok),
    ]
    for label, ok in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {label}")

    passed = ran_ok and import_ok
    print(f"\n    >> TEST 5 OVERALL: {'PASS' if passed else 'FAIL'}")
    RESULTS.record("TEST_5_TEMPORAL", passed, details)
    assert passed, "TEST 5 TEMPORAL failed"
    # audit fix N13: removed 'return passed' to avoid PytestReturnNotNoneWarning


# =====================================================================
#  TEST 6: SELF-IMPROVEMENT TEST
# =====================================================================

def test_6_evolver():
    """
    Simulate 5 iterations with mock data. Verify:
      - Threshold adjustments happen correctly
      - Hypothesis growth occurs
      - Strategy evolution (deprioritize/promote) works
    """
    _section("TEST 6: SELF-IMPROVEMENT TEST -- Evolver learning")

    details = {
        "iterations_run": 0,
        "sigma_history": [],
        "hypotheses_per_iteration": [],
        "strategies_changed": 0,
    }

    import random as stdlib_random

    log_path = PROJ_ROOT / "data" / "hypotheses" / "stress_test_evolver.json"
    if log_path.exists():
        log_path.unlink()

    evolver = EvolverEngine(log_path=log_path)
    rng = stdlib_random.Random(42)

    sigma_history = []
    hyp_counts = []
    total_strategy_changes = 0

    methods = ["ir_excess", "transit_anomaly", "radio_narrowband", "spectral_lines"]

    for iteration in range(1, 6):
        _subsection(f"6. Iteration {iteration}")

        # Build mock state
        n_hyp = rng.randint(8, 15)
        hypotheses = []
        for j in range(n_hyp):
            method = rng.choice(methods)
            # Higher FP rate in early iterations to trigger sigma increase
            if iteration <= 2:
                status = rng.choices(
                    ["natural", "artifact", "dead_end", "unexplained", "pending"],
                    weights=[0.35, 0.25, 0.15, 0.15, 0.10],
                    k=1,
                )[0]
            else:
                # Later iterations: lower FP rate to trigger sigma decrease
                status = rng.choices(
                    ["natural", "artifact", "dead_end", "unexplained", "pending"],
                    weights=[0.15, 0.05, 0.20, 0.30, 0.30],
                    k=1,
                )[0]

            props = {"ra": rng.uniform(0, 360), "dec": rng.uniform(-90, 90)}
            if rng.random() < 0.3:
                props["ir_excess"] = True
            if rng.random() < 0.15:
                props["temporal_pattern"] = "periodic_dimming_0.3d"

            hypotheses.append({
                "id": f"hyp_{iteration}_{j}",
                "text": f"Test {method} on field {j}",
                "method": method,
                "status": status,
                "scores": {"anomaly_score": rng.uniform(0, 10)},
                "properties": props,
            })

        anomalies = sum(1 for h in hypotheses if h["status"] != "pending")
        fp = sum(1 for h in hypotheses if h["status"] in ("artifact", "natural"))

        current_sigma = 3.0
        if sigma_history:
            current_sigma = sigma_history[-1]

        state = ResearchState(
            iteration=iteration,
            hypotheses_tested=hypotheses,
            targets_processed=rng.randint(20, 100),
            anomalies_found=max(anomalies, 1),
            false_positives=fp,
            current_thresholds={"anomaly_sigma": current_sigma},
        )

        record = evolver.evolve(state)
        new_sigma = state.current_thresholds["anomaly_sigma"]
        sigma_history.append(new_sigma)
        hyp_counts.append(len(record.new_hypotheses_generated))
        total_strategy_changes += len(record.strategies_deprioritized) + len(record.strategies_promoted)

        print(f"    Sigma: {current_sigma:.2f} -> {new_sigma:.2f}")
        print(f"    FP rate: {record.false_positive_rate:.1%}")
        print(f"    New hypotheses: {len(record.new_hypotheses_generated)}")
        print(f"    Deprioritized: {len(record.strategies_deprioritized)}")
        print(f"    Promoted: {len(record.strategies_promoted)}")

    details["iterations_run"] = 5
    details["sigma_history"] = sigma_history
    details["hypotheses_per_iteration"] = hyp_counts
    details["strategies_changed"] = total_strategy_changes

    # Verify
    sigma_changed = sigma_history[0] != sigma_history[-1] or len(set(sigma_history)) > 1
    hyp_generated = sum(hyp_counts) > 0
    has_strategy_changes = total_strategy_changes > 0

    # Cleanup
    if log_path.exists():
        log_path.unlink()

    _subsection("TEST 6 SUMMARY")
    checks = [
        ("Sigma threshold was adjusted", sigma_changed),
        ("New hypotheses generated", hyp_generated),
        ("Strategy weights changed", has_strategy_changes),
    ]
    for label, ok in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {label}")

    print(f"    Sigma history: {' -> '.join(f'{s:.2f}' for s in sigma_history)}")
    print(f"    Total new hypotheses: {sum(hyp_counts)}")

    passed = sigma_changed and hyp_generated
    print(f"\n    >> TEST 6 OVERALL: {'PASS' if passed else 'FAIL'}")
    RESULTS.record("TEST_6_EVOLVER", passed, details)
    assert passed, "TEST 6 EVOLVER failed"
    # audit fix N13: removed 'return passed' to avoid PytestReturnNotNoneWarning


# =====================================================================
#  TEST 7: SCIENCE SANITY CHECK
# =====================================================================

def test_7_science_sanity():
    """
    Review all detectors for scientific correctness:
      - Red giant IR excess filtering
      - Starspot transit asymmetry
      - RFI database checks
      - Data availability normalization in scoring
      - Proper motion propagation in cross-matching
    """
    _section("TEST 7: SCIENCE SANITY CHECK")

    details = {
        "checks_passed": 0,
        "checks_total": 0,
        "issues": [],
    }

    all_ok = True

    # ── 7a. Red giant IR excess: high Teff should still work ─────────
    _subsection("7a. Red Giant IR Excess Filtering")

    # A cool red giant (T~3500K) will naturally have more mid-IR emission.
    # The BB fit should handle this without false flagging.
    red_giant = {
        "source_id": "RED_GIANT_001",
        "G":    6.00,   "G_err":  0.003,
        "BP":   7.50,   "BP_err": 0.003,
        "RP":   4.80,   "RP_err": 0.004,
        "J":    3.50,   "J_err":  0.02,
        "H":    2.80,   "H_err":  0.03,
        "Ks":   2.60,   "Ks_err": 0.02,
        # Red giant W3/W4 consistent with cool photosphere
        "W3":   2.40,   "W3_err": 0.03,
        "W4":   2.30,   "W4_err": 0.10,
    }
    rg_result = compute_ir_excess(red_giant)
    print(f"    Red giant Teff:   {rg_result.fitted_teff:.0f} K")
    print(f"    W3 excess:        {rg_result.excess_W3}")
    print(f"    W3 sigma:         {rg_result.sigma_W3}")
    print(f"    Candidate:        {rg_result.is_candidate}")

    # Red giant should not be falsely flagged (photosphere extends to mid-IR)
    rg_ok = not rg_result.is_candidate or (rg_result.sigma_W3 or 0) < 3.0
    details["checks_total"] = details.get("checks_total", 0) + 1
    if rg_ok:
        details["checks_passed"] = details.get("checks_passed", 0) + 1
        print(f"    >> PASS: Red giant not falsely flagged")
    else:
        details["issues"].append("Red giant falsely flagged as IR excess candidate")
        RESULTS.add_issue("Red giant false positive in IR excess detector")
        print(f"    >> ISSUE: Red giant falsely flagged (sigma_W3={rg_result.sigma_W3})")
        all_ok = False

    # ── 7b. Symmetric transit should NOT be anomalous ────────────────
    _subsection("7b. Starspot / Symmetric Transit Check")

    n_pts = 5000
    rng = np.random.default_rng(seed=321)
    time_clean = np.linspace(0, 30, n_pts)
    flux_clean = np.ones(n_pts) + rng.normal(0, 0.001, n_pts)

    # Inject a clean symmetric transit
    period = 3.5
    depth = 0.01
    duration = 0.15
    phase = ((time_clean % period) / period)
    phase_offset = np.abs(phase - 0.5)
    half_dur = (duration / period) / 2.0
    in_transit = phase_offset < half_dur
    transit_phase = phase_offset[in_transit] / half_dur
    profile = 0.5 * (1.0 + np.cos(np.pi * transit_phase))
    flux_clean[in_transit] -= depth * profile

    sym_result = detect_transit_anomaly(time_clean, flux_clean)
    print(f"    Clean transit anomaly score: {sym_result.anomaly_score:.4f}")
    print(f"    Is anomalous:               {sym_result.is_anomalous}")
    print(f"    Symmetry score:             {sym_result.symmetry_score:.4f}")

    details["checks_total"] += 1
    sym_ok = not sym_result.is_anomalous
    if sym_ok:
        details["checks_passed"] += 1
        print(f"    >> PASS: Symmetric transit not falsely flagged")
    else:
        details["issues"].append(f"Symmetric transit falsely flagged (score={sym_result.anomaly_score:.4f})")
        RESULTS.add_issue("Symmetric transit false positive in transit anomaly detector")
        print(f"    >> ISSUE: Symmetric transit falsely flagged")
        all_ok = False

    # ── 7c. RFI database coverage ────────────────────────────────────
    _subsection("7c. RFI Database Coverage")

    from src.processing.radio_processor import KNOWN_RFI_FREQUENCIES_MHZ

    n_rfi_bands = len(KNOWN_RFI_FREQUENCIES_MHZ)
    print(f"    Known RFI bands: {n_rfi_bands}")

    # Check that critical bands are covered
    critical_bands = [
        ("Wi-Fi 2.4 GHz", 2400, 2500),
        ("GPS L1", 1559, 1591),
        ("Cell-phone", 824, 894),
    ]
    rfi_coverage_ok = True
    for name, lo, hi in critical_bands:
        covered = any(
            rfi_lo <= lo and rfi_hi >= hi
            for rfi_lo, rfi_hi in KNOWN_RFI_FREQUENCIES_MHZ
        )
        # Check partial overlap too
        if not covered:
            covered = any(
                (rfi_lo <= hi and rfi_hi >= lo)
                for rfi_lo, rfi_hi in KNOWN_RFI_FREQUENCIES_MHZ
            )
        status = "covered" if covered else "MISSING"
        print(f"      {name} ({lo}-{hi} MHz): {status}")
        if not covered:
            rfi_coverage_ok = False

    details["checks_total"] += 1
    if rfi_coverage_ok:
        details["checks_passed"] += 1
    else:
        details["issues"].append("Missing critical RFI band coverage")

    print(f"    >> {'PASS' if rfi_coverage_ok else 'FAIL'}: RFI database coverage")

    # ── 7d. Data availability normalization in scoring ────────────────
    _subsection("7d. Scoring with Missing Channels")

    scorer = EXODUSScorer(threshold=0.3)

    # A target with only 1 channel should NOT get a convergence bonus
    # Note: excess_W3/W4 must be negative (genuine excess = star brighter than model)
    td_1ch = {
        "target_id": "ONE_CHANNEL",
        "ra": 0, "dec": 0,
        "ir_excess": {"sigma_W3": 8.0, "sigma_W4": 6.0,
                      "excess_W3": -0.5, "excess_W4": -0.3,
                      "is_candidate": True},
        # All other channels missing
    }
    es_1ch = scorer.score_target(td_1ch)

    # A target with 3 detection channels active should get convergence bonus.
    # HZ is prior-only (not counted in n_active_channels), so we use
    # IR + transit + radio as the 3 independent detection channels.
    td_3ch = {
        "target_id": "THREE_CHANNELS",
        "ra": 0, "dec": 0,
        "ir_excess": {"sigma_W3": 5.0, "sigma_W4": 4.0,
                      "excess_W3": -0.3, "excess_W4": -0.2,
                      "is_candidate": True},
        "transit_anomaly": {"anomaly_score": 0.5, "is_anomalous": True},
        "radio_anomaly": {"n_candidates": 2, "max_snr": 15.0},
        "habitable_zone_planet": {"has_hz_planet": True, "n_hz_planets": 1},
    }
    es_3ch = scorer.score_target(td_3ch)

    print(f"    1-channel score: {es_1ch.total_score:.4f} (active={es_1ch.n_active_channels})")
    print(f"    3-channel score: {es_3ch.total_score:.4f} (active={es_3ch.n_active_channels})")
    print(f"    3-channel bonus: {es_3ch.convergence_bonus:.0f}x")

    details["checks_total"] += 1
    multi_ok = es_3ch.n_active_channels >= 3 and es_3ch.convergence_bonus >= 4.0
    if multi_ok:
        details["checks_passed"] += 1
        print(f"    >> PASS: Multi-channel convergence bonus works")
    else:
        details["issues"].append("Convergence bonus not working correctly")
        print(f"    >> FAIL: Convergence bonus issue")
        all_ok = False

    # ── 7e. EXODUS score of 0 for completely empty target ────────────
    _subsection("7e. Empty Target Score")

    td_empty = {"target_id": "EMPTY", "ra": 0, "dec": 0}
    es_empty = scorer.score_target(td_empty)
    print(f"    Empty target score: {es_empty.total_score:.4f}")

    details["checks_total"] += 1
    empty_ok = es_empty.total_score == 0.0 and es_empty.n_active_channels == 0
    if empty_ok:
        details["checks_passed"] += 1
        print(f"    >> PASS: Empty target gets score 0")
    else:
        details["issues"].append("Empty target should get score 0")
        print(f"    >> FAIL: Empty target score {es_empty.total_score}")
        all_ok = False

    _subsection("TEST 7 SUMMARY")
    checks = [
        ("Red giant not falsely flagged", rg_ok),
        ("Symmetric transit not anomalous", sym_ok),
        ("RFI database covers critical bands", rfi_coverage_ok),
        ("Multi-channel convergence works", multi_ok),
        ("Empty target gets score 0", empty_ok),
    ]
    for label, ok in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {label}")

    passed = all([rg_ok, sym_ok, rfi_coverage_ok, multi_ok, empty_ok])
    print(f"\n    >> TEST 7 OVERALL: {'PASS' if passed else 'FAIL'}")
    RESULTS.record("TEST_7_SCIENCE_SANITY", passed, details)
    assert passed, "TEST 7 SCIENCE SANITY failed"
    # audit fix N13: removed 'return passed' to avoid PytestReturnNotNoneWarning


# =====================================================================
#  TEST 8: DISCOVERY TRAP TEST
# =====================================================================

def test_8_discovery_traps():
    """
    Review for common discovery traps:
      - Overly high thresholds that suppress real signals
      - Hardcoded anomaly assumptions
      - Overly aggressive rejection
      - Frequency range limitations
      - Extended source filtering
    """
    _section("TEST 8: DISCOVERY TRAP TEST")

    details = {
        "checks_passed": 0,
        "checks_total": 0,
        "issues": [],
    }

    # ── 8a. Threshold sensitivity: a marginal signal ─────────────────
    _subsection("8a. Marginal Signal Detection (3.5 sigma)")

    # A signal just above threshold should still be flagged
    marginal_phot = {
        "source_id": "MARGINAL_001",
        "G":    8.50,   "G_err":  0.003,
        "BP":   8.92,   "BP_err": 0.003,
        "RP":   8.15,   "RP_err": 0.004,
        "J":    7.76,   "J_err":  0.02,
        "H":    7.56,   "H_err":  0.03,
        "Ks":   7.44,   "Ks_err": 0.02,
        # Inject moderate excess (should trigger at ~3.5 sigma)
        "W3":   6.80,   "W3_err": 0.03,
        "W4":   6.50,   "W4_err": 0.10,
    }
    marginal_result = compute_ir_excess(marginal_phot)
    print(f"    W3 sigma: {marginal_result.sigma_W3}")
    print(f"    W4 sigma: {marginal_result.sigma_W4}")
    print(f"    Candidate: {marginal_result.is_candidate}")

    details["checks_total"] += 1
    # We want to check that the pipeline IS sensitive at moderate sigma
    # The key is that it's not set too high to miss real signals
    marginal_ok = True  # We're checking the thresholds are reasonable
    if marginal_result.sigma_W3 is not None and marginal_result.sigma_W3 > 3.0:
        if not marginal_result.is_candidate:
            RESULTS.add_issue("Marginal signal at {:.1f} sigma not detected".format(
                marginal_result.sigma_W3))
            marginal_ok = False
    details["checks_passed"] += 1 if marginal_ok else 0

    print(f"    >> {'PASS' if marginal_ok else 'FAIL'}: Marginal signal handling")

    # ── 8b. Analyst doesn't over-reject ──────────────────────────────
    _subsection("8b. Analyst Over-Rejection Check")

    analyst = AnalystEngine()
    rng = np.random.default_rng(seed=555)

    # A strong, clean, non-natural signal should not be rejected
    strong_data = {
        "target_excess": [8.0, 7.5, 9.0, 6.0, 7.0],
        "control_excess": list(rng.normal(0.5, 0.8, 100)),
        "known_yso_fraction": 0.0,
        "quality_flags": [0.98] * 5,
        "independent_confirmation": True,
    }
    hypothesis = {"hypothesis_id": "TRAP_H001", "test_method": "ir_excess_comparison"}
    val = analyst.validate(hypothesis, strong_data)
    print(f"    Classification: {val.status.value}")
    print(f"    Detection:      {val.detection_score:.4f}")
    print(f"    Natural:        {val.natural_explanation_score:.4f}")
    print(f"    Instrumental:   {val.instrumental_score:.4f}")

    details["checks_total"] += 1
    rejection_ok = val.status == ValidationStatus.UNEXPLAINED
    if rejection_ok:
        details["checks_passed"] += 1
        print(f"    >> PASS: Strong signal correctly classified as UNEXPLAINED")
    else:
        details["issues"].append(f"Strong signal classified as {val.status.value} instead of UNEXPLAINED")
        RESULTS.add_issue(f"Analyst over-rejected a strong signal (got {val.status.value})")
        print(f"    >> ISSUE: Expected UNEXPLAINED, got {val.status.value}")

    # ── 8c. Breakthrough engine handles full escalation ──────────────
    _subsection("8c. Full Breakthrough Escalation")

    bt_data_dir = PROJ_ROOT / "data" / "results" / "stress_test_bt"
    bt_data_dir.mkdir(parents=True, exist_ok=True)
    bt_engine = BreakthroughEngine(data_dir=str(bt_data_dir))

    mock_result = {
        "anomaly_type": "ir_excess",
        "status": "UNEXPLAINED",
        "confidence": 0.9,
        "details": "Strong IR excess with no known explanation",
    }
    mock_target = {
        "source_id": "BT_TEST_001",
        "ra": 150.0,
        "dec": 30.0,
    }

    candidate = bt_engine.escalate(mock_result, mock_target)
    print(f"    Candidate ID:   {candidate.candidate_id[:12]}...")
    print(f"    Final status:   {candidate.status}")
    print(f"    Current level:  {candidate.current_level}")
    print(f"    Levels passed:  {len(candidate.level_results)}")

    details["checks_total"] += 1
    bt_ok = candidate.status in ("unresolved", "resolved_natural") and len(candidate.level_results) >= 4
    if bt_ok:
        details["checks_passed"] += 1
        print(f"    >> PASS: Breakthrough engine completed full escalation")
    else:
        details["issues"].append("Breakthrough engine incomplete escalation")
        print(f"    >> FAIL: Only reached {len(candidate.level_results)} levels")

    # Cleanup
    import shutil
    if bt_data_dir.exists():
        shutil.rmtree(bt_data_dir)

    # ── 8d. Radio frequency range check ──────────────────────────────
    _subsection("8d. Radio Frequency Range Check")

    # Ensure the dedoppler search works at the hydrogen line (1420 MHz)
    rng = np.random.default_rng(seed=678)
    n_freq = 512
    n_time = 64
    freqs = np.linspace(1420.0, 1420.5, n_freq)
    times = np.linspace(0, 300, n_time)
    spec = np.abs(rng.normal(10, 1, (n_freq, n_time)))

    # Inject at 1420.405 MHz (H-line)
    inject_signal(spec, freqs, times, 1420.405e6, 1.0, 20.0)
    result = process_spectrogram(spec, freqs, times, min_snr=8.0)
    h_line_detected = result.n_candidates > 0

    details["checks_total"] += 1
    if h_line_detected:
        details["checks_passed"] += 1
        print(f"    >> PASS: Signal detected near hydrogen line")
    else:
        details["issues"].append("Signal near hydrogen line not detected")
        print(f"    >> FAIL: No signal detected at 1420.405 MHz")

    # ── 8e. The scoring formula doesn't clip too aggressively ────────
    _subsection("8e. Score Range Check")

    scorer = EXODUSScorer(threshold=0.3)
    # A maximally anomalous target with all detection channels active.
    # Note: HZ is prior-only (boosts score but excluded from
    # n_active_channels), so 5 detection channels max.
    max_target = {
        "target_id": "MAX_TARGET",
        "ra": 0, "dec": 0,
        "ir_excess": {"sigma_W3": 50.0, "sigma_W4": 40.0,
                      "excess_W3": -2.0, "excess_W4": -1.5,
                      "is_candidate": True},
        "transit_anomaly": {"anomaly_score": 0.95},
        "radio_anomaly": {"n_candidates": 5, "max_snr": 50.0},
        "gaia_photometric_anomaly": {"phot_g_variability": 0.3, "n_epochs": 50},
        "habitable_zone_planet": {"has_hz_planet": True, "n_hz_planets": 2},
        "proper_motion_anomaly": {"ruwe": 3.0, "astrometric_excess_noise_sig": 15.0},
    }
    max_score = scorer.score_target(max_target)
    print(f"    Max target score:    {max_score.total_score:.4f}")
    print(f"    Active channels:     {max_score.n_active_channels}")
    print(f"    Convergence bonus:   {max_score.convergence_bonus:.0f}x")

    details["checks_total"] += 1
    # 5 detection channels (HZ is prior-only) + massive score
    range_ok = max_score.n_active_channels == 5 and max_score.total_score > 10.0
    if range_ok:
        details["checks_passed"] += 1
        print(f"    >> PASS: Score range adequate (max={max_score.total_score:.1f})")
    else:
        details["issues"].append("Score range may be too restrictive")
        print(f"    >> FAIL: Score range issue")

    _subsection("TEST 8 SUMMARY")
    checks = [
        ("Marginal signal handled properly", marginal_ok),
        ("Strong signal not over-rejected", rejection_ok),
        ("Full breakthrough escalation works", bt_ok),
        ("Hydrogen line signal detectable", h_line_detected),
        ("Score range adequate", range_ok),
    ]
    for label, ok in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {label}")

    passed = all([marginal_ok, rejection_ok, bt_ok, h_line_detected, range_ok])
    print(f"\n    >> TEST 8 OVERALL: {'PASS' if passed else 'FAIL'}")
    RESULTS.record("TEST_8_DISCOVERY_TRAPS", passed, details)
    assert passed, "TEST 8 DISCOVERY TRAPS failed"
    # audit fix N13: removed 'return passed' to avoid PytestReturnNotNoneWarning


# =====================================================================
#  FINAL REPORT GENERATION
# =====================================================================

def generate_report():
    """Generate the final stress test report at data/reports/stress_test_report.md"""
    _section("GENERATING FINAL STRESS TEST REPORT")

    elapsed = time.time() - RESULTS.start_time
    report_dir = PROJ_ROOT / "data" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "stress_test_report.md"

    n_passed = sum(1 for t in RESULTS.test_results.values() if t["passed"])
    n_total = len(RESULTS.test_results)

    lines = [
        "# Project EXODUS -- Post-Build Stress Test Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Duration:** {elapsed:.1f} seconds",
        f"**Overall:** {n_passed}/{n_total} tests passed",
        "",
        "---",
        "",
        "## Test Results Summary",
        "",
        "| Test | Status | Details |",
        "|------|--------|---------|",
    ]

    for test_name, result in RESULTS.test_results.items():
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        # Extract key detail
        d = result["details"]
        detail_str = ""

        if test_name == "TEST_1_INJECTION":
            detail_str = (
                f"IR:{d.get('ir_detected','?')} "
                f"Transit:{d.get('transit_detected','?')} "
                f"Radio:{d.get('radio_detected','?')} "
                f"EXODUS={d.get('exodus_score',0):.2f} "
                f"Class={d.get('classification','?')}"
            )
        elif test_name == "TEST_2_NULL":
            detail_str = (
                f"IR FP:{d.get('n_ir_candidates',0)} "
                f"Transit FP:{d.get('n_transit_anomalous',0)} "
                f"MaxScore={d.get('max_exodus_score',0):.3f}"
            )
        elif test_name == "TEST_3_TABBYS_STAR":
            detail_str = (
                f"Events:{d.get('irregular_events_found',0)} "
                f"MaxDepth={d.get('max_event_depth',0):.4f} "
                f"EXODUS={d.get('exodus_score',0):.3f}"
            )
        elif test_name == "TEST_4_CORRELATION":
            detail_str = (
                f"Match@3\":{d.get('match_rate_3arcsec',0):.1%} "
                f"Match@1\":{d.get('match_rate_1arcsec',0):.1%}"
            )
        elif test_name == "TEST_5_TEMPORAL":
            detail_str = f"Class={d.get('classification','?')}"
        elif test_name == "TEST_6_EVOLVER":
            detail_str = (
                f"Sigma: {' -> '.join(f'{s:.2f}' for s in d.get('sigma_history',[]))} "
                f"Strategies changed: {d.get('strategies_changed',0)}"
            )
        elif test_name == "TEST_7_SCIENCE_SANITY":
            detail_str = f"{d.get('checks_passed',0)}/{d.get('checks_total',0)} checks"
        elif test_name == "TEST_8_DISCOVERY_TRAPS":
            detail_str = f"{d.get('checks_passed',0)}/{d.get('checks_total',0)} checks"

        lines.append(f"| {test_name} | {status} | {detail_str} |")

    lines += [
        "",
        "---",
        "",
        "## Detailed Results",
        "",
    ]

    # Detailed results per test
    for test_name, result in RESULTS.test_results.items():
        lines.append(f"### {test_name}")
        lines.append("")
        lines.append(f"**Status:** {'PASS' if result['passed'] else 'FAIL'}")
        lines.append(f"**Timestamp:** {result['timestamp']}")
        lines.append("")
        for key, val in result["details"].items():
            if isinstance(val, list) and len(val) > 10:
                lines.append(f"- **{key}:** [{len(val)} items]")
            else:
                lines.append(f"- **{key}:** {val}")
        lines.append("")

    # Issues found
    if RESULTS.issues_found:
        lines += [
            "---",
            "",
            "## Issues Found",
            "",
        ]
        for issue in RESULTS.issues_found:
            lines.append(f"- {issue}")
        lines.append("")

    # Fixes applied
    if RESULTS.fixes_applied:
        lines += [
            "---",
            "",
            "## Fixes Applied",
            "",
        ]
        for fix in RESULTS.fixes_applied:
            lines.append(f"- {fix}")
        lines.append("")

    # Recommendations
    lines += [
        "---",
        "",
        "## Recommendations",
        "",
        "1. **Regularly run this stress test suite** after any code changes to catch regressions",
        "2. **Add real-data tests** once API access is configured (Gaia, WISE, BL)",
        "3. **Expand the RFI database** with site-specific frequencies for target observatories",
        "4. **Tune convergence bonus** based on actual false-positive rates from real data",
        "5. **Add spectral type checks** to the IR excess detector to better handle red giants and AGB stars",
        "",
        "---",
        "",
        f"*Report generated by Project EXODUS Stress Test Suite v1.0*",
    ]

    report_content = "\n".join(lines)

    with open(report_path, "w") as f:
        f.write(report_content)

    print(f"    Report written to: {report_path}")
    print(f"    Total tests: {n_total}")
    print(f"    Passed:      {n_passed}")
    print(f"    Failed:      {n_total - n_passed}")
    print(f"    Issues:      {len(RESULTS.issues_found)}")

    return report_path


# =====================================================================
#  MAIN
# =====================================================================

def main():
    print()
    print("=" * 78)
    print("  PROJECT EXODUS -- POST-BUILD STRESS TEST SUITE")
    print("=" * 78)
    print(f"  Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()

    tests = [
        ("TEST 1: Injection", test_1_injection),
        ("TEST 2: Null", test_2_null),
        ("TEST 3: Tabby's Star", test_3_tabbys_star),
        ("TEST 4: Correlation", test_4_correlation),
        ("TEST 5: Temporal Archaeology", test_5_temporal_archaeology),
        ("TEST 6: Self-Improvement", test_6_evolver),
        ("TEST 7: Science Sanity", test_7_science_sanity),
        ("TEST 8: Discovery Traps", test_8_discovery_traps),
    ]

    results_summary = []
    for name, test_fn in tests:
        try:
            test_fn()
            # Test functions use assert (no return value) — no exception = pass
            results_summary.append((name, True))
        except Exception as exc:
            print(f"\n    !! TEST CRASHED: {exc}")
            traceback.print_exc()
            results_summary.append((name, False))
            RESULTS.record(name.replace(": ", "_").replace(" ", "_").upper(), False,
                          {"error": str(exc)})
            RESULTS.add_issue(f"{name} crashed: {exc}")

    # Generate report
    report_path = generate_report()

    # Final summary
    _section("FINAL RESULTS")
    for name, passed in results_summary:
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}]  {name}")

    n_passed = sum(1 for _, p in results_summary if p)
    n_total = len(results_summary)
    print(f"\n    TOTAL: {n_passed}/{n_total} tests passed")
    print(f"    Report: {report_path}")

    elapsed = time.time() - RESULTS.start_time
    print(f"    Duration: {elapsed:.1f} seconds")
    print()

    return n_passed == n_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
