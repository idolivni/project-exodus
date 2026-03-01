"""
Regression tests for new detection channels and scoring.

Tests UV anomaly, radio emission, HR diagram outlier detection modules
and their integration into the EXODUS scorer (now 10 channels).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.uv_anomaly import compute_uv_anomaly, UVAnomalyResult
from src.detection.radio_emission import compute_radio_emission, RadioEmissionResult
from src.detection.hr_anomaly import compute_hr_anomaly, HRAnomalyResult
from src.scoring.exodus_score import EXODUSScorer


# ── UV Anomaly Tests ──────────────────────────────────────────────

class TestUVAnomaly:
    def test_no_data_returns_zero(self):
        result = compute_uv_anomaly(None)
        assert result.anomaly_score == 0.0
        assert result.has_data is False

    def test_empty_metrics_returns_zero(self):
        result = compute_uv_anomaly({"has_data": False})
        assert result.anomaly_score == 0.0

    def test_high_score_passes_through(self):
        result = compute_uv_anomaly({
            "has_data": True,
            "uv_anomaly_score": 0.8,
            "is_uv_deficit": True,
            "nuv_residual": 3.0,
            "data_source": "galex_vizier",
        })
        assert result.anomaly_score == 0.8
        assert result.is_anomalous is True
        assert result.has_data is True

    def test_artifact_contamination_zeros_score(self):
        result = compute_uv_anomaly(
            {"has_data": True, "uv_anomaly_score": 0.9, "data_source": "galex_vizier"},
            galex_raw={"artifact_clean": False},
        )
        assert result.anomaly_score == 0.0
        assert result.is_anomalous is False

    def test_ir_crossmatch_no_boost(self):
        """UV score must NOT be boosted by IR excess (audit fix B2).

        Channel independence requires each channel to score on its own
        merit.  The old 1.5x UV-IR bonus was removed.
        """
        base = compute_uv_anomaly(
            {"has_data": True, "uv_anomaly_score": 0.5, "data_source": "galex_vizier"},
        )
        with_ir = compute_uv_anomaly(
            {"has_data": True, "uv_anomaly_score": 0.5, "data_source": "galex_vizier"},
            ir_excess_data={"is_candidate": True},
        )
        # Scores must be identical — IR data must not influence UV score
        assert with_ir.anomaly_score == base.anomaly_score

    def test_to_dict_complete(self):
        result = compute_uv_anomaly({
            "has_data": True, "uv_anomaly_score": 0.4,
            "data_source": "galex_vizier", "nuv_residual": 1.5,
        })
        d = result.to_dict()
        assert "anomaly_score" in d
        assert "data_source" in d
        assert d["data_source"] == "galex_vizier"


# ── Radio Emission Tests ──────────────────────────────────────────

class TestRadioEmission:
    def test_no_data_returns_zero(self):
        result = compute_radio_emission(None)
        assert result.anomaly_score == 0.0
        assert result.has_data is False

    def test_weak_flux_not_detected(self):
        result = compute_radio_emission({
            "peak_flux_mjy": 0.5, "sep_arcsec": 2.0,
            "snr": 3.0, "survey": "FIRST", "data_source": "first_vizier",
        })
        assert result.is_detected is False
        assert result.anomaly_score == 0.0

    def test_strong_source_detected(self):
        result = compute_radio_emission({
            "peak_flux_mjy": 10.0, "integrated_flux_mjy": 12.0,
            "sep_arcsec": 1.5, "snr": 20.0,
            "survey": "FIRST", "data_source": "first_vizier",
        })
        assert result.is_detected is True
        assert result.anomaly_score > 0.5
        assert result.is_anomalous is True

    def test_too_far_offset_rejected(self):
        result = compute_radio_emission({
            "peak_flux_mjy": 5.0, "sep_arcsec": 15.0,
            "snr": 10.0, "survey": "NVSS", "data_source": "nvss_vizier",
        })
        assert result.is_detected is False

    def test_luminosity_proxy_computed(self):
        result = compute_radio_emission(
            {"peak_flux_mjy": 3.0, "sep_arcsec": 2.0,
             "snr": 8.0, "survey": "FIRST", "data_source": "first_vizier"},
            distance_pc=10.0,
        )
        assert result.radio_luminosity_proxy == pytest.approx(300.0, rel=0.01)

    def test_score_in_valid_range(self):
        result = compute_radio_emission({
            "peak_flux_mjy": 100.0, "sep_arcsec": 0.5,
            "snr": 50.0, "survey": "FIRST", "data_source": "first_vizier",
        })
        assert 0.0 <= result.anomaly_score <= 1.0


# ── HR Diagram Anomaly Tests ─────────────────────────────────────

class TestHRAnomaly:
    def test_no_data_returns_zero(self):
        result = compute_hr_anomaly(None)
        assert result.anomaly_score == 0.0
        assert result.has_data is False

    def test_sun_like_on_ms(self):
        """Sun should be very close to MS ridge, score ~0."""
        result = compute_hr_anomaly(
            {"bp_rp": 0.82, "phot_g_mean_mag": 4.67,
             "teff_gspphot": 5772, "logg_gspphot": 4.44, "mh_gspphot": 0.0},
            {"parallax": 100.0},  # 10 pc → abs_g ≈ 4.67
        )
        assert result.has_data is True
        assert abs(result.ms_sigma) < 1.0  # Within 1σ of MS
        assert result.anomaly_score == 0.0  # Not anomalous

    def test_above_ms_outlier(self):
        """Star much brighter than MS at its colour → anomalous."""
        result = compute_hr_anomaly(
            {"bp_rp": 1.0, "phot_g_mean_mag": 0.0,
             "teff_gspphot": 5000, "logg_gspphot": 3.0, "mh_gspphot": 0.0},
            {"parallax": 100.0},  # 10 pc → abs_g = 0.0 (expected ~5.5)
        )
        assert result.has_data == True
        assert result.is_above_ms == True
        # This is in the giant region, so score is 0 (expected evolutionary state)
        assert result.is_giant_region == True
        assert result.anomaly_score == 0.0

    def test_below_ms_outlier(self):
        """Star much fainter than expected but NOT in WD region (bp_rp > 1.5)."""
        # Use bp_rp=1.8 to avoid WD region (abs_g>10 AND bp_rp<1.5)
        result = compute_hr_anomaly(
            {"bp_rp": 1.8, "phot_g_mean_mag": 15.0,
             "teff_gspphot": 3500, "logg_gspphot": 5.0, "mh_gspphot": 0.0},
            {"parallax": 100.0},  # 10 pc → abs_g = 15.0 (expected ~9.5)
        )
        assert result.has_data == True
        assert result.ms_sigma > 5.0  # Way below MS
        assert result.is_white_dwarf_region == False
        assert result.anomaly_score > 0.5

    def test_metal_poor_subdwarf_penalized(self):
        """Metal-poor star below MS should get reduced score."""
        # Solar metallicity
        result_solar = compute_hr_anomaly(
            {"bp_rp": 0.8, "phot_g_mean_mag": 10.0,
             "teff_gspphot": 5800, "logg_gspphot": 5.0, "mh_gspphot": 0.0},
            {"parallax": 100.0},
        )
        # Metal-poor
        result_poor = compute_hr_anomaly(
            {"bp_rp": 0.8, "phot_g_mean_mag": 10.0,
             "teff_gspphot": 5800, "logg_gspphot": 5.0, "mh_gspphot": -1.0},
            {"parallax": 100.0},
        )
        assert result_poor.anomaly_score < result_solar.anomaly_score

    def test_white_dwarf_region_excluded(self):
        """Stars in WD locus should score 0 (expected evolutionary state)."""
        result = compute_hr_anomaly(
            {"bp_rp": 0.5, "phot_g_mean_mag": 12.0,
             "teff_gspphot": 10000, "logg_gspphot": 8.0, "mh_gspphot": None},
            {"parallax": 100.0},  # abs_g = 12, bp_rp < 1.5 → WD region
        )
        assert result.is_white_dwarf_region is True
        assert result.anomaly_score == 0.0

    def test_missing_parallax_returns_empty(self):
        result = compute_hr_anomaly(
            {"bp_rp": 0.8, "phot_g_mean_mag": 5.0, "teff_gspphot": 5800},
            None,
        )
        assert result.has_data is False


# ── Scorer Integration Tests ─────────────────────────────────────

class TestScorerNewChannels:
    def test_scorer_has_11_channels(self):
        """Scorer should now have 11 channels (7 original + 4 new)."""
        scorer = EXODUSScorer()
        assert len(scorer.CHANNEL_NAMES) == 11
        assert "uv_anomaly" in scorer.CHANNEL_NAMES
        assert "radio_emission" in scorer.CHANNEL_NAMES
        assert "hr_anomaly" in scorer.CHANNEL_NAMES
        assert "abundance_anomaly" in scorer.CHANNEL_NAMES

    def test_new_channels_score_correctly(self):
        """New channels should activate and contribute to scoring."""
        scorer = EXODUSScorer(threshold=0.15)
        target = {
            "target_id": "test_multi",
            "ra": 180.0, "dec": 45.0,
            "uv_anomaly": {"anomaly_score": 0.5, "data_source": "galex_vizier"},
            "radio_emission": {"anomaly_score": 0.4, "is_detected": True, "data_source": "first_vizier"},
            "hr_anomaly": {"anomaly_score": 0.35, "ms_sigma": 2.5, "data_source": "gaia_dr3"},
        }
        result = scorer.score_target(target)
        d = result.to_dict()
        assert d["n_active_channels"] == 3
        assert d["total_score"] > 0

    def test_convergence_priority_amplifies_multi_channel(self):
        """Convergence-priority with 4 channels should massively amplify."""
        scorer_std = EXODUSScorer()
        scorer_cp = EXODUSScorer(convergence_priority=True)

        target = {
            "target_id": "test_conv",
            "ra": 180.0, "dec": 45.0,
            "proper_motion_anomaly": {"ruwe": 1.8, "pm_delta_sigma": 4.0,
                                       "wise_gaia_pm": {"pm_discrepancy_sigma": 2.5}},
            "uv_anomaly": {"anomaly_score": 0.5, "data_source": "galex_vizier"},
            "radio_emission": {"anomaly_score": 0.4, "data_source": "first_vizier"},
            "hr_anomaly": {"anomaly_score": 0.35, "data_source": "gaia_dr3"},
        }
        r_std = scorer_std.score_target(target).to_dict()
        r_cp = scorer_cp.score_target(target).to_dict()

        # Convergence-priority should give higher score for multi-channel
        assert r_cp["total_score"] > r_std["total_score"]

    def test_no_data_channels_dont_penalize(self):
        """Channels without data should score 0, not penalize."""
        scorer = EXODUSScorer()
        target = {
            "target_id": "test_sparse",
            "ra": 180.0, "dec": 45.0,
            # Only PM anomaly provided
            "proper_motion_anomaly": {"ruwe": 3.0, "pm_delta_sigma": 5.0,
                                       "wise_gaia_pm": {"pm_discrepancy_sigma": 3.0}},
        }
        result = scorer.score_target(target)
        d = result.to_dict()
        # Should still get a score from PM alone
        assert d["total_score"] > 0
        # New channels should be inactive (no data)
        assert d["channel_scores"]["uv_anomaly"]["is_active"] is False
        assert d["channel_scores"]["radio_emission"]["is_active"] is False
        assert d["channel_scores"]["hr_anomaly"]["is_active"] is False
        assert d["channel_scores"]["abundance_anomaly"]["is_active"] is False


# ── Abundance Anomaly Tests ────────────────────────────────────

class TestAbundanceAnomaly:
    def test_no_data_returns_zero(self):
        """Without spectroscopic data, abundance anomaly should be zero."""
        from src.detection.abundance_anomaly import AbundanceAnomalyResult
        result = AbundanceAnomalyResult()
        assert result.anomaly_score == 0.0
        assert result.data_source == "none"
        assert result.n_anomalous_ratios == 0

    def test_thresholds_defined(self):
        """All 6 diagnostic abundance thresholds should be defined."""
        from src.detection.abundance_anomaly import ABUNDANCE_THRESHOLDS
        assert len(ABUNDANCE_THRESHOLDS) == 6
        expected = {"Eu_Fe", "Ni_Fe", "P_Fe", "Co_Ni", "Li_A", "Ce_Fe"}
        assert set(ABUNDANCE_THRESHOLDS.keys()) == expected
        for name, info in ABUNDANCE_THRESHOLDS.items():
            assert "natural_range" in info
            assert "flag_above" in info or "flag_below" in info
            assert "weight" in info

    def test_scorer_extracts_abundance_score(self):
        """Scorer should correctly extract abundance anomaly score."""
        scorer = EXODUSScorer()
        target = {
            "target_id": "TEST_ABUND", "ra": 180.0, "dec": 45.0,
            "abundance_anomaly": {
                "anomaly_score": 0.6,
                "data_source": "apogee",
                "n_anomalous_ratios": 2,
            },
        }
        result = scorer.score_target(target)
        ch = result.to_dict()["channel_scores"]["abundance_anomaly"]
        assert ch["is_active"] is True
        assert ch["score"] == 0.6

    def test_scorer_no_data_inactive(self):
        """Abundance channel with no data should be inactive."""
        scorer = EXODUSScorer()
        target = {
            "target_id": "TEST_ABUND_NONE", "ra": 180.0, "dec": 45.0,
            "abundance_anomaly": {
                "anomaly_score": 0.0,
                "data_source": "none",
            },
        }
        result = scorer.score_target(target)
        ch = result.to_dict()["channel_scores"]["abundance_anomaly"]
        assert ch["is_active"] is False

    def test_result_to_dict_complete(self):
        """AbundanceAnomalyResult.to_dict() should include all fields."""
        from src.detection.abundance_anomaly import AbundanceAnomalyResult
        result = AbundanceAnomalyResult(
            has_apogee=True, data_source="apogee",
            n_anomalous_ratios=1, anomalous_ratios=["Eu_Fe"],
            most_anomalous_ratio="Eu_Fe", most_anomalous_sigma=2.5,
            anomaly_score=0.55, is_anomalous=True,
        )
        d = result.to_dict()
        assert d["has_apogee"] is True
        assert d["n_anomalous_ratios"] == 1
        assert d["anomalous_ratios"] == ["Eu_Fe"]
        assert d["anomaly_score"] == 0.55
        assert d["most_anomalous_sigma"] == 2.5


# ── Monotonicity Checker Test ────────────────────────────────────

class TestMonotonicity:
    def test_check_monotonicity_function(self):
        """Verify the monotonicity checker from ir_variability."""
        from src.detection.ir_variability import _check_monotonicity

        # Perfectly monotonic decreasing (brightening in magnitudes)
        t = np.linspace(0, 10, 100)
        mag = np.linspace(12.0, 11.5, 100)  # decreasing = brightening
        is_mono, frac = _check_monotonicity(t, mag, n_bins=5)
        assert is_mono is True
        assert frac == 1.0

        # Oscillatory
        mag_osc = 12.0 + 0.5 * np.sin(2 * np.pi * t / 2.5)
        is_mono_osc, frac_osc = _check_monotonicity(t, mag_osc, n_bins=5)
        assert frac_osc < 0.9  # Not monotonic


# ── NEOWISE Calibration Tests ──────────────────────────────────────

class TestNEOWISECalibration:
    """Tests for NEOWISE calibration fixes from research brief."""

    def test_w2_bad_epoch_constants(self):
        """Verify bad epoch constants are defined correctly."""
        from src.ingestion.neowise_timeseries import (
            W2_BAD_MJD_START, W2_BAD_MJD_END, NEOWISE_STABLE_START_MJD,
        )
        assert W2_BAD_MJD_START == 57000
        assert W2_BAD_MJD_END == 57071
        assert NEOWISE_STABLE_START_MJD == 56700

    def test_filter_bad_epochs_removes_early_data(self):
        """Pre-reactivation data (MJD < 56700) should be removed."""
        from src.ingestion.neowise_timeseries import (
            _filter_bad_epochs, NEOWISEEpoch, NEOWISE_STABLE_START_MJD,
        )
        epochs = [
            NEOWISEEpoch(mjd=56500, w1_mag=10.0, w1_err=0.03, w2_mag=10.0,
                         w2_err=0.05, ra=0.0, dec=0.0),  # too early
            NEOWISEEpoch(mjd=56800, w1_mag=10.0, w1_err=0.03, w2_mag=10.0,
                         w2_err=0.05, ra=0.0, dec=0.0),  # after stable start
        ]
        filtered = _filter_bad_epochs(epochs)
        assert len(filtered) == 1
        assert filtered[0].mjd == 56800

    def test_filter_bad_epochs_nulls_w2_in_bad_range(self):
        """W2 data in MJD 57000-57071 should be set to NaN."""
        from src.ingestion.neowise_timeseries import (
            _filter_bad_epochs, NEOWISEEpoch,
        )
        epochs = [
            NEOWISEEpoch(mjd=57030, w1_mag=10.0, w1_err=0.03, w2_mag=10.0,
                         w2_err=0.05, ra=0.0, dec=0.0),  # in bad W2 range
            NEOWISEEpoch(mjd=57100, w1_mag=10.0, w1_err=0.03, w2_mag=10.0,
                         w2_err=0.05, ra=0.0, dec=0.0),  # after bad range
        ]
        filtered = _filter_bad_epochs(epochs)
        assert len(filtered) == 2
        # First epoch: W1 preserved, W2 nulled
        assert filtered[0].w1_mag == 10.0
        assert np.isnan(filtered[0].w2_mag)
        assert np.isnan(filtered[0].w2_err)
        # Second epoch: both preserved
        assert filtered[1].w2_mag == 10.0

    def test_epoch_averages_groups_by_sky_pass(self):
        """Epoch averages should group observations within 30-day windows."""
        from src.ingestion.neowise_timeseries import (
            compute_epoch_averages, NEOWISETimeSeries,
        )
        # Simulate 3 sky passes: 5 epochs each, 200 days apart
        mjds = []
        for pass_start in [57000, 57200, 57400]:
            mjds.extend([pass_start + i for i in range(5)])

        n = len(mjds)
        ts = NEOWISETimeSeries(
            target_ra=0.0, target_dec=0.0, n_epochs=n,
            mjd=np.array(mjds, dtype=float),
            w1_mag=np.full(n, 10.0),
            w1_err=np.full(n, 0.03),
            w2_mag=np.full(n, 10.0),
            w2_err=np.full(n, 0.05),
        )
        result = compute_epoch_averages(ts)
        assert result["n_epochs"] == 3
        assert len(result["epoch_mjd"]) == 3
        assert all(n >= 5 for n in result["n_per_epoch"])

    def test_trend_sigma_threshold_is_5(self):
        """Verify secular trend threshold was raised to 5σ."""
        from src.detection.ir_variability import TREND_SIGMA_THRESH
        assert TREND_SIGMA_THRESH == 5.0

    def test_w1_primary_band_constants(self):
        """Verify W1 noise floor is better than W2."""
        from src.detection.ir_variability import (
            W1_NOISE_FLOOR_MMAG, W2_NOISE_FLOOR_MMAG,
        )
        assert W1_NOISE_FLOOR_MMAG < W2_NOISE_FLOOR_MMAG
        assert W1_NOISE_FLOOR_MMAG == 2.6
        assert W2_NOISE_FLOOR_MMAG == 6.1

    def test_epoch_averages_reduces_scatter(self):
        """Epoch averaging should reduce scatter by ~sqrt(N) per pass."""
        from src.ingestion.neowise_timeseries import (
            compute_epoch_averages, NEOWISETimeSeries,
        )
        rng = np.random.default_rng(42)
        # 10 sky passes, 12 exposures each, ~200 days apart
        mjds = []
        w1_mags = []
        for pass_idx in range(10):
            pass_start = 57000 + pass_idx * 200
            for _ in range(12):
                mjds.append(pass_start + rng.uniform(0, 2))
                w1_mags.append(10.0 + rng.normal(0, 0.03))

        n = len(mjds)
        ts = NEOWISETimeSeries(
            target_ra=0.0, target_dec=0.0, n_epochs=n,
            mjd=np.array(mjds),
            w1_mag=np.array(w1_mags),
            w1_err=np.full(n, 0.03),
            w2_mag=np.full(n, 10.0),
            w2_err=np.full(n, 0.05),
        )
        result = compute_epoch_averages(ts)
        assert result["n_epochs"] == 10
        # Epoch-averaged errors should be ~sqrt(12) smaller than per-exposure
        mean_epoch_err = np.mean(result["epoch_w1_err"])
        assert mean_epoch_err < 0.03 / 2.0  # at least 2× improvement


# ── Spatial Clustering Tests ─────────────────────────────────────────


class TestSpatialClustering:
    """Tests for the spatial clustering analysis module."""

    def test_scored_target_cartesian(self):
        """3D Cartesian coords from RA/Dec/distance."""
        from src.detection.spatial_clustering import ScoredTarget
        t = ScoredTarget(
            target_id="TEST1", ra=0.0, dec=0.0,
            distance_pc=100.0, total_score=1.0, n_active_channels=2,
        )
        t.compute_cartesian()
        assert abs(t.x - 100.0) < 0.01
        assert abs(t.y) < 0.01
        assert abs(t.z) < 0.01

    def test_cartesian_nan_distance(self):
        """Invalid distance yields NaN coordinates."""
        from src.detection.spatial_clustering import ScoredTarget
        t = ScoredTarget(
            target_id="TEST2", ra=45.0, dec=30.0,
            distance_pc=float("nan"), total_score=0.5, n_active_channels=1,
        )
        t.compute_cartesian()
        assert np.isnan(t.x)
        assert np.isnan(t.y)
        assert np.isnan(t.z)

    def test_kulldorff_no_cluster_uniform(self):
        """Uniform scores should produce no significant cluster."""
        from src.detection.spatial_clustering import ScoredTarget, kulldorff_scan
        rng = np.random.default_rng(123)
        targets = []
        for i in range(30):
            t = ScoredTarget(
                target_id=f"UNIF_{i}",
                ra=rng.uniform(0, 360),
                dec=rng.uniform(-60, 60),
                distance_pc=rng.uniform(50, 200),
                total_score=1.0,  # all identical
                n_active_channels=2,
            )
            t.compute_cartesian()
            targets.append(t)
        clusters = kulldorff_scan(targets, n_permutations=19)
        # With uniform scores, LR should be 0 everywhere
        assert len(clusters) == 0

    def test_kulldorff_detects_injected_cluster(self):
        """Kulldorff scan should detect a planted high-score cluster."""
        from src.detection.spatial_clustering import ScoredTarget, kulldorff_scan
        rng = np.random.default_rng(42)
        targets = []
        # 40 background targets with low scores
        for i in range(40):
            t = ScoredTarget(
                target_id=f"BG_{i}",
                ra=rng.uniform(0, 360),
                dec=rng.uniform(-60, 60),
                distance_pc=rng.uniform(50, 500),
                total_score=rng.uniform(0.0, 0.3),
                n_active_channels=0,
            )
            t.compute_cartesian()
            targets.append(t)
        # 10 clustered high-score targets within 20 pc of (100, 0, 0)
        for i in range(10):
            t = ScoredTarget(
                target_id=f"CLUSTER_{i}",
                ra=rng.uniform(-2, 2),  # near RA=0
                dec=rng.uniform(-2, 2),
                distance_pc=100.0 + rng.uniform(-10, 10),
                total_score=rng.uniform(1.5, 2.5),
                n_active_channels=3,
            )
            t.compute_cartesian()
            targets.append(t)
        clusters = kulldorff_scan(targets, n_permutations=49, min_cluster_size=3)
        assert len(clusters) >= 1
        # The best cluster should contain some of our injected targets
        c = clusters[0]
        cluster_members = set(c.member_ids)
        injected = {f"CLUSTER_{i}" for i in range(10)}
        overlap = cluster_members & injected
        assert len(overlap) >= 3, f"Expected cluster overlap, got {overlap}"
        assert c.mean_score_inside > c.mean_score_outside

    def test_ripleys_k_returns_result(self):
        """Ripley's K should return a result for sufficient targets."""
        from src.detection.spatial_clustering import ScoredTarget, ripleys_k
        rng = np.random.default_rng(99)
        targets = []
        for i in range(50):
            t = ScoredTarget(
                target_id=f"RK_{i}",
                ra=rng.uniform(0, 360),
                dec=rng.uniform(-60, 60),
                distance_pc=rng.uniform(10, 300),
                total_score=rng.uniform(0.0, 2.0),
                n_active_channels=rng.integers(0, 4),
            )
            t.compute_cartesian()
            targets.append(t)
        result = ripleys_k(targets, score_threshold=0.3)
        assert result is not None
        assert result.n_anomalous >= 5
        assert len(result.distances_pc) == 20
        assert len(result.k_observed) == 20
        assert len(result.l_function) == 20

    def test_ripleys_k_too_few_targets(self):
        """Ripley's K should return None when <5 anomalous targets."""
        from src.detection.spatial_clustering import ScoredTarget, ripleys_k
        targets = []
        for i in range(10):
            t = ScoredTarget(
                target_id=f"LOW_{i}",
                ra=float(i * 30), dec=0.0,
                distance_pc=100.0, total_score=0.1,  # all below threshold
                n_active_channels=0,
            )
            t.compute_cartesian()
            targets.append(t)
        result = ripleys_k(targets, score_threshold=0.5)
        assert result is None

    def test_full_analysis_result_structure(self):
        """run_spatial_analysis returns well-formed SpatialClusterResult."""
        from src.detection.spatial_clustering import ScoredTarget, run_spatial_analysis
        rng = np.random.default_rng(7)
        targets = []
        for i in range(30):
            t = ScoredTarget(
                target_id=f"FULL_{i}",
                ra=rng.uniform(0, 360),
                dec=rng.uniform(-60, 60),
                distance_pc=rng.uniform(10, 200),
                total_score=rng.uniform(0.0, 1.5),
                n_active_channels=rng.integers(0, 3),
            )
            t.compute_cartesian()
            targets.append(t)
        result = run_spatial_analysis(targets, n_permutations=9)
        assert result.n_targets == 30
        assert result.n_with_3d_coords == 30
        d = result.to_dict()
        assert "clusters" in d
        assert "n_clusters_found" in d
        assert "ripleys_k" in d

    def test_result_to_dict_serializable(self):
        """SpatialClusterResult.to_dict() is JSON-serializable."""
        import json as _json
        from src.detection.spatial_clustering import ScoredTarget, run_spatial_analysis
        rng = np.random.default_rng(11)
        targets = []
        for i in range(20):
            t = ScoredTarget(
                target_id=f"SER_{i}",
                ra=rng.uniform(0, 360),
                dec=rng.uniform(-60, 60),
                distance_pc=rng.uniform(10, 100),
                total_score=rng.uniform(0.0, 1.0),
                n_active_channels=rng.integers(0, 2),
            )
            t.compute_cartesian()
            targets.append(t)
        result = run_spatial_analysis(targets, n_permutations=9)
        serialized = _json.dumps(result.to_dict())
        assert len(serialized) > 10


# ── Galaxy Contamination Tests ───────────────────────────────────────


class TestGalaxyContamination:
    """Tests for the background galaxy contamination check."""

    def test_no_ir_data_returns_empty(self):
        from src.vetting.galaxy_contamination import check_galaxy_contamination
        result = check_galaxy_contamination({})
        assert result.has_data is False
        assert result.contamination_score == 0.0

    def test_normal_star_clean(self):
        """A normal main-sequence star should have low contamination score."""
        from src.vetting.galaxy_contamination import check_galaxy_contamination
        result = check_galaxy_contamination({
            "ir_photometry": {
                "w1mpro": 8.5,
                "w2mpro": 8.4,  # W1-W2 = 0.1, normal
                "w3mpro": 8.3,  # W2-W3 = 0.1, normal
                "w4mpro": 8.0,  # W3-W4 = 0.3, normal
            },
            "ra": 180.0,
            "dec": 60.0,  # high galactic lat
        })
        assert result.has_data is True
        assert result.contamination_likely is False
        assert result.contamination_score < 0.3

    def test_agn_colours_flagged(self):
        """AGN-like W1-W2 colour should trigger contamination flag."""
        from src.vetting.galaxy_contamination import check_galaxy_contamination
        result = check_galaxy_contamination({
            "ir_photometry": {
                "w1mpro": 10.0,
                "w2mpro": 9.0,  # W1-W2 = 1.0, AGN
                "w3mpro": 6.5,  # W2-W3 = 2.5, galaxy
            },
            "ra": 180.0,
            "dec": 60.0,
        })
        assert result.agn_colour is True
        assert result.galaxy_w2w3_colour is True
        assert result.contamination_likely is True
        assert result.contamination_score >= 0.8

    def test_w4_only_excess_flagged(self):
        """W4-only excess pattern should increase contamination score."""
        from src.vetting.galaxy_contamination import check_galaxy_contamination
        result = check_galaxy_contamination({
            "ir_photometry": {"w1mpro": 8.0, "w2mpro": 7.9},
            "ir_excess": {"sigma_W3": 1.0, "sigma_W4": 8.0},
            "ra": 180.0,
            "dec": 60.0,
        })
        assert result.contamination_score >= 0.5
        assert any("W4-only" in f for f in result.flags)

    def test_extended_source_flagged(self):
        """Extended source flag should boost contamination score."""
        from src.vetting.galaxy_contamination import check_galaxy_contamination
        result = check_galaxy_contamination({
            "ir_photometry": {
                "w1mpro": 10.0,
                "w2mpro": 10.0,
                "ext_flg": 5,
            },
            "ra": 180.0,
            "dec": 60.0,
        })
        assert result.extended_source is True
        assert result.contamination_score >= 0.5

    def test_low_galactic_latitude_flagged(self):
        """Low |b| should contribute to contamination risk."""
        from src.vetting.galaxy_contamination import check_galaxy_contamination
        result = check_galaxy_contamination({
            "ir_photometry": {"w1mpro": 10.0, "w2mpro": 10.0},
            "ra": 266.0,   # near galactic center
            "dec": -29.0,
        })
        assert result.low_galactic_latitude is True
        assert "galactic latitude" in result.flags[0].lower() or any(
            "galactic" in f.lower() for f in result.flags
        )

    def test_to_dict_serializable(self):
        """Result.to_dict() should be JSON-serializable."""
        import json as _json
        from src.vetting.galaxy_contamination import check_galaxy_contamination
        result = check_galaxy_contamination({
            "ir_photometry": {"w1mpro": 10.0, "w2mpro": 9.0, "w3mpro": 6.0},
            "ra": 180.0, "dec": 60.0,
        })
        serialized = _json.dumps(result.to_dict())
        assert len(serialized) > 10


# ── PM-IR Correlation Tests ──────────────────────────────────────────


class TestPMIRCorrelation:
    """Tests for PM-IR correlation check."""

    def test_no_channels_no_correlation(self):
        from src.vetting.galaxy_contamination import check_pm_ir_correlation
        result = check_pm_ir_correlation({})
        assert result["pm_ir_correlated"] is False
        assert result["effective_pm_weight"] == 1.0

    def test_only_ir_no_correlation(self):
        from src.vetting.galaxy_contamination import check_pm_ir_correlation
        result = check_pm_ir_correlation({
            "channel_scores": {
                "ir_excess": {"score": 0.9, "details": {"sigma_W3": 10.0, "sigma_W4": 15.0}},
                "proper_motion_anomaly": {"score": 0.05, "details": {}},
            }
        })
        assert result["pm_ir_correlated"] is False

    def test_ruwe_pm_independent(self):
        """PM from RUWE > 1.4 is independent of IR excess."""
        from src.vetting.galaxy_contamination import check_pm_ir_correlation
        result = check_pm_ir_correlation({
            "channel_scores": {
                "ir_excess": {"score": 0.9, "details": {"sigma_W3": 20.0, "sigma_W4": 30.0}},
                "proper_motion_anomaly": {
                    "score": 0.8,
                    "details": {
                        "ruwe": 1.8,
                        "astrometric_excess_noise_sig": 10.0,
                        "wise_gaia_pm": {"pm_discrepancy_sigma": 5.0, "is_discrepant": True},
                    },
                },
            }
        })
        assert result["pm_ir_correlated"] is False
        assert "independent" in result["explanation"].lower()

    def test_wise_pm_ir_correlated(self):
        """WISE-Gaia PM discrepancy + IR excess = correlated."""
        from src.vetting.galaxy_contamination import check_pm_ir_correlation
        result = check_pm_ir_correlation({
            "channel_scores": {
                "ir_excess": {"score": 0.99, "details": {"sigma_W3": 40.0, "sigma_W4": 16.0}},
                "proper_motion_anomaly": {
                    "score": 0.95,
                    "details": {
                        "ruwe": 0.97,
                        "astrometric_excess_noise_sig": 0.0,
                        "wise_gaia_pm": {"pm_discrepancy_sigma": 12.5, "is_discrepant": True},
                    },
                },
            }
        })
        assert result["pm_ir_correlated"] is True
        assert result["effective_pm_weight"] < 1.0
        assert result["correlation_score"] > 0.3

    def test_w4_dominant_extra_penalty(self):
        """W4-dominant excess gets extra correlation penalty."""
        from src.vetting.galaxy_contamination import check_pm_ir_correlation
        # W4-dominant case
        result_w4 = check_pm_ir_correlation({
            "channel_scores": {
                "ir_excess": {"score": 0.99, "details": {"sigma_W3": 5.0, "sigma_W4": 60.0}},
                "proper_motion_anomaly": {
                    "score": 0.6,
                    "details": {
                        "ruwe": 1.0,
                        "astrometric_excess_noise_sig": 0.0,
                        "wise_gaia_pm": {"pm_discrepancy_sigma": 3.0, "is_discrepant": True},
                    },
                },
            }
        })
        # Balanced case
        result_bal = check_pm_ir_correlation({
            "channel_scores": {
                "ir_excess": {"score": 0.99, "details": {"sigma_W3": 30.0, "sigma_W4": 30.0}},
                "proper_motion_anomaly": {
                    "score": 0.6,
                    "details": {
                        "ruwe": 1.0,
                        "astrometric_excess_noise_sig": 0.0,
                        "wise_gaia_pm": {"pm_discrepancy_sigma": 3.0, "is_discrepant": True},
                    },
                },
            }
        })
        assert result_w4["correlation_score"] > result_bal["correlation_score"]


# ── Herschel Far-IR Tests ────────────────────────────────────────────


class TestHerschelSED:
    """Tests for Herschel SED interpretation (no network queries)."""

    def test_no_data_unclassified(self):
        from src.ingestion.herschel_catalog import interpret_herschel_sed
        result = interpret_herschel_sed({})
        assert result["classification"] == "unclassified"

    def test_warm_source_dyson_consistent(self):
        """Bright at W4, faint at 70µm → warm emission."""
        from src.ingestion.herschel_catalog import interpret_herschel_sed
        result = interpret_herschel_sed(
            {"pacs_70": 50.0, "pacs_100": 30.0},
            wise_w4_flux_mjy=200.0,
        )
        assert result["classification"] == "dyson_consistent"
        assert result["confidence"] > 0

    def test_rising_sed_galaxy_like(self):
        """Rising SED from 70 to 250µm → background galaxy."""
        from src.ingestion.herschel_catalog import interpret_herschel_sed
        result = interpret_herschel_sed(
            {"pacs_70": 100.0, "spire_250": 500.0},
        )
        assert result["classification"] == "galaxy_like"

    def test_moderate_sed_dust_disk(self):
        """Moderate ratio W4/70µm → debris disk."""
        from src.ingestion.herschel_catalog import interpret_herschel_sed
        result = interpret_herschel_sed(
            {"pacs_70": 100.0, "pacs_100": 80.0},
            wise_w4_flux_mjy=150.0,
        )
        assert result["classification"] == "dust_disk"

    def test_beam_sizes_defined(self):
        """Herschel beam size constants should be defined."""
        from src.ingestion.herschel_catalog import (
            PACS_70_BEAM, PACS_100_BEAM, SPIRE_250_BEAM,
        )
        assert PACS_70_BEAM < SPIRE_250_BEAM
        assert PACS_70_BEAM == 5.6
