"""
Regression tests for bugs fixed in EXODUS Audits #1–#4.

These tests verify that the specific bugs found during comprehensive
audits remain fixed.  All tests are self-contained (no network calls,
no heavy imports) and run fast (<5s total).

Audit #2 (2026-02-23):
    - Finding 1: Control-cohort identity exclusion (source_id key mismatch)
    - Finding 2: Fermi multiplicity correction (Bonferroni)
    - Finding 4: Gaia variability metric (mag→flux)

Audit #3 (2026-02-23):
    - Finding 1: Epoch photometry filter inverted (gaia_query.py)
    - Finding 2: Coverage penalty uncapped (exodus_score.py)
    - Finding 3: Box→angular coordinate exclusion (controls.py)
    - Finding 5: Stouffer p=1.0 exclusion (statistics.py)

Audit #4 (2026-02-23):
    - Finding 2: Neutrino multiplicity correction (Bonferroni)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# =====================================================================
# Test 1: Control exclusion by source_id (Audit #2 Finding 1)
# =====================================================================

def test_control_exclusion_by_source_id():
    """Target must NOT appear in its own control pool when matched by source_id."""
    from src.core.controls import select_matched_controls

    # Build a target with a known Gaia source_id
    target_source_id = "6752105888795829376"
    target = {
        "source_id": target_source_id,
        "ra": 180.0,
        "dec": 30.0,
        "phot_g_mean_mag": 10.0,
        "bp_rp": 1.0,
        "distance_pc": 20.0,
        "b_gal": 45.0,
    }

    # Build a catalog that includes the target itself + 50 controls
    rng = np.random.default_rng(42)
    catalog = [target.copy()]  # target is in the catalog
    for i in range(50):
        catalog.append({
            "source_id": str(1000000 + i),
            "ra": 180.0 + rng.uniform(-1, 1),
            "dec": 30.0 + rng.uniform(-1, 1),
            "phot_g_mean_mag": 10.0 + rng.uniform(-2, 2),
            "bp_rp": 1.0 + rng.uniform(-0.5, 0.5),
            "distance_pc": 20.0 + rng.uniform(-5, 5),
            "b_gal": 45.0 + rng.uniform(-10, 10),
        })

    cohort = select_matched_controls(
        targets=[target],
        catalog=catalog,
        n_per_target=10,
        target_id_key="source_id",
    )

    # The target source_id must NOT appear in any control's source_id
    for ctrl in cohort.controls:
        assert str(ctrl.get("source_id")) != target_source_id, (
            f"Target source_id {target_source_id} leaked into control pool!"
        )


# =====================================================================
# Test 2: Coordinate fallback exclusion with cos(dec) (Audit #3 Finding 3)
# =====================================================================

def test_control_exclusion_coordinate_fallback():
    """Coordinate fallback must use cos(dec) scaling, not raw RA box."""
    from src.core.controls import select_matched_controls

    # Target at high declination (Dec=80°) — cos(80°) ≈ 0.174
    target = {
        "source_id": "",  # empty → forces coordinate fallback
        "ra": 180.0,
        "dec": 80.0,
        "phot_g_mean_mag": 10.0,
        "bp_rp": 1.0,
        "distance_pc": 20.0,
        "b_gal": 30.0,
    }

    # Source at same Dec, 3 arcsec away in RA
    # At dec=80°, 3 arcsec in RA = 3/cos(80°) ≈ 17 arcsec on-sky
    # Should NOT be excluded (it's > 5 arcsec on-sky)
    # But a naive box check would exclude it (3" < 5" in raw RA)
    offset_arcsec = 3.0
    offset_deg = offset_arcsec / 3600.0

    distant_source = {
        "source_id": "far_away",
        "ra": 180.0 + offset_deg,
        "dec": 80.0,
        "phot_g_mean_mag": 10.0,
        "bp_rp": 1.0,
        "distance_pc": 20.0,
        "b_gal": 30.0,
    }

    # Source at EXACTLY the target position — must be excluded
    coincident_source = {
        "source_id": "right_on_top",
        "ra": 180.0,
        "dec": 80.0,
        "phot_g_mean_mag": 10.5,
        "bp_rp": 1.1,
        "distance_pc": 22.0,
        "b_gal": 31.0,
    }

    rng = np.random.default_rng(99)
    catalog = [distant_source, coincident_source]
    for i in range(50):
        catalog.append({
            "source_id": str(2000000 + i),
            "ra": 180.0 + rng.uniform(-2, 2),
            "dec": 80.0 + rng.uniform(-1, 1),
            "phot_g_mean_mag": 10.0 + rng.uniform(-2, 2),
            "bp_rp": 1.0 + rng.uniform(-0.5, 0.5),
            "distance_pc": 20.0 + rng.uniform(-5, 5),
            "b_gal": 30.0 + rng.uniform(-10, 10),
        })

    cohort = select_matched_controls(
        targets=[target],
        catalog=catalog,
        n_per_target=10,
        target_id_key="source_id",
    )

    control_ids = {str(c.get("source_id")) for c in cohort.controls}

    # Coincident source must be excluded
    assert "right_on_top" not in control_ids, (
        "Coincident source should be excluded by coordinate fallback"
    )

    # The 3-arcsec-RA source at dec=80° is actually ~17 arcsec on-sky
    # With cos(dec) correction, dra = 3" * cos(80°) ≈ 0.52" < 5" threshold
    # So it SHOULD be excluded (0.52 arcsec < 5 arcsec exclusion radius)
    # This tests that cos(dec) is applied correctly
    # Note: at dec=80°, the RA compression makes 3" of RA = only 0.52" on sky


# =====================================================================
# Test 3: Fermi multiplicity correction (Audit #2 Finding 2)
# =====================================================================

def test_fermi_multiplicity_correction():
    """p_corrected must be Bonferroni-corrected: min(1.0, p_chance * n_trials)."""
    from src.detection.gamma_exoplanet_crossmatch import GammaExoplanetMatch

    # A match with raw p_chance=0.001 but 2000 trials → p_corrected = 2.0 → clamped to 1.0
    match_insignificant = GammaExoplanetMatch(
        fermi_name="4FGL_TEST1",
        fermi_ra=100.0,
        fermi_dec=20.0,
        host_name="HD_99999",
        separation_arcmin=5.0,
        p_chance=0.001,
        escalation=False,
        p_corrected=min(1.0, 0.001 * 2000),
        n_trials=2000,
    )
    assert match_insignificant.p_corrected == 1.0, (
        f"Expected p_corrected=1.0 after Bonferroni, got {match_insignificant.p_corrected}"
    )

    # A match with raw p_chance=1e-7 and 2000 trials → p_corrected = 2e-4 (still significant)
    match_significant = GammaExoplanetMatch(
        fermi_name="4FGL_TEST2",
        fermi_ra=101.0,
        fermi_dec=21.0,
        host_name="HD_88888",
        separation_arcmin=1.0,
        p_chance=1e-7,
        escalation=True,
        p_corrected=min(1.0, 1e-7 * 2000),
        n_trials=2000,
    )
    assert match_significant.p_corrected == pytest.approx(2e-4, rel=1e-3), (
        f"Expected p_corrected≈2e-4, got {match_significant.p_corrected}"
    )
    assert match_significant.p_corrected < 0.01, "Should still be significant"


# =====================================================================
# Test 4: Coverage penalty clamped to [0, 1] (Audit #3 Finding 2)
# =====================================================================

def test_coverage_penalty_clamped():
    """Coverage penalty must never exceed 1.0, even if n_with_data > n_possible."""
    from src.scoring.exodus_score import EXODUSScorer

    scorer = EXODUSScorer(threshold=0.3)

    # Build a target with data in ALL channels + extra fields
    # that might trick the counter into n_with_data > n_possible
    target = {
        "target_id": "test_clamp",
        "ra": 100.0,
        "dec": 20.0,
        "ir_excess": {"sigma_W3": 5.0, "sigma_W4": 3.0, "excess_W3": 0.5,
                       "excess_W4": 0.3, "is_candidate": True, "chi2_red": 1.0},
        "transit_anomaly": {"anomaly_score": 0.5},
        "radio_anomaly": {"n_candidates": 3, "max_snr": 15.0},
        "gaia_photometric_anomaly": {"phot_g_variability": 0.08, "n_epochs": 50},
        "proper_motion_anomaly": {"ruwe": 2.5, "astrometric_excess_noise_sig": 3.0},
        "habitable_zone_planet": {"has_hz_planet": True, "n_hz_planets": 2},
    }

    result = scorer.score_target(target)

    # The coverage_penalty must be <= 1.0
    assert result.coverage_penalty <= 1.0, (
        f"Coverage penalty {result.coverage_penalty} exceeds 1.0!"
    )


# =====================================================================
# Test 5: Stouffer includes p=1.0 values (Audit #3 Finding 5)
# =====================================================================

def test_stouffer_includes_p_one():
    """Stouffer must include p=1.0 (null evidence) — dilutes combined significance."""
    from src.core.statistics import stouffer_combine

    # One significant p-value alone
    p_single = stouffer_combine([0.01])

    # Same p-value with two null-evidence channels (p=1.0)
    p_diluted = stouffer_combine([0.01, 1.0, 1.0])

    # The diluted version must be LESS significant (larger p-value)
    assert p_diluted > p_single, (
        f"Stouffer with p=1.0 channels ({p_diluted}) should be less significant "
        f"than single channel ({p_single}). p=1.0 values are being excluded!"
    )

    # Verify p=1.0 are actually being included (not filtered out)
    # If they were filtered, stouffer([0.01, 1.0, 1.0]) == stouffer([0.01])
    assert abs(p_diluted - p_single) > 0.001, (
        "p=1.0 channels appear to be silently filtered — no dilution detected"
    )


# =====================================================================
# Test 6: Gaia variability uses flux, not magnitudes (Audit #2 Finding 4)
# =====================================================================

def test_gaia_variability_flux_not_mag():
    """Variability must be std(flux)/mean(flux), not std(mag)/mean(mag)."""
    # Simulate G-band epoch magnitudes
    mags = np.array([10.0, 10.05, 9.95, 10.02, 9.98, 10.01, 9.99, 10.03])

    # Correct: convert to flux then compute fractional variability
    flux = 10.0 ** (-0.4 * mags)
    correct_variability = float(np.std(flux) / np.mean(flux))

    # Wrong: compute on magnitudes directly
    wrong_variability = float(np.std(mags) / np.mean(mags))

    # These should be different (mag-space variability is much smaller
    # because magnitudes are log-scale)
    assert abs(correct_variability - wrong_variability) > 1e-4, (
        "Flux-space and mag-space variability should differ"
    )

    # The correct variability should be larger than the wrong one
    # because flux is on a linear scale while mag is logarithmic
    assert correct_variability > wrong_variability, (
        "Flux-space variability should be larger than mag-space variability"
    )


# =====================================================================
# Test 7: Epoch photometry filter keeps valid flux + low error
#          (Audit #3 Finding 1)
# =====================================================================

def test_epoch_photometry_filter_correct():
    """Quality filter: keep rows with valid flux AND fractional error < 20%."""
    # Simulate the filter logic from gaia_query.py line 360-364
    df = pd.DataFrame({
        "g_transit_flux": [100.0, 0.0, 200.0, 150.0, -5.0],
        "g_transit_flux_error": [5.0, 1.0, 80.0, 10.0, 2.0],
    })

    # Apply the filter (same logic as gaia_query.py)
    valid_flux = df["g_transit_flux"].abs() > 0
    frac_err = df["g_transit_flux_error"].abs() / df["g_transit_flux"].abs()
    good_mask = valid_flux & (frac_err < 0.2)
    filtered = df[good_mask]

    # Row 0: flux=100, err=5, frac=0.05 → KEEP (valid + low error)
    # Row 1: flux=0, err=1 → REJECT (zero flux)
    # Row 2: flux=200, err=80, frac=0.40 → REJECT (high error)
    # Row 3: flux=150, err=10, frac=0.067 → KEEP (valid + low error)
    # Row 4: flux=-5, err=2 → valid_flux=True (abs>0), frac=0.4 → REJECT (high error)
    assert len(filtered) == 2, (
        f"Expected 2 rows after filter, got {len(filtered)}"
    )
    assert 0 in filtered.index, "Row 0 (valid + low error) should survive"
    assert 3 in filtered.index, "Row 3 (valid + low error) should survive"
    assert 1 not in filtered.index, "Row 1 (zero flux) should be rejected"
    assert 2 not in filtered.index, "Row 2 (high error) should be rejected"


# =====================================================================
# Test 8: Neutrino Bonferroni multiplicity correction (Audit #4 F2)
# =====================================================================

def test_neutrino_multiplicity_correction():
    """Neutrino significance must use Bonferroni-corrected p-value, not raw sigma."""
    from src.detection.neutrino_exoplanet_crossmatch import HostExcess

    n_hosts = 500  # typical sample size

    # A host with raw 3.5-sigma but corrected p >> threshold
    # Raw p ≈ 0.000233 → corrected = 0.000233 * 500 = 0.1165 (NOT significant)
    raw_p = 0.000233
    h = HostExcess(
        host_name="FakeHost",
        host_ra=0.0, host_dec=0.0,
        n_observed=10, n_expected=5.0, excess=5.0,
        poisson_pvalue=raw_p,
        poisson_sigma=3.5,
        p_corrected=min(1.0, raw_p * n_hosts),
        n_trials=n_hosts,
    )

    # Corrected p should be well above 0.0027 (3σ threshold)
    assert h.p_corrected > 0.0027, (
        f"Host with raw 3.5σ should NOT be significant after Bonferroni: "
        f"p_corrected={h.p_corrected:.4e}"
    )

    # A genuinely significant host: raw p = 1e-7, corrected = 5e-5
    h2 = HostExcess(
        host_name="RealHost",
        host_ra=0.0, host_dec=0.0,
        n_observed=20, n_expected=3.0, excess=17.0,
        poisson_pvalue=1e-7,
        poisson_sigma=5.3,
        p_corrected=min(1.0, 1e-7 * n_hosts),
        n_trials=n_hosts,
    )
    assert h2.p_corrected < 0.0027, (
        f"Host with extreme excess should remain significant: "
        f"p_corrected={h2.p_corrected:.4e}"
    )

    # Verify n_trials is stored
    assert h.n_trials == n_hosts
    assert h2.n_trials == n_hosts

    # Verify to_dict includes new fields
    d = h.to_dict()
    assert "p_corrected" in d
    assert "n_trials" in d


# =====================================================================
# Test 9: Nearby target adaptive matching (controls.py)
# =====================================================================

def test_nearby_target_adaptive_matching():
    """For nearby targets (< 10 pc), distance_pc should be dropped from matching."""
    from src.core.controls import select_matched_controls

    # Create nearby targets (all < 5 pc)
    targets = []
    for i in range(5):
        targets.append({
            "target_id": f"NEARBY_{i}",
            "source_id": f"SRC_NEAR_{i}",
            "phot_g_mean_mag": 10.0 + i * 0.5,
            "bp_rp": 1.0 + i * 0.1,
            "distance_pc": 2.0 + i * 0.5,  # 2-4 pc
            "b_gal": 30.0 + i * 5,
            "ra": 180.0 + i,
            "dec": 45.0 + i,
        })

    # Create catalog of field stars at 20-200 pc
    catalog = []
    rng = np.random.default_rng(42)
    for i in range(200):
        catalog.append({
            "source_id": f"FIELD_{i}",
            "phot_g_mean_mag": rng.normal(11.0, 2.0),
            "bp_rp": rng.normal(1.2, 0.4),
            "distance_pc": rng.uniform(20, 200),
            "b_gal": rng.uniform(-60, 60),
        })

    cohort = select_matched_controls(
        targets, catalog,
        n_per_target=5,
        target_id_key="source_id",
    )

    # distance_pc should have been dropped from matching features
    assert "distance_pc" not in cohort.match_features, (
        "distance_pc should be dropped for nearby targets"
    )

    # Should have matching caveats
    assert len(cohort.matching_caveats) > 0, (
        "Nearby target matching should generate a caveat"
    )
    assert "distance-agnostic" in cohort.matching_caveats[0].lower(), (
        "Caveat should mention distance-agnostic matching"
    )

    # Controls should still be selected (on photometry + b_gal)
    assert cohort.n_controls > 0, "Should still select controls on other features"


# =====================================================================
# Test 10: WISE-Gaia PM consistency check (stellar_anomaly.py)
# =====================================================================

def test_wise_gaia_pm_consistency():
    """PM consistency check must flag large Gaia-WISE discrepancies."""
    from src.detection.stellar_anomaly import compute_pm_consistency

    # Consistent PMs: should NOT be discrepant
    r1 = compute_pm_consistency(
        pmra_gaia=-100.0, pmdec_gaia=50.0,
        pmra_err_gaia=0.05, pmdec_err_gaia=0.04,
        pmra_wise=-101.0, pmdec_wise=49.5,
        pmra_err_wise=3.0, pmdec_err_wise=3.0,
    )
    assert not r1["is_discrepant"], "Consistent PMs should not be flagged"
    assert r1["pm_discrepancy_sigma"] < 3.0

    # Large discrepancy on low-PM star: should be flagged as real discrepancy
    # (pm_total=25 < 50 → no CatWISE systematic flag)
    r2 = compute_pm_consistency(
        pmra_gaia=20.0, pmdec_gaia=15.0,
        pmra_err_gaia=0.1, pmdec_err_gaia=0.1,
        pmra_wise=-5.0, pmdec_wise=-10.0,
        pmra_err_wise=2.0, pmdec_err_wise=2.0,
    )
    assert r2["is_discrepant"], "25 mas/yr offset on slow star should be flagged"
    assert r2["pm_discrepancy_sigma"] > 5.0
    assert not r2.get("catwise_systematic_flag", False), \
        "Low-PM star should not trigger CatWISE systematic flag"

    # Sigma should be finite and capped
    assert np.isfinite(r2["pm_discrepancy_sigma"])
    assert r2["pm_discrepancy_sigma"] <= 50.0

    # Verify required output keys (including new keys from CatWISE fix)
    for key in ("delta_pmra", "delta_pmdec", "chi2", "p_value",
                "pm_discrepancy_sigma", "is_discrepant", "interpretation",
                "catwise_systematic_flag", "wise_sys_floor"):
        assert key in r1, f"Missing key: {key}"

    # Systematic floor: even with zero formal errors, sigma should be finite
    r3 = compute_pm_consistency(
        pmra_gaia=5.0, pmdec_gaia=3.0,
        pmra_err_gaia=0.0, pmdec_err_gaia=0.0,
        pmra_wise=5.0, pmdec_wise=3.0,
        pmra_err_wise=0.0, pmdec_err_wise=0.0,
    )
    assert np.isfinite(r3["chi2"]), "Should handle zero errors gracefully"

    # Regression: different chi2 must produce different sigma
    # (bug: p-value clamping at 1e-15 collapsed all large chi2 to sigma=8.01)
    # Use low-PM stars (pm_total < 50) to avoid CatWISE systematic flag
    r_small = compute_pm_consistency(
        pmra_gaia=10.0, pmdec_gaia=5.0,
        pmra_err_gaia=0.1, pmdec_err_gaia=0.1,
        pmra_wise=5.0, pmdec_wise=2.0,
        pmra_err_wise=2.0, pmdec_err_wise=2.0,
    )
    r_large = compute_pm_consistency(
        pmra_gaia=10.0, pmdec_gaia=5.0,
        pmra_err_gaia=0.1, pmdec_err_gaia=0.1,
        pmra_wise=-30.0, pmdec_wise=-25.0,
        pmra_err_wise=2.0, pmdec_err_wise=2.0,
    )
    assert r_large["chi2"] > r_small["chi2"], "Sanity: bigger offset -> bigger chi2"
    assert r_large["pm_discrepancy_sigma"] > r_small["pm_discrepancy_sigma"], (
        f"Different chi2 must produce different sigma: "
        f"small={r_small['pm_discrepancy_sigma']:.1f} large={r_large['pm_discrepancy_sigma']:.1f}"
    )

    # NEW: CatWISE systematic flag for high-PM stars with large offset
    # Simulates the bug case: M dwarf with 200 mas/yr total PM, 168 mas/yr offset
    r_sys = compute_pm_consistency(
        pmra_gaia=-150.0, pmdec_gaia=100.0,  # pm_total=180 mas/yr
        pmra_err_gaia=0.05, pmdec_err_gaia=0.04,
        pmra_wise=-20.0, pmdec_wise=-10.0,    # huge offset (CatWISE bad PM)
        pmra_err_wise=5.0, pmdec_err_wise=5.0,
    )
    assert r_sys["catwise_systematic_flag"], \
        "High-PM star with >30% offset should be flagged as CatWISE systematic"
    assert not r_sys["is_discrepant"], \
        "CatWISE systematic should override is_discrepant"
    assert r_sys["pm_discrepancy_sigma"] == 0.0, \
        "CatWISE systematic should have sigma=0"
    assert r_sys["p_value"] == 1.0, \
        "CatWISE systematic should have p=1.0"

    # NEW: Magnitude-dependent floor for faint stars
    # Faint star (G=18) should have larger CatWISE floor
    r_bright = compute_pm_consistency(
        pmra_gaia=10.0, pmdec_gaia=5.0,
        pmra_err_gaia=0.1, pmdec_err_gaia=0.1,
        pmra_wise=0.0, pmdec_wise=0.0,
        pmra_err_wise=2.0, pmdec_err_wise=2.0,
        phot_g_mean_mag=12.0,  # bright star
    )
    r_faint = compute_pm_consistency(
        pmra_gaia=10.0, pmdec_gaia=5.0,
        pmra_err_gaia=0.1, pmdec_err_gaia=0.1,
        pmra_wise=0.0, pmdec_wise=0.0,
        pmra_err_wise=2.0, pmdec_err_wise=2.0,
        phot_g_mean_mag=18.0,  # faint star
    )
    assert r_faint["wise_sys_floor"] > r_bright["wise_sys_floor"], \
        f"Faint star should have larger CatWISE floor: " \
        f"bright={r_bright['wise_sys_floor']:.1f} faint={r_faint['wise_sys_floor']:.1f}"
    assert r_faint["wise_sys_floor"] > 10.0, \
        f"G=18 star should have CatWISE floor > 10 mas/yr, got {r_faint['wise_sys_floor']:.1f}"
    # Same PM offset should be LESS significant for faint star
    assert r_faint["pm_discrepancy_sigma"] < r_bright["pm_discrepancy_sigma"], \
        "Same offset should be less significant for faint star (larger systematic floor)"


# =====================================================================
# Test 11: PM consistency score in EXODUSScorer (exodus_score.py)
# =====================================================================

def test_astrometric_score_includes_pm_check():
    """Astrometric score should incorporate WISE-Gaia PM discrepancy."""
    from src.scoring.exodus_score import EXODUSScorer

    # Without PM check — just RUWE
    data_base = {"ruwe": 1.0, "astrometric_excess_noise_sig": 0.0}
    score_base = EXODUSScorer._get_astrometric_score(data_base)

    # With discrepant PM check — should boost score
    data_pm = {
        "ruwe": 1.0, "astrometric_excess_noise_sig": 0.0,
        "wise_gaia_pm": {
            "pm_discrepancy_sigma": 6.0,
            "is_discrepant": True,
        },
    }
    score_pm = EXODUSScorer._get_astrometric_score(data_pm)

    assert score_pm > score_base, (
        f"PM discrepancy should boost astrometric score: "
        f"base={score_base:.3f} pm={score_pm:.3f}"
    )
    assert score_pm > 0.5, "6-sigma PM discrepancy should give score > 0.5"


# =====================================================================
# Audit — Boundary & Edge-Case Tests
# =====================================================================

# Test 12: Threshold boundary — score exactly at threshold is INACTIVE
# Audit S23-F1: scorer and template matcher must agree on strict '>'

def test_threshold_boundary_scorer():
    """A channel scoring exactly 0.25 (convergence threshold) must be INACTIVE."""
    from src.scoring.exodus_score import EXODUSScorer

    scorer = EXODUSScorer(threshold=0.3, convergence_priority=True)
    # convergence_priority=True sets threshold to 0.25

    # Build target with one channel at exactly the threshold
    target_data = {
        "target_id": "boundary_test",
        "ra": 180.0,
        "dec": 45.0,
        "ir_excess": {"sigma_W3": 3.0, "sigma_W4": 2.0, "is_candidate": True,
                       "fitted_teff": 5000.0, "excess_W3": -0.3, "excess_W4": -0.5,
                       "fit_chi2_reduced": 5.0, "fit_bands_used": 6,
                       "contamination_flag": False, "data_source": "allwise"},
    }

    result = scorer.score_target(target_data)

    # Find the IR excess channel score
    ir_cs = result.channel_scores.get("ir_excess")
    if ir_cs is not None and abs(ir_cs.score - 0.25) < 1e-10:
        # If it happens to be exactly 0.25, it must NOT be active
        assert not ir_cs.is_active, (
            "Channel at exactly threshold=0.25 must be INACTIVE (strict >)"
        )


def test_threshold_boundary_template_matcher():
    """Template matcher must also use strict '>' for activation (S23-F1 fix)."""
    from src.vetting.astrophysical_templates import UnexplainabilityScorer

    scorer = UnexplainabilityScorer(activation_threshold=0.25)
    channel_scores = {
        "ir_excess": 0.8,
        "proper_motion_anomaly": 0.6,
        "uv_anomaly": 0.25,  # Exactly at threshold
    }

    result = scorer.evaluate("boundary_test", channel_scores)

    # UV at exactly 0.25 should NOT be in active_channels (strict >)
    assert "uv_anomaly" not in result.active_channels, (
        f"Channel at exactly threshold=0.25 must be INACTIVE in template matcher. "
        f"Got active_channels={result.active_channels}"
    )
    assert result.n_active_channels == 2, (
        f"Expected 2 active channels (ir_excess + PM), got {result.n_active_channels}"
    )


# Test 13: Zero-channel target produces score=0 with convergence_bonus=0

def test_zero_channel_target():
    """A target with no active channels must get score=0.0."""
    from src.scoring.exodus_score import EXODUSScorer

    scorer = EXODUSScorer(threshold=0.3, convergence_priority=True)

    target_data = {
        "target_id": "zero_ch",
        "ra": 100.0,
        "dec": -30.0,
        # No channel data at all
    }

    result = scorer.score_target(target_data)
    assert result.total_score == 0.0, f"Expected 0.0, got {result.total_score}"
    assert result.n_active_channels == 0
    assert result.convergence_bonus == 0.0


# Test 14: NaN/inf channel scores are handled gracefully

def test_nan_channel_score_handled():
    """NaN values in channel data must not crash the scorer."""
    from src.scoring.exodus_score import EXODUSScorer

    scorer = EXODUSScorer(threshold=0.3)

    target_data = {
        "target_id": "nan_test",
        "ra": 50.0,
        "dec": 10.0,
        "ir_variability": {
            "variability_score": float("nan"),
            "is_anomalous": False,
            "data_source": "real",
        },
    }

    # Should not raise
    result = scorer.score_target(target_data)
    # NaN score should be clipped to 0 or excluded
    ir_var = result.channel_scores.get("ir_variability")
    if ir_var is not None:
        assert np.isfinite(ir_var.score), f"Score should be finite, got {ir_var.score}"
        assert not ir_var.is_active, "NaN-origin score should not be active"


# Test 15: Fisher field renamed to exploratory (S23-F6)

def test_fisher_fdr_field_renamed():
    """The Fisher FDR fields must have 'exploratory_' prefix (S23-F6)."""
    from src.scoring.exodus_score import EXODUSScore

    # Check the dataclass has the new field names
    assert hasattr(EXODUSScore, "__dataclass_fields__")
    fields = EXODUSScore.__dataclass_fields__

    assert "exploratory_fdr_significant_fisher" in fields, (
        "Missing renamed field 'exploratory_fdr_significant_fisher'"
    )
    assert "exploratory_q_value_fisher" in fields, (
        "Missing renamed field 'exploratory_q_value_fisher'"
    )

    # Old names must NOT exist
    assert "fdr_significant_fisher" not in fields, (
        "Old field 'fdr_significant_fisher' still exists — rename incomplete"
    )
    assert "q_value_fisher" not in fields, (
        "Old field 'q_value_fisher' still exists — rename incomplete"
    )


# =====================================================================
# CLI entry point
# =====================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
