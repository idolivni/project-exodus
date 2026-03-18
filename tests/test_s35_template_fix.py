"""
Session 35: Validate the RS CVn template residual-clearing bug fix.

Bug: RS CVn template (fit=0.576) was clearing `radio_emission` from
Candidate 1's residuals despite X-ray silence (427× Güdel-Benz violation).
This dropped unexplainability from ~0.3 to 0.023.

Fix: (1) Added `rs_cvn_active_binary` to the Güdel-Benz X-ray conflict check.
     (2) Templates with strong physical contradictions cannot clear the
         conflicted channel from residuals.

Validation approach:
- Test 1: RS CVn with X-ray silence must NOT clear radio_emission
- Test 2: RS CVn WITH X-ray detection CAN still clear radio (no regression)
- Test 3: Candidate 1's actual channel data must produce UNEXPLAINED or
          PARTIALLY_EXPLAINED, not EXPLAINED
- Test 4: Other templates (binary, YSO) are unaffected by the fix
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vetting.astrophysical_templates import UnexplainabilityScorer


def test_rscvn_xray_silence_blocks_radio_clearing():
    """RS CVn template must NOT clear radio_emission when X-ray is absent.

    This is the core bug fix: RS CVn predicts radio from gyrosynchrotron
    emission, but this REQUIRES X-ray (Güdel-Benz relation). If the star
    is X-ray silent, RS CVn is physically ruled out and must not be
    allowed to explain away the radio emission.
    """
    scorer = UnexplainabilityScorer(activation_threshold=0.25)

    # Candidate 1-like channels: radio + IR + PM, NO X-ray
    channel_scores = {
        "ir_excess": 0.977,
        "proper_motion_anomaly": 0.444,
        "radio_emission": 0.523,
    }
    channel_details = {
        "radio_emission": {
            "xray_detected": False,   # X-ray silent — key fact
            "flux_mJy": 2.10,
            "separation_arcsec": 0.80,
        },
        "proper_motion_anomaly": {
            "ruwe": 1.10,
        },
    }

    result = scorer.evaluate(
        "TEST_RSCVN_XRAY_SILENCE",
        channel_scores,
        channel_details,
    )

    # radio_emission MUST be in residual channels (not cleared by RS CVn)
    assert "radio_emission" in result.residual_channels, (
        f"radio_emission should be in residuals when X-ray is absent. "
        f"Got residual_channels={result.residual_channels}, "
        f"best_template={result.best_template} (fit={result.best_template_fit:.3f}), "
        f"unexplainability={result.unexplainability_score:.3f}"
    )

    # Unexplainability should be meaningfully above EXPLAINED threshold (0.2)
    assert result.unexplainability_score >= 0.2, (
        f"Unexplainability should be >= 0.2 with unclearable radio. "
        f"Got {result.unexplainability_score:.3f}"
    )

    # Should NOT be classified as EXPLAINED
    assert result.classification != "EXPLAINED", (
        f"Star with X-ray-silent radio should not be EXPLAINED. "
        f"Got classification={result.classification}"
    )


def test_rscvn_with_xray_can_clear_radio():
    """RS CVn template CAN clear radio_emission when X-ray IS detected.

    Regression guard: the fix should NOT break legitimate RS CVn matches
    where X-ray is present (Güdel-Benz is satisfied).
    """
    scorer = UnexplainabilityScorer(activation_threshold=0.25)

    channel_scores = {
        "ir_excess": 0.6,
        "proper_motion_anomaly": 0.5,
        "radio_emission": 0.5,
        "gaia_photometric_anomaly": 0.4,  # starspot variability
    }
    channel_details = {
        "radio_emission": {
            "xray_detected": True,   # X-ray present — RS CVn is valid
            "flux_mJy": 2.0,
        },
        "proper_motion_anomaly": {
            "ruwe": 2.5,  # elevated — consistent with binary
        },
    }

    result = scorer.evaluate(
        "TEST_RSCVN_WITH_XRAY",
        channel_scores,
        channel_details,
    )

    # radio_emission should be clearable (RS CVn is physically valid)
    # It might or might not end up in residuals depending on which template
    # wins, but the RS CVn template should NOT be blocked from clearing it.
    # Check that unexplainability is low (well-explained by RS CVn or binary).
    assert result.unexplainability_score < 0.5, (
        f"Star with X-ray + radio + PM + phot should be well-explained. "
        f"Got unexplainability={result.unexplainability_score:.3f}, "
        f"best_template={result.best_template}"
    )


def test_candidate1_actual_data():
    """Candidate 1's real channel data must not be classified as EXPLAINED.

    This is the end-to-end validation: with the fix applied, the pipeline's
    automated assessment should keep Candidate 1 as anomalous without
    requiring a manual verification override.
    """
    scorer = UnexplainabilityScorer(activation_threshold=0.25)

    # Candidate 1 actual data from evidence bundle
    channel_scores = {
        "ir_excess": 0.977,
        "proper_motion_anomaly": 0.444,
        "radio_emission": 0.523,
        # Inactive channels included at 0 for completeness
        "ir_variability": 0.151,  # below threshold
        "uv_anomaly": 0.0,
        "hr_anomaly": 0.0,
    }
    channel_details = {
        "ir_excess": {
            "sigma_W3": 15.8,
            "sigma_W4": 18.9,
            "excess_W3": -2.1,
            "excess_W4": -4.97,
        },
        "proper_motion_anomaly": {
            "ruwe": 1.10,
            "astrometric_excess_noise_sig": 0.0,
            "pm_discrepancy_sigma": 1.5,
        },
        "radio_emission": {
            "xray_detected": False,   # X-ray silent across 5 catalogs
            "flux_mJy": 2.10,
            "separation_arcsec": 0.80,
            "spectral_index": -0.50,
        },
    }

    result = scorer.evaluate(
        "TEST_CANDIDATE_RADIO_IR_PM",
        channel_scores,
        channel_details,
    )

    # Must NOT be EXPLAINED
    assert result.classification != "EXPLAINED", (
        f"Candidate 1 must not be classified as EXPLAINED. "
        f"Got: unexplainability={result.unexplainability_score:.3f}, "
        f"classification={result.classification}, "
        f"best_template={result.best_template} (fit={result.best_template_fit:.3f}), "
        f"residual_channels={result.residual_channels}"
    )

    # radio_emission must be in residuals
    assert "radio_emission" in result.residual_channels, (
        f"radio_emission must be unexplained (X-ray silent). "
        f"Got residual_channels={result.residual_channels}"
    )

    # Unexplainability should be meaningful
    assert result.unexplainability_score >= 0.2, (
        f"Candidate 1 unexplainability should be >= 0.2. "
        f"Got {result.unexplainability_score:.3f}"
    )

    print(f"\n=== Candidate 1 Template Analysis ===")
    print(f"  Unexplainability: {result.unexplainability_score:.3f}")
    print(f"  Classification:   {result.classification}")
    print(f"  Best template:    {result.best_template} (fit={result.best_template_fit:.3f})")
    print(f"  Active channels:  {result.active_channels}")
    print(f"  Residual (unexplained): {result.residual_channels}")
    if result.template_conflicts:
        print(f"  Conflicts:")
        for c in result.template_conflicts:
            print(f"    - [{c.severity}] {c.template_name}: {c.observation}")


def test_binary_template_unaffected():
    """Binary template should still work normally (no regression).

    Binary systems don't have the Güdel-Benz constraint, so the fix
    should not change their behavior.
    """
    scorer = UnexplainabilityScorer(activation_threshold=0.25)

    # Classic binary signature: PM + IR + HR
    channel_scores = {
        "proper_motion_anomaly": 0.8,
        "ir_excess": 0.6,
        "hr_anomaly": 0.5,
    }
    channel_details = {
        "proper_motion_anomaly": {
            "ruwe": 3.5,   # elevated — binary
        },
    }

    result = scorer.evaluate(
        "TEST_BINARY_REGRESSION",
        channel_scores,
        channel_details,
    )

    # Binary should explain this well
    assert result.unexplainability_score < 0.3, (
        f"Classic binary signature should be well-explained. "
        f"Got unexplainability={result.unexplainability_score:.3f}, "
        f"best_template={result.best_template}"
    )


def test_active_flare_star_xray_check_still_works():
    """Active flare star template should still have X-ray conflict detection.

    Regression: the original active_flare_star check must still work
    after adding rs_cvn_active_binary to the same condition.
    """
    scorer = UnexplainabilityScorer(activation_threshold=0.25)

    channel_scores = {
        "radio_emission": 0.6,
        "gaia_photometric_anomaly": 0.5,
    }
    channel_details = {
        "radio_emission": {
            "xray_detected": False,
        },
    }

    result = scorer.evaluate(
        "TEST_FLARE_STAR_XRAY",
        channel_scores,
        channel_details,
    )

    # Radio should be in residuals (flare star can't explain it without X-ray)
    # Note: whether this happens depends on which template wins and fit scores
    # At minimum, there should be template conflicts detected
    assert result.has_template_conflict or "radio_emission" in result.residual_channels, (
        f"X-ray silent radio should produce conflict or remain in residuals. "
        f"Got: conflicts={result.has_template_conflict}, "
        f"residual={result.residual_channels}"
    )


if __name__ == "__main__":
    print("Running S35 template fix validation tests...\n")

    tests = [
        test_rscvn_xray_silence_blocks_radio_clearing,
        test_rscvn_with_xray_can_clear_radio,
        test_candidate1_actual_data,
        test_binary_template_unaffected,
        test_active_flare_star_xray_check_still_works,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  💥 {t.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("All tests PASSED ✅")
    else:
        print("Some tests FAILED ❌")
