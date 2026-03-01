"""
Red-Team Falsification Engine
=============================

For every multi-channel candidate, automatically run a battery of natural
explanation checks and generate a falsification scorecard.  The goal is to
*aggressively* seek reasons why an anomaly is NOT a technosignature before
anyone has to ask.

This module works entirely from data already gathered in the pipeline —
no new network queries are issued.  It reads the target dict produced by
the runner (Gaia astrometry, IR photometry, EXODUS score, unexplainability
result, multi-messenger data) and produces a structured audit trail.

Checks
------
1. **Known object type** — Is this star a known binary, YSO, or variable?
   (Inferred from Gaia params + SIMBAD type if available.)
2. **RUWE context** — Is high RUWE explained by known binary, close pair,
   or faint companion?
3. **IR excess context** — Is IR excess consistent with circumstellar dust,
   background galaxy, or photometric confusion?
4. **Galactic contamination** — Low galactic latitude → crowded field,
   higher confusion probability.
5. **Photometric reliability** — Gaia quality flags, few epochs,
   magnitude-dependent systematics.
6. **PM consistency** — WISE-Gaia proper motion discrepancy analysis.
7. **Multi-messenger skepticism** — Fermi/IceCube matches consistent with
   chance alignment given source density?
8. **Single-channel dominance** — Is the score driven by only one channel
   with no corroboration?
9. **Distance bias** — Very nearby stars (< 5 pc) get inflated RUWE and
   saturated photometry.  Very distant stars (> 100 pc) have poor parallax.
10. **Convergence quality** — Are the "converging" channels truly independent
    or likely correlated by a single mechanism?

API
---
    from src.vetting.red_team import RedTeamEngine

    engine = RedTeamEngine()
    verdict = engine.evaluate(target_data)
    print(verdict.summary)
    print(verdict.recommendation)  # "ESCALATE" | "MONITOR" | "DEMOTE"
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger

log = get_logger("vetting.red_team")


# =====================================================================
#  Data classes
# =====================================================================

@dataclass
class FalsificationCheck:
    """One falsification check result."""
    check_name: str
    risk_category: str     # "astrophysical", "instrumental", "statistical", "environmental"
    risk_level: float      # 0.0 = no concern, 1.0 = definitely a false positive
    explanation: str        # Human-readable explanation
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_name": self.check_name,
            "risk_category": self.risk_category,
            "risk_level": round(self.risk_level, 3),
            "explanation": self.explanation,
            "evidence": self.evidence,
        }


@dataclass
class RedTeamVerdict:
    """Complete red-team falsification verdict for one target."""
    target_id: str

    # Aggregate
    overall_risk: float         # 0-1, higher = more likely false positive
    risk_level: str             # "LOW" | "MODERATE" | "HIGH" | "CRITICAL"
    recommendation: str         # "ESCALATE" | "MONITOR" | "DEMOTE"

    # Individual checks
    checks: List[FalsificationCheck]
    n_risk_flags: int           # checks with risk_level > 0.5
    top_risk: Optional[str]     # name of highest-risk check

    # Natural explanations (ranked)
    natural_explanations: List[str]

    # Summary
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "overall_risk": round(self.overall_risk, 4),
            "risk_level": self.risk_level,
            "recommendation": self.recommendation,
            "n_risk_flags": self.n_risk_flags,
            "top_risk": self.top_risk,
            "natural_explanations": self.natural_explanations,
            "checks": [c.to_dict() for c in self.checks],
            "summary": self.summary,
        }


# =====================================================================
#  Red-Team Engine
# =====================================================================

class RedTeamEngine:
    """Automated falsification engine for EXODUS candidates.

    Runs a battery of skeptical checks against each target's data
    and produces a structured audit trail with risk assessment.

    Parameters
    ----------
    risk_threshold_high : float
        Overall risk above this → "HIGH" classification (default 0.5).
    risk_threshold_critical : float
        Overall risk above this → "CRITICAL" classification (default 0.75).
    """

    def __init__(
        self,
        risk_threshold_high: float = 0.5,
        risk_threshold_critical: float = 0.75,
    ):
        self.risk_threshold_high = risk_threshold_high
        self.risk_threshold_critical = risk_threshold_critical

    def evaluate(self, target_data: Dict[str, Any]) -> RedTeamVerdict:
        """Run all falsification checks on a scored target.

        Parameters
        ----------
        target_data : dict
            The target dict as produced by the runner pipeline, containing:
            - target_id, ra, dec, distance_pc
            - gaia_data (astrometry, stellar params)
            - ir_photometry (2MASS, AllWISE, CatWISE)
            - exodus_score (dict from EXODUSScore.to_dict())
            - unexplainability (dict from UnexplainabilityResult.to_dict())
            - multi_messenger (gamma, neutrino, GW, pulsar, FRB data)
            - transit_anomaly (if available)

        Returns
        -------
        RedTeamVerdict
        """
        target_id = target_data.get("target_id", "unknown")

        checks = []
        checks.append(self._check_known_object_type(target_data))
        checks.append(self._check_ruwe_context(target_data))
        checks.append(self._check_ir_excess_context(target_data))
        checks.append(self._check_galactic_contamination(target_data))
        checks.append(self._check_photometric_reliability(target_data))
        checks.append(self._check_pm_consistency(target_data))
        checks.append(self._check_mm_chance_alignment(target_data))
        checks.append(self._check_single_channel_dominance(target_data))
        checks.append(self._check_distance_bias(target_data))
        checks.append(self._check_convergence_quality(target_data))
        checks.append(self._check_xray_activity(target_data))
        checks.append(self._check_dust_extinction(target_data))
        checks.append(self._check_sb9_binary(target_data))
        checks.append(self._check_simbad_type(target_data))

        # Filter out None checks (some may not apply)
        checks = [c for c in checks if c is not None]

        # Compute overall risk
        if checks:
            # Weighted: highest-risk checks matter most
            risks = sorted([c.risk_level for c in checks], reverse=True)
            # Top-heavy weighting: top check gets 40%, next 25%, next 15%, rest 20%
            if len(risks) >= 3:
                overall_risk = (
                    0.40 * risks[0] +
                    0.25 * risks[1] +
                    0.15 * risks[2] +
                    0.20 * (sum(risks[3:]) / max(len(risks) - 3, 1))
                )
            elif len(risks) == 2:
                overall_risk = 0.55 * risks[0] + 0.45 * risks[1]
            else:
                overall_risk = risks[0]
            overall_risk = min(overall_risk, 1.0)
        else:
            overall_risk = 0.0

        # Audit fix C1: a single catastrophic check (risk >= 0.9) should
        # trigger DEMOTE regardless of the weighted average.  Before this fix,
        # the top-heavy weighting diluted even risk=1.0 to ~0.40 overall,
        # which never reached the 0.75 critical threshold.
        has_catastrophic = any(c.risk_level >= 0.9 for c in checks) if checks else False

        # Classification
        if has_catastrophic or overall_risk >= self.risk_threshold_critical:
            risk_level = "CRITICAL"
            recommendation = "DEMOTE"
        elif overall_risk >= self.risk_threshold_high:
            risk_level = "HIGH"
            recommendation = "MONITOR"
        elif overall_risk >= 0.3:
            risk_level = "MODERATE"
            recommendation = "MONITOR"
        else:
            risk_level = "LOW"
            recommendation = "ESCALATE"

        # Count risk flags
        n_risk_flags = sum(1 for c in checks if c.risk_level > 0.5)

        # Top risk
        top_risk = None
        if checks:
            worst = max(checks, key=lambda c: c.risk_level)
            if worst.risk_level > 0.3:
                top_risk = worst.check_name

        # Natural explanations (checks with risk > 0.3, ranked)
        natural_explanations = [
            c.explanation
            for c in sorted(checks, key=lambda c: -c.risk_level)
            if c.risk_level > 0.3
        ]

        # Summary
        summary = (
            f"{target_id}: risk={overall_risk:.2f} ({risk_level}), "
            f"{n_risk_flags} flags, recommendation={recommendation}"
        )
        if natural_explanations:
            summary += f". Top concern: {natural_explanations[0]}"

        verdict = RedTeamVerdict(
            target_id=target_id,
            overall_risk=overall_risk,
            risk_level=risk_level,
            recommendation=recommendation,
            checks=checks,
            n_risk_flags=n_risk_flags,
            top_risk=top_risk,
            natural_explanations=natural_explanations,
            summary=summary,
        )

        log.info("Red-team: %s", summary)
        return verdict

    def batch_evaluate(
        self, targets: List[Dict[str, Any]]
    ) -> List[RedTeamVerdict]:
        """Evaluate multiple targets. Returns verdicts sorted by risk (lowest first)."""
        verdicts = []
        for t in targets:
            try:
                v = self.evaluate(t)
                verdicts.append(v)
            except Exception as exc:
                log.warning(
                    "Red-team evaluation failed for %s: %s",
                    t.get("target_id", "unknown"), exc,
                )
        verdicts.sort(key=lambda v: v.overall_risk)
        return verdicts

    # =================================================================
    #  Individual falsification checks
    # =================================================================

    def _check_known_object_type(self, t: Dict) -> FalsificationCheck:
        """Check if the star is a known type that explains anomalies."""
        gaia = {**t.get("gaia_astrometry", {}), **t.get("gaia_params", {}), **t.get("gaia_data", {})}
        sp_type = (
            t.get("sp_type", "")
            or gaia.get("sp_type", "")
            or ""
        )
        simbad_type = t.get("simbad_type", "") or ""
        teff = gaia.get("teff_gspphot") or gaia.get("teff")
        logg = gaia.get("logg_gspphot") or gaia.get("logg")

        risk = 0.0
        explanations = []
        evidence = {}

        # Check spectral type for known confounders
        sp_upper = sp_type.upper().strip()
        if sp_upper:
            evidence["sp_type"] = sp_type

        # Known binary indicators in SIMBAD type
        simbad_upper = simbad_type.upper()
        binary_types = ["SB*", "EB*", "**", "SB1", "SB2", "ECLBIN", "SPECTROBIN"]
        for bt in binary_types:
            if bt in simbad_upper:
                risk = max(risk, 0.8)
                explanations.append(
                    f"SIMBAD classifies as '{simbad_type}' (binary) — "
                    f"RUWE and photometric anomalies are expected"
                )
                evidence["simbad_type"] = simbad_type
                break

        # YSO / T Tauri indicators
        yso_types = ["TTAU", "YSO", "PMS", "HAEBE", "HERBIG"]
        for yt in yso_types:
            if yt in simbad_upper:
                risk = max(risk, 0.7)
                explanations.append(
                    f"SIMBAD classifies as '{simbad_type}' (young stellar object) — "
                    f"IR excess and variability are intrinsic to YSOs"
                )
                evidence["simbad_type"] = simbad_type
                break

        # Variable star indicators
        var_types = ["V*", "RRCLY", "CEPHEID", "MIRA", "LPV", "BYDR", "RSCV"]
        for vt in var_types:
            if vt in simbad_upper:
                risk = max(risk, 0.6)
                explanations.append(
                    f"SIMBAD classifies as '{simbad_type}' (variable star) — "
                    f"photometric anomalies are intrinsic"
                )
                evidence["simbad_type"] = simbad_type
                break

        # Giant/AGB with IR excess is normal
        if logg is not None and logg < 3.0:
            evidence["logg"] = logg
            # Check if IR excess is the main signal
            score = t.get("exodus_score", {})
            channels = score.get("channel_scores", {})
            ir_ch = channels.get("ir_excess", {})
            if isinstance(ir_ch, dict) and ir_ch.get("is_active", False):
                risk = max(risk, 0.6)
                explanations.append(
                    f"Star has low log(g)={logg:.1f} (giant/subgiant) — "
                    f"IR excess is common from circumstellar dust and mass loss"
                )

        # Gaia DR3 Non-Single Star (NSS) catalog flag
        # non_single_star > 0 means the source appears in the NSS tables
        # (orbital solution, acceleration solution, spectroscopic binary, etc.)
        astro = t.get("gaia_astrometry", {})
        nss = astro.get("non_single_star") or gaia.get("non_single_star")
        if nss is not None:
            try:
                nss_val = int(nss)
            except (ValueError, TypeError):
                nss_val = 0
            if nss_val > 0:
                evidence["non_single_star"] = nss_val
                risk = max(risk, 0.8)
                explanations.append(
                    f"Gaia DR3 NSS catalog flag={nss_val} — confirmed "
                    f"non-single star (binary/multiple). RUWE, PM anomaly, "
                    f"and photometric variability are expected"
                )

        # Very cool stars (M-dwarfs) are magnetically active
        if teff is not None and teff < 3500:
            evidence["teff"] = teff
            score = t.get("exodus_score", {})
            channels = score.get("channel_scores", {})
            phot_ch = channels.get("gaia_photometric_anomaly", {})
            if isinstance(phot_ch, dict) and phot_ch.get("is_active", False):
                risk = max(risk, 0.5)
                explanations.append(
                    f"Star has Teff={teff:.0f}K (cool M-dwarf) — "
                    f"photometric variability from starspots and flares is expected"
                )

        explanation = explanations[0] if explanations else "No known confounding object type identified"
        return FalsificationCheck(
            check_name="known_object_type",
            risk_category="astrophysical",
            risk_level=risk,
            explanation=explanation,
            evidence=evidence,
        )

    def _check_ruwe_context(self, t: Dict) -> FalsificationCheck:
        """Contextualize RUWE: does high RUWE have a mundane explanation?"""
        gaia = {**t.get("gaia_astrometry", {}), **t.get("gaia_params", {}), **t.get("gaia_data", {})}
        ruwe = gaia.get("ruwe")
        aen_sig = gaia.get("astrometric_excess_noise_sig")
        phot_g = gaia.get("phot_g_mean_mag")
        bp_rp = gaia.get("bp_rp")

        risk = 0.0
        explanation = "No RUWE data available"
        evidence = {}

        if ruwe is None:
            return FalsificationCheck(
                check_name="ruwe_context",
                risk_category="instrumental",
                risk_level=0.0,
                explanation=explanation,
                evidence=evidence,
            )

        evidence["ruwe"] = ruwe
        if aen_sig is not None:
            evidence["aen_sig"] = aen_sig
        if phot_g is not None:
            evidence["phot_g_mean_mag"] = phot_g

        # RUWE is systematically elevated for very bright stars (G < 6)
        if phot_g is not None and phot_g < 6.0:
            risk = max(risk, 0.6)
            explanation = (
                f"Star is very bright (G={phot_g:.1f}) — RUWE={ruwe:.2f} is "
                f"likely inflated by saturation and PSF modeling artifacts"
            )

        # RUWE is systematically elevated for very red stars
        elif bp_rp is not None and bp_rp > 3.0:
            evidence["bp_rp"] = bp_rp
            risk = max(risk, 0.4)
            explanation = (
                f"Star is very red (BP-RP={bp_rp:.1f}) — RUWE={ruwe:.2f} "
                f"may be elevated by chromaticity effects in the Gaia PSF model"
            )

        # Moderate RUWE (1.4-2.0) is common and not very informative
        elif ruwe is not None and 1.4 < ruwe < 2.0:
            risk = max(risk, 0.3)
            explanation = (
                f"RUWE={ruwe:.2f} is moderately elevated — this is common "
                f"in close binaries (sep < 1\") and marginally resolved sources"
            )

        # Very high RUWE (> 4) is interesting but check if explained
        elif ruwe is not None and ruwe > 4.0:
            risk = 0.1  # Low risk — genuinely interesting
            explanation = (
                f"RUWE={ruwe:.2f} is strongly elevated — difficult to explain "
                f"by instrumental effects alone"
            )

        else:
            risk = 0.0
            explanation = f"RUWE={ruwe:.2f} is within normal range or unremarkable"

        return FalsificationCheck(
            check_name="ruwe_context",
            risk_category="instrumental",
            risk_level=risk,
            explanation=explanation,
            evidence=evidence,
        )

    def _check_ir_excess_context(self, t: Dict) -> FalsificationCheck:
        """Check if IR excess has a mundane explanation."""
        ir = t.get("ir_photometry", {})
        score = t.get("exodus_score", {})
        channels = score.get("channel_scores", {})
        ir_ch = channels.get("ir_excess", {})
        if not isinstance(ir_ch, dict):
            ir_ch = {}

        risk = 0.0
        explanation = "No IR excess detected or no IR data available"
        evidence = {}

        ir_details = ir_ch.get("details", {})
        sigma_w3 = ir_details.get("sigma_W3")
        sigma_w4 = ir_details.get("sigma_W4")

        if sigma_w3 is not None:
            evidence["sigma_W3"] = sigma_w3
        if sigma_w4 is not None:
            evidence["sigma_W4"] = sigma_w4

        # Check for photometric quality issues
        w3_snr = ir.get("w3snr") or ir.get("w3sigmpro")
        if w3_snr is not None:
            evidence["w3_quality"] = w3_snr

        # W4-only excess is often confusion noise
        if sigma_w4 is not None and sigma_w4 > 3.0:
            if sigma_w3 is None or sigma_w3 < 2.0:
                risk = max(risk, 0.6)
                explanation = (
                    f"IR excess is W4-only (sigma_W4={sigma_w4:.1f}, sigma_W3={sigma_w3 or 0:.1f}) — "
                    f"W4 has 12\" beam, very susceptible to confusion noise from "
                    f"background galaxies"
                )
                evidence["w4_only"] = True

        # Marginal excess (3-5 sigma) in a single band
        elif sigma_w3 is not None and 3.0 < sigma_w3 < 5.0:
            if sigma_w4 is None or sigma_w4 < 3.0:
                risk = max(risk, 0.4)
                explanation = (
                    f"Marginal IR excess (sigma_W3={sigma_w3:.1f}) in single band — "
                    f"could be photometric scatter at the 3-sigma level"
                )

        # Check galactic latitude — background confusion
        ra = t.get("ra", 0)
        dec = t.get("dec", 0)
        glat = _galactic_latitude(ra, dec)
        if glat is not None and abs(glat) < 10:
            evidence["galactic_lat"] = round(glat, 1)
            if sigma_w3 is not None and sigma_w3 > 3.0:
                risk = max(risk, 0.5)
                explanation = (
                    f"IR excess at low galactic latitude (b={glat:.1f}°) — "
                    f"high source density increases confusion probability"
                )

        # Strong multi-band IR excess is harder to dismiss
        if (sigma_w3 is not None and sigma_w3 > 5.0
                and sigma_w4 is not None and sigma_w4 > 5.0):
            risk = min(risk, 0.2)  # Override: this is genuinely interesting
            explanation = (
                f"Strong multi-band IR excess (W3={sigma_w3:.1f}σ, W4={sigma_w4:.1f}σ) — "
                f"difficult to dismiss as confusion or photometric error"
            )

        return FalsificationCheck(
            check_name="ir_excess_context",
            risk_category="astrophysical",
            risk_level=risk,
            explanation=explanation,
            evidence=evidence,
        )

    def _check_galactic_contamination(self, t: Dict) -> FalsificationCheck:
        """Assess crowded-field contamination risk."""
        ra = t.get("ra", 0)
        dec = t.get("dec", 0)
        glat = _galactic_latitude(ra, dec)

        risk = 0.0
        evidence = {}

        if glat is None:
            return FalsificationCheck(
                check_name="galactic_contamination",
                risk_category="environmental",
                risk_level=0.0,
                explanation="Cannot compute galactic latitude",
                evidence={},
            )

        evidence["galactic_latitude"] = round(glat, 1)

        if abs(glat) < 5:
            risk = 0.7
            explanation = (
                f"Target at galactic latitude b={glat:.1f}° — directly in the "
                f"Galactic plane. Very high source confusion, crowded photometry, "
                f"and diffuse emission contamination"
            )
        elif abs(glat) < 15:
            risk = 0.4
            explanation = (
                f"Target at galactic latitude b={glat:.1f}° — moderately "
                f"crowded field, elevated confusion risk for IR photometry"
            )
        elif abs(glat) < 30:
            risk = 0.15
            explanation = (
                f"Target at galactic latitude b={glat:.1f}° — mild "
                f"crowding, low confusion risk"
            )
        else:
            risk = 0.0
            explanation = (
                f"Target at galactic latitude b={glat:.1f}° — high "
                f"galactic latitude, minimal contamination risk"
            )

        return FalsificationCheck(
            check_name="galactic_contamination",
            risk_category="environmental",
            risk_level=risk,
            explanation=explanation,
            evidence=evidence,
        )

    def _check_photometric_reliability(self, t: Dict) -> FalsificationCheck:
        """Check Gaia photometric quality indicators."""
        gaia = {**t.get("gaia_astrometry", {}), **t.get("gaia_params", {}), **t.get("gaia_data", {})}
        risk = 0.0
        evidence = {}
        concerns = []

        # Check number of Gaia epochs
        n_obs = gaia.get("matched_transits") or gaia.get("n_obs")
        if n_obs is not None:
            evidence["n_obs"] = n_obs
            if n_obs < 20:
                risk = max(risk, 0.4)
                concerns.append(
                    f"Only {n_obs} Gaia transits — low epoch count "
                    f"reduces photometric reliability"
                )

        # Check Gaia photometric excess factor
        phot_excess = gaia.get("phot_bp_rp_excess_factor")
        if phot_excess is not None:
            evidence["phot_bp_rp_excess_factor"] = phot_excess
            # Normal range is roughly 1.0 ± 0.2 for well-behaved sources
            if phot_excess > 1.5 or phot_excess < 0.7:
                risk = max(risk, 0.5)
                concerns.append(
                    f"Gaia BP/RP excess factor={phot_excess:.2f} (normal ~1.0) — "
                    f"suggests blended or contaminated photometry"
                )

        # Gaia image blending: ipd_frac_multi_peak > 10% means
        # Gaia detected multiple peaks in the image — the source is
        # likely blended with a nearby star or background galaxy.
        # This contaminates ALL photometry (Gaia, 2MASS, WISE).
        astro = t.get("gaia_astrometry", {})
        ipd_multi = astro.get("ipd_frac_multi_peak") or gaia.get("ipd_frac_multi_peak")
        if ipd_multi is not None:
            evidence["ipd_frac_multi_peak"] = ipd_multi
            if ipd_multi > 10:
                risk = max(risk, 0.6)
                concerns.append(
                    f"Gaia ipd_frac_multi_peak={ipd_multi:.0f}% — source is "
                    f"blended; all photometric channels may be contaminated"
                )

        # Very faint or very bright targets have quality issues
        phot_g = gaia.get("phot_g_mean_mag")
        if phot_g is not None:
            evidence["phot_g_mean_mag"] = phot_g
            if phot_g > 18:
                risk = max(risk, 0.4)
                concerns.append(
                    f"Faint target (G={phot_g:.1f}) — low S/N in all bands"
                )
            elif phot_g < 4:
                risk = max(risk, 0.5)
                concerns.append(
                    f"Very bright target (G={phot_g:.1f}) — saturation "
                    f"effects in Gaia, 2MASS, and WISE"
                )

        explanation = concerns[0] if concerns else "Photometric quality indicators are nominal"
        return FalsificationCheck(
            check_name="photometric_reliability",
            risk_category="instrumental",
            risk_level=risk,
            explanation=explanation,
            evidence=evidence,
        )

    def _check_pm_consistency(self, t: Dict) -> FalsificationCheck:
        """Evaluate WISE-Gaia proper motion consistency."""
        score = t.get("exodus_score", {})
        channels = score.get("channel_scores", {})
        pm_ch = channels.get("proper_motion_anomaly", {})
        if not isinstance(pm_ch, dict):
            pm_ch = {}
        pm_details = pm_ch.get("details", {})
        wise_gaia = pm_details.get("wise_gaia_pm", {})

        risk = 0.0
        evidence = {}
        explanation = "No WISE-Gaia PM comparison available"

        if not wise_gaia:
            return FalsificationCheck(
                check_name="pm_consistency",
                risk_category="astrophysical",
                risk_level=0.0,
                explanation=explanation,
                evidence=evidence,
            )

        is_discrepant = wise_gaia.get("is_discrepant", False)
        sigma = wise_gaia.get("pm_discrepancy_sigma", 0)
        chi2 = wise_gaia.get("chi2", 0)

        evidence["pm_discrepancy_sigma"] = sigma
        evidence["is_discrepant"] = is_discrepant
        evidence["chi2"] = chi2

        if is_discrepant and sigma > 5:
            # Strong discrepancy — genuinely interesting, low risk of FP
            risk = 0.1
            explanation = (
                f"Strong WISE-Gaia PM discrepancy ({sigma:.1f}σ) — "
                f"consistent with unresolved companion or non-linear motion. "
                f"Difficult to explain by systematics alone"
            )
        elif is_discrepant and sigma > 3:
            # Moderate discrepancy — could be real or CatWISE systematic
            risk = 0.3
            explanation = (
                f"Moderate WISE-Gaia PM discrepancy ({sigma:.1f}σ) — "
                f"could indicate a companion but CatWISE PM uncertainties "
                f"have systematic floors of ~2 mas/yr"
            )
        elif not is_discrepant:
            risk = 0.0
            explanation = (
                f"WISE-Gaia PMs are consistent ({sigma:.1f}σ) — "
                f"no evidence for unresolved companion from PM analysis"
            )

        return FalsificationCheck(
            check_name="pm_consistency",
            risk_category="astrophysical",
            risk_level=risk,
            explanation=explanation,
            evidence=evidence,
        )

    def _check_mm_chance_alignment(self, t: Dict) -> FalsificationCheck:
        """Assess whether multi-messenger matches are chance alignments."""
        mm = t.get("multi_messenger", {})
        risk = 0.0
        evidence = {}
        concerns = []

        # Check gamma-ray match
        gamma = mm.get("gamma", {})
        if isinstance(gamma, dict):
            matches = gamma.get("matches") or gamma.get("coincidences") or []
            if matches:
                evidence["n_gamma_matches"] = len(matches)
                # Most gamma matches with unID Fermi sources are chance
                # The typical number of unID sources per sq deg is ~0.07
                # so for a 0.1 deg search radius, expected chance = ~2e-3
                for m in matches[:3]:
                    if isinstance(m, dict):
                        p = m.get("p_chance") or m.get("p_corrected", 1.0)
                        if p > 0.01:
                            risk = max(risk, 0.5)
                            concerns.append(
                                f"Gamma-ray match has P(chance)={p:.4f} — "
                                f"likely random alignment with background AGN"
                            )
                        elif p > 0.001:
                            risk = max(risk, 0.3)
                            concerns.append(
                                f"Gamma-ray match has P(chance)={p:.4f} — "
                                f"marginal, could be chance alignment"
                            )

        # Check neutrino match
        neutrino = mm.get("neutrino", {})
        if isinstance(neutrino, dict):
            hosts = neutrino.get("hosts_with_excess") or []
            if hosts:
                evidence["n_neutrino_hosts"] = len(hosts)
                for h in hosts[:3]:
                    if isinstance(h, dict):
                        p = h.get("p_corrected", 1.0)
                        n_trials = h.get("n_trials", 1)
                        if p > 0.01:
                            risk = max(risk, 0.4)
                            concerns.append(
                                f"Neutrino excess has P(corrected)={p:.4f} "
                                f"(Bonferroni over {n_trials} trials) — not significant"
                            )

        # Check FRB match
        frb = mm.get("frb", {})
        if isinstance(frb, dict):
            frb_matches = frb.get("matches") or []
            if frb_matches:
                evidence["n_frb_matches"] = len(frb_matches)
                # FRB matches are extremely common for nearby stars
                # because the search radius is large (few degrees)
                risk = max(risk, 0.3)
                concerns.append(
                    f"FRB spatial match ({len(frb_matches)} repeaters nearby) — "
                    f"FRB localization is poor, spatial coincidence is expected "
                    f"for many targets"
                )

        if not concerns:
            explanation = "No multi-messenger matches, or matches are statistically significant"
        else:
            explanation = concerns[0]

        return FalsificationCheck(
            check_name="mm_chance_alignment",
            risk_category="statistical",
            risk_level=risk,
            explanation=explanation,
            evidence=evidence,
        )

    def _check_single_channel_dominance(self, t: Dict) -> FalsificationCheck:
        """Check if the EXODUS score is driven by a single channel."""
        score = t.get("exodus_score", {})
        channels = score.get("channel_scores", {})
        n_active = score.get("n_active_channels", 0)
        total = score.get("total_score", 0)

        risk = 0.0
        evidence = {"n_active_channels": n_active}
        explanation = "Score not dominated by a single channel"

        if n_active <= 1:
            risk = 0.6
            # Find which channel
            dominant = "unknown"
            for name, ch in channels.items():
                if isinstance(ch, dict) and ch.get("is_active", False):
                    dominant = name
                    break
            evidence["dominant_channel"] = dominant
            explanation = (
                f"EXODUS score relies on single active channel ({dominant}) — "
                f"no independent corroboration from other channels. "
                f"Single-channel anomalies are very common"
            )
        elif n_active == 2:
            # Two channels — check if they're correlated
            active_names = [
                name for name, ch in channels.items()
                if isinstance(ch, dict) and ch.get("is_active", False)
            ]
            evidence["active_channels"] = active_names

            # Known correlated pairs that don't count as independent
            correlated_pairs = {
                frozenset({"ir_excess", "transit_anomaly"}):
                    "IR excess and transit anomaly can both arise from circumstellar dust",
                frozenset({"proper_motion_anomaly", "gaia_photometric_anomaly"}):
                    "Both RUWE and photometric anomaly can arise from an unresolved binary",
            }
            pair = frozenset(active_names)
            if pair in correlated_pairs:
                risk = 0.4
                explanation = (
                    f"Two active channels ({', '.join(active_names)}) but they "
                    f"are astrophysically correlated: {correlated_pairs[pair]}"
                )
            else:
                risk = 0.1
                explanation = (
                    f"Two independent active channels ({', '.join(active_names)}) — "
                    f"genuine multi-channel convergence is encouraging"
                )

        return FalsificationCheck(
            check_name="single_channel_dominance",
            risk_category="statistical",
            risk_level=risk,
            explanation=explanation,
            evidence=evidence,
        )

    def _check_distance_bias(self, t: Dict) -> FalsificationCheck:
        """Check for distance-dependent systematic effects."""
        distance = t.get("distance_pc")
        gaia = {**t.get("gaia_astrometry", {}), **t.get("gaia_params", {}), **t.get("gaia_data", {})}
        parallax = gaia.get("parallax")
        parallax_error = gaia.get("parallax_error")

        risk = 0.0
        evidence = {}
        explanation = "Distance is in nominal range for reliable measurements"

        if distance is not None:
            evidence["distance_pc"] = distance

            if distance < 3:
                risk = 0.5
                explanation = (
                    f"Extremely nearby star ({distance:.1f} pc) — "
                    f"saturated in 2MASS/WISE, large angular size may "
                    f"affect Gaia PSF fitting and inflate RUWE"
                )
            elif distance < 5:
                risk = 0.3
                explanation = (
                    f"Very nearby star ({distance:.1f} pc) — some risk "
                    f"of photometric saturation and resolved structure "
                    f"inflating anomaly scores"
                )
            elif distance > 100:
                risk = 0.3
                explanation = (
                    f"Distant star ({distance:.1f} pc) — parallax-based "
                    f"distance may be uncertain, IR photometry more "
                    f"susceptible to confusion"
                )
                if parallax_error is not None and parallax is not None and parallax > 0:
                    frac_err = parallax_error / parallax
                    evidence["parallax_frac_error"] = round(frac_err, 3)
                    if frac_err > 0.2:
                        risk = 0.5
                        explanation = (
                            f"Distant star ({distance:.1f} pc) with poor "
                            f"parallax ({frac_err:.0%} relative error) — "
                            f"distance and derived quantities are unreliable"
                        )

        return FalsificationCheck(
            check_name="distance_bias",
            risk_category="instrumental",
            risk_level=risk,
            explanation=explanation,
            evidence=evidence,
        )

    def _check_convergence_quality(self, t: Dict) -> FalsificationCheck:
        """Assess whether multi-channel convergence is genuine or correlated."""
        score = t.get("exodus_score", {})
        unex = t.get("unexplainability", {})
        channels = score.get("channel_scores", {})
        n_active = score.get("n_active_channels", 0)

        risk = 0.0
        evidence = {}
        explanation = "Convergence quality is adequate"

        if n_active < 2:
            # No convergence to assess
            return FalsificationCheck(
                check_name="convergence_quality",
                risk_category="statistical",
                risk_level=0.0,
                explanation="No multi-channel convergence to assess",
                evidence={"n_active": n_active},
            )

        active_names = [
            name for name, ch in channels.items()
            if isinstance(ch, dict) and ch.get("is_active", False)
        ]
        evidence["active_channels"] = active_names
        evidence["n_active"] = n_active

        # Check unexplainability — if EXPLAINED, convergence is from known physics
        unex_class = unex.get("classification", "")
        unex_score = unex.get("unexplainability_score", 0)
        if unex_class == "EXPLAINED":
            risk = 0.7
            best = unex.get("best_template", "unknown")
            explanation = (
                f"Multi-channel convergence fully explained by "
                f"'{best}' template (unexplainability={unex_score:.3f}). "
                f"Channels are correlated through known astrophysics, "
                f"not independent detections"
            )
            evidence["unexplainability"] = unex_score
            evidence["best_template"] = best
        elif unex_class == "PARTIALLY_EXPLAINED":
            risk = 0.4
            residual = unex.get("residual_channels", [])
            explanation = (
                f"Convergence partially explained (unexplainability={unex_score:.3f}). "
                f"Residual unexplained channels: {residual}. "
                f"Some channels may be correlated through known physics"
            )
            evidence["unexplainability"] = unex_score
            evidence["residual_channels"] = residual
        elif unex_class == "UNEXPLAINED":
            risk = 0.1
            explanation = (
                f"Multi-channel convergence is genuinely unexplained "
                f"(unexplainability={unex_score:.3f}). No known astrophysical "
                f"template accounts for the observed channel pattern"
            )
            evidence["unexplainability"] = unex_score

        # Check if FDR-significant
        fdr_sig = score.get("fdr_significant", False)
        q_val = score.get("q_value")
        if fdr_sig:
            evidence["fdr_significant"] = True
            evidence["q_value"] = q_val
            # FDR significance slightly reduces risk
            risk = max(0, risk - 0.1)

        return FalsificationCheck(
            check_name="convergence_quality",
            risk_category="statistical",
            risk_level=risk,
            explanation=explanation,
            evidence=evidence,
        )


    def _check_xray_activity(self, t: Dict) -> Optional[FalsificationCheck]:
        """Check eROSITA eRASS1 for X-ray activity near the target.

        X-ray bright sources strongly suggest stellar activity (YSO,
        flare star, active binary, RS CVn) — not technosignatures.
        A Dyson sphere around a mature star would be X-ray quiet.
        """
        erosita = t.get("erosita")
        if erosita is None:
            # Audit fix D2: return explicit "no data" instead of silently passing
            return FalsificationCheck(
                check_name="xray_activity",
                risk_category="data_gap",
                risk_level=0.0,
                explanation="No eROSITA data available — X-ray check NOT performed (data gap)",
                evidence={"data_available": False},
            )

        risk = 0.0
        evidence = {}
        explanation = "No X-ray counterpart in eRASS1 — consistent with quiet star"

        flux = erosita.get("flux_0p2_2p3")
        sep = erosita.get("sep_arcsec", 999)
        det_like = erosita.get("det_like")

        evidence["erosita_sep_arcsec"] = sep
        evidence["erosita_flux"] = flux
        evidence["erosita_det_like"] = det_like

        if flux is not None and flux > 0:
            if flux > 1e-12:
                # Very bright X-ray source — definitely active
                risk = 0.8
                explanation = (
                    f"Strong X-ray source at {sep:.0f}\" (flux={flux:.1e} erg/s/cm²) "
                    f"— highly active star (flare star, RS CVn, or YSO). "
                    f"Technosignatures do not produce coronal X-ray emission"
                )
            elif flux > 1e-13:
                # Moderately X-ray active
                risk = 0.6
                explanation = (
                    f"X-ray counterpart at {sep:.0f}\" (flux={flux:.1e} erg/s/cm²) "
                    f"— stellar coronal activity detected. Most likely an active "
                    f"star, weakening the case for artificial origin"
                )
            else:
                # Faint X-ray detection — mild concern
                risk = 0.2
                explanation = (
                    f"Faint X-ray source at {sep:.0f}\" (flux={flux:.1e} erg/s/cm²) "
                    f"— low-level activity, not strongly diagnostic"
                )

        return FalsificationCheck(
            check_name="xray_activity",
            risk_category="astrophysical",
            risk_level=risk,
            explanation=explanation,
            evidence=evidence,
        )

    def _check_dust_extinction(self, t: Dict) -> Optional[FalsificationCheck]:
        """Check 3D dust map for interstellar reddening along sightline.

        High E(B-V) means interstellar dust is significant — optical
        photometry may be biased, and IR excess interpretation requires
        care to distinguish circumstellar from interstellar reddening.
        """
        ra = t.get("ra")
        dec = t.get("dec")
        distance = t.get("distance_pc")

        if ra is None or dec is None or distance is None:
            # Audit fix D2: return explicit "no data" instead of silently passing
            return FalsificationCheck(
                check_name="dust_extinction",
                risk_category="data_gap",
                risk_level=0.0,
                explanation="No position/distance data — dust check NOT performed (data gap)",
                evidence={"data_available": False},
            )

        try:
            from src.vetting.dust_extinction import get_extinction_context
        except ImportError:
            return None

        ctx = get_extinction_context(ra, dec, distance)
        if not ctx.get("available", False):
            return None

        ebv = ctx.get("ebv", 0)
        if ebv is None:
            return None

        evidence = {"ebv": ebv, "source": "bayestar2019"}
        concern = ctx.get("concern_level", "NEGLIGIBLE")

        if concern == "HIGH":
            risk = 0.6
            explanation = (
                f"High interstellar extinction E(B-V)={ebv:.2f} — "
                f"optical photometry severely affected, blackbody Teff may be "
                f"underestimated. IR excess could be partly interstellar"
            )
            evidence["extinctions"] = ctx.get("extinctions", {})
        elif concern == "MODERATE":
            risk = 0.3
            explanation = (
                f"Moderate interstellar extinction E(B-V)={ebv:.3f} — "
                f"some optical bias expected, mid-IR excess still reliable"
            )
        elif concern == "LOW":
            risk = 0.1
            explanation = (
                f"Low interstellar extinction E(B-V)={ebv:.4f} — "
                f"no significant impact on photometry"
            )
        else:
            risk = 0.0
            explanation = (
                f"Negligible interstellar extinction E(B-V)={ebv:.4f} — "
                f"sightline is dust-free"
            )

        return FalsificationCheck(
            check_name="dust_extinction",
            risk_category="environmental",
            risk_level=risk,
            explanation=explanation,
            evidence=evidence,
        )

    def _check_sb9_binary(self, t: Dict) -> Optional[FalsificationCheck]:
        """Check SB9 Spectroscopic Binary catalog via VizieR.

        SB9 contains ~3,600 spectroscopic binary orbits.  A match means
        the target is a CONFIRMED binary — RUWE anomaly, PM anomaly, and
        photometric variability are all expected.
        """
        sb9 = t.get("sb9")
        if sb9 is None:
            # Audit fix D2: return explicit "no data" instead of silently passing
            return FalsificationCheck(
                check_name="sb9_binary",
                risk_category="data_gap",
                risk_level=0.0,
                explanation="No SB9 data available — binary check NOT performed (data gap)",
                evidence={"data_available": False},
            )

        risk = 0.0
        evidence = {}
        explanation = "No SB9 match — not a known spectroscopic binary"

        if sb9.get("match"):
            risk = 0.9
            period = sb9.get("period_days")
            sep = sb9.get("sep_arcsec")
            evidence["sb9_match"] = True
            evidence["sb9_sep_arcsec"] = sep
            if period:
                evidence["sb9_period_days"] = period

            explanation = (
                f"SB9 spectroscopic binary match at {sep:.1f}\" — "
                f"confirmed binary system"
            )
            if period:
                explanation += f" (P={period:.1f} d)"
            explanation += (
                ". RUWE, PM anomaly, and photometric variability "
                "are all expected for binaries"
            )

        return FalsificationCheck(
            check_name="sb9_binary",
            risk_category="astrophysical",
            risk_level=risk,
            explanation=explanation,
            evidence=evidence,
        )

    def _check_simbad_type(self, t: Dict) -> Optional[FalsificationCheck]:
        """Check SIMBAD object type for known astrophysical explanations.

        SIMBAD classification can immediately flag targets as YSO, CV,
        AGN, spectroscopic binary, etc. — all of which have known
        mechanisms for multi-channel anomalies (audit fix D1).
        """
        simbad = t.get("simbad")
        if simbad is None:
            return FalsificationCheck(
                check_name="simbad_type",
                risk_category="data_gap",
                risk_level=0.0,
                explanation="No SIMBAD data available — check NOT performed (data gap)",
                evidence={"data_available": False},
            )

        if not simbad.get("match"):
            return FalsificationCheck(
                check_name="simbad_type",
                risk_category="astrophysical",
                risk_level=0.0,
                explanation="No SIMBAD match within 5\" — not a catalogued object",
                evidence={"simbad_match": False},
            )

        risk = simbad.get("risk_level", 0.0)
        explanation = simbad.get("risk_explanation", "Unknown SIMBAD type")
        evidence = {
            "simbad_match": True,
            "main_id": simbad.get("main_id"),
            "otype": simbad.get("otype"),
            "sp_type": simbad.get("sp_type"),
            "sep_arcsec": simbad.get("sep_arcsec"),
        }
        rv = simbad.get("rvz_radvel")
        if rv is not None:
            evidence["radial_velocity_km_s"] = rv

        return FalsificationCheck(
            check_name="simbad_type",
            risk_category="astrophysical",
            risk_level=risk,
            explanation=explanation,
            evidence=evidence,
        )


# =====================================================================
#  Helpers
# =====================================================================

def _galactic_latitude(ra_deg: float, dec_deg: float) -> Optional[float]:
    """Compute galactic latitude from equatorial coordinates (J2000).

    Uses the standard spherical trigonometry transformation.
    Returns galactic latitude in degrees, or None on error.
    """
    try:
        ra = math.radians(ra_deg)
        dec = math.radians(dec_deg)

        # North Galactic Pole: RA=192.8595°, Dec=27.1284° (J2000)
        ra_ngp = math.radians(192.8595)
        dec_ngp = math.radians(27.1284)

        sin_b = (
            math.sin(dec) * math.sin(dec_ngp)
            + math.cos(dec) * math.cos(dec_ngp) * math.cos(ra - ra_ngp)
        )
        return math.degrees(math.asin(max(-1, min(1, sin_b))))
    except Exception:
        return None


# =====================================================================
#  CLI & demo
# =====================================================================

if __name__ == "__main__":
    import json

    print("=" * 70)
    print("  EXODUS Red-Team Falsification Engine — Demo")
    print("=" * 70)

    engine = RedTeamEngine()

    # --- Scenario 1: Known binary with multi-channel anomaly ---
    print("\n--- Scenario 1: Known binary (should be DEMOTED) ---")
    verdict = engine.evaluate({
        "target_id": "KNOWN_BINARY_TEST",
        "ra": 100.0,
        "dec": 20.0,
        "distance_pc": 15.0,
        "simbad_type": "SB*",
        "gaia_data": {
            "ruwe": 3.5,
            "phot_g_mean_mag": 8.5,
            "bp_rp": 1.2,
            "teff_gspphot": 5200,
            "logg_gspphot": 4.3,
        },
        "exodus_score": {
            "total_score": 1.8,
            "n_active_channels": 2,
            "channel_scores": {
                "proper_motion_anomaly": {
                    "score": 0.7, "is_active": True,
                    "details": {"ruwe": 3.5, "wise_gaia_pm": {}},
                },
                "ir_excess": {
                    "score": 0.4, "is_active": True,
                    "details": {"sigma_W3": 3.5, "sigma_W4": 2.0},
                },
                "transit_anomaly": {"score": 0.1, "is_active": False, "details": {}},
                "gaia_photometric_anomaly": {"score": 0.2, "is_active": False, "details": {}},
                "radio_anomaly": {"score": 0.0, "is_active": False, "details": {}},
            },
        },
        "unexplainability": {
            "classification": "EXPLAINED",
            "unexplainability_score": 0.05,
            "best_template": "binary_system",
        },
        "multi_messenger": {},
    })
    print(f"  {verdict.summary}")
    print(f"  Recommendation: {verdict.recommendation}")
    for c in verdict.checks:
        if c.risk_level > 0.3:
            print(f"    [{c.risk_level:.2f}] {c.check_name}: {c.explanation}")

    # --- Scenario 2: Genuinely weird multi-channel target ---
    print("\n--- Scenario 2: Unexplained multi-channel (should ESCALATE) ---")
    verdict = engine.evaluate({
        "target_id": "WEIRD_STAR_TEST",
        "ra": 45.0,
        "dec": -30.0,
        "distance_pc": 25.0,
        "gaia_data": {
            "ruwe": 5.2,
            "phot_g_mean_mag": 10.5,
            "bp_rp": 1.5,
            "teff_gspphot": 4800,
            "logg_gspphot": 4.5,
        },
        "exodus_score": {
            "total_score": 3.2,
            "n_active_channels": 3,
            "fdr_significant": True,
            "q_value": 0.005,
            "channel_scores": {
                "proper_motion_anomaly": {
                    "score": 0.6, "is_active": True,
                    "details": {
                        "ruwe": 5.2,
                        "wise_gaia_pm": {
                            "is_discrepant": True,
                            "pm_discrepancy_sigma": 6.2,
                            "chi2": 38.4,
                        },
                    },
                },
                "ir_excess": {
                    "score": 0.8, "is_active": True,
                    "details": {"sigma_W3": 8.0, "sigma_W4": 6.5},
                },
                "radio_anomaly": {
                    "score": 0.7, "is_active": True,
                    "details": {"n_candidates": 1, "max_snr": 12.0},
                },
                "transit_anomaly": {"score": 0.2, "is_active": False, "details": {}},
                "gaia_photometric_anomaly": {"score": 0.1, "is_active": False, "details": {}},
            },
        },
        "unexplainability": {
            "classification": "UNEXPLAINED",
            "unexplainability_score": 0.92,
            "best_template": "debris_disk",
            "residual_channels": ["radio_anomaly", "proper_motion_anomaly"],
        },
        "multi_messenger": {},
    })
    print(f"  {verdict.summary}")
    print(f"  Recommendation: {verdict.recommendation}")
    for c in verdict.checks:
        if c.risk_level > 0.0:
            print(f"    [{c.risk_level:.2f}] {c.check_name}: {c.explanation}")

    # --- Scenario 3: Galactic plane target with marginal IR ---
    print("\n--- Scenario 3: Galactic plane + marginal IR (should be cautious) ---")
    verdict = engine.evaluate({
        "target_id": "GALPLANE_TEST",
        "ra": 270.0,
        "dec": -29.0,  # near Galactic center
        "distance_pc": 50.0,
        "gaia_data": {
            "ruwe": 1.1,
            "phot_g_mean_mag": 12.0,
            "teff_gspphot": 5500,
            "logg_gspphot": 4.4,
        },
        "exodus_score": {
            "total_score": 0.8,
            "n_active_channels": 1,
            "channel_scores": {
                "ir_excess": {
                    "score": 0.5, "is_active": True,
                    "details": {"sigma_W3": 4.0, "sigma_W4": 2.5},
                },
                "proper_motion_anomaly": {"score": 0.1, "is_active": False, "details": {}},
                "transit_anomaly": {"score": 0.0, "is_active": False, "details": {}},
                "gaia_photometric_anomaly": {"score": 0.0, "is_active": False, "details": {}},
                "radio_anomaly": {"score": 0.0, "is_active": False, "details": {}},
            },
        },
        "unexplainability": {
            "classification": "EXPLAINED",
            "unexplainability_score": 0.05,
            "best_template": "background_contamination",
        },
        "multi_messenger": {},
    })
    print(f"  {verdict.summary}")
    print(f"  Recommendation: {verdict.recommendation}")
    for c in verdict.checks:
        if c.risk_level > 0.3:
            print(f"    [{c.risk_level:.2f}] {c.check_name}: {c.explanation}")

    print("\n" + "=" * 70)
    print("  Demo complete")
    print("=" * 70)
