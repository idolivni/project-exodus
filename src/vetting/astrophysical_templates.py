"""
Astrophysical Template Matching & Unexplainability Score
=======================================================

For each multi-channel EXODUS candidate, test whether the observed
anomaly pattern can be explained by known astrophysical phenomena.
The "Unexplainability Score" is the residual after subtracting the
best-fit astrophysical template.

This is EXODUS's single most important credibility mechanism:
If a candidate's anomaly pattern is well-explained by a binary star,
debris disk, or young stellar object, it is NOT a technosignature
candidate — regardless of its convergence score.

Templates
---------
1. **Binary/multiple system** — predicts RUWE anomaly + RV signal +
   possible IR excess from unresolved companion.

2. **Debris disk** — predicts mid-IR excess + possible transit dips
   (dust transits, cometary tails) but NOT astrometric anomaly.

3. **Young stellar object (YSO)** — predicts IR excess + photometric
   variability + sometimes X-ray and astrometric anomaly (disk
   accretion/outflow + gravitational perturbation by forming planets).

4. **Background contamination** — predicts IR excess (background
   galaxy in photometric aperture) + possible astrometric shift
   (centroid pulled by contaminant) but NOT transit anomaly.

5. **Active/flare star** — predicts photometric variability (spots,
   flares) + possible radio emission (coherent flare radio bursts)
   but NOT IR excess or transit anomaly.

6. **Instrumental/systematic** — predicts single-channel anomaly
   in any one channel (photometric pipeline artifact, bad epoch,
   catalogue cross-match error).

API
---
    from src.vetting.astrophysical_templates import UnexplainabilityScorer

    scorer = UnexplainabilityScorer()
    result = scorer.evaluate(candidate_channels)
    print(result.unexplainability_score)   # 0-1, higher = more unexplainable
    print(result.best_template)            # "binary" | "debris_disk" | etc.
    print(result.best_template_fit)        # 0-1, goodness of fit
    print(result.residual_channels)        # channels not explained by best template
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger

log = get_logger("vetting.astrophysical_templates")


# =====================================================================
#  Data classes
# =====================================================================

@dataclass
class TemplateMatch:
    """Result of matching a single astrophysical template to a candidate."""
    template_name: str
    template_description: str
    fit_score: float           # 0-1, how well this template explains the observation
    explained_channels: List[str]   # channels that this template would predict
    unexplained_channels: List[str] # active channels NOT predicted by this template
    channel_fits: Dict[str, float]  # per-channel fit quality (0-1)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_name": self.template_name,
            "template_description": self.template_description,
            "fit_score": round(self.fit_score, 4),
            "explained_channels": self.explained_channels,
            "unexplained_channels": self.unexplained_channels,
            "channel_fits": {k: round(v, 4) for k, v in self.channel_fits.items()},
            "notes": self.notes,
        }


@dataclass
class TemplateConflict:
    """A contradiction between a template's prediction and observed data."""
    template_name: str
    prediction: str       # what the template predicts
    observation: str      # what is actually observed
    severity: str         # "strong" | "moderate" | "weak"
    channel: str          # which channel or data source is involved

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_name": self.template_name,
            "prediction": self.prediction,
            "observation": self.observation,
            "severity": self.severity,
            "channel": self.channel,
        }


@dataclass
class UnexplainabilityResult:
    """Full unexplainability analysis for one candidate."""
    target_id: str

    # Core result
    unexplainability_score: float    # 0-1, higher = more genuinely anomalous
    classification: str               # "EXPLAINED" | "PARTIALLY_EXPLAINED" | "UNEXPLAINED"

    # Best template
    best_template: str
    best_template_fit: float          # 0-1
    best_template_match: TemplateMatch

    # All template results
    all_templates: List[TemplateMatch]

    # Residual analysis
    residual_channels: List[str]      # channels not explained by ANY template
    n_residual_channels: int
    n_active_channels: int
    active_channels: List[str]

    # Channel-level details
    channel_scores: Dict[str, float]  # input channel scores

    # Template conflict detection (Audit #5 F7)
    template_conflicts: List[TemplateConflict] = field(default_factory=list)
    has_template_conflict: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "target_id": self.target_id,
            "unexplainability_score": round(self.unexplainability_score, 4),
            "classification": self.classification,
            "best_template": self.best_template,
            "best_template_fit": round(self.best_template_fit, 4),
            "residual_channels": self.residual_channels,
            "n_residual_channels": self.n_residual_channels,
            "n_active_channels": self.n_active_channels,
            "active_channels": self.active_channels,
            "channel_scores": {k: round(v, 4) for k, v in self.channel_scores.items()},
            "all_templates": [t.to_dict() for t in self.all_templates],
        }
        if self.template_conflicts:
            d["template_conflicts"] = [c.to_dict() for c in self.template_conflicts]
            d["has_template_conflict"] = self.has_template_conflict
        return d

    def summary(self) -> str:
        return (
            f"{self.target_id}: unexplainability={self.unexplainability_score:.3f} "
            f"({self.classification}), best_template={self.best_template} "
            f"(fit={self.best_template_fit:.3f}), "
            f"residual_channels={self.residual_channels}"
        )


# =====================================================================
#  Astrophysical Template Definitions
# =====================================================================

# Each template defines:
#   - name: identifier
#   - description: what astrophysical scenario it models
#   - predicted_channels: channels this scenario would activate
#   - predicted_strength: typical activation strength per channel (0-1)
#   - correlation_pattern: expected correlations between channels
#     (e.g., binary: RUWE and IR tend to co-occur)
#   - distinguishing_features: what makes this scenario identifiable

TEMPLATES = {
    "binary_system": {
        "name": "binary_system",
        "description": (
            "Unresolved binary or multiple star system.  Gravitational "
            "interaction causes astrometric wobble (high RUWE), radial "
            "velocity variations, and sometimes IR excess from a faint "
            "companion.  Can also cause apparent transit-like events "
            "(grazing eclipses)."
        ),
        "predicted_channels": {
            "proper_motion_anomaly": 0.9,  # RUWE almost always elevated
            "ir_excess": 0.3,               # sometimes, from M-dwarf companion
            "transit_anomaly": 0.4,         # eclipsing binaries can mimic
            "gaia_photometric_anomaly": 0.5, # eclipsing → variability
        },
        "anti_predicted_channels": {
            "radio_anomaly": 0.0,            # binaries don't emit narrowband radio
        },
        # Correlation: if RUWE is high AND IR excess is moderate, binary is very likely
        "correlation_weight": {
            ("proper_motion_anomaly", "ir_excess"): 1.5,
            ("proper_motion_anomaly", "transit_anomaly"): 1.3,
        },
    },

    "debris_disk": {
        "name": "debris_disk",
        "description": (
            "Circumstellar debris disk (planetary system remnant).  Thermal "
            "emission from warm dust produces mid-IR excess.  Transiting "
            "dust clumps or cometary tails can cause irregular dimming.  "
            "Does NOT affect astrometry or produce radio emission."
        ),
        "predicted_channels": {
            "ir_excess": 0.9,                # primary signature
            "transit_anomaly": 0.4,          # dust transits
            "gaia_photometric_anomaly": 0.3, # possible variability from transiting dust
        },
        "anti_predicted_channels": {
            "proper_motion_anomaly": 0.0,    # no astrometric effect
            "radio_anomaly": 0.0,            # no radio emission
        },
        "correlation_weight": {
            ("ir_excess", "transit_anomaly"): 1.5,
        },
    },

    "young_stellar_object": {
        "name": "young_stellar_object",
        "description": (
            "Pre-main-sequence star with accretion disk and active outflows.  "
            "Produces strong IR excess (circumstellar disk), photometric "
            "variability (accretion bursts, disk obscuration), and "
            "astrometric anomaly (gravitational perturbation by forming "
            "planets, or disk asymmetry).  Multiple channels can be "
            "simultaneously active through a single astrophysical process.  "
            "Does NOT produce narrowband drifting radio signals."
        ),
        "predicted_channels": {
            "ir_excess": 0.9,                # strong disk emission
            "ir_variability": 0.7,           # accretion-driven IR changes
            "gaia_photometric_anomaly": 0.8, # accretion variability
            "proper_motion_anomaly": 0.5,    # disk/companion perturbation
            "transit_anomaly": 0.5,          # disk structure transits
            "uv_anomaly": 0.6,              # accretion produces UV excess
            "hr_anomaly": 0.5,               # PMS stars sit above MS
        },
        "anti_predicted_channels": {
            "radio_anomaly": 0.0,            # YSOs do NOT produce narrowband radio
        },
        "correlation_weight": {
            ("ir_excess", "gaia_photometric_anomaly"): 1.8,
            ("ir_excess", "proper_motion_anomaly"): 1.3,
            ("ir_excess", "transit_anomaly"): 1.3,
            ("ir_excess", "uv_anomaly"): 1.5,
        },
    },

    "background_contamination": {
        "name": "background_contamination",
        "description": (
            "Unresolved background source (galaxy, AGN, nebulosity) "
            "contaminating the photometric aperture.  Produces spurious "
            "IR excess (background galaxy SED) and can shift the "
            "astrometric centroid.  The Hephaistos Dyson sphere candidates "
            "(Suazo et al. 2024) were ALL caused by this phenomenon."
        ),
        "predicted_channels": {
            "ir_excess": 0.9,                # primary effect
            "proper_motion_anomaly": 0.4,    # centroid shift
        },
        "anti_predicted_channels": {
            "transit_anomaly": 0.0,          # contaminants don't transit
            "radio_anomaly": 0.0,            # AGN radio is diffuse, not narrowband
            "gaia_photometric_anomaly": 0.0, # contaminants are constant
        },
        "correlation_weight": {
            ("ir_excess", "proper_motion_anomaly"): 1.5,
        },
    },

    "active_flare_star": {
        "name": "active_flare_star",
        "description": (
            "Magnetically active star (M-dwarf, RS CVn, BY Dra).  "
            "Produces photometric variability from starspots and flares, "
            "and can emit coherent radio bursts.  Does NOT produce "
            "sustained IR excess or astrometric anomaly."
        ),
        "predicted_channels": {
            "gaia_photometric_anomaly": 0.9, # primary: spots + flares
            "radio_anomaly": 0.4,            # coherent radio bursts
            "radio_emission": 0.5,           # continuum from active corona
            "transit_anomaly": 0.3,          # spot crossings mimic transits
            "uv_anomaly": 0.6,              # chromospheric UV emission
        },
        "anti_predicted_channels": {
            "ir_excess": 0.0,                # no dust emission
            "proper_motion_anomaly": 0.0,    # no astrometric effect
        },
        "correlation_weight": {
            ("gaia_photometric_anomaly", "radio_anomaly"): 1.5,
            ("gaia_photometric_anomaly", "transit_anomaly"): 1.3,
            ("gaia_photometric_anomaly", "uv_anomaly"): 1.4,
        },
    },

    "instrumental_systematic": {
        "name": "instrumental_systematic",
        "description": (
            "Instrumental artifact or pipeline systematic affecting a "
            "single channel.  Bad CCD column, catalogue cross-match "
            "error, pipeline glitch.  Predicts anomaly in exactly ONE "
            "channel with no correlation to any other channel."
        ),
        "predicted_channels": {
            # Any single channel can be affected
            "ir_excess": 0.5,
            "transit_anomaly": 0.5,
            "radio_anomaly": 0.5,
            "gaia_photometric_anomaly": 0.5,
            "proper_motion_anomaly": 0.5,
        },
        "anti_predicted_channels": {},
        "correlation_weight": {},
        # Special flag: this template only fits if exactly 1 channel is active
        "_single_channel_only": True,
    },

    # Audit fix A7: RS CVn / chromospherically active binary template.
    # RS CVn systems are tidally-locked binaries with active chromospheres.
    # They predict: IR excess (circumbinary dust or companion),
    # PM anomaly (binary orbital motion), radio emission (gyrosynchrotron
    # from active corona), UV excess (chromospheric emission), and
    # photometric variability (starspots).
    # CRITICAL: RS CVn also predicts STRONG X-ray emission (Güdel-Benz
    # relation).  X-ray silence is a strong COUNTER-indicator.
    "rs_cvn_active_binary": {
        "name": "rs_cvn_active_binary",
        "description": (
            "RS CVn or chromospherically active binary system.  Tidal "
            "locking drives vigorous magnetic activity producing radio "
            "emission (gyrosynchrotron), UV excess (chromospheric), IR "
            "excess (circumbinary dust or cool companion), photometric "
            "variability (starspots), and astrometric wobble (binary orbit).  "
            "Critically, RS CVn systems obey the Güdel-Benz relation: "
            "radio-luminous RS CVn are ALWAYS X-ray-luminous.  X-ray "
            "silence with radio detection is a strong counter-indicator."
        ),
        "predicted_channels": {
            "ir_excess": 0.6,                # dust or companion
            "proper_motion_anomaly": 0.7,    # binary orbital wobble
            "radio_emission": 0.8,           # gyrosynchrotron, key signature
            "gaia_photometric_anomaly": 0.7, # starspot modulation
            "uv_anomaly": 0.5,              # chromospheric emission
        },
        "anti_predicted_channels": {
            "radio_anomaly": 0.0,            # no narrowband drifting signals
        },
        "correlation_weight": {
            ("radio_emission", "gaia_photometric_anomaly"): 1.6,
            ("radio_emission", "uv_anomaly"): 1.5,
            ("proper_motion_anomaly", "ir_excess"): 1.4,
            ("radio_emission", "proper_motion_anomaly"): 1.3,
        },
        # NOTE: Güdel-Benz X-ray prediction cannot be encoded in the
        # template system (no X-ray channel).  The red_team module's
        # guedel_benz_check handles this separately.  An RS CVn match
        # WITHOUT X-ray detection is suspicious and should be flagged.
    },

    # Audit fix D3: two new binary templates covering detection gaps
    "spectroscopic_binary": {
        "name": "spectroscopic_binary",
        "description": (
            "Spectroscopic binary (SB1/SB2) — detected via RV variations, "
            "not necessarily via astrometry.  Close binaries can produce "
            "IR excess from mass transfer or irradiation, HR anomaly from "
            "composite spectrum, and transit-like eclipses.  May NOT "
            "show elevated RUWE if orbital period is long."
        ),
        "predicted_channels": {
            "ir_excess": 0.6,                # companion or mass transfer
            "proper_motion_anomaly": 0.5,    # RUWE if period < few years
            "hr_anomaly": 0.5,               # composite spectrum shifts HR position
            "transit_anomaly": 0.3,          # eclipsing subset
            "gaia_photometric_anomaly": 0.4, # ellipsoidal variations
        },
        "anti_predicted_channels": {
            "radio_anomaly": 0.0,            # no narrowband radio
        },
        "correlation_weight": {
            ("ir_excess", "hr_anomaly"): 1.5,
            ("proper_motion_anomaly", "hr_anomaly"): 1.4,
        },
    },

    "face_on_binary": {
        "name": "face_on_binary",
        "description": (
            "Binary system seen nearly face-on (inclination ~ 0 deg).  "
            "Orbital motion is primarily in the plane of the sky, "
            "so RUWE stays low and no eclipses occur.  However, the "
            "companion still contributes to composite IR excess and "
            "photometric variability.  This is the HARDEST binary "
            "configuration to detect — specifically the one that "
            "survives RUWE < 1.2 cuts."
        ),
        "predicted_channels": {
            "ir_excess": 0.7,                # companion contributes
            "gaia_photometric_anomaly": 0.3, # reflection/ellipsoidal
            "hr_anomaly": 0.4,               # composite spectrum
        },
        "anti_predicted_channels": {
            "proper_motion_anomaly": 0.0,    # face-on → low RUWE
            "transit_anomaly": 0.0,          # no eclipses
        },
        "correlation_weight": {
            ("ir_excess", "gaia_photometric_anomaly"): 1.3,
        },
    },
}


# =====================================================================
#  Unexplainability Scorer
# =====================================================================

class UnexplainabilityScorer:
    """Evaluate how unexplainable a candidate's anomaly pattern is.

    For each multi-channel candidate, tests the observed anomaly
    pattern against a library of known astrophysical templates.
    The "unexplainability score" is the fraction of the anomaly
    that no known template can account for.

    Parameters
    ----------
    activation_threshold : float
        Channel score threshold for considering a channel "active"
        (same as EXODUSScorer threshold, default 0.3).
    templates : dict, optional
        Custom template definitions.  Defaults to built-in library.
    """

    # Audit fix D3: expanded from 5 to 10 channels (all detection channels)
    DETECTION_CHANNELS = [
        "ir_excess",
        "ir_variability",
        "transit_anomaly",
        "radio_anomaly",
        "radio_emission",
        "gaia_photometric_anomaly",
        "proper_motion_anomaly",
        "uv_anomaly",
        "hr_anomaly",
        "abundance_anomaly",
    ]

    def __init__(
        self,
        activation_threshold: float = 0.3,
        templates: Optional[Dict[str, Dict]] = None,
    ):
        self.activation_threshold = activation_threshold
        self.templates = templates or TEMPLATES

    def evaluate(
        self,
        target_id: str,
        channel_scores: Dict[str, float],
        channel_details: Optional[Dict[str, Dict]] = None,
    ) -> UnexplainabilityResult:
        """Evaluate unexplainability of a candidate's anomaly pattern.

        Parameters
        ----------
        target_id : str
            Target identifier.
        channel_scores : dict
            Channel name → score (0-1).  Only detection channels are
            used (HZ is excluded).
        channel_details : dict, optional
            Additional per-channel details for refined matching.
            E.g., {"ir_excess": {"sigma_W3": 5.0, "excess_W3": -0.5}}.

        Returns
        -------
        UnexplainabilityResult
            Complete analysis including best template, residuals,
            and the final unexplainability score.
        """
        if channel_details is None:
            channel_details = {}

        # Identify active channels
        # Audit fix S23-F1: use strict '>' to match exodus_score.py line 342.
        # Previously used '>=', creating inconsistency where a channel at exactly
        # the threshold was active in template matching but inactive in scoring.
        active_channels = []
        for ch in self.DETECTION_CHANNELS:
            score = channel_scores.get(ch, 0.0)
            if score > self.activation_threshold:
                active_channels.append(ch)

        # Special case: 0 or 1 active channels
        if len(active_channels) == 0:
            return self._trivial_result(
                target_id, channel_scores, active_channels,
                classification="EXPLAINED",
                notes="No active channels — nothing to explain",
            )

        if len(active_channels) == 1:
            # Single channel is almost always explainable
            return self._single_channel_result(
                target_id, channel_scores, active_channels,
                channel_details,
            )

        # Multi-channel: test against each template
        template_matches = []
        for tname, tdef in self.templates.items():
            match = self._match_template(
                tname, tdef, active_channels, channel_scores, channel_details,
            )
            template_matches.append(match)

        # Sort by fit score (best explanation first)
        template_matches.sort(key=lambda m: -m.fit_score)
        best = template_matches[0]

        # Compute residual channels: channels not explained by ANY template
        all_explained = set()
        for m in template_matches:
            if m.fit_score > 0.3:  # only count templates with decent fit
                all_explained.update(m.explained_channels)

        residual_channels = [
            ch for ch in active_channels if ch not in all_explained
        ]

        # Compute base unexplainability score
        unexplainability = self._compute_unexplainability(
            active_channels, channel_scores, best, residual_channels,
        )

        # --- Template conflict detection (Audit #5 F7 + C3 fix) ---
        # Check whether the best template's predictions are contradicted
        # by actual observations.  If so, the template fit is unreliable
        # and unexplainability should be boosted.
        #
        # Audit fix C3: also check alternative templates.  If the 2nd/3rd
        # best template has a reasonable fit (>= 70% of best) AND fewer
        # conflicts, prefer it for the conflict assessment.  This prevents
        # a poorly-fitting best template from artificially boosting
        # unexplainability when a better-matching template exists.
        best_tdef = self.templates.get(best.template_name, {})
        conflicts = self._detect_template_conflicts(
            best, best_tdef, active_channels, channel_scores,
            channel_details,
        )

        # Check top alternatives for fewer conflicts
        fit_threshold = best.fit_score * 0.70  # must be within 70% of best
        for alt_match in template_matches[1:4]:  # check up to 3 alternatives
            if alt_match.fit_score < fit_threshold:
                break
            alt_tdef = self.templates.get(alt_match.template_name, {})
            alt_conflicts = self._detect_template_conflicts(
                alt_match, alt_tdef, active_channels, channel_scores,
                channel_details,
            )
            if len(alt_conflicts) < len(conflicts):
                log.debug(
                    "%s: alternative template '%s' (fit=%.2f) has fewer "
                    "conflicts (%d vs %d) than best '%s' (fit=%.2f)",
                    target_id, alt_match.template_name, alt_match.fit_score,
                    len(alt_conflicts), len(conflicts),
                    best.template_name, best.fit_score,
                )
                conflicts = alt_conflicts

        has_conflict = len(conflicts) > 0

        if conflicts:
            n_strong = sum(1 for c in conflicts if c.severity == "strong")
            n_moderate = sum(1 for c in conflicts if c.severity == "moderate")

            # Boost unexplainability proportional to conflict severity
            conflict_boost = 0.15 * n_strong + 0.08 * n_moderate
            unexplainability = min(1.0, unexplainability + conflict_boost)

            conflict_desc = ", ".join(
                f"{c.channel}({c.severity})" for c in conflicts
            )
            log.warning(
                "%s: template '%s' has %d conflict(s): %s. "
                "Unexplainability boosted by %.2f",
                target_id, best.template_name,
                len(conflicts), conflict_desc, conflict_boost,
            )

        # Classification
        if unexplainability < 0.2:
            classification = "EXPLAINED"
        elif unexplainability < 0.5:
            classification = "PARTIALLY_EXPLAINED"
        else:
            classification = "UNEXPLAINED"

        result = UnexplainabilityResult(
            target_id=target_id,
            unexplainability_score=unexplainability,
            classification=classification,
            best_template=best.template_name,
            best_template_fit=best.fit_score,
            best_template_match=best,
            all_templates=template_matches,
            residual_channels=residual_channels,
            n_residual_channels=len(residual_channels),
            n_active_channels=len(active_channels),
            active_channels=active_channels,
            channel_scores=dict(channel_scores),
            template_conflicts=conflicts,
            has_template_conflict=has_conflict,
        )

        conflict_str = f", conflicts={len(conflicts)}" if conflicts else ""
        log.info(
            "%s: unexplainability=%.3f (%s), best=%s (fit=%.3f), "
            "residual=%s%s",
            target_id, unexplainability, classification,
            best.template_name, best.fit_score,
            residual_channels, conflict_str,
        )

        return result

    def evaluate_from_exodus_score(
        self,
        exodus_score_dict: Dict[str, Any],
    ) -> UnexplainabilityResult:
        """Evaluate from an EXODUSScore.to_dict() output.

        Convenience method that extracts channel scores from the
        standard EXODUS score dictionary format.
        """
        target_id = exodus_score_dict.get("target_id", "unknown")

        channel_scores = {}
        channel_details = {}
        channels = exodus_score_dict.get("channel_scores", {})

        for ch_name, ch_data in channels.items():
            if isinstance(ch_data, dict):
                channel_scores[ch_name] = ch_data.get("score", 0.0)
                channel_details[ch_name] = ch_data.get("details", {})
            elif hasattr(ch_data, "score"):
                channel_scores[ch_name] = ch_data.score
                channel_details[ch_name] = ch_data.details if hasattr(ch_data, "details") else {}

        return self.evaluate(target_id, channel_scores, channel_details)

    def batch_evaluate(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[UnexplainabilityResult]:
        """Evaluate a batch of candidates from EXODUS score dicts.

        Returns results sorted by unexplainability (most unexplainable first).
        """
        results = []
        for c in candidates:
            try:
                result = self.evaluate_from_exodus_score(c)
                results.append(result)
            except Exception as exc:
                log.warning(
                    "Failed to evaluate %s: %s",
                    c.get("target_id", "unknown"), exc,
                )

        results.sort(key=lambda r: -r.unexplainability_score)
        return results

    # =================================================================
    #  Template matching
    # =================================================================

    def _match_template(
        self,
        template_name: str,
        template_def: Dict[str, Any],
        active_channels: List[str],
        channel_scores: Dict[str, float],
        channel_details: Dict[str, Dict],
    ) -> TemplateMatch:
        """Match a single astrophysical template against observed channels.

        Returns a TemplateMatch with fit_score indicating how well
        the template explains the observed pattern.
        """
        predicted = template_def.get("predicted_channels", {})
        anti_predicted = template_def.get("anti_predicted_channels", {})
        correlations = template_def.get("correlation_weight", {})
        single_only = template_def.get("_single_channel_only", False)

        # Special handling for instrumental template
        if single_only and len(active_channels) > 1:
            return TemplateMatch(
                template_name=template_name,
                template_description=template_def["description"],
                fit_score=0.0,
                explained_channels=[],
                unexplained_channels=list(active_channels),
                channel_fits={},
                notes="Instrumental template requires single active channel",
            )

        explained = []
        unexplained = []
        channel_fits = {}

        # Score each active channel against the template
        for ch in active_channels:
            pred_strength = predicted.get(ch, 0.0)

            if pred_strength > 0.0:
                # This template predicts activity in this channel
                obs_score = channel_scores.get(ch, 0.0)

                # Fit quality: how closely does the observed score match
                # the expected strength?
                # Perfect fit = observed is near or below predicted
                # Over-strong = observed >> predicted (still explained but
                #   the template under-predicts)
                if obs_score <= pred_strength:
                    fit = 1.0
                else:
                    # Penalty for over-strength (template under-predicts)
                    ratio = obs_score / (pred_strength + 0.01)
                    fit = max(0.0, 1.0 - 0.3 * (ratio - 1.0))

                channel_fits[ch] = fit
                if fit > 0.3:
                    explained.append(ch)
                else:
                    unexplained.append(ch)
            else:
                # Template does NOT predict activity in this channel
                unexplained.append(ch)
                channel_fits[ch] = 0.0

        # Missing-prediction penalty (audit fix A7b): if a template predicts
        # a channel with high strength but that channel is NOT active, the
        # template is less likely correct.  E.g., RS CVn predicts strong
        # photometric variability (0.7) but the target is photometrically
        # constant → template fit should be reduced.
        missing_penalty = 1.0
        for ch, pred_str in predicted.items():
            if ch not in active_channels and pred_str >= 0.5:
                # Strong prediction (>= 0.5) that's not observed
                # Penalty scales with prediction strength
                missing_penalty *= (1.0 - 0.4 * pred_str)
        missing_penalty = max(0.2, missing_penalty)

        # Anti-prediction penalty: if an anti-predicted channel IS active,
        # reduce the fit (this template says this channel should NOT fire)
        anti_penalty = 1.0
        for ch, anti_strength in anti_predicted.items():
            if ch in active_channels and anti_strength == 0.0:
                obs_score = channel_scores.get(ch, 0.0)
                anti_penalty *= max(0.1, 1.0 - obs_score)

        # Correlation bonus: if two correlated channels are both active,
        # this template is more likely correct
        corr_bonus = 1.0
        for (ch_a, ch_b), weight in correlations.items():
            if ch_a in active_channels and ch_b in active_channels:
                corr_bonus *= weight

        # Compute overall fit score
        if not channel_fits:
            fit_score = 0.0
        else:
            # Base fit: weighted average of per-channel fits
            weights = [predicted.get(ch, 0.1) for ch in channel_fits]
            fits = list(channel_fits.values())
            base_fit = np.average(fits, weights=weights) if weights else 0.0

            # Fraction of active channels explained
            coverage = len(explained) / max(len(active_channels), 1)

            # Combined score
            # Apply missing_penalty AFTER clip so correlation bonus can't
            # absorb it: predicted-but-absent channels always reduce fit.
            fit_score = float(
                base_fit * coverage * anti_penalty
                * min(corr_bonus, 3.0)
            )
            fit_score = np.clip(fit_score, 0.0, 1.0)
            fit_score *= missing_penalty

        # Generate notes
        notes_parts = []
        if explained:
            notes_parts.append(f"Explains: {', '.join(explained)}")
        if unexplained:
            notes_parts.append(f"Cannot explain: {', '.join(unexplained)}")
        if corr_bonus > 1.0:
            notes_parts.append(f"Correlation bonus: {corr_bonus:.2f}x")
        if anti_penalty < 1.0:
            notes_parts.append(f"Anti-prediction penalty: {anti_penalty:.2f}x")
        if missing_penalty < 1.0:
            missing_chs = [ch for ch, ps in predicted.items()
                           if ch not in active_channels and ps >= 0.5]
            notes_parts.append(
                f"Missing predictions ({', '.join(missing_chs)}): "
                f"{missing_penalty:.2f}x"
            )

        return TemplateMatch(
            template_name=template_name,
            template_description=template_def["description"],
            fit_score=float(fit_score),
            explained_channels=explained,
            unexplained_channels=unexplained,
            channel_fits=channel_fits,
            notes="; ".join(notes_parts),
        )

    # =================================================================
    #  Template conflict detection (Audit #5 F7)
    # =================================================================

    def _detect_template_conflicts(
        self,
        best_template: TemplateMatch,
        template_def: Dict[str, Any],
        active_channels: List[str],
        channel_scores: Dict[str, float],
        channel_details: Dict[str, Dict],
    ) -> List[TemplateConflict]:
        """Detect contradictions between the best template's predictions
        and observed auxiliary data.

        For example, a binary template predicts elevated RUWE, but if
        RUWE=1.06 the prediction is contradicted.  A YSO template predicts
        photometric variability, but if ZTF shows 7 years of constant
        photometry the prediction is contradicted.

        Returns a list of TemplateConflict objects (may be empty).
        """
        conflicts: List[TemplateConflict] = []
        predicted = template_def.get("predicted_channels", {})
        tname = best_template.template_name

        # --- Binary/spectroscopic_binary/face_on_binary: RUWE contradiction ---
        # Binary templates predict proper_motion_anomaly ≥ 0.5, meaning
        # RUWE should be elevated.  If RUWE < 1.4 and AEN_sig < 5, the
        # prediction fails (star is astrometrically clean).
        if tname in ("binary_system", "spectroscopic_binary"):
            pm_predicted = predicted.get("proper_motion_anomaly", 0.0)
            pm_score = channel_scores.get("proper_motion_anomaly", 0.0)
            pm_details = channel_details.get("proper_motion_anomaly", {})
            ruwe = pm_details.get("ruwe", pm_details.get("RUWE"))
            aen_sig = pm_details.get("astrometric_excess_noise_sig")

            if pm_predicted >= 0.5 and pm_score < self.activation_threshold:
                # Template strongly predicts PM anomaly but channel is inactive
                note = f"RUWE={ruwe:.2f}" if ruwe is not None else "PM channel inactive"
                conflicts.append(TemplateConflict(
                    template_name=tname,
                    prediction=f"Elevated RUWE/astrometric anomaly (predicted strength {pm_predicted:.1f})",
                    observation=f"PM channel inactive (score={pm_score:.2f}). {note}",
                    severity="strong",
                    channel="proper_motion_anomaly",
                ))

        # --- Face-on binary: specifically anti-predicts PM anomaly ---
        # If face_on_binary is the best template BUT proper_motion_anomaly IS
        # active, that contradicts the face-on geometry.
        if tname == "face_on_binary":
            pm_anti = template_def.get("anti_predicted_channels", {}).get("proper_motion_anomaly", -1)
            pm_score = channel_scores.get("proper_motion_anomaly", 0.0)
            if pm_anti == 0.0 and pm_score >= self.activation_threshold:
                conflicts.append(TemplateConflict(
                    template_name=tname,
                    prediction="Face-on binary: RUWE should be LOW (no radial motion detected)",
                    observation=f"PM anomaly IS active (score={pm_score:.2f}) — contradicts face-on geometry",
                    severity="strong",
                    channel="proper_motion_anomaly",
                ))

        # --- YSO: variability contradiction ---
        # YSO template predicts gaia_photometric_anomaly ≥ 0.8 (accretion
        # variability).  If photometric channels show constant lightcurve
        # (e.g., ZTF chi2_red < 2.0), the prediction fails.
        if tname == "young_stellar_object":
            phot_predicted = predicted.get("gaia_photometric_anomaly", 0.0)
            phot_score = channel_scores.get("gaia_photometric_anomaly", 0.0)
            phot_details = channel_details.get("gaia_photometric_anomaly", {})
            chi2_red = phot_details.get("chi2_red")
            phot_variable = phot_details.get("phot_variable_flag")

            if phot_predicted >= 0.5 and phot_score < self.activation_threshold:
                obs_note = f"chi2_red={chi2_red:.1f}" if chi2_red is not None else "photometry channel inactive"
                if phot_variable == "NOT_AVAILABLE":
                    obs_note += ", Gaia: NOT_AVAILABLE"
                conflicts.append(TemplateConflict(
                    template_name=tname,
                    prediction=f"Photometric variability from accretion (predicted strength {phot_predicted:.1f})",
                    observation=f"Photometry channel inactive (score={phot_score:.2f}). {obs_note}",
                    severity="strong",
                    channel="gaia_photometric_anomaly",
                ))

            # Also check IR variability for YSO
            irv_predicted = predicted.get("ir_variability", 0.0)
            irv_score = channel_scores.get("ir_variability", 0.0)
            if irv_predicted >= 0.5 and irv_score < self.activation_threshold:
                conflicts.append(TemplateConflict(
                    template_name=tname,
                    prediction=f"IR variability from accretion (predicted strength {irv_predicted:.1f})",
                    observation=f"IR variability channel inactive (score={irv_score:.2f})",
                    severity="moderate",
                    channel="ir_variability",
                ))

        # --- Active flare star: X-ray contradiction ---
        # Active stars should be X-ray bright (Güdel-Benz relation).
        # If radio_emission is active but no X-ray detection, that
        # contradicts the flare star template.
        if tname == "active_flare_star":
            radio_score = channel_scores.get("radio_emission", 0.0)
            radio_details = channel_details.get("radio_emission", {})
            xray_detected = radio_details.get("xray_detected", None)

            if radio_score >= self.activation_threshold and xray_detected is False:
                conflicts.append(TemplateConflict(
                    template_name=tname,
                    prediction="Active/flare star: radio + X-ray correlated (Güdel-Benz)",
                    observation="Radio emission detected but no X-ray counterpart — violates Güdel-Benz",
                    severity="strong",
                    channel="radio_emission",
                ))

        # --- General: template predicts a channel that is anti-observed ---
        # If a template strongly predicts a channel (≥0.5) but that channel
        # has data and is not just inactive but has score ≈ 0, that's a
        # contradiction (the star was checked and found clean).
        for ch, pred_strength in predicted.items():
            if pred_strength < 0.5:
                continue
            ch_score = channel_scores.get(ch, -1)
            if ch_score < 0:
                continue  # no data for this channel
            if ch_score < 0.1 and ch not in active_channels:
                # Channel has data, template predicts it, but it's near-zero
                # Only flag if we haven't already flagged this channel above
                already_flagged = any(c.channel == ch for c in conflicts)
                if not already_flagged:
                    conflicts.append(TemplateConflict(
                        template_name=tname,
                        prediction=f"Channel {ch} should be active (predicted strength {pred_strength:.1f})",
                        observation=f"Channel has data but score={ch_score:.2f} (near zero)",
                        severity="moderate",
                        channel=ch,
                    ))

        return conflicts

    # =================================================================
    #  Unexplainability computation
    # =================================================================

    def _compute_unexplainability(
        self,
        active_channels: List[str],
        channel_scores: Dict[str, float],
        best_template: TemplateMatch,
        residual_channels: List[str],
    ) -> float:
        """Compute the final unexplainability score.

        The score combines:
        1. How much of the anomaly pattern the best template leaves unexplained
        2. The signal strength of the unexplained channels
        3. Bonus for channels that are anti-predicted by the best template

        Score = 0.0 means fully explained by known astrophysics.
        Score = 1.0 means no known astrophysical model fits at all.
        """
        n_active = len(active_channels)
        if n_active == 0:
            return 0.0

        # Component 1: Fraction of channels unexplained
        n_residual = len(residual_channels)
        fraction_unexplained = n_residual / n_active

        # Component 2: Signal strength of residual channels
        # (Weighted by how anomalous the unexplained channels are)
        if residual_channels:
            residual_strength = np.mean([
                channel_scores.get(ch, 0.0) for ch in residual_channels
            ])
        else:
            residual_strength = 0.0

        # Component 3: Inverse of best template fit
        # (Poor template fit → more unexplainable)
        template_residual = 1.0 - best_template.fit_score

        # Component 4: Multi-channel bonus
        # If 3+ channels are SIMULTANEOUSLY unexplained, that's very significant
        multi_bonus = 1.0
        if n_residual >= 3:
            multi_bonus = 1.5
        elif n_residual >= 2:
            multi_bonus = 1.2

        # Weighted combination
        unexplainability = (
            0.35 * fraction_unexplained +
            0.25 * residual_strength +
            0.25 * template_residual +
            0.15 * (n_residual / len(self.DETECTION_CHANNELS))  # audit fix C4: was /5.0, should be /10 (all detection channels)
        ) * multi_bonus

        return float(np.clip(unexplainability, 0.0, 1.0))

    # =================================================================
    #  Special cases
    # =================================================================

    def _trivial_result(
        self,
        target_id: str,
        channel_scores: Dict[str, float],
        active_channels: List[str],
        classification: str,
        notes: str,
    ) -> UnexplainabilityResult:
        """Return a trivial result for 0-channel cases."""
        trivial_match = TemplateMatch(
            template_name="none",
            template_description="No template needed",
            fit_score=1.0,
            explained_channels=[],
            unexplained_channels=[],
            channel_fits={},
            notes=notes,
        )
        return UnexplainabilityResult(
            target_id=target_id,
            unexplainability_score=0.0,
            classification=classification,
            best_template="none",
            best_template_fit=1.0,
            best_template_match=trivial_match,
            all_templates=[trivial_match],
            residual_channels=[],
            n_residual_channels=0,
            n_active_channels=len(active_channels),
            active_channels=active_channels,
            channel_scores=dict(channel_scores),
        )

    def _single_channel_result(
        self,
        target_id: str,
        channel_scores: Dict[str, float],
        active_channels: List[str],
        channel_details: Dict[str, Dict],
    ) -> UnexplainabilityResult:
        """Handle single-channel active cases.

        A single active channel is almost always explainable by either
        a specific astrophysical template or the instrumental template.
        The unexplainability score is low unless the signal is
        extraordinarily strong.
        """
        ch = active_channels[0]
        score = channel_scores.get(ch, 0.0)

        # Find the best-matching template for this single channel
        best_fit = 0.0
        best_template = "instrumental_systematic"

        for tname, tdef in self.templates.items():
            if tdef.get("_single_channel_only"):
                continue
            pred = tdef.get("predicted_channels", {})
            if ch in pred and pred[ch] > 0.3:
                fit = pred[ch]
                if fit > best_fit:
                    best_fit = fit
                    best_template = tname

        # Single-channel unexplainability is capped at 0.3
        # (single-channel anomalies are too common to be interesting)
        unexplainability = min(0.3, 0.1 * score)

        template_match = TemplateMatch(
            template_name=best_template,
            template_description=(
                self.templates.get(best_template, {}).get(
                    "description", "Single-channel anomaly"
                )
            ),
            fit_score=max(best_fit, 0.5),
            explained_channels=active_channels,
            unexplained_channels=[],
            channel_fits={ch: max(best_fit, 0.5)},
            notes=f"Single-channel ({ch}) — likely {best_template.replace('_', ' ')}",
        )

        return UnexplainabilityResult(
            target_id=target_id,
            unexplainability_score=unexplainability,
            classification="EXPLAINED",
            best_template=best_template,
            best_template_fit=max(best_fit, 0.5),
            best_template_match=template_match,
            all_templates=[template_match],
            residual_channels=[],
            n_residual_channels=0,
            n_active_channels=1,
            active_channels=active_channels,
            channel_scores=dict(channel_scores),
        )


# =====================================================================
#  Convenience functions
# =====================================================================

def compute_unexplainability(
    target_id: str,
    channel_scores: Dict[str, float],
    channel_details: Optional[Dict[str, Dict]] = None,
) -> UnexplainabilityResult:
    """One-call convenience function.

    Parameters
    ----------
    target_id : str
        Target identifier.
    channel_scores : dict
        Channel name → score (0-1).
    channel_details : dict, optional
        Additional per-channel info.

    Returns
    -------
    UnexplainabilityResult
    """
    scorer = UnexplainabilityScorer()
    return scorer.evaluate(target_id, channel_scores, channel_details)


# =====================================================================
#  CLI & demo
# =====================================================================

if __name__ == "__main__":
    import json

    print("=" * 70)
    print("  EXODUS Unexplainability Score — Demo")
    print("=" * 70)

    scorer = UnexplainabilityScorer()

    # --- Scenario 1: Binary star (should be EXPLAINED) ---
    print("\n--- Scenario 1: Binary star pattern ---")
    result = scorer.evaluate(
        "BINARY_TEST",
        channel_scores={
            "proper_motion_anomaly": 0.8,   # high RUWE
            "ir_excess": 0.4,                # moderate (M-dwarf companion)
            "transit_anomaly": 0.0,
            "gaia_photometric_anomaly": 0.35, # eclipsing
            "radio_anomaly": 0.0,
        },
    )
    print(result.summary())

    # --- Scenario 2: Debris disk (should be EXPLAINED) ---
    print("\n--- Scenario 2: Debris disk pattern ---")
    result = scorer.evaluate(
        "DISK_TEST",
        channel_scores={
            "ir_excess": 0.9,                # strong
            "transit_anomaly": 0.35,          # dust transit
            "proper_motion_anomaly": 0.0,
            "gaia_photometric_anomaly": 0.0,
            "radio_anomaly": 0.0,
        },
    )
    print(result.summary())

    # --- Scenario 3: Genuinely weird (should be UNEXPLAINED) ---
    print("\n--- Scenario 3: Multi-channel anomaly (no template fits) ---")
    result = scorer.evaluate(
        "WEIRD_TEST",
        channel_scores={
            "ir_excess": 0.7,                # strong IR excess
            "proper_motion_anomaly": 0.6,    # astrometric wobble
            "radio_anomaly": 0.8,            # narrowband radio ← nothing natural does this
            "transit_anomaly": 0.5,          # asymmetric transits
            "gaia_photometric_anomaly": 0.4, # variability
        },
    )
    print(result.summary())
    print("\n  All templates:")
    for t in result.all_templates:
        print(f"    {t.template_name:30s} fit={t.fit_score:.3f}  "
              f"explains={t.explained_channels}")

    # --- Scenario 4: YSO (should be EXPLAINED) ---
    print("\n--- Scenario 4: Young stellar object ---")
    result = scorer.evaluate(
        "YSO_TEST",
        channel_scores={
            "ir_excess": 0.9,
            "gaia_photometric_anomaly": 0.8,
            "proper_motion_anomaly": 0.4,
            "transit_anomaly": 0.35,
            "radio_anomaly": 0.0,
        },
    )
    print(result.summary())

    # --- Scenario 5: Single-channel noise (should be EXPLAINED) ---
    print("\n--- Scenario 5: Single channel only ---")
    result = scorer.evaluate(
        "SINGLE_TEST",
        channel_scores={
            "ir_excess": 0.6,
            "proper_motion_anomaly": 0.0,
            "transit_anomaly": 0.0,
            "gaia_photometric_anomaly": 0.0,
            "radio_anomaly": 0.0,
        },
    )
    print(result.summary())

    # --- Scenario 6: Technosignature-like (IR + radio, no astrometric) ---
    print("\n--- Scenario 6: IR + radio (no natural model explains both) ---")
    result = scorer.evaluate(
        "TECHNOSIG_TEST",
        channel_scores={
            "ir_excess": 0.8,
            "radio_anomaly": 0.7,
            "proper_motion_anomaly": 0.0,
            "transit_anomaly": 0.0,
            "gaia_photometric_anomaly": 0.0,
        },
    )
    print(result.summary())
    print(f"\n  Key insight: radio_anomaly is anti-predicted by debris_disk")
    print(f"  and binary templates → drives unexplainability UP")
