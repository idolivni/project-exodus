"""
Multi-modal convergence EXODUS Score for technosignature candidate ranking.

The EXODUS Score quantifies how many independent observational channels
converge on a single stellar target as anomalous.  A star that is flagged
by only one detector could easily be a natural outlier or an instrumental
artefact; a star flagged simultaneously in IR excess, transit shape,
radio emission, astrometric jitter, and epoch photometry variability is
far more likely to harbour genuine anomalous activity -- natural *or*
artificial.

Scoring formula
---------------
For each target we evaluate ten independent detection channels plus one
prior, each returning a normalised score in [0, 1].  Channels scoring
above a configurable threshold (default 0.3) are deemed "active".  The
final EXODUS Score is the geometric mean of the active channel scores
multiplied by an exponential convergence bonus:

    geo_mean = exp( mean( log(active_scores) ) )
    convergence_bonus = 4 ** (n_active - 1)
    EXODUS_score = geo_mean * convergence_bonus * coverage_penalty

This rewards multi-channel convergence superlinearly: a target with
three active channels at moderate significance will out-rank a target
with a single channel at very high significance.

Detection Channels
------------------
1. ir_excess              -- mid-IR excess above photospheric prediction
2. transit_anomaly        -- asymmetric / variable-depth transit events
3. radio_anomaly          -- narrowband Doppler-drifting radio signals
4. gaia_photometric_anomaly -- epoch photometry variability from Gaia DR3
5. proper_motion_anomaly  -- astrometric excess noise / CatWISE-Gaia PM discrepancy
6. ir_variability         -- NEOWISE secular IR brightening / excess scatter over 10+ years
7. uv_anomaly             -- GALEX FUV/NUV excess or deficit
8. radio_emission         -- FIRST/NVSS/VLASS radio continuum detection
9. hr_anomaly             -- HR diagram position anomaly
10. abundance_anomaly     -- APOGEE/GALAH chemical peculiarities

Prior (not counted as detection)
---------------------------------
+  habitable_zone_planet  -- confirmed HZ planet present (boost, not detection)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Project utilities
# ---------------------------------------------------------------------------
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))

from src.utils import get_logger, get_config, save_result, PROJECT_ROOT

log = get_logger("scoring.exodus_score")


# ============================================================================
#  Data classes
# ============================================================================

@dataclass
class ChannelScore:
    """Score from a single observational channel for one target.

    Attributes
    ----------
    channel_name : str
        Identifier for the channel (e.g. ``"ir_excess"``).
    score : float
        Normalised score in [0, 1], where 0 means no anomaly and 1 means
        maximally anomalous.
    is_active : bool
        Whether the score exceeds the activation threshold.
    calibrated_p : float or None
        Calibrated p-value from matched control distribution.
        None if no controls were provided.
    details : dict
        Arbitrary channel-specific metadata (sigma values, counts, etc.)
    """

    channel_name: str
    score: float
    is_active: bool
    calibrated_p: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EXODUSScore:
    """Aggregate EXODUS Score for a single stellar target.

    Attributes
    ----------
    target_id : str
        Unique identifier for the target (e.g. Gaia source_id).
    ra : float
        Right ascension in degrees (ICRS).
    dec : float
        Declination in degrees (ICRS).
    total_score : float
        Final EXODUS score (geo_mean * convergence_bonus).
    channel_scores : dict
        Mapping of channel_name -> ChannelScore for every evaluated channel.
    n_active_channels : int
        Number of channels whose score exceeded the threshold.
    convergence_bonus : float
        Multiplicative bonus: ``4 ** (n_active - 1)`` (convergence-priority mode).
    geo_mean : float
        Geometric mean of the active channel scores.
    rank : int or None
        Rank among all scored targets (1 = highest score).  Populated by
        :meth:`EXODUSScorer.score_all`.
    """

    target_id: str
    ra: float
    dec: float
    total_score: float
    channel_scores: Dict[str, ChannelScore]
    n_active_channels: int
    convergence_bonus: float
    geo_mean: float
    rank: Optional[int] = None
    combined_p: Optional[float] = None
    stouffer_p: Optional[float] = None  # Conservative: includes all channels (p=1.0 for inactive)
    fdr_significant: Optional[bool] = None       # Based on stouffer_p (conservative gate)
    q_value: Optional[float] = None               # BH q-value from stouffer_p
    # Audit fix S23-F6: renamed from fdr_significant_fisher / q_value_fisher
    # to make the exploratory (anti-conservative) nature structurally obvious.
    # Fisher p-value is post-selection (only active channels) and MUST NOT be
    # used for claims — use stouffer-based fdr_significant instead.
    exploratory_fdr_significant_fisher: Optional[bool] = None
    exploratory_q_value_fisher: Optional[float] = None
    global_p: Optional[float] = None
    coverage_penalty: float = 1.0
    n_channels_with_data: int = 0
    n_channels_possible: int = 6
    # Annotation flag for single-channel extremes.
    # Set when any calibrated channel has p < 0.001 but n_active <= 1.
    # Does NOT change scores, FDR, or rankings — purely for human review.
    single_channel_extreme: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["channel_scores"] = {k: v.to_dict() for k, v in self.channel_scores.items()}
        # Statistical methodology notes (Audit #4 F1, Audit #5 F4/F10):
        # FDR gating uses Stouffer Z-combination (conservative, includes all
        # channels with p=1.0 for inactive) as primary.  Fisher combination
        # (post-activation-selection, exploratory) is reported as secondary.
        caveats = []
        if self.combined_p is not None and self.n_active_channels > 0:
            caveats.append(
                "Fisher combination (combined_p) applied post-activation-selection: "
                "only channels with score > threshold contribute p-values. "
                "This is exploratory and may be anti-conservative."
            )
        if self.stouffer_p is not None:
            caveats.append(
                "Stouffer Z-combination (stouffer_p) includes all channels "
                "(p=1.0 for inactive) and is the conservative FDR gate. "
                "fdr_significant is based on stouffer_p."
            )
        if caveats:
            d["statistical_caveats"] = caveats
        return d

    def summary(self) -> str:
        """One-line human-readable summary."""
        active_names = [
            name for name, cs in self.channel_scores.items() if cs.is_active
        ]
        # Show Stouffer (conservative) as primary, Fisher as secondary
        p_str = ""
        if self.stouffer_p is not None:
            p_str = f"  sp={self.stouffer_p:.2e}"
        elif self.combined_p is not None:
            p_str = f"  fp={self.combined_p:.2e}"
        fdr_str = ""
        if self.fdr_significant is not None:
            fdr_str = "  FDR:YES" if self.fdr_significant else "  FDR:no"
        gp_str = ""
        if self.global_p is not None:
            gp_str = f"  gp={self.global_p:.2e}"
        cov_str = ""
        if self.coverage_penalty < 1.0:
            cov_str = f"  cov={self.n_channels_with_data}/{self.n_channels_possible}"
        return (
            f"[Rank {self.rank or '?':>3}]  {self.target_id:<24s}  "
            f"EXODUS={self.total_score:8.3f}  "
            f"active={self.n_active_channels}/6+1prior  "
            f"geo_mean={self.geo_mean:.3f}  "
            f"bonus={self.convergence_bonus:.0f}x  "
            f"channels=[{', '.join(active_names)}]"
            f"{p_str}{fdr_str}{gp_str}{cov_str}"
        )


# ============================================================================
#  Scorer
# ============================================================================

class EXODUSScorer:
    """Compute the multi-modal convergence EXODUS Score for stellar targets.

    Parameters
    ----------
    threshold : float, optional
        Minimum channel score to be considered "active".  Default 0.3.
    control_scores : dict, optional
        Mapping of channel_name -> list of float control scores.
        When provided, each channel's heuristic score is calibrated
        to a p-value against this empirical null distribution.
    coverage_matrix : CoverageMatrix, optional
        When provided, applies a missingness penalty so targets cannot
        rank highly solely due to having more available data.
    """

    # Canonical channel names in evaluation order.
    CHANNEL_NAMES = [
        "ir_excess",
        "transit_anomaly",
        "radio_anomaly",
        "gaia_photometric_anomaly",
        "habitable_zone_planet",
        "proper_motion_anomaly",
        "ir_variability",
        "uv_anomaly",
        "radio_emission",
        "hr_anomaly",
        "abundance_anomaly",
    ]

    def __init__(
        self,
        threshold: float = 0.3,
        control_scores: Optional[Dict[str, List[float]]] = None,
        coverage_matrix: Optional[Any] = None,
        population_fdr: bool = False,
        convergence_priority: bool = False,
    ):
        self._convergence_priority = convergence_priority
        if convergence_priority:
            # Lower threshold to catch weak-but-real signals across channels.
            # Standard mode: 0.3 (catches ~4.8σ in single channel).
            # Convergence mode: 0.15 (catches ~1.5σ per channel), but
            # requires 3+ channels to score meaningfully.
            # Audit fix B3: raised from 0.15 → 0.25.
            # 0.15 catches ~1.5σ noise per channel, which combined with the
            # 4^(n-1) convergence bonus amplifies noise into signal.
            # 0.25 catches ~2.5σ — still sub-standard but noise-resistant.
            self.threshold = 0.25 if threshold == 0.3 else threshold
        else:
            self.threshold = threshold
        self.control_scores = control_scores or {}
        self.coverage_matrix = coverage_matrix
        self._population_fdr = population_fdr
        self._results: List[EXODUSScore] = []
        mode_str = "CONVERGENCE-PRIORITY" if convergence_priority else "standard"
        log.info(
            "EXODUSScorer initialised  (mode=%s, activation threshold=%.2f, "
            "controls=%s, coverage=%s)",
            mode_str, self.threshold,
            "yes" if self.control_scores else "no",
            "yes" if self.coverage_matrix else "no",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_target(self, target_data: Dict[str, Any]) -> EXODUSScore:
        """Compute the EXODUS score for a single target.

        Parameters
        ----------
        target_data : dict
            Must contain at minimum:

            - ``"target_id"`` : str
            - ``"ra"`` : float
            - ``"dec"`` : float

            Plus optional keys for each channel's raw result data:

            - ``"ir_excess"`` : dict with IR excess results
            - ``"transit_anomaly"`` : dict with transit anomaly results
            - ``"radio_anomaly"`` : dict with radio processing results
            - ``"gaia_photometric_anomaly"`` : dict with Gaia epoch photometry
            - ``"habitable_zone_planet"`` : dict with exoplanet HZ data
            - ``"proper_motion_anomaly"`` : dict with Gaia astrometry data

        Returns
        -------
        EXODUSScore
        """
        target_id = str(target_data.get("target_id", "unknown"))
        ra = float(target_data.get("ra") or 0.0)
        dec = float(target_data.get("dec") or 0.0)

        # --- Evaluate each channel ---
        channel_extractors = {
            "ir_excess": self._get_ir_excess_score,
            "transit_anomaly": self._get_transit_anomaly_score,
            "radio_anomaly": self._get_radio_score,
            "gaia_photometric_anomaly": self._get_gaia_variability_score,
            "habitable_zone_planet": self._get_hz_score,
            "proper_motion_anomaly": self._get_astrometric_score,
            "ir_variability": self._get_ir_variability_score,
            "uv_anomaly": self._get_uv_anomaly_score,
            "radio_emission": self._get_radio_emission_score,
            "hr_anomaly": self._get_hr_anomaly_score,
            "abundance_anomaly": self._get_abundance_anomaly_score,
        }

        channel_scores: Dict[str, ChannelScore] = {}
        for ch_name, extractor in channel_extractors.items():
            raw = target_data.get(ch_name)

            # Treat simulated-data channels as "no data" for scoring integrity.
            # Simulated data must never enter Fisher/Stouffer combination.
            if (isinstance(raw, dict)
                    and raw.get("data_source") in ("simulated",)):
                cs = ChannelScore(
                    channel_name=ch_name,
                    score=0.0,
                    is_active=False,
                    calibrated_p=None,
                    details={"reason": "simulated data excluded",
                             "data_source": raw.get("data_source")},
                )
                channel_scores[ch_name] = cs
                continue

            if raw is None:
                # Channel data not available -- score 0, inactive.
                cs = ChannelScore(
                    channel_name=ch_name,
                    score=0.0,
                    is_active=False,
                    calibrated_p=None,
                    details={"reason": "no data provided"},
                )
            else:
                score_val = extractor(raw)
                # Clamp to [0, 1]
                score_val = float(np.clip(score_val, 0.0, 1.0))
                is_active = score_val > self.threshold

                # Calibrate against controls if available.
                # Skip p-value calibration for habitable_zone_planet: it is
                # a binary prior (0.0 or 0.8/0.9) and control stars will
                # always score 0.0, producing artificially tiny p-values
                # that inflate Fisher combination.  HZ still contributes to
                # the heuristic EXODUS score (geo_mean + convergence bonus).
                cal_p = None
                if (
                    ch_name != "habitable_zone_planet"
                    and ch_name in self.control_scores
                    and self.control_scores[ch_name]
                ):
                    try:
                        from src.core.statistics import calibrate_score_to_pvalue
                        cal_p = calibrate_score_to_pvalue(
                            score_val, self.control_scores[ch_name]
                        )
                    except ImportError:
                        pass

                cs = ChannelScore(
                    channel_name=ch_name,
                    score=score_val,
                    is_active=is_active,
                    calibrated_p=cal_p,
                    details=raw if isinstance(raw, dict) else {"raw": raw},
                )
            channel_scores[ch_name] = cs

        # ── Apply PM-IR correlation weight (audit fix B1) ──────────────
        # When the PM anomaly comes from WISE-Gaia offset (not Gaia-internal
        # RUWE/AEN), it may be correlated with the IR excess (e.g. background
        # galaxy in WISE beam causes BOTH).  galaxy_contamination module
        # computes a correlation score → effective weight.  Apply it here.
        pm_ir = target_data.get("pm_ir_correlation")
        if pm_ir and isinstance(pm_ir, dict) and pm_ir.get("pm_ir_correlated"):
            eff_weight = pm_ir.get("effective_pm_weight", 1.0)
            pm_cs = channel_scores.get("proper_motion_anomaly")
            if pm_cs is not None and pm_cs.score > 0:
                old_score = pm_cs.score
                new_score = old_score * eff_weight
                # Recalibrate p-value for the reduced score (audit fix A3:
                # old code preserved stale pre-weight calibrated_p)
                new_cal_p = pm_cs.calibrated_p
                pm_controls = self.control_scores.get("proper_motion_anomaly")
                if pm_controls:
                    try:
                        from src.core.statistics import calibrate_score_to_pvalue
                        new_cal_p = calibrate_score_to_pvalue(
                            new_score, pm_controls
                        )
                    except ImportError:
                        pass
                channel_scores["proper_motion_anomaly"] = ChannelScore(
                    channel_name=pm_cs.channel_name,
                    score=new_score,
                    is_active=new_score > self.threshold,
                    calibrated_p=new_cal_p,
                    details={**pm_cs.details,
                             "pm_ir_weight_applied": eff_weight,
                             "original_pm_score": old_score,
                             "original_calibrated_p": pm_cs.calibrated_p},
                )
                log.debug(
                    "PM-IR weight applied: %.3f → %.3f (weight=%.2f)",
                    old_score, new_score, eff_weight,
                )

        # --- Compute aggregate EXODUS score ---
        # HZ is a Bayesian prior, not an independent detection channel.
        # It boosts the score but does NOT count toward convergence
        # (n_active_channels, geo_mean, convergence_bonus).
        PRIOR_CHANNELS = {"habitable_zone_planet"}
        detection_active = {
            k: v.score for k, v in channel_scores.items()
            if v.is_active and k not in PRIOR_CHANNELS
        }
        prior_active = {
            k: v.score for k, v in channel_scores.items()
            if v.is_active and k in PRIOR_CHANNELS
        }
        n_active = len(detection_active)

        if n_active == 0:
            total_score = 0.0
            geo_mean = 0.0
            convergence_bonus = 0.0
        elif self._convergence_priority:
            # CONVERGENCE-PRIORITY MODE
            # Key insight: natural astrophysics produces strong single-channel
            # anomalies (binaries → PM, dust → IR). Technosignatures produce
            # weak-but-convergent multi-channel signals.
            #
            # Standard: bonus = 2^(n-1)     → 1ch=1x, 2ch=2x, 3ch=4x
            # Convergence: bonus = 4^(n-1)  → 1ch=1x, 2ch=4x, 3ch=16x, 4ch=64x
            #
            # Also apply single-channel penalty: targets with only 1 active
            # channel get scored at 25% of their raw value, pushing them down
            # the ranking in favor of multi-channel targets.
            geo_mean = float(np.exp(np.mean(np.log(list(detection_active.values())))))
            if n_active == 1:
                convergence_bonus = 0.25  # Penalty for single-channel
            else:
                convergence_bonus = float(4 ** (n_active - 1))  # 4x/16x/64x
            total_score = geo_mean * convergence_bonus
        else:
            geo_mean = float(np.exp(np.mean(np.log(list(detection_active.values())))))
            convergence_bonus = float(2 ** (n_active - 1))
            total_score = geo_mean * convergence_bonus

        # Apply HZ prior boost (multiplicative, not counted as convergence)
        if prior_active:
            hz_boost = 1.0 + max(prior_active.values())
            total_score *= hz_boost

        # --- Coverage penalty (Core Directive §3) ---
        # Targets with fewer available detection channels cannot rank as
        # highly as fully-observed targets.  Penalty = sqrt(n_with_data / n_possible).
        #
        # Auto-computed from channel data availability: a channel "has data"
        # if its raw input was not None (the scorer set reason="no data
        # provided" for missing channels).
        DETECTION_CHANNELS = [
            ch for ch in self.CHANNEL_NAMES
            if ch not in PRIOR_CHANNELS
        ]
        n_possible = len(DETECTION_CHANNELS)
        n_with_data = sum(
            1 for ch in DETECTION_CHANNELS
            if target_data.get(ch) is not None
        )

        coverage_penalty = 1.0
        if self.coverage_matrix is not None:
            # External coverage matrix takes precedence (dataset-level).
            try:
                coverage_penalty = float(
                    self.coverage_matrix.missingness_penalty(target_id)
                )
            except Exception:
                pass
        elif n_possible > 0:
            # Auto-compute from channel data availability.
            coverage_penalty = float(min(1.0, np.sqrt(n_with_data / n_possible)))

        total_score *= coverage_penalty

        # --- Combined p-value (Fisher) across active channels ---
        # NOTE: Fisher is applied post-activation-selection (Audit #4 F1).
        # Stouffer (below) includes all channels as the conservative alternative.
        combined_p = None
        stouffer_p = None
        active_p_values = [
            cs.calibrated_p for cs in channel_scores.values()
            if cs.is_active and cs.calibrated_p is not None
        ]
        if active_p_values:
            try:
                from src.core.statistics import fisher_combine
                combined_p = fisher_combine(active_p_values)
            except ImportError:
                log.warning("scipy not available; Fisher combined p-value not computed")  # CX-6

        # Stouffer Z-combination: uses ALL calibrated channels, substituting
        # p=1.0 for inactive ones.  This avoids post-selection bias by
        # diluting the signal with non-detections (conservative).
        all_p_for_stouffer = []
        for ch_name, cs in channel_scores.items():
            if ch_name in PRIOR_CHANNELS:
                continue  # HZ is prior-only, excluded from both combiners
            if cs.calibrated_p is not None:
                all_p_for_stouffer.append(cs.calibrated_p)
            elif ch_name in self.control_scores and self.control_scores[ch_name]:
                all_p_for_stouffer.append(1.0)  # has controls but inactive → p=1.0
        if all_p_for_stouffer:
            try:
                from src.core.statistics import stouffer_combine
                stouffer_p = stouffer_combine(all_p_for_stouffer)
            except ImportError:
                log.warning("scipy not available; Stouffer combined p-value not computed")  # CX-6

        # Detect single-channel extremes for human review.
        # Fires when any calibrated channel has p < 0.001 but there's only
        # 0-1 active channels.  The pipeline is architecturally blind to these
        # (convergence-priority design), so flagging them ensures no genuine
        # extreme anomaly is silently buried.  Does NOT change scores or FDR.
        sce_flag = False
        if n_active <= 1:
            for cs in channel_scores.values():
                if cs.calibrated_p is not None and cs.calibrated_p < 0.001:
                    sce_flag = True
                    log.info(
                        "SINGLE-CHANNEL EXTREME: %s has %s cal_p=%.2e "
                        "but only %d active channels — flagged for human review",
                        target_id, cs.channel_name,
                        cs.calibrated_p, n_active,
                    )
                    break

        result = EXODUSScore(
            target_id=target_id,
            ra=ra,
            dec=dec,
            total_score=total_score,
            channel_scores=channel_scores,
            n_active_channels=n_active,
            convergence_bonus=convergence_bonus,
            geo_mean=geo_mean,
            rank=None,
            combined_p=combined_p,
            stouffer_p=stouffer_p,
            fdr_significant=None,  # set in score_all()
            coverage_penalty=coverage_penalty,
            n_channels_with_data=n_with_data,
            n_channels_possible=n_possible,
            single_channel_extreme=sce_flag,
        )

        log.debug(
            "Scored %s: total=%.4f  active=%d/6+1prior  combined_p=%s",
            target_id, total_score, n_active,
            f"{combined_p:.2e}" if combined_p is not None else "N/A",
        )
        return result

    def score_all(
        self, targets_data: List[Dict[str, Any]]
    ) -> List[EXODUSScore]:
        """Score and rank all targets.

        Parameters
        ----------
        targets_data : list of dict
            Each element is a target_data dict suitable for
            :meth:`score_target`.

        Returns
        -------
        list of EXODUSScore
            Sorted descending by ``total_score``, with ``rank`` populated
            starting at 1.
        """
        log.info("Scoring %d targets ...", len(targets_data))

        if not self.control_scores:
            log.warning(
                "UNCALIBRATED SCORING: No matched control scores provided. "
                "All calibrated_p values will be null, combined_p will be null, "
                "and FDR correction will be skipped. Scores reflect relative "
                "anomaly ranking only, NOT calibrated statistical significance. "
                "Wire matched controls via EXODUSScorer(control_scores=...) "
                "for publication-grade p-values."
            )

        results = [self.score_target(td) for td in targets_data]

        # Sort descending by total_score, breaking ties by n_active_channels
        # then by geo_mean (both descending).
        results.sort(
            key=lambda r: (r.total_score, r.n_active_channels, r.geo_mean),
            reverse=True,
        )

        # Assign ranks
        for idx, r in enumerate(results, start=1):
            r.rank = idx

        # --- FDR correction (Audit #5 F4/F10) ---
        # PRIMARY gate: Stouffer Z-combination (conservative, all channels).
        # SECONDARY (exploratory): Fisher combination (post-selection, active only).
        #
        # Supports two modes:
        #   1. Global (default): single BH correction across all targets
        #   2. Population-conditional: separate BH per population group,
        #      using `population_tag` from target_data. This accounts for
        #      different base rates across populations (HZ planets vs
        #      VASCO vanished vs calibration binaries etc).
        has_any_p = any(
            r.stouffer_p is not None or r.combined_p is not None
            for r in results
        )
        if has_any_p:
            try:
                from src.core.statistics import benjamini_hochberg

                # Check if population-conditional FDR is enabled
                pop_tags = [td.get("population_tag") for td in targets_data]
                has_pop_tags = any(t is not None for t in pop_tags)
                use_pop_fdr = has_pop_tags and self._population_fdr

                if use_pop_fdr:
                    self._apply_population_fdr(results, pop_tags)
                else:
                    # --- Primary FDR: Stouffer (conservative) ---
                    all_stouffer_p = np.array([
                        r.stouffer_p if r.stouffer_p is not None else 1.0
                        for r in results
                    ])
                    rejected_s, q_values_s = benjamini_hochberg(
                        all_stouffer_p, alpha=0.05
                    )
                    for i, r in enumerate(results):
                        r.fdr_significant = bool(rejected_s[i])
                        r.q_value = float(q_values_s[i])
                    n_fdr_stouffer = int(np.sum(rejected_s))
                    log.info(
                        "FDR correction (Stouffer, conservative): "
                        "%d/%d targets significant at alpha=0.05",
                        n_fdr_stouffer, len(results),
                    )

                    # --- Secondary FDR: Fisher (exploratory) ---
                    all_fisher_p = np.array([
                        r.combined_p if r.combined_p is not None else 1.0
                        for r in results
                    ])
                    rejected_f, q_values_f = benjamini_hochberg(
                        all_fisher_p, alpha=0.05
                    )
                    for i, r in enumerate(results):
                        r.exploratory_fdr_significant_fisher = bool(rejected_f[i])
                        r.exploratory_q_value_fisher = float(q_values_f[i])
                    n_fdr_fisher = int(np.sum(rejected_f))
                    log.info(
                        "FDR correction (Fisher, exploratory): "
                        "%d/%d targets significant at alpha=0.05",
                        n_fdr_fisher, len(results),
                    )

                    if n_fdr_fisher > n_fdr_stouffer:
                        log.warning(
                            "Fisher FDR (%d) > Stouffer FDR (%d): "
                            "difference of %d targets likely due to "
                            "post-activation selection bias in Fisher.",
                            n_fdr_fisher, n_fdr_stouffer,
                            n_fdr_fisher - n_fdr_stouffer,
                        )

            except ImportError:
                log.warning("Could not import statistics module for FDR correction")

        # --- Global p-values / look-elsewhere correction (Core Directive §5) ---
        # Šidák correction: p_global = 1 - (1 - p_local)^N
        # Answers: "given N targets tested, what is the probability of
        # seeing at least one result this extreme by chance?"
        # Uses stouffer_p (conservative) as the base for global correction.
        n_tested = len(results)
        if n_tested > 1:
            for r in results:
                # Use Stouffer (conservative) for global p; fall back to Fisher
                base_p = r.stouffer_p if r.stouffer_p is not None else r.combined_p
                if base_p is not None:
                    # Šidák: exact correction for independent tests
                    r.global_p = 1.0 - (1.0 - base_p) ** n_tested
                    # Clamp to valid range
                    r.global_p = float(np.clip(r.global_p, 1e-300, 1.0))
            n_global_sig = sum(
                1 for r in results
                if r.global_p is not None and r.global_p < 0.05
            )
            log.info(
                "Look-elsewhere: %d/%d targets have global_p < 0.05 "
                "(Šidák correction, N=%d)",
                n_global_sig, n_tested, n_tested,
            )

        self._results = results

        n_active_any = sum(1 for r in results if r.n_active_channels > 0)
        log.info(
            "Scoring complete.  %d/%d targets have >= 1 active channel.",
            n_active_any, len(results),
        )

        return results

    def get_top_targets(self, n: int = 20) -> List[EXODUSScore]:
        """Return the top *n* targets by EXODUS score.

        Must call :meth:`score_all` first.

        Parameters
        ----------
        n : int
            Number of top targets to return.  Default 20.

        Returns
        -------
        list of EXODUSScore

        Raises
        ------
        RuntimeError
            If :meth:`score_all` has not been called yet.
        """
        if not self._results:
            raise RuntimeError(
                "No scored results available. Call score_all() first."
            )
        return self._results[:n]

    def get_channel_breakdown(self, target_id: str) -> Optional[EXODUSScore]:
        """Return the detailed channel breakdown for a specific target.

        Parameters
        ----------
        target_id : str
            The target identifier to look up.

        Returns
        -------
        EXODUSScore or None
            The full score object including per-channel breakdowns,
            or None if the target was not found in the scored results.
        """
        for r in self._results:
            if r.target_id == target_id:
                return r
        return None

    # ------------------------------------------------------------------
    # Population-conditional FDR
    # ------------------------------------------------------------------

    def _apply_population_fdr(
        self,
        results: List[EXODUSScore],
        pop_tags: List[Optional[str]],
    ) -> None:
        """Apply BH FDR correction separately within each population group.

        Uses Stouffer Z-combination (conservative) as primary FDR gate and
        Fisher combination (exploratory) as secondary.  (Audit #5 F4/F10.)

        Targets with the same ``population_tag`` form a group. BH correction
        is applied within each group independently, accounting for different
        base rates (e.g. calibration binaries have high PM false positive
        rates, while VASCO targets have near-zero base rates for IR detection).

        Targets without a population_tag are pooled into an "untagged" group.
        """
        from src.core.statistics import benjamini_hochberg

        # Group targets by population tag
        groups: Dict[str, List[int]] = {}
        for i, tag in enumerate(pop_tags):
            key = tag if tag is not None else "__untagged__"
            groups.setdefault(key, []).append(i)

        total_fdr_stouffer = 0
        total_fdr_fisher = 0
        for pop_name, indices in groups.items():
            display_name = pop_name if pop_name != "__untagged__" else "untagged"

            # --- Primary: Stouffer (conservative) ---
            stouffer_pv = np.array([
                results[i].stouffer_p if results[i].stouffer_p is not None else 1.0
                for i in indices
            ])
            rejected_s, q_values_s = benjamini_hochberg(stouffer_pv, alpha=0.05)
            for j, idx in enumerate(indices):
                results[idx].fdr_significant = bool(rejected_s[j])
                results[idx].q_value = float(q_values_s[j])
            n_sig_s = int(np.sum(rejected_s))
            total_fdr_stouffer += n_sig_s

            # --- Secondary: Fisher (exploratory) ---
            fisher_pv = np.array([
                results[i].combined_p if results[i].combined_p is not None else 1.0
                for i in indices
            ])
            rejected_f, q_values_f = benjamini_hochberg(fisher_pv, alpha=0.05)
            for j, idx in enumerate(indices):
                results[idx].exploratory_fdr_significant_fisher = bool(rejected_f[j])
                results[idx].exploratory_q_value_fisher = float(q_values_f[j])
            n_sig_f = int(np.sum(rejected_f))
            total_fdr_fisher += n_sig_f

            log.info(
                "FDR (population '%s'): Stouffer=%d/%d, Fisher=%d/%d at alpha=0.05",
                display_name, n_sig_s, len(indices), n_sig_f, len(indices),
            )

        log.info(
            "Population-conditional FDR: Stouffer=%d/%d, Fisher=%d/%d "
            "across %d groups",
            total_fdr_stouffer, len(results),
            total_fdr_fisher, len(results), len(groups),
        )

    # ------------------------------------------------------------------
    # Individual channel score extractors
    # ------------------------------------------------------------------
    # Each method receives the raw data dict for its channel and returns
    # a float in [0, 1].

    @staticmethod
    def _get_ir_excess_score(data: Dict[str, Any]) -> float:
        """Extract a normalised IR excess score.

        Expected keys in *data*:
            - ``sigma_W3`` : significance of W3 excess (float or None)
            - ``sigma_W4`` : significance of W4 excess (float or None)
            - ``excess_W3`` : obs - pred magnitude (negative = genuine excess)
            - ``excess_W4`` : obs - pred magnitude (negative = genuine excess)
            - ``is_candidate`` : bool (from IR excess module)

        **Direction matters.**  Only *negative* excess (observed brighter
        than model) indicates genuine IR emission -- the Dyson-sphere
        signature.  Positive excess (star dimmer than model) is just
        photometric scatter or extinction and must NOT inflate the score.
        This distinction was the root cause of false positives in Phase 0
        calibration (51 Peg, HD 209458, KIC 8462852).

        The score is the maximum *direction-qualified* sigma across W3/W4,
        mapped to [0, 1] via ``score = 1 - exp(-sigma_max / 5)``.
        """
        sigma_w3 = data.get("sigma_W3")
        sigma_w4 = data.get("sigma_W4")
        excess_w3 = data.get("excess_W3")
        excess_w4 = data.get("excess_W4")

        # HARD GATE: If the IR excess module explicitly marked this as
        # NOT a candidate (via chi2_red quality gate, NIR anchor
        # requirement, or contamination demotion), the scorer must
        # respect that decision.  The raw sigma values are meaningless
        # when the underlying model is unreliable.
        if "is_candidate" in data and not data["is_candidate"]:
            return 0.0

        sigmas = []

        # W3: only count if excess is NEGATIVE (star brighter than model)
        if (sigma_w3 is not None and np.isfinite(sigma_w3)
                and excess_w3 is not None and excess_w3 < 0):
            sigmas.append(float(sigma_w3))

        # W4: same direction check
        if (sigma_w4 is not None and np.isfinite(sigma_w4)
                and excess_w4 is not None and excess_w4 < 0):
            sigmas.append(float(sigma_w4))

        if not sigmas:
            return 0.0

        sigma_max = max(sigmas)
        # Saturating exponential mapping: 0-sigma -> 0, inf-sigma -> 1
        score = 1.0 - np.exp(-sigma_max / 5.0)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _get_transit_anomaly_score(data: Dict[str, Any]) -> float:
        """Extract a normalised transit anomaly score.

        Expected keys in *data*:
            - ``anomaly_score`` : float in [0, 1] (composite transit score)
            - ``is_anomalous`` : bool (optional)

        If ``anomaly_score`` is present it is used directly.  Otherwise
        falls back to a simple binary from ``is_anomalous``.
        """
        if "anomaly_score" in data:
            val = data["anomaly_score"]
            if val is not None and np.isfinite(val):
                return float(np.clip(val, 0.0, 1.0))

        # Fallback: binary from is_anomalous flag
        if data.get("is_anomalous"):
            return 0.7  # moderately high default for flagged anomalies
        return 0.0

    @staticmethod
    def _get_radio_score(data: Dict[str, Any]) -> float:
        """Extract a normalised radio anomaly score.

        Expected keys in *data*:
            - ``n_candidates`` : int -- total non-RFI candidate detections
            - ``max_snr`` : float -- peak SNR among candidates
            - ``candidates`` : list of dicts (optional, for richer scoring)

        Score combines candidate count and peak SNR:
            count_score = 1 - exp(-n_candidates / 3)
            snr_score   = 1 - exp(-max_snr / 20)
            score       = 0.4 * count_score + 0.6 * snr_score
        """
        n_candidates = int(data.get("n_candidates", 0))

        # Filter out RFI if full candidate list available
        if "candidates" in data and isinstance(data["candidates"], list):
            non_rfi = [
                c for c in data["candidates"]
                if not c.get("is_rfi", False)
            ]
            n_candidates = len(non_rfi)
            snrs = [c.get("snr", 0.0) for c in non_rfi]
            max_snr = max(snrs) if snrs else 0.0
        else:
            max_snr = float(data.get("max_snr", 0.0))

        if n_candidates == 0 and max_snr <= 0:
            return 0.0

        count_score = 1.0 - np.exp(-n_candidates / 3.0)
        snr_score = 1.0 - np.exp(-max_snr / 20.0)
        score = 0.4 * count_score + 0.6 * snr_score
        return float(score)

    @staticmethod
    def _get_gaia_variability_score(data: Dict[str, Any]) -> float:
        """Extract a normalised Gaia epoch photometry variability score.

        Expected keys in *data*:
            - ``phot_g_variability`` : float -- std(flux)/mean(flux) of
              G-band epoch photometry (fractional flux variability).
              Magnitudes are converted to flux (10^{-0.4 mag}) before
              computing std/mean, because magnitudes are logarithmic.
            - ``n_epochs`` : int -- number of good epochs (for weighting).
            - ``variability_flag`` : bool (optional Gaia-provided flag).

        Score is the fractional variability mapped via a saturating
        exponential.  Typical quiet stars have variability ~ 0.001-0.01;
        genuinely variable sources reach 0.05-0.5+.  We map so that
        a 5% fractional variability gives a score of ~ 0.63.
        """
        var = data.get("phot_g_variability")
        if var is None or not np.isfinite(var):
            # Fall back to a binary flag if present
            if data.get("variability_flag"):
                return 0.5
            return 0.0

        var = abs(float(var))
        # Saturating map: 0 -> 0,  0.05 -> 0.63,  0.15 -> 0.95
        score = 1.0 - np.exp(-var / 0.05)

        # Weight down if very few epochs (less reliable)
        n_epochs = int(data.get("n_epochs") or 50)
        if n_epochs < 10:
            score *= n_epochs / 10.0

        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _get_hz_score(data: Dict[str, Any]) -> float:
        """Extract a habitable-zone planet score.

        This is fundamentally a binary channel: does the target host
        a confirmed planet in the habitable zone?

        Expected keys in *data*:
            - ``has_hz_planet`` : bool
            - ``n_hz_planets`` : int (optional, for bonus)
            - ``hz_confidence`` : float in [0, 1] (optional)

        Returns 0.5 for a single confirmed HZ planet, 0.6 for multiple,
        or the ``hz_confidence`` value (capped at 0.6) if provided and > 0.

        Weight reduced from 0.8/0.9 to 0.5/0.6 after
        quantitative analysis showed 80% score inflation: 16 targets
        scored > 1.0 only because of HZ boost, masking single-channel
        detections as multi-channel anomalies.  FDR is unaffected
        (HZ has calibrated_p=null).  See data/reports/hz_boost_analysis.json.
        """
        if data.get("hz_confidence") is not None:
            conf = float(data["hz_confidence"])
            if np.isfinite(conf) and conf > 0:
                return float(np.clip(conf, 0.0, 0.6))

        if data.get("has_hz_planet"):
            n_hz = int(data.get("n_hz_planets", 1))
            if n_hz >= 2:
                return 0.6
            return 0.5
        return 0.0

    @staticmethod
    def _get_astrometric_score(data: Dict[str, Any]) -> float:
        """Extract a normalised astrometric anomaly score from Gaia data.

        Expected keys in *data*:
            - ``ruwe`` : float -- Renormalised Unit Weight Error.
              Values > 1.4 indicate poor single-star astrometric fit
              (possible unseen companion or extended structure).
            - ``astrometric_excess_noise`` : float -- in mas.
            - ``astrometric_excess_noise_sig`` : float -- significance.

        Score combines RUWE departure and excess noise significance.
        """
        ruwe = data.get("ruwe")
        excess_noise_sig = data.get("astrometric_excess_noise_sig")

        scores = []

        # RUWE score: 1.0 is perfect, >1.4 is anomalous, >2.0 is very anomalous
        if ruwe is not None and np.isfinite(ruwe):
            ruwe = float(ruwe)
            if ruwe > 1.0:
                # Map RUWE 1.0-3.0 to 0-1 approximately
                ruwe_score = 1.0 - np.exp(-(ruwe - 1.0) / 1.0)
                scores.append(float(ruwe_score))

        # Excess noise significance score
        if excess_noise_sig is not None and np.isfinite(excess_noise_sig):
            sig = float(excess_noise_sig)
            if sig > 0:
                noise_score = 1.0 - np.exp(-sig / 5.0)
                scores.append(float(noise_score))

        # WISE-Gaia PM consistency (wavelength-dependent photocentre shift)
        pm_check = data.get("wise_gaia_pm")
        if pm_check and isinstance(pm_check, dict):
            pm_sigma = pm_check.get("pm_discrepancy_sigma", 0.0)
            if pm_sigma is not None and np.isfinite(pm_sigma) and pm_sigma > 0:
                # Map sigma: 3→0.45, 5→0.82, 8→0.98
                pm_score = 1.0 - np.exp(-pm_sigma / 4.0)
                scores.append(float(pm_score))

        if not scores:
            return 0.0

        # Return the maximum of the indicators
        return float(max(scores))

    @staticmethod
    def _get_ir_variability_score(data: Dict[str, Any]) -> float:
        """Extract a normalised IR variability score from NEOWISE time-series.

        Expected keys in *data*:
            - ``variability_score`` : float in [0, 1] — combined score from
              excess scatter + secular trend analysis.
            - ``is_anomalous`` : bool — whether variability exceeds thresholds.
            - ``data_source`` : str — "real", "simulated", or "none".

        The ``variability_score`` field is already in [0, 1] from the
        ir_variability module, combining:
          - Excess scatter: observed std / photometric noise - 1
          - Secular trend: significance of linear mag/yr slope

        A Dyson sphere under construction would show secular IR brightening
        (negative slope in magnitudes). Variable dust emission would show
        stochastic excess scatter. Both are genuinely INDEPENDENT from static
        IR excess (which measures the current W3/W4 SED anomaly, not changes
        over time).
        """
        # Reject simulated/no-data results — simulated data MUST NOT
        # enter Fisher/Stouffer combination (integrity requirement)
        source = data.get("data_source", "none")
        if source in ("none", "insufficient", "simulated"):
            return 0.0

        vs = data.get("variability_score")
        if vs is not None and np.isfinite(vs):
            return float(np.clip(vs, 0.0, 1.0))

        # Fallback: binary from is_anomalous flag
        if data.get("is_anomalous"):
            return 0.5

        return 0.0

    @staticmethod
    def _get_uv_anomaly_score(data: Dict[str, Any]) -> float:
        """Extract a normalised UV anomaly score from GALEX data.

        Expected keys in *data*:
            - ``anomaly_score`` : float in [0, 1]
            - ``data_source`` : str — "galex_vizier" or "none"

        UV anomaly is HIGHLY independent of IR and PM channels.
        A star with UV + IR convergence rules out background galaxy
        confusion (galaxies don't emit UV at the same position).
        """
        source = data.get("data_source", "none")
        if source == "none":
            return 0.0

        score = data.get("anomaly_score")
        if score is not None and np.isfinite(score):
            return float(np.clip(score, 0.0, 1.0))

        # Fallback: use uv_anomaly_score if present
        score2 = data.get("uv_anomaly_score")
        if score2 is not None and np.isfinite(score2):
            return float(np.clip(score2, 0.0, 1.0))

        return 0.0

    @staticmethod
    def _get_radio_emission_score(data: Dict[str, Any]) -> float:
        """Extract a normalised radio continuum emission score.

        Expected keys in *data*:
            - ``anomaly_score`` : float in [0, 1]
            - ``is_detected`` : bool
            - ``data_source`` : str

        Radio at 1.4 GHz is independent of IR/PM. Most MS stars are
        radio-quiet; detection indicates something unusual.
        """
        source = data.get("data_source", "none")
        if source == "none":
            return 0.0

        score = data.get("anomaly_score")
        if score is not None and np.isfinite(score):
            return float(np.clip(score, 0.0, 1.0))

        # Fallback: binary detected flag → 0.4
        if data.get("is_detected"):
            return 0.4

        return 0.0

    @staticmethod
    def _get_hr_anomaly_score(data: Dict[str, Any]) -> float:
        """Extract a normalised HR diagram outlier score.

        Expected keys in *data*:
            - ``anomaly_score`` : float in [0, 1]
            - ``ms_sigma`` : float — deviation from MS in sigma
            - ``data_source`` : str

        Stars that don't belong on the HR diagram. Combined with the
        binary veto (RUWE check), HR outliers become unexplained.
        """
        source = data.get("data_source", "none")
        if source == "none":
            return 0.0

        score = data.get("anomaly_score")
        if score is not None and np.isfinite(score):
            return float(np.clip(score, 0.0, 1.0))

        return 0.0

    @staticmethod
    def _get_abundance_anomaly_score(data: Dict[str, Any]) -> float:
        """Extract a normalised stellar abundance anomaly score.

        Expected keys in *data*:
            - ``anomaly_score`` : float in [0, 1]
            - ``n_anomalous_ratios`` : int
            - ``data_source`` : str — "apogee", "galah", "both", "none"

        Chemical abundances are MAXIMALLY INDEPENDENT of all other channels:
        IR excess = thermal, PM = gravitational, UV/radio = emission,
        abundance = chemical composition of the star itself.
        A star with anomalous chemistry AND multi-channel anomaly is
        extremely compelling.
        """
        source = data.get("data_source", "none")
        if source == "none":
            return 0.0

        score = data.get("anomaly_score")
        if score is not None and np.isfinite(score):
            return float(np.clip(score, 0.0, 1.0))

        # Fallback: derive from n_anomalous_ratios
        n_anom = data.get("n_anomalous_ratios", 0)
        if n_anom >= 3:
            return 0.7
        elif n_anom >= 2:
            return 0.5
        elif n_anom >= 1:
            return 0.3
        return 0.0


# ============================================================================
#  Main -- demonstration with mock data
# ============================================================================

if __name__ == "__main__":
    import random

    random.seed(42)
    np.random.seed(42)

    log.setLevel("INFO")

    print("=" * 78)
    print("  Project EXODUS  --  Multi-Modal Convergence Score Demo")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Generate 100 mock targets with varying channel activations
    # ------------------------------------------------------------------
    N_TARGETS = 100
    mock_targets: List[Dict[str, Any]] = []

    for i in range(N_TARGETS):
        target_id = f"GAIA_DR3_{1000000000 + i}"
        ra = random.uniform(0, 360)
        dec = random.uniform(-90, 90)

        td: Dict[str, Any] = {
            "target_id": target_id,
            "ra": ra,
            "dec": dec,
        }

        # Randomly populate channels with varying probability
        # Most targets get 0-1 channels; a few get multiple.
        #
        # We structure the mock data so that:
        #   - ~40% of targets have 0 active channels
        #   - ~30% have exactly 1
        #   - ~15% have 2
        #   - ~10% have 3
        #   - ~5%  have 4+

        n_channels_to_activate = np.random.choice(
            [0, 1, 2, 3, 4, 5, 6],
            p=[0.35, 0.30, 0.15, 0.10, 0.05, 0.03, 0.02],
        )

        # Decide which channels to activate
        activated = set()
        if n_channels_to_activate > 0:
            activated = set(
                random.sample(
                    EXODUSScorer.CHANNEL_NAMES,
                    k=min(n_channels_to_activate, 6),
                )
            )

        # IR excess
        if "ir_excess" in activated:
            sigma = random.uniform(3.0, 15.0)
            td["ir_excess"] = {
                "sigma_W3": sigma,
                "sigma_W4": sigma * random.uniform(0.5, 1.5),
                "is_candidate": True,
            }
        else:
            # Quiet source -- low sigma
            td["ir_excess"] = {
                "sigma_W3": random.uniform(0.0, 1.5),
                "sigma_W4": random.uniform(0.0, 1.5),
                "is_candidate": False,
            }

        # Transit anomaly
        if "transit_anomaly" in activated:
            td["transit_anomaly"] = {
                "anomaly_score": random.uniform(0.4, 0.95),
                "is_anomalous": True,
            }
        else:
            td["transit_anomaly"] = {
                "anomaly_score": random.uniform(0.0, 0.2),
                "is_anomalous": False,
            }

        # Radio anomaly
        if "radio_anomaly" in activated:
            n_cand = random.randint(1, 8)
            td["radio_anomaly"] = {
                "n_candidates": n_cand,
                "max_snr": random.uniform(10.0, 50.0),
            }
        else:
            td["radio_anomaly"] = {
                "n_candidates": 0,
                "max_snr": 0.0,
            }

        # Gaia photometric anomaly
        if "gaia_photometric_anomaly" in activated:
            td["gaia_photometric_anomaly"] = {
                "phot_g_variability": random.uniform(0.03, 0.3),
                "n_epochs": random.randint(20, 100),
            }
        else:
            td["gaia_photometric_anomaly"] = {
                "phot_g_variability": random.uniform(0.001, 0.01),
                "n_epochs": random.randint(30, 80),
            }

        # Habitable-zone planet
        if "habitable_zone_planet" in activated:
            td["habitable_zone_planet"] = {
                "has_hz_planet": True,
                "n_hz_planets": random.randint(1, 3),
            }
        else:
            td["habitable_zone_planet"] = {
                "has_hz_planet": False,
            }

        # Proper motion anomaly
        if "proper_motion_anomaly" in activated:
            td["proper_motion_anomaly"] = {
                "ruwe": random.uniform(1.5, 4.0),
                "astrometric_excess_noise_sig": random.uniform(3.0, 20.0),
            }
        else:
            td["proper_motion_anomaly"] = {
                "ruwe": random.uniform(0.9, 1.3),
                "astrometric_excess_noise_sig": random.uniform(0.0, 1.0),
            }

        mock_targets.append(td)

    # ------------------------------------------------------------------
    # Score all targets
    # ------------------------------------------------------------------
    scorer = EXODUSScorer(threshold=0.3)
    results = scorer.score_all(mock_targets)

    # ------------------------------------------------------------------
    # Verification: multi-anomaly targets must rank above single-anomaly
    # ------------------------------------------------------------------
    print("\n--- Verification ---")

    multi_results = [r for r in results if r.n_active_channels >= 3]
    single_results = [r for r in results if r.n_active_channels == 1]

    if multi_results and single_results:
        best_single = max(r.total_score for r in single_results)
        worst_multi = min(r.total_score for r in multi_results)

        if worst_multi > best_single:
            print(
                f"PASS: All multi-channel targets (>= 3) rank above single-"
                f"channel targets.\n"
                f"      Worst multi-channel score:  {worst_multi:.4f}\n"
                f"      Best single-channel score:  {best_single:.4f}"
            )
        else:
            # This is expected in edge cases when a single channel has a
            # very high score and the multi-channel set has moderate scores.
            # The convergence bonus helps but does not guarantee dominance
            # in all cases.
            print(
                f"NOTE: Overlap detected between multi- and single-channel "
                f"targets.\n"
                f"      Worst multi-channel score:  {worst_multi:.4f}\n"
                f"      Best single-channel score:  {best_single:.4f}\n"
                f"      This can occur when a single channel has very high "
                f"significance."
            )
    else:
        print(
            "SKIP: Not enough multi- and single-channel targets to compare."
        )

    # Distribution summary
    from collections import Counter

    dist = Counter(r.n_active_channels for r in results)
    print("\nChannel activation distribution:")
    for n_ch in sorted(dist.keys()):
        print(f"  {n_ch} active channels: {dist[n_ch]:>3d} targets")

    # ------------------------------------------------------------------
    # Print top 20
    # ------------------------------------------------------------------
    top = scorer.get_top_targets(n=20)

    print(f"\n{'=' * 78}")
    print(f"  Top 20 Targets by EXODUS Score")
    print(f"{'=' * 78}")
    for r in top:
        print(r.summary())

    # ------------------------------------------------------------------
    # Detailed breakdown of the #1 target
    # ------------------------------------------------------------------
    if top:
        best = top[0]
        print(f"\n{'=' * 78}")
        print(f"  Detailed Channel Breakdown for #1: {best.target_id}")
        print(f"{'=' * 78}")
        print(f"  RA, Dec          : {best.ra:.4f}, {best.dec:.4f}")
        print(f"  Total EXODUS     : {best.total_score:.4f}")
        print(f"  Active channels  : {best.n_active_channels} / 6")
        print(f"  Geometric mean   : {best.geo_mean:.4f}")
        print(f"  Convergence bonus: {best.convergence_bonus:.0f}x")
        print()
        for ch_name, cs in best.channel_scores.items():
            status = "ACTIVE" if cs.is_active else "      "
            print(f"    [{status}]  {ch_name:<30s}  score={cs.score:.4f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    try:
        output = {
            "n_targets": len(results),
            "threshold": scorer.threshold,
            "top_20": [r.to_dict() for r in top],
            "distribution": dict(dist),
        }
        path = save_result("exodus_scores_demo", output)
        print(f"\nResults saved to: {path}")
    except Exception as e:
        print(f"\n(Could not save results: {e})")

    print("\nDone.")
