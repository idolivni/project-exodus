"""
Evolver Engine -- self-improvement loop for Project EXODUS.

After each iteration of the main research loop, the Evolver analyzes which
hypotheses produced interesting (unexplained) results and which were dead ends,
adjusts scoring thresholds to control false-positive rates, generates new
hypotheses from observed patterns, and deprioritizes unproductive search
strategies.  Every decision is logged to ``data/hypotheses/evolution_log.json``
for human review.

The loop implements the simplest possible version of automated scientific
reasoning:

    observe results -> evaluate methods -> tune parameters
                    -> generate new questions -> repeat

This is NOT unsupervised.  A human reviews the recommendations at the end of
every batch and decides which new hypotheses to actually pursue.

Key references
--------------
- Project EXODUS design doc, Section 5 (self-improvement loop)
- Langley et al. 1987, "Scientific Discovery" (BACON/GLAUBER paradigm)
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import get_config, get_logger, save_result, PROJECT_ROOT

log = get_logger("engines.evolver")

# ── Defaults ─────────────────────────────────────────────────────────
_DEFAULT_LOG_PATH = PROJECT_ROOT / "data" / "hypotheses" / "evolution_log.json"

# Hypothesis status constants
STATUS_CONFIRMED = "confirmed"       # hypothesis validated by data
STATUS_UNEXPLAINED = "unexplained"   # anomaly remains unexplained -- interesting!
STATUS_NATURAL = "natural"           # explained by natural astrophysics
STATUS_ARTIFACT = "artifact"         # instrumental / data artifact
STATUS_DEAD_END = "dead_end"         # tested, nothing found
STATUS_PENDING = "pending"           # not yet tested

_DEAD_END_STATUSES = {STATUS_ARTIFACT, STATUS_DEAD_END, STATUS_NATURAL}
_INTERESTING_STATUSES = {STATUS_UNEXPLAINED, STATUS_CONFIRMED}

# Threshold tuning parameters
_FP_HIGH = 0.30          # above this: thresholds too loose
_FP_LOW = 0.05           # below this: thresholds too tight
_SIGMA_INCREASE = 0.5    # how much to tighten when FP rate is high
_SIGMA_DECREASE = 0.25   # how much to loosen when FP rate is low
_SIGMA_FLOOR = 2.0       # never go below this sigma

# Strategy evaluation
_DEAD_END_FRACTION = 0.50  # deprioritize if more than 50% dead ends


# =====================================================================
#  Data classes
# =====================================================================

@dataclass
class EvolutionRecord:
    """Single evolution step produced by the Evolver after one iteration.

    Captures every decision made -- threshold adjustments, new hypotheses,
    strategy weight changes -- so a human can audit the reasoning.

    Attributes
    ----------
    iteration : int
        Research loop iteration number.
    timestamp : str
        ISO-8601 UTC timestamp of this evolution step.
    threshold_changes : dict
        Mapping of threshold name -> {old, new, reason}.
    new_hypotheses_generated : list of dict
        Each entry: {id, text, source_pattern, priority}.
    strategies_deprioritized : list of dict
        Each entry: {method, old_weight, new_weight, reason}.
    strategies_promoted : list of dict
        Each entry: {method, old_weight, new_weight, reason}.
    false_positive_rate : float
        Overall false-positive rate this iteration.
    true_positive_rate : float
        Fraction of tested hypotheses that produced interesting results.
    recommendations : list of str
        Human-readable recommendations for the operator.
    """

    iteration: int = 0
    timestamp: str = ""
    threshold_changes: Dict[str, Any] = field(default_factory=dict)
    new_hypotheses_generated: List[Dict[str, Any]] = field(default_factory=list)
    strategies_deprioritized: List[Dict[str, Any]] = field(default_factory=list)
    strategies_promoted: List[Dict[str, Any]] = field(default_factory=list)
    false_positive_rate: float = 0.0
    true_positive_rate: float = 0.0
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ResearchState:
    """Snapshot of the research pipeline state after one iteration.

    This is the input the Evolver receives from the main loop.

    Attributes
    ----------
    iteration : int
        Current loop iteration number.
    hypotheses_tested : list of dict
        Each entry: {id, text, method, status, scores, properties}.
        ``status`` is one of: confirmed, unexplained, natural, artifact,
        dead_end, pending.  ``scores`` is a dict of metric -> value.
        ``properties`` is an optional dict of target attributes observed.
    targets_processed : int
        Number of targets analyzed this iteration.
    anomalies_found : int
        Number of anomalies flagged this iteration (before vetting).
    false_positives : int
        Number of flagged anomalies later classified as artifact/natural.
    current_thresholds : dict
        Mapping of threshold name -> current value.  At minimum
        contains ``anomaly_sigma``.
    """

    iteration: int = 0
    hypotheses_tested: List[Dict[str, Any]] = field(default_factory=list)
    targets_processed: int = 0
    anomalies_found: int = 0
    false_positives: int = 0
    current_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "anomaly_sigma": 3.0,
    })


# =====================================================================
#  Evolver Engine
# =====================================================================

class EvolverEngine:
    """Self-improvement engine for the Project EXODUS research loop.

    After each iteration, call :meth:`evolve` with the current
    :class:`ResearchState`.  The engine returns an :class:`EvolutionRecord`
    containing threshold adjustments, new hypotheses, and strategy weight
    changes.  All decisions are appended to a persistent JSON log for
    human review.

    Parameters
    ----------
    log_path : str or Path, optional
        Path to the evolution log file.  Defaults to
        ``data/hypotheses/evolution_log.json``.
    """

    def __init__(self, log_path: Optional[str | Path] = None) -> None:
        self.log_path = Path(log_path) if log_path else _DEFAULT_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory history
        self._history: List[EvolutionRecord] = []

        # Strategy weights: method_name -> weight (1.0 = default).
        # Weights < 1.0 are deprioritized; > 1.0 are promoted.
        self._strategy_weights: Dict[str, float] = {}

        # Running hypothesis counter for generating unique IDs
        self._hypothesis_counter: int = 0

        # Load existing log if present
        self.load_log()

        log.info(
            "EvolverEngine initialized (log=%s, history=%d records, strategies=%d)",
            self.log_path,
            len(self._history),
            len(self._strategy_weights),
        )

    # ─── Main entry point ────────────────────────────────────────────

    def evolve(self, state: ResearchState) -> EvolutionRecord:
        """Analyze results from one iteration and produce improvements.

        This is the core self-improvement step.  It:

        1. Evaluates which hypotheses produced interesting results.
        2. Computes false-positive rates per detection method.
        3. Adjusts scoring thresholds accordingly.
        4. Generates new hypotheses from patterns in the results.
        5. Deprioritizes dead-end strategies.
        6. Promotes productive strategies.
        7. Compiles human-readable recommendations.

        Parameters
        ----------
        state : ResearchState
            Current pipeline state snapshot.

        Returns
        -------
        EvolutionRecord
            Full record of all changes made this step.
        """
        log.info(
            "=== Evolution step: iteration %d  (%d hypotheses, %d targets, "
            "%d anomalies, %d FP) ===",
            state.iteration,
            len(state.hypotheses_tested),
            state.targets_processed,
            state.anomalies_found,
            state.false_positives,
        )

        record = EvolutionRecord(
            iteration=state.iteration,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Step 1 & 2: analyze performance
        performance = self._analyze_hypothesis_performance(state)
        fp_rates = self._compute_false_positive_rate(state)

        # Overall rates
        total_tested = len([
            h for h in state.hypotheses_tested
            if h.get("status") != STATUS_PENDING
        ])
        interesting = len([
            h for h in state.hypotheses_tested
            if h.get("status") in _INTERESTING_STATUSES
        ])

        record.true_positive_rate = (
            interesting / total_tested if total_tested > 0 else 0.0
        )
        record.false_positive_rate = (
            state.false_positives / state.anomalies_found
            if state.anomalies_found > 0 else 0.0
        )

        # Step 3: adjust thresholds
        record.threshold_changes = self._adjust_thresholds(state)

        # Step 4: generate new hypotheses
        record.new_hypotheses_generated = self._generate_new_hypotheses(state)

        # Step 5: deprioritize dead ends
        record.strategies_deprioritized = self._deprioritize_dead_ends(state)

        # Step 6: promote productive strategies
        record.strategies_promoted = self._promote_productive(state)

        # Step 7: compile recommendations
        record.recommendations = self._compile_recommendations(
            state, record, performance, fp_rates
        )

        # Persist
        self._history.append(record)
        self.save_log()

        log.info(
            "Evolution step complete: %d threshold changes, %d new hypotheses, "
            "%d deprioritized, %d promoted",
            len(record.threshold_changes),
            len(record.new_hypotheses_generated),
            len(record.strategies_deprioritized),
            len(record.strategies_promoted),
        )

        return record

    # ─── Analysis methods ────────────────────────────────────────────

    def _analyze_hypothesis_performance(
        self, state: ResearchState
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate which hypotheses worked and which were dead ends.

        Groups hypotheses by detection method and computes per-method
        statistics: total tested, interesting, dead ends, and the list
        of unexplained results.

        Returns
        -------
        dict
            method_name -> {total, interesting, dead_ends, unexplained_ids}
        """
        method_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"total": 0, "interesting": 0, "dead_ends": 0,
                     "unexplained_ids": []}
        )

        for h in state.hypotheses_tested:
            method = h.get("method", "unknown")
            status = h.get("status", STATUS_PENDING)

            if status == STATUS_PENDING:
                continue

            method_stats[method]["total"] += 1

            if status in _INTERESTING_STATUSES:
                method_stats[method]["interesting"] += 1
            if status in _DEAD_END_STATUSES:
                method_stats[method]["dead_ends"] += 1
            if status == STATUS_UNEXPLAINED:
                method_stats[method]["unexplained_ids"].append(
                    h.get("id", "?")
                )

        for method, stats in method_stats.items():
            log.info(
                "  [%s] total=%d  interesting=%d  dead_ends=%d  unexplained=%s",
                method,
                stats["total"],
                stats["interesting"],
                stats["dead_ends"],
                stats["unexplained_ids"],
            )

        return dict(method_stats)

    def _compute_false_positive_rate(
        self, state: ResearchState
    ) -> Dict[str, float]:
        """Compute false-positive rate per detection method.

        A "false positive" here means a hypothesis that was flagged as
        anomalous but resolved as artifact or natural.

        Returns
        -------
        dict
            method_name -> false_positive_rate (0.0 to 1.0).
        """
        method_flagged: Dict[str, int] = defaultdict(int)
        method_fp: Dict[str, int] = defaultdict(int)

        for h in state.hypotheses_tested:
            method = h.get("method", "unknown")
            status = h.get("status", STATUS_PENDING)

            if status == STATUS_PENDING:
                continue

            # Count anything that was tested as "flagged"
            method_flagged[method] += 1

            # Count artifacts and naturals as false positives
            if status in {STATUS_ARTIFACT, STATUS_NATURAL}:
                method_fp[method] += 1

        fp_rates: Dict[str, float] = {}
        for method in method_flagged:
            total = method_flagged[method]
            fp = method_fp[method]
            rate = fp / total if total > 0 else 0.0
            fp_rates[method] = rate
            log.info(
                "  FP rate [%s]: %d / %d = %.2f",
                method, fp, total, rate,
            )

        return fp_rates

    def _adjust_thresholds(
        self, state: ResearchState
    ) -> Dict[str, Dict[str, Any]]:
        """Tune detection thresholds based on false-positive rate.

        Rules:
        - If FP rate > 0.30: increase anomaly_sigma by 0.5 (tighten)
        - If FP rate < 0.05: decrease anomaly_sigma by 0.25 (loosen)
        - anomaly_sigma never drops below 2.0

        Returns
        -------
        dict
            threshold_name -> {old, new, reason}
        """
        changes: Dict[str, Dict[str, Any]] = {}

        overall_fp_rate = (
            state.false_positives / state.anomalies_found
            if state.anomalies_found > 0 else 0.0
        )

        current_sigma = state.current_thresholds.get("anomaly_sigma", 3.0)
        new_sigma = current_sigma

        if overall_fp_rate > _FP_HIGH:
            new_sigma = current_sigma + _SIGMA_INCREASE
            reason = (
                f"FP rate {overall_fp_rate:.2f} > {_FP_HIGH} threshold; "
                f"tightening sigma by {_SIGMA_INCREASE}"
            )
            log.info("Threshold adjustment: %s", reason)
        elif overall_fp_rate < _FP_LOW:
            new_sigma = max(current_sigma - _SIGMA_DECREASE, _SIGMA_FLOOR)
            if new_sigma < current_sigma:
                reason = (
                    f"FP rate {overall_fp_rate:.2f} < {_FP_LOW} threshold; "
                    f"loosening sigma by {_SIGMA_DECREASE} "
                    f"(floor={_SIGMA_FLOOR})"
                )
                log.info("Threshold adjustment: %s", reason)
            else:
                reason = (
                    f"FP rate {overall_fp_rate:.2f} < {_FP_LOW} but sigma "
                    f"already at floor ({_SIGMA_FLOOR}); no change"
                )
                log.info("Threshold note: %s", reason)
        else:
            reason = (
                f"FP rate {overall_fp_rate:.2f} is within acceptable range "
                f"[{_FP_LOW}, {_FP_HIGH}]; no change"
            )
            log.debug("Threshold: %s", reason)

        if new_sigma != current_sigma:
            changes["anomaly_sigma"] = {
                "old": current_sigma,
                "new": round(new_sigma, 2),
                "reason": reason,
            }
            # Update the state so downstream code sees the new value
            state.current_thresholds["anomaly_sigma"] = round(new_sigma, 2)

        return changes

    def _generate_new_hypotheses(
        self, state: ResearchState
    ) -> List[Dict[str, Any]]:
        """Create new hypotheses from patterns in the current results.

        Pattern-matching rules:
        1. Property co-occurrence: if targets with property A also have
           property B, generate "Do all targets with A show B?"
        2. Spatial clustering: if multiple anomalies share similar sky
           positions, generate "Is this region of sky special?"
        3. Temporal patterns: if a temporal pattern is found in one
           target, generate "Does this temporal pattern appear in other
           targets?"

        Returns
        -------
        list of dict
            Each: {id, text, source_pattern, priority}.
        """
        new_hypotheses: List[Dict[str, Any]] = []

        # Collect properties from interesting hypotheses
        interesting = [
            h for h in state.hypotheses_tested
            if h.get("status") in _INTERESTING_STATUSES
        ]

        if not interesting:
            log.info("No interesting results this iteration; no new hypotheses.")
            return new_hypotheses

        # ── Rule 1: Property co-occurrence ────────────────────────────
        # Gather all properties from interesting hypotheses and look for
        # pairs that appear together more than once.
        property_sets: List[set] = []
        for h in interesting:
            props = h.get("properties", {})
            if props:
                property_sets.append(set(props.keys()))

        if len(property_sets) >= 2:
            # Find properties that appear in multiple interesting targets
            all_props: Dict[str, int] = defaultdict(int)
            for ps in property_sets:
                for p in ps:
                    all_props[p] += 1

            common_props = [
                p for p, count in all_props.items() if count >= 2
            ]

            # Look for co-occurring property pairs
            for i, prop_a in enumerate(common_props):
                for prop_b in common_props[i + 1:]:
                    # Count how many targets have both
                    both_count = sum(
                        1 for ps in property_sets
                        if prop_a in ps and prop_b in ps
                    )
                    if both_count >= 2:
                        self._hypothesis_counter += 1
                        hyp = {
                            "id": f"evo_hyp_{self._hypothesis_counter:04d}",
                            "text": (
                                f"Do all targets with '{prop_a}' also show "
                                f"'{prop_b}'?  ({both_count} co-occurrences "
                                f"observed)"
                            ),
                            "source_pattern": "property_cooccurrence",
                            "priority": min(both_count / len(interesting), 1.0),
                        }
                        new_hypotheses.append(hyp)
                        log.info(
                            "New hypothesis (co-occurrence): %s", hyp["text"]
                        )

        # ── Rule 2: Spatial clustering ────────────────────────────────
        # If multiple anomalies share similar RA/Dec, flag the region.
        positions = []
        for h in interesting:
            props = h.get("properties", {})
            ra = props.get("ra")
            dec = props.get("dec")
            if ra is not None and dec is not None:
                positions.append((float(ra), float(dec), h.get("id", "?")))

        if len(positions) >= 2:
            # Simple clustering: check if any pair is within 1 degree
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    ra_i, dec_i, id_i = positions[i]
                    ra_j, dec_j, id_j = positions[j]
                    sep = (
                        (ra_i - ra_j) ** 2 + (dec_i - dec_j) ** 2
                    ) ** 0.5
                    if sep < 1.0:  # within ~1 degree
                        center_ra = (ra_i + ra_j) / 2
                        center_dec = (dec_i + dec_j) / 2
                        self._hypothesis_counter += 1
                        hyp = {
                            "id": f"evo_hyp_{self._hypothesis_counter:04d}",
                            "text": (
                                f"Is the sky region near "
                                f"(RA={center_ra:.2f}, Dec={center_dec:.2f}) "
                                f"special?  Anomalies {id_i} and {id_j} are "
                                f"within {sep:.2f} deg."
                            ),
                            "source_pattern": "spatial_cluster",
                            "priority": 0.8,
                        }
                        new_hypotheses.append(hyp)
                        log.info(
                            "New hypothesis (spatial cluster): %s",
                            hyp["text"],
                        )

        # ── Rule 3: Temporal patterns ─────────────────────────────────
        # If any interesting hypothesis has a temporal_pattern property,
        # suggest searching for it in other targets.
        for h in interesting:
            props = h.get("properties", {})
            temporal = props.get("temporal_pattern")
            if temporal:
                self._hypothesis_counter += 1
                hyp = {
                    "id": f"evo_hyp_{self._hypothesis_counter:04d}",
                    "text": (
                        f"Does the temporal pattern '{temporal}' found in "
                        f"{h.get('id', '?')} appear in other targets?"
                    ),
                    "source_pattern": "temporal_generalization",
                    "priority": 0.7,
                }
                new_hypotheses.append(hyp)
                log.info(
                    "New hypothesis (temporal): %s", hyp["text"]
                )

        log.info(
            "Generated %d new hypotheses from %d interesting results",
            len(new_hypotheses),
            len(interesting),
        )
        return new_hypotheses

    def _deprioritize_dead_ends(
        self, state: ResearchState
    ) -> List[Dict[str, Any]]:
        """Reduce weight of detection methods with >50% dead-end rate.

        A method is deprioritized by halving its strategy weight (but
        never below 0.1).

        Returns
        -------
        list of dict
            Each: {method, old_weight, new_weight, reason}.
        """
        changes: List[Dict[str, Any]] = []

        # Group by method
        method_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "dead_ends": 0}
        )

        for h in state.hypotheses_tested:
            method = h.get("method", "unknown")
            status = h.get("status", STATUS_PENDING)
            if status == STATUS_PENDING:
                continue
            method_stats[method]["total"] += 1
            if status in _DEAD_END_STATUSES:
                method_stats[method]["dead_ends"] += 1

        for method, stats in method_stats.items():
            total = stats["total"]
            dead = stats["dead_ends"]
            if total < 2:
                continue  # not enough data to judge

            dead_fraction = dead / total
            if dead_fraction > _DEAD_END_FRACTION:
                old_weight = self._strategy_weights.get(method, 1.0)
                new_weight = max(old_weight * 0.5, 0.1)

                if new_weight < old_weight:
                    self._strategy_weights[method] = round(new_weight, 3)
                    reason = (
                        f"{dead}/{total} hypotheses ({dead_fraction:.0%}) "
                        f"were dead ends (> {_DEAD_END_FRACTION:.0%} threshold)"
                    )
                    changes.append({
                        "method": method,
                        "old_weight": round(old_weight, 3),
                        "new_weight": round(new_weight, 3),
                        "reason": reason,
                    })
                    log.info(
                        "Deprioritized [%s]: weight %.3f -> %.3f (%s)",
                        method, old_weight, new_weight, reason,
                    )

        return changes

    def _promote_productive(
        self, state: ResearchState
    ) -> List[Dict[str, Any]]:
        """Increase weight of methods that produced UNEXPLAINED results.

        Any method that yielded at least one unexplained anomaly this
        iteration gets its weight increased by 25% (capped at 3.0).

        Returns
        -------
        list of dict
            Each: {method, old_weight, new_weight, reason}.
        """
        changes: List[Dict[str, Any]] = []

        # Find methods with unexplained results
        methods_with_unexplained: Dict[str, List[str]] = defaultdict(list)

        for h in state.hypotheses_tested:
            if h.get("status") == STATUS_UNEXPLAINED:
                method = h.get("method", "unknown")
                methods_with_unexplained[method].append(
                    h.get("id", "?")
                )

        for method, hyp_ids in methods_with_unexplained.items():
            old_weight = self._strategy_weights.get(method, 1.0)
            new_weight = min(old_weight * 1.25, 3.0)

            if new_weight > old_weight:
                self._strategy_weights[method] = round(new_weight, 3)
                reason = (
                    f"Produced {len(hyp_ids)} unexplained result(s): "
                    f"{', '.join(hyp_ids)}"
                )
                changes.append({
                    "method": method,
                    "old_weight": round(old_weight, 3),
                    "new_weight": round(new_weight, 3),
                    "reason": reason,
                })
                log.info(
                    "Promoted [%s]: weight %.3f -> %.3f (%s)",
                    method, old_weight, new_weight, reason,
                )

        return changes

    def _compile_recommendations(
        self,
        state: ResearchState,
        record: EvolutionRecord,
        performance: Dict[str, Dict[str, Any]],
        fp_rates: Dict[str, float],
    ) -> List[str]:
        """Build human-readable recommendations from all analysis.

        Returns
        -------
        list of str
            Prioritized list of recommendations for the operator.
        """
        recs: List[str] = []

        # Overall summary
        recs.append(
            f"Iteration {state.iteration}: processed {state.targets_processed} "
            f"targets, found {state.anomalies_found} anomalies, "
            f"{state.false_positives} false positives "
            f"(FP rate={record.false_positive_rate:.1%})."
        )

        # Threshold changes
        for name, change in record.threshold_changes.items():
            recs.append(
                f"THRESHOLD CHANGE: {name} adjusted from {change['old']} "
                f"to {change['new']}. Reason: {change['reason']}"
            )

        # High FP rate methods
        for method, rate in fp_rates.items():
            if rate > _FP_HIGH:
                recs.append(
                    f"WARNING: Method '{method}' has a high false-positive "
                    f"rate ({rate:.0%}).  Consider reviewing its detection "
                    f"criteria or adding filters."
                )

        # Productive methods
        for method, stats in performance.items():
            if stats["unexplained_ids"]:
                recs.append(
                    f"PROMISING: Method '{method}' produced unexplained "
                    f"results: {', '.join(stats['unexplained_ids'])}.  "
                    f"Investigate further."
                )

        # New hypotheses
        if record.new_hypotheses_generated:
            recs.append(
                f"GENERATED {len(record.new_hypotheses_generated)} new "
                f"hypotheses from observed patterns.  Review before adding "
                f"to the research queue."
            )
            for hyp in record.new_hypotheses_generated:
                recs.append(
                    f"  -> [{hyp['id']}] {hyp['text']} "
                    f"(pattern: {hyp['source_pattern']}, "
                    f"priority: {hyp['priority']:.2f})"
                )

        # Deprioritized strategies
        for dep in record.strategies_deprioritized:
            recs.append(
                f"DEPRIORITIZED: '{dep['method']}' weight reduced "
                f"{dep['old_weight']} -> {dep['new_weight']}.  "
                f"Reason: {dep['reason']}"
            )

        # Strategy weights summary
        if self._strategy_weights:
            weight_summary = ", ".join(
                f"{m}={w:.3f}" for m, w in
                sorted(self._strategy_weights.items(), key=lambda x: x[1])
            )
            recs.append(f"Current strategy weights: {weight_summary}")

        # Low-data warning
        if state.targets_processed < 10:
            recs.append(
                "NOTE: Very few targets processed this iteration.  "
                "Statistical conclusions are unreliable."
            )

        return recs

    # ─── Public query methods ────────────────────────────────────────

    def get_recommendations(self) -> List[str]:
        """Return human-readable recommendations from the most recent
        evolution step.

        Returns
        -------
        list of str
            Recommendations, or a note if no evolution has run yet.
        """
        if not self._history:
            return ["No evolution steps have been performed yet."]
        return self._history[-1].recommendations

    def get_evolution_history(self) -> List[EvolutionRecord]:
        """Return the full evolution log as a list of records.

        Returns
        -------
        list of EvolutionRecord
        """
        return list(self._history)

    def get_strategy_weights(self) -> Dict[str, float]:
        """Return current strategy weights.

        Returns
        -------
        dict
            method_name -> weight
        """
        return dict(self._strategy_weights)

    # ─── Persistence ─────────────────────────────────────────────────

    def save_log(self) -> Path:
        """Persist the evolution history and strategy weights to disk.

        Returns
        -------
        Path
            The file path written.
        """
        data = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "total_records": len(self._history),
            "strategy_weights": self._strategy_weights,
            "hypothesis_counter": self._hypothesis_counter,
            "records": [asdict(r) for r in self._history],
        }

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        log.debug("Evolution log saved to %s (%d records)", self.log_path,
                  len(self._history))
        return self.log_path

    def load_log(self) -> None:
        """Load evolution history and strategy weights from disk.

        If the log file does not exist, the engine starts fresh.
        """
        if not self.log_path.exists():
            log.debug("No existing evolution log at %s; starting fresh.",
                      self.log_path)
            return

        try:
            with open(self.log_path, "r") as f:
                data = json.load(f)

            self._strategy_weights = data.get("strategy_weights", {})
            self._hypothesis_counter = data.get("hypothesis_counter", 0)

            records_raw = data.get("records", [])
            self._history = []
            for raw in records_raw:
                rec = EvolutionRecord(
                    iteration=raw.get("iteration", 0),
                    timestamp=raw.get("timestamp", ""),
                    threshold_changes=raw.get("threshold_changes", {}),
                    new_hypotheses_generated=raw.get(
                        "new_hypotheses_generated", []
                    ),
                    strategies_deprioritized=raw.get(
                        "strategies_deprioritized", []
                    ),
                    strategies_promoted=raw.get("strategies_promoted", []),
                    false_positive_rate=raw.get("false_positive_rate", 0.0),
                    true_positive_rate=raw.get("true_positive_rate", 0.0),
                    recommendations=raw.get("recommendations", []),
                )
                self._history.append(rec)

            log.info(
                "Loaded evolution log: %d records, %d strategy weights",
                len(self._history),
                len(self._strategy_weights),
            )

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            log.warning(
                "Failed to load evolution log at %s: %s.  Starting fresh.",
                self.log_path,
                exc,
            )
            self._history = []
            self._strategy_weights = {}
            self._hypothesis_counter = 0


# =====================================================================
#  CLI demo -- simulate 10 iterations of self-improvement
# =====================================================================

def _make_simulated_state(iteration: int, rng: random.Random) -> ResearchState:
    """Generate a plausible simulated ResearchState for testing.

    Creates a mix of hypothesis outcomes: some natural, some artifact,
    some unexplained, some dead ends.  The false-positive rate varies
    by iteration to exercise the threshold adjustment logic.
    """
    methods = ["ir_excess", "transit_anomaly", "radio_narrowband",
               "spectral_lines", "proper_motion"]

    n_hypotheses = rng.randint(5, 15)
    n_targets = rng.randint(20, 200)

    # Weighted status distribution -- mostly dead ends and naturals,
    # with occasional unexplained gems
    status_weights = {
        STATUS_NATURAL:     0.30,
        STATUS_ARTIFACT:    0.25,
        STATUS_DEAD_END:    0.20,
        STATUS_UNEXPLAINED: 0.10 + 0.02 * iteration,  # gets slightly better
        STATUS_CONFIRMED:   0.02,
        STATUS_PENDING:     0.13 - 0.02 * min(iteration, 6),
    }
    statuses = list(status_weights.keys())
    weights = list(status_weights.values())

    # Simulated sky properties
    base_ra = rng.uniform(100, 200)
    base_dec = rng.uniform(-30, 30)

    hypotheses = []
    for i in range(n_hypotheses):
        method = rng.choice(methods)
        status = rng.choices(statuses, weights=weights, k=1)[0]

        props: Dict[str, Any] = {
            "ra": base_ra + rng.gauss(0, 2.0),
            "dec": base_dec + rng.gauss(0, 2.0),
        }

        # Sometimes add extra properties for pattern detection
        if rng.random() < 0.3:
            props["ir_excess"] = True
        if rng.random() < 0.2:
            props["variable"] = True
        if rng.random() < 0.15:
            props["temporal_pattern"] = rng.choice([
                "periodic_dimming_0.3d",
                "gradual_brightening",
                "irregular_flicker",
            ])

        scores = {
            "anomaly_score": rng.uniform(0, 10),
            "confidence": rng.uniform(0.1, 1.0),
        }

        hypotheses.append({
            "id": f"hyp_{iteration:02d}_{i:03d}",
            "text": f"Test {method} on target field {i}",
            "method": method,
            "status": status,
            "scores": scores,
            "properties": props,
        })

    # Count false positives (artifact + natural resolved)
    anomalies = sum(
        1 for h in hypotheses
        if h["status"] in (_INTERESTING_STATUSES | _DEAD_END_STATUSES)
    )
    false_pos = sum(
        1 for h in hypotheses
        if h["status"] in {STATUS_ARTIFACT, STATUS_NATURAL}
    )

    # Vary the sigma across iterations for demonstration
    current_sigma = 3.0 + 0.0  # evolver will adjust this

    return ResearchState(
        iteration=iteration,
        hypotheses_tested=hypotheses,
        targets_processed=n_targets,
        anomalies_found=max(anomalies, 1),
        false_positives=false_pos,
        current_thresholds={"anomaly_sigma": current_sigma},
    )


if __name__ == "__main__":
    import textwrap

    print()
    print("=" * 72)
    print("  Project EXODUS -- Evolver Engine Demo")
    print("  Self-improvement loop simulation (10 iterations)")
    print("=" * 72)

    # Use a fixed seed for reproducibility
    rng = random.Random(42)

    # Use a temporary log so we don't pollute the real one
    demo_log = PROJECT_ROOT / "data" / "hypotheses" / "evolution_log_demo.json"
    evolver = EvolverEngine(log_path=demo_log)

    sigma_history: List[float] = []
    hypothesis_counts: List[int] = []

    for iteration in range(1, 11):
        print(f"\n{'- ' * 36}")
        print(f"  ITERATION {iteration}")
        print(f"{'- ' * 36}")

        state = _make_simulated_state(iteration, rng)

        print(f"  Targets processed : {state.targets_processed}")
        print(f"  Hypotheses tested : {len(state.hypotheses_tested)}")
        print(f"  Anomalies found   : {state.anomalies_found}")
        print(f"  False positives   : {state.false_positives}")
        print(f"  Current sigma     : {state.current_thresholds['anomaly_sigma']}")

        record = evolver.evolve(state)

        # Track sigma changes
        new_sigma = state.current_thresholds["anomaly_sigma"]
        sigma_history.append(new_sigma)

        # Display threshold adjustments
        if record.threshold_changes:
            print(f"\n  Threshold adjustments:")
            for name, change in record.threshold_changes.items():
                print(f"    {name}: {change['old']} -> {change['new']}")
                print(f"      Reason: {change['reason']}")
        else:
            print(f"\n  No threshold adjustments this iteration.")

        # Display new hypotheses
        n_new = len(record.new_hypotheses_generated)
        hypothesis_counts.append(n_new)
        if record.new_hypotheses_generated:
            print(f"\n  New hypotheses generated ({n_new}):")
            for hyp in record.new_hypotheses_generated[:3]:  # show top 3
                print(f"    [{hyp['id']}] {hyp['text']}")
            if n_new > 3:
                print(f"    ... and {n_new - 3} more")
        else:
            print(f"\n  No new hypotheses generated.")

        # Display strategy changes
        if record.strategies_deprioritized:
            print(f"\n  Strategies deprioritized:")
            for dep in record.strategies_deprioritized:
                print(
                    f"    {dep['method']}: "
                    f"{dep['old_weight']} -> {dep['new_weight']}"
                )
        if record.strategies_promoted:
            print(f"\n  Strategies promoted:")
            for promo in record.strategies_promoted:
                print(
                    f"    {promo['method']}: "
                    f"{promo['old_weight']} -> {promo['new_weight']}"
                )

        print(f"\n  FP rate: {record.false_positive_rate:.1%}")
        print(f"  TP rate: {record.true_positive_rate:.1%}")

    # ── Final summary ────────────────────────────────────────────────
    print(f"\n\n{'=' * 72}")
    print("  FINAL SUMMARY AFTER 10 ITERATIONS")
    print(f"{'=' * 72}")

    print(f"\n  Sigma history: {' -> '.join(f'{s:.2f}' for s in sigma_history)}")
    print(f"  Hypotheses generated per iteration: {hypothesis_counts}")
    print(f"  Total hypotheses generated: {sum(hypothesis_counts)}")

    # Strategy weights
    weights = evolver.get_strategy_weights()
    if weights:
        print(f"\n  Final strategy weights:")
        for method, weight in sorted(weights.items(), key=lambda x: x[1]):
            bar = "#" * int(weight * 20)
            print(f"    {method:25s}  {weight:.3f}  {bar}")

    # Recommendations
    print(f"\n  Final recommendations:")
    for rec in evolver.get_recommendations():
        wrapped = textwrap.fill(rec, width=68, initial_indent="    ",
                                subsequent_indent="      ")
        print(wrapped)

    # Verify log was saved
    history = evolver.get_evolution_history()
    print(f"\n  Evolution log: {len(history)} records saved to:")
    print(f"    {demo_log}")

    # Also save a summary result via the project utility
    summary = {
        "demo": "evolver_engine_10_iterations",
        "sigma_history": sigma_history,
        "hypothesis_counts": hypothesis_counts,
        "total_hypotheses": sum(hypothesis_counts),
        "final_weights": weights,
        "final_recommendations": evolver.get_recommendations(),
    }
    result_path = save_result("evolver_demo", summary)
    print(f"  Summary result saved to: {result_path}")

    print()
    print("=" * 72)
    print("  Done.")
    print("=" * 72)
    print()
