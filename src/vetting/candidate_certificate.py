"""
Telescope Candidate Certificate — formal escalation gate for Project EXODUS.

A target must pass ALL criteria below before it earns the label "telescope
candidate".  This prevents self-deception and ensures only robust anomalies
reach telescope-time proposals.

Certificate Criteria (ALL required)
------------------------------------
1. REAL DATA ONLY — every active channel used real data (no simulated fallback)
2. MULTI-CHANNEL — 2+ independent detection channels active
3. FDR SIGNIFICANT — stouffer_p < 0.01 (conservative, all-channel)
4. UNEXPLAINED — unexplainability_score > 0.3 (no natural template fits)
5. RED-TEAM SURVIVES — red_team recommendation != "DEMOTE"
6. NO SINGLE-CHANNEL DOMINANCE — no single channel contributes > 80% of score
7. DISTANCE REASONABLE — target within 2000 pc (audit fix D4: raised from 500)
8. REPLICATION — same anomaly detected in 2+ independent pipeline runs
   (tracked via certificate history, optional for first-pass)

Output: CandidateCertificate dataclass with pass/fail per criterion,
overall certified=True/False, and machine-readable JSON.

Usage
-----
    from src.vetting.candidate_certificate import certify_candidate
    cert = certify_candidate(scored_target, red_team_verdict, unexplainability)
    if cert.certified:
        print(f"TELESCOPE CANDIDATE: {cert.target_id}")
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger

log = get_logger("vetting.candidate_certificate")


# ── Thresholds ────────────────────────────────────────────────────────
MIN_REAL_CHANNELS = 2          # At least 2 channels must have real data
MAX_STOUFFER_P = 0.01          # Conservative significance threshold
MIN_UNEXPLAINABILITY = 0.3     # Must be at least partially unexplained
MAX_SINGLE_CHANNEL_FRAC = 0.80 # No one channel > 80% of total score
MAX_DISTANCE_PC = 2000.0       # Within 2000 pc (audit fix D4: raised from 500)
MIN_ACTIVE_CHANNELS = 2        # 2+ independent detections

# Channels where simulated data is known to exist
SIMULATED_DATA_SOURCES = ("simulated", "simulation")


@dataclass
class CertificateCriterion:
    """A single pass/fail criterion."""
    name: str
    passed: bool
    value: Any = None      # The measured value
    threshold: Any = None  # The required threshold
    note: str = ""         # Human-readable explanation


@dataclass
class CandidateCertificate:
    """Machine-readable escalation gate result."""
    target_id: str
    certified: bool = False
    criteria: List[CertificateCriterion] = field(default_factory=list)
    n_passed: int = 0
    n_total: int = 0
    grade: str = "FAIL"    # FAIL, PARTIAL, CERTIFIED
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "certified": self.certified,
            "grade": self.grade,
            "n_passed": self.n_passed,
            "n_total": self.n_total,
            "summary": self.summary,
            "criteria": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "value": c.value,
                    "threshold": c.threshold,
                    "note": c.note,
                }
                for c in self.criteria
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


def certify_candidate(
    scored_target: Dict[str, Any],
    red_team_verdict: Optional[Dict[str, Any]] = None,
    unexplainability: Optional[Dict[str, Any]] = None,
    pipeline_run_count: int = 1,
) -> CandidateCertificate:
    """Evaluate a scored target against all certificate criteria.

    Parameters
    ----------
    scored_target : dict
        A target dict from the scorer output (has channel_scores, total_score, etc.)
    red_team_verdict : dict, optional
        Red-Team result for this target (has recommendation, overall_risk, etc.)
    unexplainability : dict, optional
        Unexplainability score result (has score, best_template, etc.)
    pipeline_run_count : int
        Number of independent pipeline runs that detected this anomaly.

    Returns
    -------
    CandidateCertificate
    """
    target_id = scored_target.get("target_id", "unknown")
    channel_scores = scored_target.get("channel_scores", {})
    total_score = scored_target.get("total_score", 0.0)
    stouffer_p = scored_target.get("stouffer_p")
    distance_pc = scored_target.get("distance_pc")

    criteria = []

    # ── 1. REAL DATA ONLY ──────────────────────────────────────────
    real_channels = 0
    simulated_channels = []
    for ch_name, ch_data in channel_scores.items():
        if not isinstance(ch_data, dict):
            continue
        details = ch_data.get("details", {})
        if isinstance(details, dict):
            ds = details.get("data_source", "")
            reason = details.get("reason", "")
            if reason == "simulated data excluded":
                simulated_channels.append(ch_name)
            elif ds in SIMULATED_DATA_SOURCES:
                simulated_channels.append(ch_name)
            elif ch_data.get("is_active") and reason != "no data provided":
                real_channels += 1

    real_data_ok = real_channels >= MIN_REAL_CHANNELS and len(simulated_channels) == 0
    criteria.append(CertificateCriterion(
        name="real_data_only",
        passed=real_data_ok,
        value={"real_channels": real_channels, "simulated": simulated_channels},
        threshold=f"{MIN_REAL_CHANNELS}+ real, 0 simulated",
        note=f"{real_channels} real channel(s), {len(simulated_channels)} simulated"
             + (f" ({', '.join(simulated_channels)})" if simulated_channels else ""),
    ))

    # ── 2. MULTI-CHANNEL ──────────────────────────────────────────
    # Audit fix CX-3: n_active_channels already excludes HZ prior
    # (set at exodus_score.py line 535 via len(detection_active), which
    # filters out PRIOR_CHANNELS at lines 420-424).  No additional
    # subtraction is needed here — the previous code double-subtracted HZ.
    n_detection_active = scored_target.get("n_active_channels", 0)

    multi_ok = n_detection_active >= MIN_ACTIVE_CHANNELS
    criteria.append(CertificateCriterion(
        name="multi_channel",
        passed=multi_ok,
        value=n_detection_active,
        threshold=f">= {MIN_ACTIVE_CHANNELS}",
        note=f"{n_detection_active} independent detection channel(s) active"
             " (excl. HZ prior)",
    ))

    # ── 3. FDR SIGNIFICANT ────────────────────────────────────────
    stouffer_ok = stouffer_p is not None and stouffer_p < MAX_STOUFFER_P
    criteria.append(CertificateCriterion(
        name="fdr_significant",
        passed=stouffer_ok,
        value=stouffer_p,
        threshold=f"< {MAX_STOUFFER_P}",
        note=f"stouffer_p = {stouffer_p:.2e}" if stouffer_p is not None else "no stouffer_p",
    ))

    # ── 4. UNEXPLAINED ────────────────────────────────────────────
    if unexplainability is not None:
        unex_score = unexplainability.get("unexplainability_score",
                                         unexplainability.get("score", 0.0))
        best_template = unexplainability.get("best_template", "none")
    else:
        unex_score = 0.0
        best_template = "not computed"

    unex_ok = unex_score > MIN_UNEXPLAINABILITY
    criteria.append(CertificateCriterion(
        name="unexplained",
        passed=unex_ok,
        value=unex_score,
        threshold=f"> {MIN_UNEXPLAINABILITY}",
        note=f"unexplainability = {unex_score:.3f}, best template = {best_template}",
    ))

    # ── 5. RED-TEAM SURVIVES ──────────────────────────────────────
    if red_team_verdict is not None:
        recommendation = red_team_verdict.get("recommendation", "UNKNOWN")
        risk_level = red_team_verdict.get("risk_level", "UNKNOWN")
    else:
        recommendation = "NOT_COMPUTED"
        risk_level = "UNKNOWN"

    # Audit fix C2: require ESCALATE, not just not-DEMOTE.
    # MONITOR means "further investigation needed" — insufficient for certification.
    # Only ESCALATE (low risk, all checks passed) should gate certification.
    rt_ok = recommendation == "ESCALATE"
    criteria.append(CertificateCriterion(
        name="red_team_survives",
        passed=rt_ok,
        value=recommendation,
        threshold="== ESCALATE",
        note=f"Red-Team: {recommendation} (risk={risk_level})",
    ))

    # ── 6. NO SINGLE-CHANNEL DOMINANCE ────────────────────────────
    max_channel_frac = 0.0
    dominant_channel = "none"
    if total_score > 0:
        for ch_name, ch_data in channel_scores.items():
            if not isinstance(ch_data, dict):
                continue
            ch_score = ch_data.get("score", 0.0)
            frac = ch_score / total_score if total_score > 0 else 0.0
            if frac > max_channel_frac:
                max_channel_frac = frac
                dominant_channel = ch_name

    dominance_ok = max_channel_frac <= MAX_SINGLE_CHANNEL_FRAC or total_score == 0
    criteria.append(CertificateCriterion(
        name="no_single_channel_dominance",
        passed=dominance_ok,
        value=round(max_channel_frac, 3),
        threshold=f"<= {MAX_SINGLE_CHANNEL_FRAC}",
        note=f"dominant channel: {dominant_channel} ({max_channel_frac:.1%} of score)",
    ))

    # ── 7. DISTANCE REASONABLE ────────────────────────────────────
    if distance_pc is not None:
        dist_ok = 0 < distance_pc <= MAX_DISTANCE_PC
    else:
        dist_ok = False  # Unknown distance is disqualifying

    criteria.append(CertificateCriterion(
        name="distance_reasonable",
        passed=dist_ok,
        value=distance_pc,
        threshold=f"<= {MAX_DISTANCE_PC} pc",
        note=f"d = {distance_pc:.1f} pc" if distance_pc else "distance unknown",
    ))

    # ── 8. REPLICATION ────────────────────────────────────────────
    repl_ok = pipeline_run_count >= 2
    criteria.append(CertificateCriterion(
        name="replication",
        passed=repl_ok,
        value=pipeline_run_count,
        threshold=">= 2 independent runs",
        note=f"{pipeline_run_count} pipeline run(s)"
             + (" (first pass — replication pending)" if pipeline_run_count < 2 else ""),
    ))

    # ── Grade ─────────────────────────────────────────────────────
    n_passed = sum(1 for c in criteria if c.passed)
    n_total = len(criteria)

    # Certified requires ALL criteria (except replication on first pass)
    # On first pass (run_count=1), we allow replication to be pending
    required_for_cert = [c for c in criteria if c.name != "replication"]
    all_required_pass = all(c.passed for c in required_for_cert)

    if all_required_pass and repl_ok:
        certified = True
        grade = "CERTIFIED"
    elif all_required_pass:
        certified = False
        grade = "PROVISIONAL"  # Passes everything except replication
    elif n_passed >= n_total * 0.75:
        certified = False
        grade = "PARTIAL"
    else:
        certified = False
        grade = "FAIL"

    failed = [c.name for c in criteria if not c.passed]
    if certified:
        summary = f"TELESCOPE CANDIDATE: {target_id} passes all {n_total} criteria"
    elif grade == "PROVISIONAL":
        summary = (f"PROVISIONAL: {target_id} passes {n_passed}/{n_total} "
                   f"(pending: {', '.join(failed)})")
    else:
        summary = (f"{grade}: {target_id} passes {n_passed}/{n_total} "
                   f"(failed: {', '.join(failed)})")

    cert = CandidateCertificate(
        target_id=target_id,
        certified=certified,
        criteria=criteria,
        n_passed=n_passed,
        n_total=n_total,
        grade=grade,
        summary=summary,
    )

    log.info("Certificate for %s: %s (%d/%d)", target_id, grade, n_passed, n_total)
    return cert


def certify_batch(
    scored_targets: List[Dict[str, Any]],
    red_team_results: Optional[Dict[str, Dict]] = None,
    unexplainability_results: Optional[Dict[str, Dict]] = None,
) -> List[CandidateCertificate]:
    """Certify a batch of scored targets.

    Parameters
    ----------
    scored_targets : list of dict
    red_team_results : dict, optional
        target_id -> red_team_verdict
    unexplainability_results : dict, optional
        target_id -> unexplainability result

    Returns
    -------
    list of CandidateCertificate, sorted by n_passed descending
    """
    red_team_results = red_team_results or {}
    unexplainability_results = unexplainability_results or {}

    certs = []
    for t in scored_targets:
        tid = t.get("target_id", "unknown")
        cert = certify_candidate(
            scored_target=t,
            red_team_verdict=red_team_results.get(tid),
            unexplainability=unexplainability_results.get(tid),
        )
        certs.append(cert)

    # Sort: CERTIFIED first, then PROVISIONAL, then by n_passed
    grade_order = {"CERTIFIED": 0, "PROVISIONAL": 1, "PARTIAL": 2, "FAIL": 3}
    certs.sort(key=lambda c: (grade_order.get(c.grade, 9), -c.n_passed))

    return certs


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate targets against Telescope Candidate Certificate"
    )
    parser.add_argument("--report", required=True,
                        help="Path to quick_run report JSON")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top targets to evaluate")
    parser.add_argument("--output",
                        help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    with open(args.report) as f:
        report = json.load(f)

    targets = report.get("all_scored", report.get("top_targets", []))[:args.top]

    print(f"Evaluating {len(targets)} targets from {args.report}")
    print("=" * 70)

    certs = certify_batch(targets)

    for cert in certs:
        icon = {
            "CERTIFIED": "[PASS]",
            "PROVISIONAL": "[PROV]",
            "PARTIAL": "[ -- ]",
            "FAIL": "[FAIL]",
        }.get(cert.grade, "[????]")
        print(f"  {icon} {cert.target_id}: {cert.n_passed}/{cert.n_total} — {cert.summary}")

    # Summary
    n_cert = sum(1 for c in certs if c.certified)
    n_prov = sum(1 for c in certs if c.grade == "PROVISIONAL")
    n_part = sum(1 for c in certs if c.grade == "PARTIAL")
    n_fail = sum(1 for c in certs if c.grade == "FAIL")

    print(f"\n{'='*70}")
    print(f"  CERTIFIED: {n_cert}  |  PROVISIONAL: {n_prov}  |  PARTIAL: {n_part}  |  FAIL: {n_fail}")
    print(f"{'='*70}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump([c.to_dict() for c in certs], f, indent=2, default=str)
        print(f"\n  Saved to: {args.output}")
