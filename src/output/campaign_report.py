"""
Campaign reporting for Project EXODUS research campaigns.

Generates calibration validation reports (expected vs observed per-channel)
and campaign summary reports (multi-channel convergence, coverage, FDR).

Public API
----------
generate_calibration_report(results, campaign, output_dir)
    Compare observed channel scores against expected behavior.

generate_campaign_report(results, campaign, output_dir)
    Summary report for golden sample / systematic survey campaigns.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# -- Project imports ----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger, safe_json_dump, PROJECT_ROOT

log = get_logger("output.campaign_report")

_UTC = timezone.utc


# =====================================================================
#  Constants
# =====================================================================

# Channel score threshold: above this the channel is considered "triggered"
TRIGGER_THRESHOLD = 0.3

# Canonical channel names (must match src/scoring/exodus_score.py)
CANONICAL_CHANNELS = {
    "ir_excess",
    "transit_anomaly",
    "radio_anomaly",
    "gaia_photometric_anomaly",
    "habitable_zone_planet",
    "proper_motion_anomaly",
}


# =====================================================================
#  Enums
# =====================================================================

class ChannelOutcome(str, Enum):
    """Classification of a single channel-target comparison."""

    TRUE_POSITIVE = "TRUE_POSITIVE"
    TRUE_NEGATIVE = "TRUE_NEGATIVE"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    FALSE_NEGATIVE = "FALSE_NEGATIVE"
    NEUTRAL = "NEUTRAL"
    NOT_EVALUATED = "NOT_EVALUATED"


# =====================================================================
#  Data classes
# =====================================================================

@dataclass
class CalibrationReport:
    """Result of a calibration campaign validation.

    Attributes
    ----------
    campaign : str
        Campaign identifier.
    n_targets : int
        Total number of calibration targets evaluated.
    per_target : list of dict
        Per-target, per-channel classification results.
    n_true_positive : int
        Channels correctly triggered on positive controls.
    n_true_negative : int
        Channels correctly silent on negative controls.
    n_false_positive : int
        Channels incorrectly triggered on negative controls.
    n_false_negative : int
        Channels that failed to trigger on positive controls.
    pass_rate : float
        (TP + TN) / total tested, in [0, 1].
    all_passed : bool
        True if there are zero false positives and zero false negatives.
    timestamp : str
        ISO-format timestamp of report generation.
    """

    campaign: str
    n_targets: int
    per_target: List[Dict[str, Any]] = field(default_factory=list)
    n_true_positive: int = 0
    n_true_negative: int = 0
    n_false_positive: int = 0
    n_false_negative: int = 0
    n_not_evaluated: int = 0
    pass_rate: float = 0.0
    all_passed: bool = False
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(tz=_UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign": self.campaign,
            "n_targets": self.n_targets,
            "per_target": self.per_target,
            "n_true_positive": self.n_true_positive,
            "n_true_negative": self.n_true_negative,
            "n_false_positive": self.n_false_positive,
            "n_false_negative": self.n_false_negative,
            "n_not_evaluated": self.n_not_evaluated,
            "pass_rate": self.pass_rate,
            "all_passed": self.all_passed,
            "timestamp": self.timestamp,
        }


@dataclass
class CampaignReport:
    """Result of a non-calibration campaign summary.

    Attributes
    ----------
    campaign : str
        Campaign identifier.
    n_targets : int
        Total number of targets evaluated.
    convergent_targets : list of dict
        Targets with 2+ active channels.
    coverage_matrix : dict
        target_id -> list of channel names that had data.
    fdr_targets : list of dict
        Targets flagged as FDR-significant.
    timestamp : str
        ISO-format timestamp.
    """

    campaign: str
    n_targets: int = 0
    convergent_targets: List[Dict[str, Any]] = field(default_factory=list)
    coverage_matrix: Dict[str, List[str]] = field(default_factory=dict)
    fdr_targets: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(tz=_UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign": self.campaign,
            "n_targets": self.n_targets,
            "convergent_targets": self.convergent_targets,
            "coverage_matrix": self.coverage_matrix,
            "fdr_targets": self.fdr_targets,
            "timestamp": self.timestamp,
        }


# =====================================================================
#  Internal helpers
# =====================================================================

def _get_channel_score(target: Dict[str, Any], channel_name: str) -> Optional[float]:
    """Extract a numeric channel score from a target result dict.

    Looks inside ``target["exodus_score"]["channel_scores"][channel_name]``
    and returns the ``score`` value, or ``None`` if unavailable.
    """
    exodus_score = target.get("exodus_score", {})
    if not exodus_score:
        return None

    channel_scores = exodus_score.get("channel_scores", {})
    channel = channel_scores.get(channel_name)
    if channel is None:
        return None

    if isinstance(channel, dict):
        return channel.get("score")
    # Support dataclass-style objects
    return getattr(channel, "score", None)


def _channel_has_data(target: Dict[str, Any], channel_name: str) -> bool:
    """Check whether a channel had real observational data for scoring.

    Returns False if the channel score was produced with no underlying
    data (i.e. the scorer filled in a 0.0 placeholder because the
    channel's raw input was ``None``).  This is critical for honest
    calibration: a channel that was never exercised must not be counted
    as a TRUE_NEGATIVE just because its placeholder score is below
    threshold.
    """
    exodus_score = target.get("exodus_score", {})
    if not exodus_score:
        return False

    channel_scores = exodus_score.get("channel_scores", {})
    channel = channel_scores.get(channel_name)
    if channel is None:
        return False

    # Check the details dict for the "no data provided" sentinel
    if isinstance(channel, dict):
        details = channel.get("details", {})
        if isinstance(details, dict) and details.get("reason") == "no data provided":
            return False
        return True
    # Support dataclass-style objects
    details = getattr(channel, "details", {})
    if isinstance(details, dict) and details.get("reason") == "no data provided":
        return False
    return True


def _classify_channel(
    expected: str,
    score: Optional[float],
    threshold: float = TRIGGER_THRESHOLD,
    has_data: bool = True,
) -> ChannelOutcome:
    """Classify a single channel-target pair.

    Parameters
    ----------
    expected : str
        One of ``"positive"``, ``"negative"``, or ``"neutral"``.
    score : float or None
        The observed channel score.
    threshold : float
        Score above which the channel is deemed "triggered".
    has_data : bool
        Whether the channel had real observational data.  When False,
        the channel was never exercised and must not be counted as
        a TRUE_NEGATIVE.  This prevents false confidence in calibration.

    Returns
    -------
    ChannelOutcome
    """
    if expected == "neutral" or score is None:
        return ChannelOutcome.NEUTRAL

    # Channel had no real data — cannot claim TP/TN/FP/FN
    if not has_data:
        return ChannelOutcome.NOT_EVALUATED

    triggered = score > threshold

    if expected == "positive":
        return ChannelOutcome.TRUE_POSITIVE if triggered else ChannelOutcome.FALSE_NEGATIVE
    elif expected == "negative":
        return ChannelOutcome.FALSE_POSITIVE if triggered else ChannelOutcome.TRUE_NEGATIVE

    return ChannelOutcome.NEUTRAL


# =====================================================================
#  Calibration report generation
# =====================================================================

def generate_calibration_report(
    results: List[Dict[str, Any]],
    campaign_targets: Any,
    output_dir: Optional[Path] = None,
) -> CalibrationReport:
    """Compare observed channel scores against expected calibration behavior.

    Parameters
    ----------
    results : list of dict
        Target dicts after scoring, each containing ``exodus_score`` with
        channel scores.
    campaign_targets : CampaignTargets
        The loaded campaign targets dataclass with expected_channels.
    output_dir : Path, optional
        Directory to write JSON and Markdown reports.  If ``None``, defaults
        to ``data/reports/``.

    Returns
    -------
    CalibrationReport
        Full calibration validation results.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "reports"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    campaign_name = getattr(campaign_targets, "campaign", "calibration")
    log.info(
        "Generating calibration report for campaign '%s' (%d targets)",
        campaign_name, len(results),
    )

    # Build a lookup from target_id -> expected_channels
    expected_lookup: Dict[str, Dict[str, str]] = {}
    if hasattr(campaign_targets, "targets"):
        for t in campaign_targets.targets:
            tid = t.get("target_id", "")
            expected_lookup[tid] = t.get("expected_channels", {})

    # Evaluate each target
    per_target: List[Dict[str, Any]] = []
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    total_not_evaluated = 0

    for target in results:
        tid = target.get("target_id", "unknown")
        expected_channels = expected_lookup.get(tid, {})

        channel_results: List[Dict[str, Any]] = []

        for ch_name in sorted(CANONICAL_CHANNELS):
            expected = expected_channels.get(ch_name, "neutral")
            score = _get_channel_score(target, ch_name)
            has_data = _channel_has_data(target, ch_name)
            outcome = _classify_channel(expected, score, has_data=has_data)

            channel_results.append({
                "channel": ch_name,
                "expected": expected,
                "score": round(score, 4) if score is not None else None,
                "triggered": score is not None and score > TRIGGER_THRESHOLD,
                "has_data": has_data,
                "outcome": outcome.value,
            })

            if outcome == ChannelOutcome.TRUE_POSITIVE:
                total_tp += 1
            elif outcome == ChannelOutcome.TRUE_NEGATIVE:
                total_tn += 1
            elif outcome == ChannelOutcome.FALSE_POSITIVE:
                total_fp += 1
            elif outcome == ChannelOutcome.FALSE_NEGATIVE:
                total_fn += 1
            elif outcome == ChannelOutcome.NOT_EVALUATED:
                total_not_evaluated += 1

        per_target.append({
            "target_id": tid,
            "channels": channel_results,
        })

    # Compute pass rate
    total_tested = total_tp + total_tn + total_fp + total_fn
    pass_rate = (total_tp + total_tn) / total_tested if total_tested > 0 else 0.0

    # Calibration MUST fail if required channels (expected="positive" or
    # "negative") were never exercised.  A NOT_EVALUATED required channel
    # means the pipeline was never tested for that detection mode.
    has_untested_required = total_not_evaluated > 0

    report = CalibrationReport(
        campaign=campaign_name,
        n_targets=len(results),
        per_target=per_target,
        n_true_positive=total_tp,
        n_true_negative=total_tn,
        n_false_positive=total_fp,
        n_false_negative=total_fn,
        n_not_evaluated=total_not_evaluated,
        pass_rate=pass_rate,
        all_passed=(
            total_fp == 0
            and total_fn == 0
            and not has_untested_required
        ),
    )

    if total_not_evaluated > 0:
        log.warning(
            "Calibration BLOCKED: %d required channel-target pairs had NO "
            "DATA (NOT_EVALUATED). Calibration cannot pass with untested "
            "required channels -- these represent coverage gaps, not "
            "true negatives.",
            total_not_evaluated,
        )

    # Save JSON report
    ts = datetime.now(tz=_UTC).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"calibration_{campaign_name}_{ts}.json"
    with open(json_path, "w") as f:
        safe_json_dump(report.to_dict(), f, indent=2)
    log.info("Calibration JSON report: %s", json_path)

    # Save Markdown report
    md = _render_calibration_markdown(report)
    md_path = output_dir / f"calibration_{campaign_name}_{ts}.md"
    md_path.write_text(md, encoding="utf-8")
    log.info("Calibration Markdown report: %s", md_path)

    return report


def _render_calibration_markdown(report: CalibrationReport) -> str:
    """Render a human-readable Markdown calibration report."""
    lines = [
        f"# Calibration Report: {report.campaign}",
        "",
        f"**Generated:** {report.timestamp}",
        f"**Targets evaluated:** {report.n_targets}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Metric          | Count |",
        f"|-----------------|-------|",
        f"| True Positive   | {report.n_true_positive:>5d} |",
        f"| True Negative   | {report.n_true_negative:>5d} |",
        f"| False Positive  | {report.n_false_positive:>5d} |",
        f"| False Negative  | {report.n_false_negative:>5d} |",
        f"| Not Evaluated   | {report.n_not_evaluated:>5d} |",
        f"| **Pass Rate**   | **{report.pass_rate:.1%}** |",
        "",
        f"**Overall:** {'ALL PASSED' if report.all_passed else 'FAILURES DETECTED'}",
        "",
        "---",
        "",
        "## Per-Target Results",
        "",
    ]

    for entry in report.per_target:
        tid = entry["target_id"]
        lines.append(f"### {tid}")
        lines.append("")
        lines.append("| Channel | Expected | Score | Triggered | Outcome |")
        lines.append("|---------|----------|-------|-----------|---------|")

        for ch in entry["channels"]:
            score_str = f"{ch['score']:.4f}" if ch["score"] is not None else "N/A"
            trig_str = "YES" if ch["triggered"] else "no"
            outcome = ch["outcome"]
            # Mark failures and unevaluated with emphasis
            if outcome in ("FALSE_POSITIVE", "FALSE_NEGATIVE"):
                outcome = f"**{outcome}**"
            elif outcome == "NOT_EVALUATED":
                outcome = f"*{outcome}*"
            lines.append(
                f"| {ch['channel']:<30s} | {ch['expected']:<8s} | "
                f"{score_str:>7s} | {trig_str:<9s} | {outcome} |"
            )
        lines.append("")

    lines.extend([
        "---",
        f"*Generated by Project EXODUS Campaign Reporting Pipeline v1.0*",
    ])

    return "\n".join(lines) + "\n"


# =====================================================================
#  Campaign report generation (non-calibration)
# =====================================================================

def generate_campaign_report(
    results: List[Dict[str, Any]],
    campaign_targets: Any,
    output_dir: Optional[Path] = None,
) -> CampaignReport:
    """Generate a campaign summary for golden sample or systematic surveys.

    Produces a multi-channel convergence summary, coverage matrix, and
    highlights FDR-significant targets.

    Parameters
    ----------
    results : list of dict
        Target dicts after scoring, each containing ``exodus_score``.
    campaign_targets : CampaignTargets
        The loaded campaign targets dataclass.
    output_dir : Path, optional
        Directory to write JSON and Markdown reports.

    Returns
    -------
    CampaignReport
        Campaign summary report.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "reports"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    campaign_name = getattr(campaign_targets, "campaign", "campaign")
    log.info(
        "Generating campaign report for '%s' (%d targets)",
        campaign_name, len(results),
    )

    convergent_targets: List[Dict[str, Any]] = []
    coverage_matrix: Dict[str, List[str]] = {}
    fdr_targets: List[Dict[str, Any]] = []

    for target in results:
        tid = target.get("target_id", "unknown")
        exodus_score = target.get("exodus_score", {})
        channel_scores = exodus_score.get("channel_scores", {})

        # Determine which channels had real observational data.
        # Use _channel_has_data() to exclude placeholder scores (0.0 with
        # reason="no data provided") that would overstate coverage.
        channels_with_data: List[str] = []
        active_channels: List[str] = []

        for ch_name in sorted(CANONICAL_CHANNELS):
            if not _channel_has_data(target, ch_name):
                continue
            channels_with_data.append(ch_name)
            score = _get_channel_score(target, ch_name)
            if score is not None and score > TRIGGER_THRESHOLD:
                active_channels.append(ch_name)

        coverage_matrix[tid] = channels_with_data

        # Multi-channel convergence: 2+ active channels
        if len(active_channels) >= 2:
            total = exodus_score.get("total_score", 0)
            convergent_targets.append({
                "target_id": tid,
                "n_active": len(active_channels),
                "active_channels": active_channels,
                "total_score": total,
            })

        # FDR significance
        fdr_sig = exodus_score.get("fdr_significant", False)
        if fdr_sig:
            fdr_targets.append({
                "target_id": tid,
                "total_score": exodus_score.get("total_score", 0),
                "combined_p": exodus_score.get("combined_p"),
                "n_active": exodus_score.get("n_active_channels", 0),
            })

    # Sort convergent targets by number of active channels, then score
    convergent_targets.sort(
        key=lambda x: (x["n_active"], x["total_score"]),
        reverse=True,
    )
    fdr_targets.sort(key=lambda x: x["total_score"], reverse=True)

    report = CampaignReport(
        campaign=campaign_name,
        n_targets=len(results),
        convergent_targets=convergent_targets,
        coverage_matrix=coverage_matrix,
        fdr_targets=fdr_targets,
    )

    # Save JSON report
    ts = datetime.now(tz=_UTC).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"campaign_{campaign_name}_{ts}.json"
    with open(json_path, "w") as f:
        safe_json_dump(report.to_dict(), f, indent=2)
    log.info("Campaign JSON report: %s", json_path)

    # Save Markdown report
    md = _render_campaign_markdown(report)
    md_path = output_dir / f"campaign_{campaign_name}_{ts}.md"
    md_path.write_text(md, encoding="utf-8")
    log.info("Campaign Markdown report: %s", md_path)

    return report


def _render_campaign_markdown(report: CampaignReport) -> str:
    """Render a human-readable Markdown campaign report."""
    lines = [
        f"# Campaign Report: {report.campaign}",
        "",
        f"**Generated:** {report.timestamp}",
        f"**Targets evaluated:** {report.n_targets}",
        "",
        "---",
        "",
        "## Multi-Channel Convergence",
        "",
        f"Targets with 2+ active channels: **{len(report.convergent_targets)}**",
        "",
    ]

    if report.convergent_targets:
        lines.append("| Target | Active | Score | Channels |")
        lines.append("|--------|--------|-------|----------|")
        for ct in report.convergent_targets:
            ch_str = ", ".join(ct["active_channels"])
            lines.append(
                f"| {ct['target_id']:<24s} | {ct['n_active']:>6d} | "
                f"{ct['total_score']:>7.3f} | {ch_str} |"
            )
        lines.append("")
    else:
        lines.append("No targets with multi-channel convergence.")
        lines.append("")

    # Coverage matrix
    lines.extend([
        "---",
        "",
        "## Coverage Matrix",
        "",
    ])

    all_channels = sorted(CANONICAL_CHANNELS)
    header = "| Target | " + " | ".join(ch[:12] for ch in all_channels) + " |"
    sep = "|--------|" + "|".join("-" * 14 for _ in all_channels) + "|"
    lines.append(header)
    lines.append(sep)

    for tid, channels in sorted(report.coverage_matrix.items()):
        ch_set = set(channels)
        cols = " | ".join(
            ("  Y " if ch in ch_set else "  . ").center(12)
            for ch in all_channels
        )
        lines.append(f"| {tid:<24s} | {cols} |")

    lines.append("")

    # FDR-significant targets
    lines.extend([
        "---",
        "",
        "## FDR-Significant Targets",
        "",
    ])

    if report.fdr_targets:
        lines.append("| Target | Score | p-value | Active |")
        lines.append("|--------|-------|---------|--------|")
        for ft in report.fdr_targets:
            p_str = f"{ft['combined_p']:.2e}" if ft["combined_p"] is not None else "N/A"
            lines.append(
                f"| {ft['target_id']:<24s} | {ft['total_score']:>7.3f} | "
                f"{p_str:>7s} | {ft['n_active']:>6d} |"
            )
        lines.append("")
    else:
        lines.append("No FDR-significant targets in this campaign.")
        lines.append("")

    lines.extend([
        "---",
        f"*Generated by Project EXODUS Campaign Reporting Pipeline v1.0*",
    ])

    return "\n".join(lines) + "\n"


# =====================================================================
#  CLI demo / self-test
# =====================================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 70)
    print("  Project EXODUS -- Campaign Report Demo")
    print("=" * 70)

    rng = np.random.default_rng(seed=42)

    # ------------------------------------------------------------------
    # Build mock CampaignTargets (mimics target_loader.CampaignTargets)
    # ------------------------------------------------------------------
    class MockCampaignTargets:
        def __init__(self, campaign, targets):
            self.campaign = campaign
            self.targets = targets

    mock_targets_raw = [
        {
            "target_id": "Vega",
            "host_star": "alpha_Lyr",
            "ra": 279.2347,
            "dec": 38.7837,
            "is_positive_control": True,
            "expected_channels": {"ir_excess": "positive"},
            "notes": "Known debris disk -- should trigger IR excess",
        },
        {
            "target_id": "51_Peg",
            "host_star": "51_Peg",
            "ra": 344.3667,
            "dec": 20.7689,
            "is_negative_control": True,
            "expected_channels": {
                "ir_excess": "negative",
                "transit_anomaly": "negative",
            },
            "notes": "Negative control",
        },
        {
            "target_id": "Proxima_Cen_b",
            "host_star": "Proxima_Centauri",
            "ra": 217.4290,
            "dec": -62.6794,
            "expected_channels": {},
            "notes": "Neutral target",
        },
    ]

    campaign_targets = MockCampaignTargets(
        campaign="test_calibration",
        targets=mock_targets_raw,
    )

    # ------------------------------------------------------------------
    # Build mock scored results
    # ------------------------------------------------------------------
    mock_results = []
    for t in mock_targets_raw:
        tid = t["target_id"]
        # Simulate channel scores
        channel_scores = {}
        for ch in sorted(CANONICAL_CHANNELS):
            # Vega should have high IR excess, 51 Peg should have low
            if tid == "Vega" and ch == "ir_excess":
                score = 0.85
            elif tid == "51_Peg" and ch in ("ir_excess", "transit_anomaly"):
                score = 0.10
            else:
                score = float(rng.uniform(0.05, 0.25))

            channel_scores[ch] = {
                "channel_name": ch,
                "score": score,
                "is_active": score > TRIGGER_THRESHOLD,
                "details": {},
            }

        active = [v for v in channel_scores.values() if v["is_active"]]
        active_scores = [v["score"] for v in active]
        geo = float(np.exp(np.mean(np.log(active_scores)))) if active_scores else 0
        bonus = 2 ** max(len(active) - 1, 0)

        mock_results.append({
            "target_id": tid,
            "ra": t["ra"],
            "dec": t["dec"],
            "exodus_score": {
                "total_score": geo * bonus,
                "channel_scores": channel_scores,
                "n_active_channels": len(active),
                "geo_mean": geo,
                "convergence_bonus": bonus,
                "fdr_significant": len(active) >= 3,
                "combined_p": rng.uniform(0.001, 0.05) if len(active) >= 2 else None,
            },
        })

    # ------------------------------------------------------------------
    # Test 1: Calibration report
    # ------------------------------------------------------------------
    print("\n[1] Calibration Report")
    print("-" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        cal_report = generate_calibration_report(
            mock_results, campaign_targets, output_dir=Path(tmpdir)
        )

        print(f"  Campaign:   {cal_report.campaign}")
        print(f"  Targets:    {cal_report.n_targets}")
        print(f"  TP: {cal_report.n_true_positive}  TN: {cal_report.n_true_negative}  "
              f"FP: {cal_report.n_false_positive}  FN: {cal_report.n_false_negative}")
        print(f"  Pass rate:  {cal_report.pass_rate:.1%}")
        print(f"  All passed: {cal_report.all_passed}")

        # Verify Vega IR excess triggered as TP
        vega_entry = next(
            e for e in cal_report.per_target if e["target_id"] == "Vega"
        )
        ir_ch = next(
            c for c in vega_entry["channels"] if c["channel"] == "ir_excess"
        )
        assert ir_ch["outcome"] == "TRUE_POSITIVE", \
            f"Expected Vega IR excess to be TP, got {ir_ch['outcome']}"
        print("  Vega IR excess correctly classified as TRUE_POSITIVE: PASS")

        # Verify 51 Peg IR excess is TN
        peg_entry = next(
            e for e in cal_report.per_target if e["target_id"] == "51_Peg"
        )
        ir_ch = next(
            c for c in peg_entry["channels"] if c["channel"] == "ir_excess"
        )
        assert ir_ch["outcome"] == "TRUE_NEGATIVE", \
            f"Expected 51 Peg IR excess to be TN, got {ir_ch['outcome']}"
        print("  51 Peg IR excess correctly classified as TRUE_NEGATIVE: PASS")

        # Verify JSON was written
        json_files = list(Path(tmpdir).glob("calibration_*.json"))
        assert len(json_files) == 1, f"Expected 1 JSON file, got {len(json_files)}"
        print(f"  JSON report written: PASS")

        # Verify Markdown was written
        md_files = list(Path(tmpdir).glob("calibration_*.md"))
        assert len(md_files) == 1, f"Expected 1 MD file, got {len(md_files)}"
        print(f"  Markdown report written: PASS")

    # ------------------------------------------------------------------
    # Test 2: Campaign report (non-calibration)
    # ------------------------------------------------------------------
    print("\n[2] Campaign Report")
    print("-" * 50)

    golden_targets = MockCampaignTargets(
        campaign="golden_sample",
        targets=mock_targets_raw,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        camp_report = generate_campaign_report(
            mock_results, golden_targets, output_dir=Path(tmpdir)
        )

        print(f"  Campaign:        {camp_report.campaign}")
        print(f"  Targets:         {camp_report.n_targets}")
        print(f"  Convergent (2+): {len(camp_report.convergent_targets)}")
        print(f"  FDR-significant: {len(camp_report.fdr_targets)}")
        print(f"  Coverage keys:   {len(camp_report.coverage_matrix)}")

        # Verify JSON and Markdown written
        json_files = list(Path(tmpdir).glob("campaign_*.json"))
        md_files = list(Path(tmpdir).glob("campaign_*.md"))
        assert len(json_files) == 1
        assert len(md_files) == 1
        print(f"  JSON + Markdown written: PASS")

    # ------------------------------------------------------------------
    # Test 3: Edge cases
    # ------------------------------------------------------------------
    print("\n[3] Edge Cases")
    print("-" * 50)

    # Empty results
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_campaign = MockCampaignTargets(campaign="empty", targets=[])
        empty_report = generate_calibration_report(
            [], empty_campaign, output_dir=Path(tmpdir)
        )
        assert empty_report.n_targets == 0
        assert empty_report.pass_rate == 0.0
        print("  Empty results handled: PASS")

    # Target with no exodus_score
    with tempfile.TemporaryDirectory() as tmpdir:
        no_score_results = [{"target_id": "no_score_star", "ra": 0, "dec": 0}]
        no_score_campaign = MockCampaignTargets(
            campaign="no_score",
            targets=[{"target_id": "no_score_star", "expected_channels": {}}],
        )
        ns_report = generate_calibration_report(
            no_score_results, no_score_campaign, output_dir=Path(tmpdir)
        )
        assert ns_report.n_targets == 1
        print("  Missing exodus_score handled gracefully: PASS")

    print("\n" + "=" * 70)
    print("  All tests passed.")
    print("=" * 70)
