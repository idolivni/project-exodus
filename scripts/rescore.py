#!/usr/bin/env python3
"""
Project EXODUS — Re-Score Existing Data with New Scoring Modes
==============================================================

Takes an existing report JSON (with per-target channel data) and re-scores
using the convergence-priority mode. This avoids re-querying any data —
we just re-run the scoring formula on the already-gathered channel results.

The trick: the scored target dicts in reports contain channel_scores with
full details, which we can feed back into the scorer. For Tier 0 reports
where the old format only saved top_targets (not all_scored), we re-score
those available.

Usage
-----
    # Re-score a specific report with convergence-priority mode
    python scripts/rescore.py --report data/reports/quick_run_YYYYMMDD_HHMMSS.json --convergence-priority

    # Re-score ALL reports in a directory
    python scripts/rescore.py --report-dir data/reports/ --convergence-priority

    # Re-score with custom threshold
    python scripts/rescore.py --report data/reports/quick_run_YYYYMMDD_HHMMSS.json --threshold 0.15
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger, safe_json_dump

log = get_logger("rescore")


def reconstruct_target_data(scored_target: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct a target_data dict from a scored result in a report.

    The scorer's score_target() expects:
        {"target_id": ..., "ra": ..., "dec": ...,
         "ir_excess": {...}, "proper_motion_anomaly": {...}, ...}

    The stored scored result has:
        {"target_id": ..., "ra": ..., "dec": ...,
         "channel_scores": {"ir_excess": {"details": {...}, ...}, ...}}

    We extract the details from each channel_score to reconstruct the input.
    """
    td = {
        "target_id": scored_target.get("target_id"),
        "ra": scored_target.get("ra"),
        "dec": scored_target.get("dec"),
        "distance_pc": scored_target.get("distance_pc"),
        "host_star": scored_target.get("host_star"),
    }

    for ch_name, ch_data in scored_target.get("channel_scores", {}).items():
        details = ch_data.get("details", {})
        if details and details.get("reason") not in ("no data provided", "simulated data excluded"):
            td[ch_name] = details

    return td


def rescore_targets(
    scored_targets: List[Dict[str, Any]],
    convergence_priority: bool = False,
    threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Re-score a list of previously-scored targets with new parameters."""
    from src.scoring.exodus_score import EXODUSScorer

    kwargs = {"convergence_priority": convergence_priority}
    if threshold is not None:
        kwargs["threshold"] = threshold

    scorer = EXODUSScorer(**kwargs)

    # Reconstruct target_data dicts
    target_data_list = [reconstruct_target_data(st) for st in scored_targets]

    # Score all
    results = scorer.score_all(target_data_list)

    # Convert to dicts and enrich
    rescored = []
    for r in results:
        d = r.to_dict()
        # Carry forward distance/host from original
        orig_lookup = {st["target_id"]: st for st in scored_targets}
        orig = orig_lookup.get(d.get("target_id"), {})
        d["distance_pc"] = orig.get("distance_pc")
        d["host_star"] = orig.get("host_star")
        rescored.append(d)

    return rescored


def rescore_report(
    report_path: Path,
    convergence_priority: bool = False,
    threshold: Optional[float] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Re-score a single report file."""
    log.info("Re-scoring %s (convergence_priority=%s)", report_path.name, convergence_priority)

    with open(report_path) as f:
        rpt = json.load(f)

    # Get scored targets (prefer all_scored, fall back to top_targets)
    scored = rpt.get("all_scored") or rpt.get("top_targets") or []
    if not scored:
        log.warning("No scored targets in %s, skipping", report_path.name)
        return {}

    rescored = rescore_targets(scored, convergence_priority=convergence_priority, threshold=threshold)

    # Build comparison summary
    original_by_id = {t["target_id"]: t for t in scored}
    comparisons = []
    for t in rescored[:50]:
        tid = t["target_id"]
        orig = original_by_id.get(tid, {})
        comparisons.append({
            "target_id": tid,
            "host_star": t.get("host_star"),
            "original_score": round(orig.get("total_score", 0), 4),
            "rescored": round(t.get("total_score", 0), 4),
            "original_active": orig.get("n_active_channels", 0),
            "rescored_active": t.get("n_active_channels", 0),
            "delta": round(t.get("total_score", 0) - orig.get("total_score", 0), 4),
            "rank_change": (orig.get("rank") or 999) - (t.get("rank") or 999),
        })

    result = {
        "analysis": "rescore",
        "source_report": report_path.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "convergence_priority": convergence_priority,
            "threshold": threshold,
        },
        "n_rescored": len(rescored),
        "n_with_data": sum(1 for t in rescored if t.get("n_active_channels", 0) > 0),
        "n_multi_channel_2plus": sum(1 for t in rescored if t.get("n_active_channels", 0) >= 2),
        "n_multi_channel_3plus": sum(1 for t in rescored if t.get("n_active_channels", 0) >= 3),
        "top_20_rescored": rescored[:20],
        "comparisons": comparisons,
    }

    # Print
    _print_rescore_report(result)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fp:
            safe_json_dump(result, fp)
        log.info("Rescored results saved to %s", output_path)

    return result


def rescore_all_reports(
    report_dir: Path,
    convergence_priority: bool = False,
    threshold: Optional[float] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Re-score all reports, merge unique targets, rank globally."""
    log.info("Re-scoring all reports in %s", report_dir)

    # Collect all unique targets (highest original score wins)
    all_targets = {}
    report_files = sorted(report_dir.glob("quick_run_*.json"))

    for rpath in report_files:
        try:
            with open(rpath) as f:
                rpt = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Skipping %s: %s", rpath.name, exc)
            continue

        scored = rpt.get("all_scored") or rpt.get("top_targets") or []
        for t in scored:
            tid = t.get("target_id")
            if not tid:
                continue
            t["_source_report"] = rpath.name
            if tid not in all_targets or t.get("total_score", 0) > all_targets[tid].get("total_score", 0):
                all_targets[tid] = t

    log.info("Collected %d unique targets from %d reports", len(all_targets), len(report_files))

    targets_list = list(all_targets.values())
    rescored = rescore_targets(targets_list, convergence_priority=convergence_priority, threshold=threshold)

    # Build comparison
    original_by_id = {t["target_id"]: t for t in targets_list}
    comparisons = []
    for t in rescored[:50]:
        tid = t["target_id"]
        orig = original_by_id.get(tid, {})
        comparisons.append({
            "target_id": tid,
            "host_star": t.get("host_star"),
            "original_score": round(orig.get("total_score", 0), 4),
            "rescored": round(t.get("total_score", 0), 4),
            "original_active": orig.get("n_active_channels", 0),
            "rescored_active": t.get("n_active_channels", 0),
            "source_report": orig.get("_source_report"),
        })

    result = {
        "analysis": "rescore_all",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "convergence_priority": convergence_priority,
            "threshold": threshold,
        },
        "n_reports": len(report_files),
        "n_unique_targets": len(all_targets),
        "n_rescored": len(rescored),
        "n_with_data": sum(1 for t in rescored if t.get("n_active_channels", 0) > 0),
        "n_multi_channel_2plus": sum(1 for t in rescored if t.get("n_active_channels", 0) >= 2),
        "n_multi_channel_3plus": sum(1 for t in rescored if t.get("n_active_channels", 0) >= 3),
        "top_30_rescored": rescored[:30],
        "comparisons": comparisons,
    }

    _print_rescore_report(result)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fp:
            safe_json_dump(result, fp)
        log.info("Global rescored results saved to %s", output_path)

    return result


def _print_rescore_report(result: Dict[str, Any]) -> None:
    """Print human-readable rescore comparison."""
    p = result["parameters"]
    mode = "CONVERGENCE-PRIORITY" if p.get("convergence_priority") else "standard"
    thresh = p.get("threshold") or (0.15 if p.get("convergence_priority") else 0.3)

    print()
    print("=" * 78)
    print(f"  PROJECT EXODUS — RE-SCORE ANALYSIS ({mode})")
    print("=" * 78)
    n_rpts = result.get("n_reports", 0)
    source = result.get("source_report", f"{n_rpts} reports")
    print(f"  Source: {source}")
    print(f"  Mode: {mode}  |  Threshold: {thresh}")
    print(f"  Targets: {result.get('n_rescored', 0)} total, "
          f"{result.get('n_with_data', 0)} with data")
    print(f"  Multi-channel (2+): {result.get('n_multi_channel_2plus', 0)}")
    print(f"  Multi-channel (3+): {result.get('n_multi_channel_3plus', 0)}")
    print()

    comparisons = result.get("comparisons", [])
    if comparisons:
        print(f"  {'Target':<30} {'Orig':>8} {'New':>8} {'Orig Ch':>7} {'New Ch':>6}")
        print("  " + "-" * 65)
        for c in comparisons[:30]:
            arrow = ">>>" if c.get("rescored", 0) > c.get("original_score", 0) else "   "
            print(f"  {c['target_id']:<30} {c.get('original_score', 0):>8.4f} "
                  f"{c.get('rescored', 0):>8.4f} {c.get('original_active', 0):>7} "
                  f"{c.get('rescored_active', 0):>6} {arrow}")

    print()
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="EXODUS Re-Score Analysis")
    parser.add_argument("--report", type=str, help="Path to a single report JSON")
    parser.add_argument("--report-dir", type=str, help="Directory of report JSONs (re-score all)")
    parser.add_argument("--convergence-priority", action="store_true",
                        help="Use convergence-priority scoring mode")
    parser.add_argument("--threshold", type=float, help="Custom activation threshold")
    parser.add_argument("--output", type=str, help="Output JSON path")
    args = parser.parse_args()

    if not args.report and not args.report_dir:
        # Default: re-score all reports
        args.report_dir = str(PROJECT_ROOT / "data" / "reports")

    if args.report:
        output = Path(args.output) if args.output else (
            PROJECT_ROOT / "data" / "reports" / f"rescored_{Path(args.report).stem}.json"
        )
        rescore_report(
            Path(args.report),
            convergence_priority=args.convergence_priority,
            threshold=args.threshold,
            output_path=output,
        )
    else:
        output = Path(args.output) if args.output else (
            PROJECT_ROOT / "data" / "reports" / "rescored_global.json"
        )
        rescore_all_reports(
            Path(args.report_dir),
            convergence_priority=args.convergence_priority,
            threshold=args.threshold,
            output_path=output,
        )


if __name__ == "__main__":
    main()
