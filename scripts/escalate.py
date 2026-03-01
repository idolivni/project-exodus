#!/usr/bin/env python
"""
EXODUS Escalation Script — Extract top targets from a Tier 0 report and
prepare a Tier 1 follow-up target file.

Usage
-----
  # Extract top 50 targets from a Tier 0 report:
  python scripts/escalate.py --report data/reports/quick_run_YYYYMMDD.json --top 50

  # Extract and immediately launch Tier 1:
  python scripts/escalate.py --report data/reports/quick_run_YYYYMMDD.json --top 50 --run

  # Use custom escalation criteria:
  python scripts/escalate.py --report data/reports/quick_run_YYYYMMDD.json --score-min 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_report(report_path: str) -> Dict[str, Any]:
    """Load a quick_run report JSON file."""
    with open(report_path) as f:
        return json.load(f)


def extract_escalation_candidates(
    report: Dict[str, Any],
    top_n: int = 50,
    score_min: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Extract escalation candidates from a Tier 0 report.

    Selection criteria (union — any one qualifies):
      1. Total EXODUS score >= score_min (default: any score > 0)
      2. Has IR excess candidate (is_candidate=True)
      3. Has astrometric anomaly (RUWE > 1.4)
      4. Has multi-messenger match (gamma, neutrino, GW, pulsar)
      5. Is flagged UNEXPLAINED by the Unexplainability Score

    Returns list of target dicts sorted by EXODUS score (descending),
    limited to top_n.
    """
    targets = report.get("top_targets", [])
    if not targets:
        print("WARNING: No top_targets found in report.")
        return []

    # Build a lookup of unexplainability results from the report
    unexplained_lookup = {}
    for ut in report.get("unexplained_targets", []):
        unexplained_lookup[ut.get("target_id")] = ut

    candidates = []
    for t in targets:
        target_id = t.get("target_id", "unknown")
        total_score = t.get("total_score", 0.0)
        channels = t.get("channel_scores", {})

        # Check escalation reasons
        reasons = []

        # Score-based
        if score_min and total_score >= score_min:
            reasons.append(f"score={total_score:.3f}")
        elif not score_min:
            reasons.append(f"score={total_score:.3f}")

        # IR excess candidate
        ir = channels.get("ir_excess", {})
        ir_details = ir.get("details", {})
        if ir_details.get("is_candidate", False):
            reasons.append("ir_candidate")

        # Astrometric anomaly (RUWE > 1.4 or PM discrepancy)
        astro = channels.get("proper_motion_anomaly", {})
        astro_details = astro.get("details", {})
        ruwe = astro_details.get("ruwe") or 1.0
        if ruwe > 1.4:
            reasons.append(f"ruwe={ruwe:.2f}")
        pm_check = astro_details.get("wise_gaia_pm", {})
        if pm_check.get("is_discrepant"):
            pm_sig = pm_check.get("pm_discrepancy_sigma", 0)
            reasons.append(f"pm_discrepancy={pm_sig:.1f}sigma")

        # Multi-channel convergence
        n_active = t.get("n_active_channels", 0)
        if n_active >= 2:
            reasons.append(f"convergent({n_active}ch)")

        # FDR significance
        if t.get("fdr_significant", False):
            reasons.append("fdr_significant")

        # Unexplainability Score
        unex = unexplained_lookup.get(target_id, {})
        unex_class = unex.get("classification")
        if unex_class == "UNEXPLAINED":
            reasons.append(f"UNEXPLAINED(score={unex.get('unexplainability_score', 0):.3f})")
        elif unex_class == "PARTIALLY_EXPLAINED":
            reasons.append(f"partial_unexplained")

        if reasons:
            candidates.append({
                "target_id": target_id,
                "ra": t.get("ra", 0.0),
                "dec": t.get("dec", 0.0),
                "distance_pc": t.get("distance_pc"),
                "host_star": t.get("host_star"),
                "total_score": total_score,
                "n_active_channels": n_active,
                "escalation_reasons": reasons,
                "tier0_rank": t.get("rank", 0),
                "combined_p": t.get("combined_p"),
                "q_value": t.get("q_value"),
                "unexplainability": unex.get("unexplainability_score"),
            })

    # Sort by total_score descending, take top N
    candidates.sort(key=lambda x: x["total_score"], reverse=True)
    return candidates[:top_n]


def write_target_file(
    candidates: List[Dict[str, Any]],
    output_path: str,
    source_report: str,
) -> str:
    """Write candidates as a campaign target JSON file."""
    targets = []
    for c in candidates:
        t = {
            "target_id": c["target_id"],
            "host_star": c.get("host_star")
                or (c["target_id"].rsplit("_", 1)[0].replace("_", " ")
                    if "_" in c["target_id"]
                    else c["target_id"].replace("_", " ")),
            "ra": c["ra"],
            "dec": c["dec"],
            "hz_flag": False,  # will be enriched at runtime
            "notes": f"Escalated from Tier 0: {', '.join(c['escalation_reasons'])}. "
                     f"Tier 0 score: {c['total_score']:.4f}, rank: {c['tier0_rank']}",
        }
        if c.get("distance_pc"):
            t["distance_pc"] = c["distance_pc"]
        targets.append(t)

    campaign = {
        "campaign": "tier1_escalation",
        "phase": "phase_2",
        "description": (
            f"Tier 1 follow-up on top {len(targets)} targets from Tier 0 blitz. "
            f"Source report: {Path(source_report).name}"
        ),
        "targets": targets,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(campaign, f, indent=2)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract top targets from a Tier 0 report for Tier 1 follow-up."
    )
    parser.add_argument(
        "--report", required=True,
        help="Path to Tier 0 quick_run report JSON",
    )
    parser.add_argument(
        "--top", type=int, default=50,
        help="Maximum number of targets to escalate (default: 50)",
    )
    parser.add_argument(
        "--score-min", type=float, default=None,
        help="Minimum EXODUS score for escalation (default: include all)",
    )
    parser.add_argument(
        "--output", default="data/targets/tier1_escalation.json",
        help="Output target file path",
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Immediately launch Tier 1 on the escalation targets",
    )
    args = parser.parse_args()

    # Load report
    print(f"Loading report: {args.report}")
    report = load_report(args.report)
    tier = report.get("tier", "?")
    n_targets = report.get("n_targets", "?")
    print(f"  Tier: {tier}, Targets in report: {n_targets}")

    # Extract candidates
    candidates = extract_escalation_candidates(
        report, top_n=args.top, score_min=args.score_min,
    )
    print(f"\nEscalation candidates: {len(candidates)}")

    if not candidates:
        print("No candidates found. Exiting.")
        sys.exit(0)

    # Print summary
    print(f"\n{'Rank':<6} {'Target':<30} {'Score':>8} {'Dist(pc)':>8} {'Ch':>3} {'Reasons'}")
    print("-" * 100)
    for i, c in enumerate(candidates, 1):
        dist_str = f"{c['distance_pc']:.1f}" if c.get("distance_pc") else "?"
        print(
            f"{i:<6} {c['target_id']:<30} {c['total_score']:8.4f} "
            f"{dist_str:>8} {c['n_active_channels']:>3} "
            f"{', '.join(c['escalation_reasons'])}"
        )

    # Write target file
    output = write_target_file(candidates, args.output, args.report)
    print(f"\nTarget file written: {output}")

    # Optionally launch Tier 1
    if args.run:
        cmd = (
            f"./venv/bin/python scripts/run_quick.py "
            f"--target-file {args.output} --tier 1"
        )
        print(f"\nLaunching Tier 1: {cmd}")
        os.system(cmd)


if __name__ == "__main__":
    main()
