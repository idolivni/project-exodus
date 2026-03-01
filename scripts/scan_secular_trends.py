#!/usr/bin/env python3
"""
Project EXODUS — Secular IR Trend Scanner
==========================================

Scans all completed report data for targets with secular (monotonic)
IR brightening trends in NEOWISE time-series. This is the "construction
in progress" signal — a civilization building a Dyson swarm produces a
star getting steadily brighter in mid-IR.

Stars don't monotonically brighten in IR over 10 years:
  - Dust clouds don't form that fast
  - Variable stars oscillate, they don't trend
  - AGN vary stochastically, not monotonically

This script:
1. Loads all scored targets from existing reports
2. Finds those with NEOWISE time-series data
3. Extracts secular trend metrics from ir_variability channel
4. Ranks by secular_trend_score and monotonic fraction
5. Reports any targets with significant monotonic trends

Usage
-----
    python scripts/scan_secular_trends.py
    python scripts/scan_secular_trends.py --report-dir data/reports/ --mono-threshold 0.8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger, safe_json_dump

log = get_logger("secular_scan")


def load_all_targets_with_ir_var(report_dir: Path) -> List[Dict[str, Any]]:
    """Load scored targets that have ir_variability data."""
    targets_by_id: Dict[str, Dict[str, Any]] = {}

    report_files = sorted(report_dir.glob("quick_run_*.json"))
    log.info("Found %d report files", len(report_files))

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

            # Check for ir_variability channel data
            cs = t.get("channel_scores", {})
            irv = cs.get("ir_variability", {})
            details = irv.get("details", {})

            if not details or details.get("reason") in ("no data provided", "simulated data excluded"):
                continue

            if details.get("data_source") != "real":
                continue

            t["_report_source"] = rpath.name
            # Keep highest variability score instance
            existing = targets_by_id.get(tid)
            if existing is None or details.get("variability_score", 0) > (
                existing.get("channel_scores", {}).get("ir_variability", {}).get("details", {}).get("variability_score", 0)
            ):
                targets_by_id[tid] = t

    log.info("Found %d unique targets with real IR variability data", len(targets_by_id))
    return list(targets_by_id.values())


def extract_trend_metrics(target: Dict[str, Any]) -> Dict[str, Any]:
    """Extract secular trend metrics from a scored target."""
    cs = target.get("channel_scores", {})
    irv = cs.get("ir_variability", {}).get("details", {})

    return {
        "target_id": target.get("target_id"),
        "host_star": target.get("host_star"),
        "distance_pc": target.get("distance_pc"),
        "total_score": target.get("total_score", 0),
        "n_epochs": irv.get("n_epochs", 0),
        "time_baseline_years": irv.get("time_baseline_years", 0),
        "variability_score": irv.get("variability_score", 0),
        "w1_trend_mag_per_year": irv.get("w1_trend_mag_per_year", 0),
        "w1_trend_sigma": irv.get("w1_trend_sigma", 0),
        "w2_trend_mag_per_year": irv.get("w2_trend_mag_per_year", 0),
        "w2_trend_sigma": irv.get("w2_trend_sigma", 0),
        "w1_monotonic_frac": irv.get("w1_monotonic_frac", 0),
        "w2_monotonic_frac": irv.get("w2_monotonic_frac", 0),
        "cross_band_consistent": irv.get("cross_band_consistent", False),
        "is_brightening": irv.get("is_brightening", False),
        "secular_trend_score": irv.get("secular_trend_score", 0),
        "has_secular_trend": irv.get("has_secular_trend", False),
        "data_source": irv.get("data_source", "unknown"),
        "report_source": target.get("_report_source"),
    }


def scan_secular_trends(
    report_dir: Path,
    mono_threshold: float = 0.8,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Main scan: find targets with monotonic IR trends."""

    targets = load_all_targets_with_ir_var(report_dir)
    if not targets:
        log.warning("No targets with real IR variability data found.")
        return {"n_targets": 0, "n_with_trend": 0}

    # Extract metrics for all targets
    metrics = [extract_trend_metrics(t) for t in targets]

    # Classify
    with_trend = [m for m in metrics if m.get("has_secular_trend")]
    monotonic = [m for m in metrics
                 if max(m.get("w1_monotonic_frac", 0), m.get("w2_monotonic_frac", 0)) >= mono_threshold]
    brightening = [m for m in metrics if m.get("is_brightening")]
    cross_band = [m for m in metrics if m.get("cross_band_consistent")]

    # Sort by secular_trend_score
    metrics.sort(key=lambda m: m.get("secular_trend_score", 0), reverse=True)

    result = {
        "analysis": "secular_trend_scan",
        "parameters": {
            "mono_threshold": mono_threshold,
        },
        "summary": {
            "n_targets_with_neowise": len(metrics),
            "n_with_secular_trend": len(with_trend),
            "n_monotonic": len(monotonic),
            "n_brightening": len(brightening),
            "n_cross_band_consistent": len(cross_band),
        },
        "top_30_by_trend_score": metrics[:30],
        "monotonic_targets": sorted(monotonic, key=lambda m: m.get("secular_trend_score", 0), reverse=True),
        "brightening_targets": sorted(brightening, key=lambda m: m.get("secular_trend_score", 0), reverse=True),
    }

    # Print report
    _print_report(result)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fp:
            safe_json_dump(result, fp)
        log.info("Results saved to %s", output_path)

    return result


def _print_report(result: Dict[str, Any]) -> None:
    s = result["summary"]

    print()
    print("=" * 78)
    print("  PROJECT EXODUS — SECULAR IR TREND SCAN")
    print("=" * 78)
    print(f"  Targets with real NEOWISE data: {s['n_targets_with_neowise']}")
    print(f"  With secular trend (3σ+):       {s['n_with_secular_trend']}")
    print(f"  Monotonic (>80% consistent):    {s['n_monotonic']}")
    print(f"  Brightening (neg mag slope):    {s['n_brightening']}")
    print(f"  Cross-band consistent (W1+W2):  {s['n_cross_band_consistent']}")
    print()

    top = result.get("top_30_by_trend_score", [])
    if top:
        print(f"  TOP SECULAR TRENDS:")
        print(f"  {'Rank':>4}  {'Target':<30} {'TrendSc':>8} {'W1σ':>6} {'W2σ':>6} {'MonoW1':>7} {'MonoW2':>7} {'XBand':>5} {'Bright':>6}")
        print("  " + "-" * 95)
        for i, m in enumerate(top[:20], 1):
            xband = "yes" if m.get("cross_band_consistent") else "no"
            bright = "YES" if m.get("is_brightening") else "no"
            print(f"  {i:>4}  {m['target_id']:<30} "
                  f"{m.get('secular_trend_score', 0):>8.4f} "
                  f"{m.get('w1_trend_sigma', 0):>6.1f} "
                  f"{m.get('w2_trend_sigma', 0):>6.1f} "
                  f"{m.get('w1_monotonic_frac', 0):>7.2f} "
                  f"{m.get('w2_monotonic_frac', 0):>7.2f} "
                  f"{xband:>5} {bright:>6}")

    # Highlight construction-in-progress candidates
    mono = result.get("monotonic_targets", [])
    bright_mono = [m for m in mono if m.get("is_brightening") and m.get("cross_band_consistent")]
    if bright_mono:
        print()
        print("  *** CONSTRUCTION-IN-PROGRESS CANDIDATES ***")
        print("  (Monotonic + Brightening + Cross-band consistent)")
        for m in bright_mono:
            print(f"    → {m['target_id']}: trend_score={m.get('secular_trend_score', 0):.4f}, "
                  f"W1σ={m.get('w1_trend_sigma', 0):.1f}, W2σ={m.get('w2_trend_sigma', 0):.1f}")
    elif not mono:
        print()
        print("  No monotonic IR trends found in current data.")
        print("  This constrains the rate of active Dyson swarm construction")
        print("  among nearby stars with NEOWISE coverage.")

    print()
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="EXODUS Secular IR Trend Scanner")
    parser.add_argument("--report-dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "reports"))
    parser.add_argument("--mono-threshold", type=float, default=0.8,
                        help="Monotonic fraction threshold (default: 0.8)")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "data" / "reports" / "secular_trend_scan.json"))
    args = parser.parse_args()

    scan_secular_trends(
        report_dir=Path(args.report_dir),
        mono_threshold=args.mono_threshold,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
