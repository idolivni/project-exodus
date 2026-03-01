#!/usr/bin/env python3
"""
Project EXODUS — Binary Subtraction Analysis
=============================================

Load all scored targets from existing report JSONs, remove known/likely
binaries, and rank what remains.  The key insight: every high-scoring
target so far is dominated by the binary_system template (RUWE + IR
from unresolved companions).  If we subtract those, *anything* left
at even modest scores becomes far more interesting than a high-scoring
binary.

Binary indicators (any one is sufficient to flag):
  1. RUWE > 1.4   (Gaia astrometric excess → unresolved companion)
  2. astrometric_excess_noise_sig > 5
  3. WISE-Gaia PM discrepancy > 3σ  (wavelength-dependent photocentre)
  4. Known SB9 binary   (from data gathering, if available)
  5. Known Gaia NSS flag (non-single-star solution)

Output: ranked list of "clean" targets — candidates where the anomaly
is NOT trivially explained by a stellar companion.

Usage
-----
    python scripts/binary_subtraction.py
    python scripts/binary_subtraction.py --ruwe-cut 1.4
    python scripts/binary_subtraction.py --ruwe-cut 2.0 --output data/reports/binary_subtracted.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger, safe_json_dump

log = get_logger("binary_subtraction")

# ── Binary detection thresholds ─────────────────────────────────────

DEFAULT_RUWE_CUT = 1.4          # Standard Gaia threshold for binaries
DEFAULT_AEN_SIG_CUT = 5.0       # Astrometric excess noise significance
DEFAULT_PM_DISC_SIGMA_CUT = 3.0 # WISE-Gaia PM discrepancy


def load_all_scored_targets(report_dir: Path) -> List[Dict[str, Any]]:
    """Load scored targets from all report JSONs in directory.

    Deduplicates by target_id (keeps highest-scoring instance).
    """
    targets_by_id: Dict[str, Dict[str, Any]] = {}

    report_files = sorted(report_dir.glob("quick_run_*.json"))
    log.info("Found %d report files in %s", len(report_files), report_dir)

    for rpath in report_files:
        try:
            with open(rpath) as f:
                rpt = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Skipping %s: %s", rpath.name, exc)
            continue

        # Prefer all_scored, fall back to top_targets
        scored = rpt.get("all_scored") or rpt.get("top_targets") or []
        source_tag = rpath.stem  # e.g. quick_run_YYYYMMDD_HHMMSS

        for t in scored:
            tid = t.get("target_id")
            if not tid:
                continue
            t["_report_source"] = source_tag

            # Keep highest-scoring instance
            existing = targets_by_id.get(tid)
            if existing is None or t.get("total_score", 0) > existing.get("total_score", 0):
                targets_by_id[tid] = t

    all_targets = list(targets_by_id.values())
    log.info("Loaded %d unique scored targets from %d reports",
             len(all_targets), len(report_files))
    return all_targets


def classify_binary(
    target: Dict[str, Any],
    ruwe_cut: float = DEFAULT_RUWE_CUT,
    aen_sig_cut: float = DEFAULT_AEN_SIG_CUT,
    pm_disc_sigma_cut: float = DEFAULT_PM_DISC_SIGMA_CUT,
) -> Tuple[bool, List[str]]:
    """Classify whether a target is likely a binary system.

    Returns
    -------
    is_binary : bool
    reasons : list of str
        Human-readable reasons for binary classification.
    """
    reasons = []

    # Extract PM anomaly details
    cs = target.get("channel_scores", {})
    pm_data = cs.get("proper_motion_anomaly", {})
    details = pm_data.get("details", {})

    # 1. RUWE check
    ruwe = details.get("ruwe")
    if ruwe is not None and ruwe > ruwe_cut:
        reasons.append(f"RUWE={ruwe:.2f} > {ruwe_cut}")

    # 2. Astrometric excess noise significance
    aen_sig = details.get("astrometric_excess_noise_sig")
    if aen_sig is not None and aen_sig > aen_sig_cut:
        reasons.append(f"aen_sig={aen_sig:.1f} > {aen_sig_cut}")

    # 3. WISE-Gaia PM discrepancy
    wise_gaia = details.get("wise_gaia_pm", {})
    pm_disc = wise_gaia.get("pm_discrepancy_sigma")
    if pm_disc is not None and pm_disc > pm_disc_sigma_cut:
        reasons.append(f"PM_disc={pm_disc:.1f}σ > {pm_disc_sigma_cut}σ")

    # 4. Unexplainability template check (if available)
    unex = target.get("unexplainability", {})
    if unex.get("best_template") == "binary_system":
        reasons.append(f"template=binary_system (fit={unex.get('best_template_fit', '?')})")

    # 5. Red-team binary flag (if available)
    rt = target.get("red_team", {})
    for flag in rt.get("risk_flags", []):
        if "binary" in str(flag).lower() or "companion" in str(flag).lower():
            reasons.append(f"red_team_flag: {flag}")

    is_binary = len(reasons) > 0
    return is_binary, reasons


def run_binary_subtraction(
    report_dir: Path,
    ruwe_cut: float = DEFAULT_RUWE_CUT,
    aen_sig_cut: float = DEFAULT_AEN_SIG_CUT,
    pm_disc_sigma_cut: float = DEFAULT_PM_DISC_SIGMA_CUT,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Main entry: load targets, subtract binaries, rank remainder."""

    all_targets = load_all_scored_targets(report_dir)
    if not all_targets:
        log.error("No scored targets found.")
        return {}

    # Classify each target
    clean_targets = []
    binary_targets = []
    for t in all_targets:
        is_bin, reasons = classify_binary(
            t, ruwe_cut=ruwe_cut, aen_sig_cut=aen_sig_cut,
            pm_disc_sigma_cut=pm_disc_sigma_cut,
        )
        t["_binary_classification"] = {
            "is_binary": is_bin,
            "reasons": reasons,
        }
        if is_bin:
            binary_targets.append(t)
        else:
            clean_targets.append(t)

    # Sort clean targets by EXODUS score
    clean_targets.sort(key=lambda t: t.get("total_score", 0), reverse=True)
    binary_targets.sort(key=lambda t: t.get("total_score", 0), reverse=True)

    # Analyze channel composition of clean targets
    clean_channel_stats = _channel_stats(clean_targets)
    binary_channel_stats = _channel_stats(binary_targets)

    # Build summary
    result = {
        "analysis": "binary_subtraction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "ruwe_cut": ruwe_cut,
            "aen_sig_cut": aen_sig_cut,
            "pm_disc_sigma_cut": pm_disc_sigma_cut,
        },
        "summary": {
            "total_targets": len(all_targets),
            "n_binaries": len(binary_targets),
            "n_clean": len(clean_targets),
            "binary_fraction": round(len(binary_targets) / max(1, len(all_targets)), 3),
            "clean_max_score": clean_targets[0]["total_score"] if clean_targets else 0,
            "clean_max_target": clean_targets[0].get("target_id") if clean_targets else None,
            "binary_max_score": binary_targets[0]["total_score"] if binary_targets else 0,
        },
        "clean_channel_stats": clean_channel_stats,
        "binary_channel_stats": binary_channel_stats,
        "clean_targets": [_target_summary(t) for t in clean_targets[:100]],
        "binary_targets_top20": [_target_summary(t) for t in binary_targets[:20]],
    }

    # Print report
    _print_report(result)

    # Save if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fp:
            safe_json_dump(result, fp)
        log.info("Results saved to %s", output_path)

    return result


def _target_summary(t: Dict[str, Any]) -> Dict[str, Any]:
    """Create a compact summary of a scored target."""
    cs = t.get("channel_scores", {})
    pm_details = cs.get("proper_motion_anomaly", {}).get("details", {})
    ir_details = cs.get("ir_excess", {}).get("details", {})

    active_channels = [
        name for name, ch in cs.items()
        if ch.get("is_active") and name != "habitable_zone_planet"
    ]
    channel_scores_compact = {
        name: round(ch.get("score", 0), 4)
        for name, ch in cs.items()
        if ch.get("score", 0) > 0
    }

    return {
        "target_id": t.get("target_id"),
        "host_star": t.get("host_star"),
        "total_score": round(t.get("total_score", 0), 4),
        "n_active": t.get("n_active_channels", 0),
        "active_channels": active_channels,
        "channel_scores": channel_scores_compact,
        "ruwe": pm_details.get("ruwe"),
        "aen_sig": pm_details.get("astrometric_excess_noise_sig"),
        "pm_disc_sigma": pm_details.get("wise_gaia_pm", {}).get("pm_discrepancy_sigma"),
        "ir_sigma_W4": ir_details.get("sigma_W4"),
        "distance_pc": t.get("distance_pc"),
        "fdr_significant": t.get("fdr_significant"),
        "stouffer_p": t.get("stouffer_p"),
        "report_source": t.get("_report_source"),
        "binary_reasons": t.get("_binary_classification", {}).get("reasons", []),
    }


def _channel_stats(targets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute channel activation statistics for a target set."""
    if not targets:
        return {"n": 0}

    n = len(targets)
    ch_counts = {}
    for t in targets:
        cs = t.get("channel_scores", {})
        for name, ch in cs.items():
            if ch.get("is_active") and name != "habitable_zone_planet":
                ch_counts[name] = ch_counts.get(name, 0) + 1

    multi_ch = sum(1 for t in targets if t.get("n_active_channels", 0) >= 2)
    three_ch = sum(1 for t in targets if t.get("n_active_channels", 0) >= 3)

    return {
        "n": n,
        "channel_activation": {k: f"{v}/{n} ({100*v/n:.0f}%)" for k, v in sorted(ch_counts.items())},
        "multi_channel_2plus": f"{multi_ch}/{n} ({100*multi_ch/n:.0f}%)",
        "multi_channel_3plus": f"{three_ch}/{n} ({100*three_ch/n:.0f}%)",
        "score_range": f"{min(t['total_score'] for t in targets):.4f} — {max(t['total_score'] for t in targets):.4f}",
        "median_score": round(sorted(t["total_score"] for t in targets)[n // 2], 4),
    }


def _print_report(result: Dict[str, Any]) -> None:
    """Print human-readable binary subtraction report."""
    s = result["summary"]
    p = result["parameters"]

    print()
    print("=" * 74)
    print("  PROJECT EXODUS — BINARY SUBTRACTION ANALYSIS")
    print("=" * 74)
    print(f"  Total scored targets:  {s['total_targets']}")
    print(f"  Binaries removed:      {s['n_binaries']} ({100*s['binary_fraction']:.1f}%)")
    print(f"  Clean targets:         {s['n_clean']}")
    print(f"  Thresholds:  RUWE > {p['ruwe_cut']},  AEN_sig > {p['aen_sig_cut']},  PM_disc > {p['pm_disc_sigma_cut']}σ")
    print()

    # Clean targets
    clean = result["clean_targets"]
    if clean:
        print(f"  TOP CLEAN TARGETS (non-binary, n={len(clean)}):")
        print(f"  {'Rank':>4}  {'Target':<30} {'Score':>8}  {'Ch':>2}  {'Active Channels':<40}  {'RUWE':>6}")
        print("  " + "-" * 100)
        for i, t in enumerate(clean[:30], 1):
            ruwe_str = f"{t['ruwe']:.2f}" if t.get("ruwe") else "N/A"
            channels = ", ".join(t.get("active_channels", []))
            print(f"  {i:>4}  {t['target_id']:<30} {t['total_score']:>8.4f}  {t.get('n_active', 0):>2}  {channels:<40}  {ruwe_str:>6}")
    else:
        print("  *** NO CLEAN TARGETS FOUND — every scored target has binary indicators ***")
        print("  This is the core finding: binary contamination is total.")

    print()

    # Channel stats comparison
    cs_clean = result.get("clean_channel_stats", {})
    cs_binary = result.get("binary_channel_stats", {})
    print("  CHANNEL ACTIVATION COMPARISON:")
    print(f"  {'Channel':<30} {'Clean':>20}  {'Binary':>20}")
    print("  " + "-" * 74)
    all_channels = set(
        list(cs_clean.get("channel_activation", {}).keys()) +
        list(cs_binary.get("channel_activation", {}).keys())
    )
    for ch in sorted(all_channels):
        c_val = cs_clean.get("channel_activation", {}).get(ch, "0/0 (0%)")
        b_val = cs_binary.get("channel_activation", {}).get(ch, "0/0 (0%)")
        print(f"  {ch:<30} {c_val:>20}  {b_val:>20}")

    if cs_clean.get("n", 0) > 0:
        print(f"  {'Multi-channel (2+)':<30} {cs_clean.get('multi_channel_2plus', '?'):>20}  {cs_binary.get('multi_channel_2plus', '?'):>20}")
        print(f"  {'Multi-channel (3+)':<30} {cs_clean.get('multi_channel_3plus', '?'):>20}  {cs_binary.get('multi_channel_3plus', '?'):>20}")

    print()

    # Top binaries for comparison
    bin_top = result.get("binary_targets_top20", [])
    if bin_top:
        print(f"  TOP BINARIES (removed, n={s['n_binaries']}, showing top 10):")
        print(f"  {'Rank':>4}  {'Target':<30} {'Score':>8}  {'RUWE':>8}  {'Reasons'}")
        print("  " + "-" * 90)
        for i, t in enumerate(bin_top[:10], 1):
            ruwe_str = f"{t['ruwe']:.2f}" if t.get("ruwe") else "N/A"
            reasons = "; ".join(t.get("binary_reasons", [])[:2])
            print(f"  {i:>4}  {t['target_id']:<30} {t['total_score']:>8.4f}  {ruwe_str:>8}  {reasons}")

    print()
    print("=" * 74)


def main():
    parser = argparse.ArgumentParser(description="EXODUS Binary Subtraction Analysis")
    parser.add_argument(
        "--report-dir", type=str,
        default=str(PROJECT_ROOT / "data" / "reports"),
        help="Directory containing report JSONs",
    )
    parser.add_argument(
        "--ruwe-cut", type=float, default=DEFAULT_RUWE_CUT,
        help=f"RUWE threshold for binary classification (default: {DEFAULT_RUWE_CUT})",
    )
    parser.add_argument(
        "--aen-sig-cut", type=float, default=DEFAULT_AEN_SIG_CUT,
        help=f"Astrometric excess noise sig threshold (default: {DEFAULT_AEN_SIG_CUT})",
    )
    parser.add_argument(
        "--pm-disc-cut", type=float, default=DEFAULT_PM_DISC_SIGMA_CUT,
        help=f"WISE-Gaia PM discrepancy sigma threshold (default: {DEFAULT_PM_DISC_SIGMA_CUT})",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "data" / "reports" / "binary_subtraction.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    run_binary_subtraction(
        report_dir=Path(args.report_dir),
        ruwe_cut=args.ruwe_cut,
        aen_sig_cut=args.aen_sig_cut,
        pm_disc_sigma_cut=args.pm_disc_cut,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
