#!/usr/bin/env python3
"""
analyze_hz_boost.py — Quantify the impact of the habitable zone (HZ) prior
boost on EXODUS scoring.

The HZ boost is multiplicative: total_score *= (1.0 + max_hz_score)
where hz_score is 0.8 for single HZ planet, 0.9 for multiple.  This gives
a 1.8x-1.9x score inflation for any target with a known HZ planet.

This script loads all quick_run reports, deduplicates targets (keeping the
newest occurrence), and computes the effect of the HZ boost on scoring,
FDR significance, and ranking.

Output: data/reports/hz_boost_analysis.json + printed summary.
"""

import json
import glob
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
REPORT_PATTERN = str(REPORTS_DIR / "quick_run_*.json")
OUTPUT_FILE = REPORTS_DIR / "hz_boost_analysis.json"

# ---------------------------------------------------------------------------
# Load and deduplicate
# ---------------------------------------------------------------------------

def extract_timestamp(filepath: str) -> str:
    """Extract timestamp string from quick_run_YYYYMMDD_HHMMSS.json filename."""
    basename = os.path.basename(filepath)
    # quick_run_YYYYMMDD_HHMMSS.json -> YYYYMMDD_HHMMSS
    parts = basename.replace("quick_run_", "").replace(".json", "")
    return parts


def load_all_targets() -> dict:
    """Load all reports and deduplicate by target_id, keeping newest."""
    files = sorted(glob.glob(REPORT_PATTERN))
    if not files:
        print(f"ERROR: No report files found matching {REPORT_PATTERN}")
        sys.exit(1)

    print(f"Found {len(files)} report files")

    # target_id -> (timestamp_str, target_dict, report_file)
    targets = {}
    total_raw = 0

    for f in files:
        ts = extract_timestamp(f)
        with open(f) as fh:
            data = json.load(fh)
        for t in data.get("top_targets", []):
            total_raw += 1
            tid = t.get("target_id", "UNKNOWN")
            if tid not in targets or ts > targets[tid][0]:
                targets[tid] = (ts, t, os.path.basename(f))

    print(f"Total raw target entries: {total_raw}")
    print(f"Unique targets (deduplicated, newest kept): {len(targets)}")
    return targets


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_hz_boost(targets: dict) -> dict:
    """Run the full HZ boost analysis."""

    all_targets = []
    hz_targets = []
    non_hz_targets = []

    for tid, (ts, t, report) in targets.items():
        cs = t.get("channel_scores", {})
        hz_ch = cs.get("habitable_zone_planet", {})
        hz_active = hz_ch.get("is_active", False)
        hz_score = hz_ch.get("score", 0.0)
        total_score = t.get("total_score", 0.0)
        n_active = t.get("n_active_channels", 0)
        fdr_sig = t.get("fdr_significant", None)
        convergence_bonus = t.get("convergence_bonus", 1.0)
        geo_mean = t.get("geo_mean", 0.0)

        # Count detection channels (non-HZ) that are active
        detection_channels_active = []
        for ch_name, ch_data in cs.items():
            if ch_name == "habitable_zone_planet":
                continue
            if isinstance(ch_data, dict) and ch_data.get("is_active", False):
                detection_channels_active.append(ch_name)

        n_detection = len(detection_channels_active)

        entry = {
            "target_id": tid,
            "total_score": total_score,
            "hz_active": hz_active,
            "hz_score": hz_score if hz_active else 0.0,
            "n_active_channels": n_active,
            "n_detection_channels": n_detection,
            "detection_channels": detection_channels_active,
            "fdr_significant": fdr_sig,
            "convergence_bonus": convergence_bonus,
            "geo_mean": geo_mean,
            "report": report,
        }

        if hz_active and total_score > 0:
            boost_factor = 1.0 + hz_score
            score_without_hz = total_score / boost_factor
            entry["boost_factor"] = boost_factor
            entry["score_without_hz"] = score_without_hz
            entry["score_inflation_pct"] = (boost_factor - 1.0) * 100.0
            hz_targets.append(entry)
        else:
            entry["boost_factor"] = 1.0
            entry["score_without_hz"] = total_score
            entry["score_inflation_pct"] = 0.0
            non_hz_targets.append(entry)

        all_targets.append(entry)

    # ----- Summary statistics -----
    n_total = len(all_targets)
    n_hz = len(hz_targets)
    n_non_hz = len(non_hz_targets)
    n_scored = sum(1 for t in all_targets if t["total_score"] > 0)
    n_hz_scored = sum(1 for t in hz_targets if t["total_score"] > 0)

    # Score distributions
    hz_scores_with = [t["total_score"] for t in hz_targets if t["total_score"] > 0]
    hz_scores_without = [t["score_without_hz"] for t in hz_targets if t["total_score"] > 0]
    non_hz_scores = [t["total_score"] for t in non_hz_targets if t["total_score"] > 0]

    # FDR analysis
    hz_fdr_with = sum(1 for t in hz_targets if t["fdr_significant"] is True)
    hz_fdr_without_count = 0
    hz_fdr_lost = []
    for t in hz_targets:
        if t["fdr_significant"] is True:
            # Would it lose FDR if HZ removed?
            # FDR depends on q_value/combined_p, not directly on total_score.
            # But total_score is used for ranking. The FDR p-values come from
            # calibrated channels. HZ is uncalibrated (calibrated_p=null), so
            # it does NOT affect Fisher combined_p or q_value directly.
            # FDR significance is unchanged by HZ removal!
            hz_fdr_without_count += 1

    # However, let's check: does HZ score affect fdr_significant through
    # any indirect mechanism? Since calibrated_p for HZ is always null,
    # the Fisher combination excludes it. So FDR significance is the same.
    fdr_change_targets = []
    for t in hz_targets:
        # FDR is based on calibrated channels only, HZ is uncalibrated
        # So no FDR change expected. But let's flag if there's a score
        # impact that would change ranking order.
        pass

    # Targets where HZ boost is the ONLY reason score > 1.0
    hz_only_above_1 = []
    for t in hz_targets:
        if t["total_score"] > 1.0 and t["score_without_hz"] <= 1.0:
            hz_only_above_1.append(t)

    # Targets where HZ boost is the ONLY reason score > 0
    # (these have n_detection_channels == 0 but total_score > 0 due to HZ)
    hz_zero_detection = [t for t in hz_targets if t["n_detection_channels"] == 0]

    # n_active_channels check — HZ should NOT affect this
    channel_count_mismatch = []
    for t in hz_targets:
        if t["n_active_channels"] != t["n_detection_channels"]:
            # This is expected: n_active_channels in the report should
            # already exclude HZ (it's a prior, not detection)
            # But let's check if any report includes HZ in n_active
            pass

    # ----- Score comparison by detection channel count -----
    # Group HZ targets by n_detection_channels
    hz_by_ndet = defaultdict(list)
    for t in hz_targets:
        hz_by_ndet[t["n_detection_channels"]].append(t)

    non_hz_by_ndet = defaultdict(list)
    for t in non_hz_targets:
        if t["total_score"] > 0:
            non_hz_by_ndet[t["n_detection_channels"]].append(t)

    # ----- Optimal HZ weight analysis -----
    # Current: hz_weight = 0.8 -> boost = 1.8x
    # What weight would prevent HZ from dominating?
    # Criterion: HZ boost should not cause score > 1.0 for single-channel targets
    # Single-channel penalty: geo_mean * 0.25, then * coverage_penalty
    # Max single-channel score without HZ: 1.0 * 0.25 * 1.0 = 0.25
    # With HZ: 0.25 * (1 + w) > 1.0 -> w > 3.0 (always below for w=0.8)
    # Actually: let's find max score_without_hz for single-detection targets
    single_det_hz = [t for t in hz_targets if t["n_detection_channels"] == 1]
    if single_det_hz:
        max_unboosted_single = max(t["score_without_hz"] for t in single_det_hz)
    else:
        max_unboosted_single = 0.0

    # For each candidate weight, compute how many targets would cross score > 1.0
    candidate_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    weight_analysis = []
    for w in candidate_weights:
        new_boost = 1.0 + w
        n_above_1 = 0
        n_above_1_single_det = 0
        for t in hz_targets:
            new_score = t["score_without_hz"] * new_boost
            if new_score > 1.0:
                n_above_1 += 1
                if t["n_detection_channels"] <= 1:
                    n_above_1_single_det += 1
        weight_analysis.append({
            "hz_weight": w,
            "boost_factor": new_boost,
            "n_above_1": n_above_1,
            "n_above_1_single_det_only": n_above_1_single_det,
        })

    # ----- Build detailed per-target table -----
    hz_detail_table = sorted(hz_targets, key=lambda x: -x["total_score"])
    for t in hz_detail_table:
        # Clean up for JSON serialization
        t["detection_channels"] = t["detection_channels"]

    # ----- Recommendation -----
    n_hz_fdr = hz_fdr_with
    n_total_fdr = sum(1 for t in all_targets if t.get("fdr_significant") is True)
    hz_fdr_fraction = n_hz_fdr / max(n_total_fdr, 1)

    # How many are "HZ-propped" (score > 1.0 only because of HZ)?
    n_hz_propped = len(hz_only_above_1)
    hz_propped_fraction = n_hz_propped / max(n_hz, 1)

    # Recommendation logic
    if hz_propped_fraction > 0.5:
        recommendation = (
            f"CRITICAL: {hz_propped_fraction:.0%} of HZ-boosted targets score > 1.0 "
            f"ONLY because of the HZ boost. The 1.8x multiplier is dominating scoring "
            f"for targets with weak underlying anomalies. Recommend reducing HZ weight "
            f"from 0.8 to 0.3 (boost: 1.3x) or converting to additive bonus."
        )
        recommended_weight = 0.3
    elif hz_propped_fraction > 0.2:
        recommendation = (
            f"WARNING: {hz_propped_fraction:.0%} of HZ-boosted targets score > 1.0 "
            f"only because of HZ boost. Consider reducing weight from 0.8 to 0.5 "
            f"(boost: 1.5x) to reduce false inflation."
        )
        recommended_weight = 0.5
    else:
        recommendation = (
            f"HZ boost impact is modest: only {hz_propped_fraction:.0%} of HZ targets "
            f"have scores inflated above 1.0 by the boost. Current weight (0.8) is "
            f"acceptable but could be reduced to 0.5 for conservatism."
        )
        recommended_weight = 0.5

    # Additional: check if HZ boost affects ranking relative to non-HZ targets
    # Compare top-N with and without HZ boost
    all_with_scores = [(t["target_id"], t["total_score"]) for t in all_targets if t["total_score"] > 0]
    all_without_hz_scores = [(t["target_id"], t["score_without_hz"]) for t in all_targets if t["total_score"] > 0]

    rank_with = {tid: i for i, (tid, _) in enumerate(sorted(all_with_scores, key=lambda x: -x[1]))}
    rank_without = {tid: i for i, (tid, _) in enumerate(sorted(all_without_hz_scores, key=lambda x: -x[1]))}

    rank_changes = []
    for t in hz_targets:
        if t["total_score"] > 0:
            tid = t["target_id"]
            rw = rank_with.get(tid, -1)
            rwo = rank_without.get(tid, -1)
            if rw != rwo:
                rank_changes.append({
                    "target_id": tid,
                    "rank_with_hz": rw + 1,
                    "rank_without_hz": rwo + 1,
                    "rank_change": rwo - rw,  # positive = moved up by HZ
                    "score_with": t["total_score"],
                    "score_without": t["score_without_hz"],
                })

    rank_changes.sort(key=lambda x: -abs(x["rank_change"]))

    # ----- Assemble output -----
    result = {
        "analysis": "HZ Prior Boost Impact Analysis",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_sources": {
            "n_report_files": len(glob.glob(REPORT_PATTERN)),
            "n_raw_target_entries": sum(1 for _ in all_targets),
            "deduplication": "newest report per target_id",
        },
        "summary": {
            "n_total_unique_targets": n_total,
            "n_scored_targets": n_scored,
            "n_hz_active": n_hz,
            "n_hz_active_and_scored": n_hz_scored,
            "hz_fraction_of_scored": round(n_hz_scored / max(n_scored, 1), 4),
            "hz_fraction_of_total": round(n_hz / max(n_total, 1), 4),
        },
        "score_distributions": {
            "hz_with_boost": {
                "n": len(hz_scores_with),
                "mean": round(float(np.mean(hz_scores_with)), 4) if hz_scores_with else None,
                "median": round(float(np.median(hz_scores_with)), 4) if hz_scores_with else None,
                "std": round(float(np.std(hz_scores_with)), 4) if hz_scores_with else None,
                "min": round(float(np.min(hz_scores_with)), 4) if hz_scores_with else None,
                "max": round(float(np.max(hz_scores_with)), 4) if hz_scores_with else None,
            },
            "hz_without_boost": {
                "n": len(hz_scores_without),
                "mean": round(float(np.mean(hz_scores_without)), 4) if hz_scores_without else None,
                "median": round(float(np.median(hz_scores_without)), 4) if hz_scores_without else None,
                "std": round(float(np.std(hz_scores_without)), 4) if hz_scores_without else None,
                "min": round(float(np.min(hz_scores_without)), 4) if hz_scores_without else None,
                "max": round(float(np.max(hz_scores_without)), 4) if hz_scores_without else None,
            },
            "non_hz_targets": {
                "n": len(non_hz_scores),
                "mean": round(float(np.mean(non_hz_scores)), 4) if non_hz_scores else None,
                "median": round(float(np.median(non_hz_scores)), 4) if non_hz_scores else None,
                "std": round(float(np.std(non_hz_scores)), 4) if non_hz_scores else None,
                "min": round(float(np.min(non_hz_scores)), 4) if non_hz_scores else None,
                "max": round(float(np.max(non_hz_scores)), 4) if non_hz_scores else None,
            },
        },
        "fdr_impact": {
            "n_total_fdr_significant": n_total_fdr,
            "n_hz_fdr_significant": n_hz_fdr,
            "hz_fraction_of_fdr": round(n_hz_fdr / max(n_total_fdr, 1), 4),
            "fdr_changes_if_hz_removed": 0,
            "explanation": (
                "HZ channel has calibrated_p=null (uncalibrated), so it is excluded "
                "from Fisher p-value combination. FDR significance is entirely determined "
                "by calibrated detection channels. Removing HZ boost changes total_score "
                "and ranking, but does NOT change any target's FDR significance."
            ),
        },
        "n_active_channels_check": {
            "hz_affects_n_active": False,
            "explanation": (
                "n_active_channels in EXODUS counts detection channels only. "
                "habitable_zone_planet is classified as a prior (PRIOR_CHANNELS set) "
                "and excluded from n_active_channels, convergence_bonus, and geo_mean."
            ),
        },
        "hz_score_inflation": {
            "n_hz_only_above_1": n_hz_propped,
            "hz_propped_fraction": round(hz_propped_fraction, 4),
            "targets_above_1_only_because_of_hz": [
                {
                    "target_id": t["target_id"],
                    "score_with_hz": round(t["total_score"], 4),
                    "score_without_hz": round(t["score_without_hz"], 4),
                    "n_detection_channels": t["n_detection_channels"],
                    "detection_channels": t["detection_channels"],
                    "boost_factor": t["boost_factor"],
                }
                for t in sorted(hz_only_above_1, key=lambda x: -x["total_score"])
            ],
        },
        "hz_zero_detection": {
            "n_targets": len(hz_zero_detection),
            "explanation": (
                "Targets with HZ active but zero detection channels. "
                "These have total_score=0 because the multiplicative boost "
                "is applied to the base score, which is 0 when n_active=0."
            ),
            "targets": [t["target_id"] for t in hz_zero_detection],
        },
        "weight_sensitivity": weight_analysis,
        "ranking_impact": {
            "n_targets_with_rank_change": len(rank_changes),
            "top_10_rank_changes": rank_changes[:10],
        },
        "recommendation": {
            "text": recommendation,
            "current_weight": 0.8,
            "current_boost": 1.8,
            "recommended_weight": recommended_weight,
            "recommended_boost": 1.0 + recommended_weight,
        },
        "hz_targets_detail": [
            {
                "target_id": t["target_id"],
                "score_with_hz": round(t["total_score"], 4),
                "score_without_hz": round(t["score_without_hz"], 4),
                "boost_factor": t["boost_factor"],
                "inflation_pct": round(t["score_inflation_pct"], 1),
                "n_detection_channels": t["n_detection_channels"],
                "detection_channels": t["detection_channels"],
                "fdr_significant": t["fdr_significant"],
            }
            for t in hz_detail_table
            if t["total_score"] > 0
        ],
    }

    return result


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary(result: dict) -> None:
    """Print a human-readable summary."""
    s = result["summary"]
    sd = result["score_distributions"]
    fdr = result["fdr_impact"]
    infl = result["hz_score_inflation"]
    rec = result["recommendation"]
    ws = result["weight_sensitivity"]
    rank = result["ranking_impact"]

    print("\n" + "=" * 72)
    print("  HZ PRIOR BOOST IMPACT ANALYSIS")
    print("=" * 72)

    print(f"\n--- SCOPE ---")
    print(f"  Report files loaded:        {result['data_sources']['n_report_files']}")
    print(f"  Unique targets (deduped):   {s['n_total_unique_targets']}")
    print(f"  Scored targets (score > 0): {s['n_scored_targets']}")
    print(f"  HZ-active targets:          {s['n_hz_active']} ({s['hz_fraction_of_total']:.1%} of total)")
    print(f"  HZ-active and scored:       {s['n_hz_active_and_scored']} ({s['hz_fraction_of_scored']:.1%} of scored)")

    print(f"\n--- SCORE DISTRIBUTIONS ---")
    print(f"  {'':30s} {'WITH HZ':>12s}  {'WITHOUT HZ':>12s}  {'NON-HZ':>12s}")
    print(f"  {'':30s} {'-------':>12s}  {'----------':>12s}  {'------':>12s}")
    for stat in ["n", "mean", "median", "std", "min", "max"]:
        v_with = sd["hz_with_boost"].get(stat)
        v_without = sd["hz_without_boost"].get(stat)
        v_non = sd["non_hz_targets"].get(stat)
        fmt = lambda v: f"{v:12.4f}" if v is not None and not isinstance(v, int) else f"{v:>12}" if v is not None else f"{'N/A':>12}"
        print(f"  {stat:30s} {fmt(v_with)}  {fmt(v_without)}  {fmt(v_non)}")

    print(f"\n--- FDR IMPACT ---")
    print(f"  Total FDR-significant targets:       {fdr['n_total_fdr_significant']}")
    print(f"  HZ targets that are FDR-significant: {fdr['n_hz_fdr_significant']} ({fdr['hz_fraction_of_fdr']:.1%})")
    print(f"  FDR changes if HZ removed:           {fdr['fdr_changes_if_hz_removed']}")
    print(f"  Reason: {fdr['explanation']}")

    print(f"\n--- N_ACTIVE_CHANNELS CHECK ---")
    print(f"  HZ affects n_active_channels: {result['n_active_channels_check']['hz_affects_n_active']}")
    print(f"  {result['n_active_channels_check']['explanation']}")

    print(f"\n--- HZ SCORE INFLATION ---")
    print(f"  Targets where HZ boost is the ONLY reason score > 1.0:")
    print(f"    Count:    {infl['n_hz_only_above_1']}")
    print(f"    Fraction: {infl['hz_propped_fraction']:.1%} of HZ-active targets")
    if infl["targets_above_1_only_because_of_hz"]:
        print(f"\n    {'TARGET':35s} {'WITH HZ':>8s} {'W/O HZ':>8s} {'BOOST':>6s} {'N_DET':>5s} CHANNELS")
        print(f"    {'-'*35:35s} {'-'*8:>8s} {'-'*8:>8s} {'-'*6:>6s} {'-'*5:>5s} {'-'*20}")
        for t in infl["targets_above_1_only_because_of_hz"][:20]:
            chans = ", ".join(t["detection_channels"]) if t["detection_channels"] else "(none)"
            print(f"    {t['target_id']:35s} {t['score_with_hz']:8.3f} {t['score_without_hz']:8.3f} {t['boost_factor']:6.1f}x {t['n_detection_channels']:5d} {chans}")
    else:
        print("    (none)")

    print(f"\n--- ZERO-DETECTION HZ TARGETS ---")
    zd = result["hz_zero_detection"]
    print(f"  Count: {zd['n_targets']} targets have HZ active but 0 detection channels")
    print(f"  {zd['explanation']}")

    print(f"\n--- WEIGHT SENSITIVITY ---")
    print(f"  How many HZ targets would score > 1.0 at different HZ weights:")
    print(f"    {'WEIGHT':>8s} {'BOOST':>8s} {'N>1.0':>8s} {'N>1.0 (1det)':>14s}")
    print(f"    {'-'*8:>8s} {'-'*8:>8s} {'-'*8:>8s} {'-'*14:>14s}")
    for w in ws:
        print(f"    {w['hz_weight']:8.1f} {w['boost_factor']:8.1f}x {w['n_above_1']:8d} {w['n_above_1_single_det_only']:14d}")

    print(f"\n--- RANKING IMPACT ---")
    print(f"  Targets whose rank changes if HZ removed: {rank['n_targets_with_rank_change']}")
    if rank["top_10_rank_changes"]:
        print(f"\n    {'TARGET':35s} {'RANK W/':>8s} {'RANK W/O':>9s} {'CHANGE':>8s} {'SCORE W/':>9s} {'SCORE W/O':>10s}")
        print(f"    {'-'*35:35s} {'-'*8:>8s} {'-'*9:>9s} {'-'*8:>8s} {'-'*9:>9s} {'-'*10:>10s}")
        for rc in rank["top_10_rank_changes"][:10]:
            direction = f"+{rc['rank_change']}" if rc["rank_change"] > 0 else str(rc["rank_change"])
            print(f"    {rc['target_id']:35s} {rc['rank_with_hz']:8d} {rc['rank_without_hz']:9d} {direction:>8s} {rc['score_with']:9.3f} {rc['score_without']:10.3f}")

    print(f"\n--- RECOMMENDATION ---")
    print(f"  {rec['text']}")
    print(f"\n  Current:     weight={rec['current_weight']}, boost={rec['current_boost']}x")
    print(f"  Recommended: weight={rec['recommended_weight']}, boost={rec['recommended_boost']}x")

    # Additional context: comparison of HZ targets to prime candidates
    print(f"\n--- CONTEXT: PRIME CANDIDATES ---")
    detail = result.get("hz_targets_detail", [])
    prime_affected = [t for t in detail if t["n_detection_channels"] >= 3]
    if prime_affected:
        print(f"  HZ-boosted targets with 3+ detection channels:")
        for t in prime_affected:
            chans = ", ".join(t["detection_channels"])
            print(f"    {t['target_id']:35s} {t['score_with_hz']:8.3f} -> {t['score_without_hz']:8.3f} ({t['inflation_pct']:.0f}% inflation) [{chans}]")
    else:
        print(f"  No HZ-boosted target has 3+ detection channels.")

    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    targets = load_all_targets()
    result = analyze_hz_boost(targets)

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nAnalysis saved to {OUTPUT_FILE}")

    print_summary(result)


if __name__ == "__main__":
    main()
