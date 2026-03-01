#!/usr/bin/env python3
"""
build_aacs_catalog.py — Anomalous Astrophysics Catalog of Stars (AACS)

Aggregates ALL scored targets from every EXODUS campaign report into a single
deduplicated catalog with research-grade AACS taxonomy classification.

Taxonomy v2 — 10 classes in 4 tiers:
  TIER 0 (No Anomaly):   GATHERED
  TIER 1 (Sub-threshold): SINGLE_CHANNEL, HZ_PRIOR
  TIER 2 (Explained):     BINARY, YSO, CIRCUMSTELLAR, ACTIVE_STAR, CONTAMINATION
  TIER 3 (Anomalous):     PARTIALLY_EXPLAINED, UNEXPLAINED
  FALLBACK:               UNCLASSIFIED

Outputs:
  data/reports/aacs_catalog.json  — full catalog (flat list)
  data/reports/aacs_catalog.csv   — CSV for easy viewing
"""

import json
import csv
import glob
import os
import re
import sys
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "data", "reports")
REPORT_GLOB = os.path.join(REPORTS_DIR, "quick_run_*.json")
OUT_JSON = os.path.join(REPORTS_DIR, "aacs_catalog.json")
OUT_CSV = os.path.join(REPORTS_DIR, "aacs_catalog.csv")
OVERRIDES_FILE = os.path.join(PROJECT_ROOT, "data", "verification_overrides.json")


# ── AACS Taxonomy v2 ──────────────────────────────────────────────────────────
#
# Tier 0 — No Anomaly
#   GATHERED            0 active detection channels; data gathered, no anomaly
#
# Tier 1 — Sub-threshold (single channel, low credibility)
#   SINGLE_CHANNEL      Exactly 1 detection channel active
#   HZ_PRIOR            Only habitable_zone_planet active (prior, not detection)
#
# Tier 2 — Explained by known astrophysics
#   BINARY              Binary/multiple system template match, unexplainability < 0.2
#   YSO                 Young stellar object template match, unexplainability < 0.2
#   CIRCUMSTELLAR       Debris disk / circumstellar material template match
#   ACTIVE_STAR         Active / flare star template match
#   CONTAMINATION       Background contamination template match
#
# Tier 3 — Anomalous
#   PARTIALLY_EXPLAINED Template matches but 0.2 <= unexplainability < 0.5
#   UNEXPLAINED         unexplainability >= 0.5 or no template fits multi-channel
#
# Fallback
#   UNCLASSIFIED        No template data available (early reports) + multi-channel

# Maps template names → AACS class
TEMPLATE_CLASS_MAP = {
    "binary_system": "BINARY",
    "spectroscopic_binary": "BINARY",
    "face_on_binary": "BINARY",
    "young_stellar_object": "YSO",
    "debris_disk": "CIRCUMSTELLAR",
    "active_flare_star": "ACTIVE_STAR",
    "background_contamination": "CONTAMINATION",
    "instrumental_systematic": "SINGLE_CHANNEL",
}

# Maps AACS class → paper-friendly tier
TIER_MAP = {
    "GATHERED": "NOISE",
    "SINGLE_CHANNEL": "NOISE",
    "HZ_PRIOR": "NOISE",
    "BINARY": "EXPLAINED",
    "YSO": "EXPLAINED",
    "CIRCUMSTELLAR": "EXPLAINED",
    "ACTIVE_STAR": "EXPLAINED",
    "CONTAMINATION": "EXPLAINED",
    "PARTIALLY_EXPLAINED": "BORDERLINE",
    "UNEXPLAINED": "ANOMALOUS",
    "UNCLASSIFIED": "UNKNOWN",
}

# Well-known multi-channel patterns that indicate binaries
# (fallback heuristic for reports without template data)
BINARY_CHANNEL_PATTERNS = [
    frozenset({"proper_motion_anomaly", "ir_excess"}),
    frozenset({"proper_motion_anomaly", "hr_anomaly"}),
    frozenset({"proper_motion_anomaly", "hr_anomaly", "uv_anomaly"}),
    frozenset({"proper_motion_anomaly", "ir_excess", "hr_anomaly"}),
    frozenset({"proper_motion_anomaly", "ir_excess", "uv_anomaly"}),
    frozenset({"proper_motion_anomaly", "hr_anomaly", "uv_anomaly", "ir_excess"}),
    frozenset({"uv_anomaly", "hr_anomaly"}),  # composite spectrum → spectroscopic binary
    frozenset({"proper_motion_anomaly", "uv_anomaly"}),  # astrometric wobble + UV from companion
]

# Circumstellar channel patterns (debris disk when no template available)
CIRCUMSTELLAR_CHANNEL_PATTERNS = [
    frozenset({"ir_excess", "transit_anomaly"}),  # dust transits + thermal IR
]

# Display ordering for AACS classes (tier order, then alphabetical)
CLASS_DISPLAY_ORDER = [
    "GATHERED", "SINGLE_CHANNEL", "HZ_PRIOR",
    "BINARY", "YSO", "CIRCUMSTELLAR", "ACTIVE_STAR", "CONTAMINATION",
    "PARTIALLY_EXPLAINED", "UNEXPLAINED",
    "UNCLASSIFIED",
]


# ── Helper Functions ──────────────────────────────────────────────────────────

def extract_timestamp_from_filename(filepath: str) -> datetime:
    """Extract datetime from filename like quick_run_YYYYMMDD_HHMMSS.json"""
    basename = os.path.basename(filepath)
    match = re.search(r"quick_run_(\d{8}_\d{6})\.json", basename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    return datetime.min


def extract_campaign_label(filepath: str) -> str:
    """Derive a human-readable campaign label from the report filename."""
    return os.path.basename(filepath).replace(".json", "")


def parse_template_from_explanations(explanations: List[str]) -> Optional[str]:
    """Extract template match name from natural_explanations text (fallback)."""
    if not explanations:
        return None
    for text in explanations:
        m = re.search(r"'(\w+)'\s*template", text)
        if m:
            return m.group(1)
    return None


def parse_unexplainability_from_explanations(explanations: List[str]) -> Optional[float]:
    """Extract unexplainability score from natural_explanations text (fallback)."""
    if not explanations:
        return None
    for text in explanations:
        m = re.search(r"unexplainability[=:]?\s*([\d.]+)", text)
        if m:
            return float(m.group(1))
    return None


def extract_template_data(red_team_data: Dict) -> Dict[str, Any]:
    """Extract template match + unexplainability from per-target red_team checks.

    Primary source: convergence_quality check → evidence dict.
    Fallback: regex parse from natural_explanations text (old reports).
    """
    checks = red_team_data.get("checks", [])
    for check in checks:
        if check.get("check_name") == "convergence_quality":
            ev = check.get("evidence", {})
            template = ev.get("best_template")
            unex = ev.get("unexplainability")
            return {
                "template_match": template,
                "unexplainability": unex,
                "residual_channels": ev.get("residual_channels", []),
                "source": "red_team_checks",
            }

    # Fallback: regex parse from natural_explanations
    ne = red_team_data.get("natural_explanations", [])
    return {
        "template_match": parse_template_from_explanations(ne),
        "unexplainability": parse_unexplainability_from_explanations(ne),
        "residual_channels": [],
        "source": "natural_explanations_regex",
    }


def load_verification_overrides() -> Dict[str, Dict]:
    """Load manual verification overrides from JSON file."""
    if not os.path.exists(OVERRIDES_FILE):
        return {}
    try:
        with open(OVERRIDES_FILE) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"WARNING: Could not load verification overrides: {exc}")
        return {}


# ── Classification Engine ─────────────────────────────────────────────────────

def classify_target(
    n_active_channels: int,
    active_channel_names: List[str],
    template_match: Optional[str],
    unexplainability: Optional[float],
    natural_explanations: List[str],
) -> str:
    """Assign AACS taxonomy class based on all available data.

    Classification priority:
    1. Channel count (0 → GATHERED, 1 → SINGLE_CHANNEL/HZ_PRIOR)
    2. Template match + unexplainability (multi-channel)
    3. Channel-pattern heuristics (fallback for old reports)
    4. Natural explanations text (last resort)
    """
    # ── Tier 0: No anomaly ──
    if n_active_channels == 0:
        return "GATHERED"

    # ── Tier 1: Sub-threshold (single channel) ──
    channels = set(active_channel_names)
    detection_channels = channels - {"habitable_zone_planet"}

    if len(detection_channels) == 0:
        # Only HZ prior (or truly empty)
        if "habitable_zone_planet" in channels:
            return "HZ_PRIOR"
        return "GATHERED"

    if len(detection_channels) == 1:
        return "SINGLE_CHANNEL"

    # ── Multi-channel (2+ detection channels) ──

    # Normalize "none" string to None
    if template_match and template_match.lower() == "none":
        template_match = None

    # Case A: Template data available
    if template_match and template_match in TEMPLATE_CLASS_MAP:
        base_class = TEMPLATE_CLASS_MAP[template_match]

        # Check unexplainability threshold
        if unexplainability is not None:
            if unexplainability >= 0.5:
                return "UNEXPLAINED"
            if unexplainability >= 0.2:
                return "PARTIALLY_EXPLAINED"

        return base_class

    # Case B: Template not in our map but unexplainability available
    if template_match and unexplainability is not None:
        # Unknown template — classify by unexplainability
        if unexplainability >= 0.5:
            return "UNEXPLAINED"
        if unexplainability >= 0.2:
            return "PARTIALLY_EXPLAINED"
        # Low unexplainability + unknown template → treat as explained
        return "UNCLASSIFIED"

    # Case C: No template but unexplainability available
    if unexplainability is not None:
        if unexplainability >= 0.5:
            return "UNEXPLAINED"
        if unexplainability >= 0.2:
            return "PARTIALLY_EXPLAINED"

    # ── Fallback heuristics (for old reports without template data) ──

    # Channel-pattern heuristics
    detection_frozen = frozenset(detection_channels)
    if detection_frozen in BINARY_CHANNEL_PATTERNS:
        return "BINARY"
    if detection_frozen in CIRCUMSTELLAR_CHANNEL_PATTERNS:
        return "CIRCUMSTELLAR"

    # Natural explanations text
    if natural_explanations:
        joined = " ".join(natural_explanations).lower()
        if "binary" in joined and ("explained" in joined or "template" in joined):
            return "BINARY"
        if "yso" in joined or "young stellar" in joined:
            return "YSO"
        if "debris" in joined or "circumstellar" in joined or "disk" in joined:
            return "CIRCUMSTELLAR"

    return "UNCLASSIFIED"


# ── Catalog Builder ───────────────────────────────────────────────────────────

def build_catalog():
    report_files = sorted(glob.glob(REPORT_GLOB))
    if not report_files:
        print(f"ERROR: No report files found at {REPORT_GLOB}")
        sys.exit(1)

    print(f"Found {len(report_files)} report files")

    # Load verification overrides
    overrides = load_verification_overrides()
    if overrides:
        print(f"Loaded {len(overrides)} verification override(s)")
    print()

    # ── Phase 1: Load all reports, extract targets ─────────────────────────
    # Process reports chronologically; for dedup, keep the newest.
    target_registry: Dict[str, Tuple[Dict, datetime, str]] = {}
    report_meta: Dict[str, Dict] = {}

    for filepath in report_files:
        with open(filepath) as fh:
            data = json.load(fh)

        ts = extract_timestamp_from_filename(filepath)
        campaign_label = extract_campaign_label(filepath)
        n_targets = data.get("n_targets", 0)

        # Build top-level red-team lookup (aggregated — used as fallback)
        red_team_by_id = {}
        rt = data.get("red_team", {})
        for entry in rt.get("results", []):
            red_team_by_id[entry["target_id"]] = entry

        report_meta[filepath] = {
            "timestamp": ts,
            "campaign_label": campaign_label,
            "n_targets": n_targets,
            "red_team_by_id": red_team_by_id,
        }

        # Get ALL scored targets
        all_targets = data.get("all_scored", [])
        if not all_targets:
            all_targets = data.get("top_targets", [])

        for target in all_targets:
            tid = target["target_id"]
            if tid in target_registry:
                existing_ts = target_registry[tid][1]
                if ts <= existing_ts:
                    continue
            target_registry[tid] = (target, ts, filepath)

    print(f"Unique targets after dedup: {len(target_registry)}\n")

    # ── Phase 2: Build catalog entries ─────────────────────────────────────
    catalog = []
    campaign_counter = Counter()
    channel_activation_counter = Counter()
    all_channels_seen = set()

    for tid, (target, ts, filepath) in target_registry.items():
        meta = report_meta[filepath]
        campaign_label = meta["campaign_label"]

        # ── Channel analysis ──
        channel_scores = target.get("channel_scores", {})
        active_channels = {}
        inactive_channels = []
        for ch_name, ch_data in channel_scores.items():
            all_channels_seen.add(ch_name)
            if ch_data.get("is_active", False):
                active_channels[ch_name] = {
                    "score": ch_data.get("score", 0.0),
                    "calibrated_p": ch_data.get("calibrated_p"),
                    "details": ch_data.get("details", {}),
                }
                channel_activation_counter[ch_name] += 1
            else:
                inactive_channels.append(ch_name)

        # ── Template + unexplainability extraction ──
        # Primary: per-target embedded red_team (has full checks with evidence)
        embedded_rt = target.get("red_team", {})
        template_data = extract_template_data(embedded_rt)

        template_match = template_data["template_match"]
        unexplainability = template_data["unexplainability"]
        residual_channels = template_data["residual_channels"]
        template_source = template_data["source"]

        # If embedded red_team had no data, try top-level red_team (aggregated)
        if template_match is None and unexplainability is None:
            fallback_rt = meta["red_team_by_id"].get(tid, {})
            ne = fallback_rt.get("natural_explanations", [])
            template_match = parse_template_from_explanations(ne)
            unexplainability = parse_unexplainability_from_explanations(ne)
            if template_match or unexplainability is not None:
                template_source = "top_level_red_team_regex"

        # ── Red-team aggregated fields ──
        # Prefer embedded, fall back to top-level
        rt_agg = embedded_rt if embedded_rt else meta["red_team_by_id"].get(tid, {})
        natural_explanations = rt_agg.get("natural_explanations", [])
        risk_level = rt_agg.get("risk_level")
        recommendation = rt_agg.get("recommendation")
        overall_risk = rt_agg.get("overall_risk")
        top_risk = rt_agg.get("top_risk")
        n_risk_flags = rt_agg.get("n_risk_flags")

        # ── AACS classification ──
        n_active = target.get("n_active_channels", 0)
        active_ch_names = list(active_channels.keys())

        aacs_class = classify_target(
            n_active_channels=n_active,
            active_channel_names=active_ch_names,
            template_match=template_match,
            unexplainability=unexplainability,
            natural_explanations=natural_explanations,
        )

        # ── Verification overrides ──
        override = overrides.get(tid, {})
        verification_status = override.get("verification_status")
        verification_notes = override.get("verification_notes")
        override_class = override.get("override_class")
        if override_class:
            aacs_class = override_class

        # ── Derive tier ──
        aacs_tier = TIER_MAP.get(aacs_class, "UNKNOWN")

        # ── Build entry ──
        entry = {
            "target_id": tid,
            "ra": target.get("ra"),
            "dec": target.get("dec"),
            "distance_pc": target.get("distance_pc"),
            "host_star": target.get("host_star"),
            "total_score": target.get("total_score"),
            "n_active_channels": n_active,
            "stouffer_p": target.get("stouffer_p"),
            "fdr_significant": target.get("fdr_significant", False),
            "q_value": target.get("q_value"),
            "combined_p": target.get("combined_p"),
            "convergence_bonus": target.get("convergence_bonus"),
            "geo_mean": target.get("geo_mean"),
            "coverage_penalty": target.get("coverage_penalty"),
            "n_channels_with_data": target.get("n_channels_with_data"),
            "n_channels_possible": target.get("n_channels_possible"),
            "active_channels": active_channels,
            "inactive_channels": sorted(inactive_channels),
            "red_team_risk_level": risk_level,
            "red_team_recommendation": recommendation,
            "red_team_overall_risk": overall_risk,
            "red_team_top_risk": top_risk,
            "red_team_n_risk_flags": n_risk_flags,
            "natural_explanations": natural_explanations,
            "template_match": template_match,
            "unexplainability": unexplainability,
            "residual_channels": residual_channels,
            "template_source": template_source,
            "aacs_class": aacs_class,
            "aacs_tier": aacs_tier,
            "verification_status": verification_status,
            "verification_notes": verification_notes,
            "source_report": os.path.basename(filepath),
            "report_timestamp": ts.isoformat(),
        }

        catalog.append(entry)
        campaign_counter[campaign_label] += 1

    # Sort catalog by total_score descending
    catalog.sort(key=lambda x: (x["total_score"] or 0), reverse=True)

    # ── Phase 3: Save JSON ─────────────────────────────────────────────────
    with open(OUT_JSON, "w") as fh:
        json.dump(catalog, fh, indent=2, default=str)
    print(f"Saved JSON catalog: {OUT_JSON}")
    print(f"  {len(catalog)} unique targets\n")

    # ── Phase 4: Save CSV ──────────────────────────────────────────────────
    csv_columns = [
        "target_id", "ra", "dec", "distance_pc", "host_star",
        "total_score", "n_active_channels", "stouffer_p",
        "fdr_significant", "q_value",
        "active_channel_names", "active_channel_scores",
        "inactive_channels",
        "red_team_risk_level", "red_team_recommendation",
        "red_team_overall_risk", "red_team_top_risk",
        "template_match", "unexplainability",
        "residual_channels",
        "aacs_class", "aacs_tier",
        "verification_status", "verification_notes",
        "source_report",
    ]

    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_columns)
        writer.writeheader()
        for entry in catalog:
            row = {k: entry.get(k) for k in csv_columns}
            # Flatten active channels for CSV
            row["active_channel_names"] = "|".join(sorted(entry["active_channels"].keys()))
            row["active_channel_scores"] = "|".join(
                f"{ch}={entry['active_channels'][ch]['score']:.4f}"
                for ch in sorted(entry["active_channels"].keys())
            )
            row["inactive_channels"] = "|".join(entry.get("inactive_channels", []))
            row["residual_channels"] = "|".join(entry.get("residual_channels") or [])
            writer.writerow(row)

    print(f"Saved CSV catalog: {OUT_CSV}\n")

    # ── Phase 5: Summary Statistics ────────────────────────────────────────
    print("=" * 78)
    print("AACS CATALOG — SUMMARY STATISTICS")
    print("=" * 78)

    print(f"\nTotal unique targets: {len(catalog)}")

    # Per-campaign breakdown
    print(f"\n{'Campaign Report':<50} {'Targets':>8}")
    print("-" * 60)
    for campaign, count in sorted(campaign_counter.items()):
        print(f"  {campaign:<48} {count:>8}")
    print(f"  {'TOTAL':<48} {sum(campaign_counter.values()):>8}")

    # Channel activation frequency
    print(f"\n{'Channel':<30} {'Activations':>12} {'Fraction':>10}")
    print("-" * 55)
    for ch in sorted(all_channels_seen):
        cnt = channel_activation_counter.get(ch, 0)
        frac = cnt / len(catalog) if catalog else 0
        print(f"  {ch:<28} {cnt:>12} {frac:>10.1%}")

    # Score distribution
    scores = [e["total_score"] for e in catalog if e["total_score"] is not None]
    if scores:
        scores_sorted = sorted(scores)
        n = len(scores_sorted)
        mean_s = sum(scores_sorted) / n
        median_idx = n // 2
        median_s = (
            scores_sorted[median_idx]
            if n % 2 == 1
            else (scores_sorted[median_idx - 1] + scores_sorted[median_idx]) / 2
        )
        p95_idx = int(n * 0.95)
        p95_s = scores_sorted[min(p95_idx, n - 1)]

        print(f"\nScore Distribution (n={n}):")
        print(f"  Min:    {scores_sorted[0]:.6f}")
        print(f"  Median: {median_s:.6f}")
        print(f"  Mean:   {mean_s:.6f}")
        print(f"  95th %%: {p95_s:.6f}")
        print(f"  Max:    {scores_sorted[-1]:.6f}")

    # n_active_channels distribution
    ch_dist = Counter(e["n_active_channels"] for e in catalog)
    print(f"\nChannel Count Distribution:")
    for k in sorted(ch_dist.keys()):
        print(f"  {k} active channels: {ch_dist[k]:>6} targets")

    # ── AACS Classification Breakdown ──
    class_counter = Counter(e["aacs_class"] for e in catalog)
    tier_counter = Counter(e["aacs_tier"] for e in catalog)

    print(f"\n{'─' * 78}")
    print("AACS CLASSIFICATION (Taxonomy v2)")
    print(f"{'─' * 78}")
    print(f"\n{'Class':<25} {'Count':>8} {'Pct':>8}  {'Tier':<12}")
    print("-" * 58)
    for cls in CLASS_DISPLAY_ORDER:
        cnt = class_counter.get(cls, 0)
        if cnt == 0:
            continue
        pct = cnt / len(catalog) * 100
        tier = TIER_MAP.get(cls, "UNKNOWN")
        print(f"  {cls:<23} {cnt:>8} {pct:>7.1f}%  {tier:<12}")
    print(f"  {'─' * 23} {'─' * 8} {'─' * 8}  {'─' * 12}")
    print(f"  {'TOTAL':<23} {len(catalog):>8} {'100.0':>7}%")

    print(f"\nTier Summary:")
    for tier_name in ["NOISE", "EXPLAINED", "BORDERLINE", "ANOMALOUS", "UNKNOWN"]:
        cnt = tier_counter.get(tier_name, 0)
        pct = cnt / len(catalog) * 100
        print(f"  {tier_name:<16} {cnt:>8} ({pct:>5.1f}%)")

    # Multi-channel analysis
    multi_ch = [e for e in catalog if e["n_active_channels"] >= 2]
    if multi_ch:
        mc_class = Counter(e["aacs_class"] for e in multi_ch)
        print(f"\nAmong multi-channel (>=2) targets (n={len(multi_ch)}):")
        for cls in CLASS_DISPLAY_ORDER:
            cnt = mc_class.get(cls, 0)
            if cnt == 0:
                continue
            print(f"  {cls:<23} {cnt:>6}")

    # FDR significant
    n_fdr = sum(1 for e in catalog if e["fdr_significant"])
    print(f"\nFDR-significant targets: {n_fdr}")

    # Verification overrides
    verified = [e for e in catalog if e.get("verification_status")]
    if verified:
        print(f"\nVerification Overrides ({len(verified)}):")
        for e in verified:
            print(
                f"  {e['target_id']:<40} class={e['aacs_class']:<20} "
                f"status={e['verification_status']}"
            )

    # Top 20 by score
    print(f"\n{'─' * 78}")
    print("Top 20 Targets by Score:")
    print(
        f"{'Rank':<5} {'Target':<35} {'Score':>8} {'Ch':>3} "
        f"{'Class':<20} {'Tier':<12} {'Verif'}"
    )
    print("-" * 105)
    for i, e in enumerate(catalog[:20], 1):
        verif = e.get("verification_status") or ""
        print(
            f"  {i:<3} {e['target_id']:<35} {(e['total_score'] or 0):>8.3f} "
            f"{e['n_active_channels']:>3} "
            f"{e['aacs_class']:<20} {e['aacs_tier']:<12} {verif}"
        )

    print(f"\n{'=' * 78}")
    print("Catalog build complete.")


if __name__ == "__main__":
    build_catalog()
