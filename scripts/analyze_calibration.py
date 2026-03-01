#!/usr/bin/env python3
"""
Analyze calibration campaign results for Fisher channel independence.

This script answers the critical question for publication:
  "Are the EXODUS detection channels independent?"

If channels are correlated (e.g., binaries always trigger BOTH IR and PM),
then Fisher combination of p-values is invalid and over-counts significance.

Analyses:
  1. Per-population channel activation rates
  2. Inter-channel correlation matrices (Spearman + Pearson)
  3. Fisher independence test (empirical vs theoretical chi2)
  4. Expected vs observed channel behavior
  5. False positive rates (giants = negative control)
  6. Population-level score distributions

Usage:
  ./venv/bin/python scripts/analyze_calibration.py [--report-dir data/reports/]

Requires calibration campaigns to have completed with all_scored data.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Detection channels (excluding HZ prior) ──────────────────
# Must match EXODUSScorer.CHANNEL_NAMES minus "habitable_zone_planet".
# Audit #5 F6: expanded from 6 to all 10 detection channels.
DETECTION_CHANNELS = [
    "ir_excess",
    "transit_anomaly",
    "radio_anomaly",
    "gaia_photometric_anomaly",
    "proper_motion_anomaly",
    "ir_variability",
    "uv_anomaly",
    "radio_emission",
    "hr_anomaly",
    "abundance_anomaly",
]

# ── Expected behavior per population ─────────────────────────
# Which channels SHOULD fire for each population?
# Audit #5 F6: expanded to cover all 10 detection channels.
EXPECTED = {
    "binary": {
        "should_fire": ["proper_motion_anomaly"],
        "may_fire": ["gaia_photometric_anomaly", "hr_anomaly"],
        "should_not_fire": ["ir_excess", "radio_anomaly", "radio_emission",
                            "abundance_anomaly"],
    },
    "disk": {
        "should_fire": ["ir_excess"],
        "may_fire": ["ir_variability", "uv_anomaly"],
        "should_not_fire": ["proper_motion_anomaly", "radio_anomaly",
                            "radio_emission"],
    },
    "yso": {
        "should_fire": ["ir_excess", "ir_variability"],
        "may_fire": ["gaia_photometric_anomaly", "uv_anomaly", "hr_anomaly"],
        "should_not_fire": ["radio_anomaly"],
    },
    "giant": {
        "should_fire": [],
        "may_fire": ["ir_excess", "hr_anomaly"],  # circumstellar dust in AGB; HR offset
        "should_not_fire": ["proper_motion_anomaly", "radio_anomaly",
                            "radio_emission", "abundance_anomaly"],
    },
}


def load_report(path: str) -> Optional[Dict]:
    """Load a quick_run report JSON."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  WARNING: Could not load {path}: {e}")
        return None


def identify_calibration_reports(report_dir: str) -> Dict[str, str]:
    """Find calibration reports by matching target file names."""
    reports = {}
    report_path = Path(report_dir)

    for json_file in sorted(report_path.glob("quick_run_*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)

            target_file = data.get("target_file", "")
            n_targets = data.get("n_targets", 0)
            has_all_scored = "all_scored" in data

            # Match by target file name
            for pop in ["binary", "disk", "yso", "giant"]:
                if f"calibration_{pop}" in target_file:
                    # Prefer reports with all_scored and more targets
                    existing = reports.get(pop)
                    if existing is None:
                        reports[pop] = str(json_file)
                    else:
                        # Prefer larger n_targets and all_scored
                        old_data = load_report(existing)
                        old_n = old_data.get("n_targets", 0) if old_data else 0
                        old_has_all = "all_scored" in old_data if old_data else False
                        if (has_all_scored and not old_has_all) or (n_targets > old_n):
                            reports[pop] = str(json_file)
        except Exception:
            continue

    return reports


def extract_channel_data(targets: List[Dict]) -> Dict[str, np.ndarray]:
    """Extract per-channel calibrated p-values and scores for a target list.

    Returns dict with keys like 'ir_excess_p', 'ir_excess_score', 'ir_excess_active'.
    """
    n = len(targets)
    result = {}

    for ch in DETECTION_CHANNELS:
        p_vals = np.full(n, np.nan)
        scores = np.full(n, 0.0)
        active = np.zeros(n, dtype=bool)
        has_data = np.zeros(n, dtype=bool)

        for i, t in enumerate(targets):
            ch_data = t.get("channel_scores", {}).get(ch, {})
            if isinstance(ch_data, dict):
                # Check if channel has data
                details = ch_data.get("details", {})
                if isinstance(details, dict) and details.get("reason") == "no data provided":
                    continue

                has_data[i] = True
                scores[i] = ch_data.get("score", 0.0)
                active[i] = ch_data.get("is_active", False)

                cal_p = ch_data.get("calibrated_p")
                if cal_p is not None:
                    p_vals[i] = cal_p

        result[f"{ch}_p"] = p_vals
        result[f"{ch}_score"] = scores
        result[f"{ch}_active"] = active
        result[f"{ch}_has_data"] = has_data

    # Also extract composite scores
    result["total_score"] = np.array([t.get("total_score", 0.0) for t in targets])
    result["stouffer_p"] = np.array([
        t.get("stouffer_p", 1.0) if t.get("stouffer_p") is not None else 1.0
        for t in targets
    ])
    result["n_active"] = np.array([t.get("n_active_channels", 0) for t in targets])

    return result


def analyze_activation_rates(data: Dict[str, np.ndarray], pop_name: str, n_targets: int):
    """Print channel activation rates for a population."""
    print(f"\n  Channel activation rates ({pop_name}, N={n_targets}):")
    print(f"  {'Channel':<30s} {'Has Data':>10s} {'Active':>10s} {'Rate':>10s}")
    print(f"  {'-'*60}")

    for ch in DETECTION_CHANNELS:
        has_data = data[f"{ch}_has_data"]
        active = data[f"{ch}_active"]
        n_data = int(np.sum(has_data))
        n_active = int(np.sum(active))
        rate = n_active / n_data * 100 if n_data > 0 else 0.0
        print(f"  {ch:<30s} {n_data:>10d} {n_active:>10d} {rate:>9.1f}%")

    # Multi-channel
    n_multi = int(np.sum(data["n_active"] >= 2))
    print(f"\n  Multi-channel (2+): {n_multi}/{n_targets} ({n_multi/n_targets*100:.1f}%)")


def analyze_correlations(data: Dict[str, np.ndarray], pop_name: str):
    """Compute and print inter-channel correlation matrix."""
    from scipy import stats

    print(f"\n  Inter-channel SCORE correlations ({pop_name}):")

    # Build score matrix (only channels with sufficient data)
    channels_with_data = []
    score_matrix = []

    for ch in DETECTION_CHANNELS:
        scores = data[f"{ch}_score"]
        has_data = data[f"{ch}_has_data"]
        if np.sum(has_data) > 10:  # Need at least 10 targets with data
            channels_with_data.append(ch)
            score_matrix.append(scores)

    if len(channels_with_data) < 2:
        print("  Insufficient data for correlation analysis")
        return {}

    score_matrix = np.array(score_matrix)  # shape: (n_channels, n_targets)

    # Spearman correlations (robust to non-normality)
    n_ch = len(channels_with_data)
    corr_matrix = np.zeros((n_ch, n_ch))
    p_matrix = np.zeros((n_ch, n_ch))

    short_names = [ch.replace("_anomaly", "").replace("_excess", "")[:10] for ch in channels_with_data]

    print(f"\n  Spearman rho (p-value):")
    header = f"  {'':>12s}" + "".join(f" {s:>12s}" for s in short_names)
    print(header)

    for i in range(n_ch):
        row_str = f"  {short_names[i]:>12s}"
        for j in range(n_ch):
            # Use only targets where BOTH channels have data
            mask = data[f"{channels_with_data[i]}_has_data"] & data[f"{channels_with_data[j]}_has_data"]
            if np.sum(mask) > 10:
                rho, p = stats.spearmanr(score_matrix[i][mask], score_matrix[j][mask])
                corr_matrix[i, j] = rho
                p_matrix[i, j] = p
                if i == j:
                    row_str += f" {'1.000':>12s}"
                else:
                    sig = "*" if p < 0.05 else " "
                    row_str += f" {rho:>5.3f}{sig}({p:.2f})"
            else:
                row_str += f" {'---':>12s}"
        print(row_str)

    # Flag concerning correlations
    print(f"\n  Correlations |rho| > 0.3 AND p < 0.01 (concerning for independence):")
    found_concerning = False
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            if abs(corr_matrix[i, j]) > 0.3 and p_matrix[i, j] < 0.01:
                found_concerning = True
                print(f"    {channels_with_data[i]} <-> {channels_with_data[j]}: "
                      f"rho={corr_matrix[i,j]:.3f}, p={p_matrix[i,j]:.2e}")

    if not found_concerning:
        print("    None found -- channels appear independent!")

    return {
        "channels": channels_with_data,
        "corr_matrix": corr_matrix.tolist(),
        "p_matrix": p_matrix.tolist(),
    }


def analyze_fisher_validity(data: Dict[str, np.ndarray], pop_name: str):
    """Test whether Fisher combination p-values follow expected chi2 distribution.

    Under the null (independent channels), -2*sum(log(p_i)) ~ chi2(2k).
    If channels are correlated, the distribution will be shifted.
    """
    from scipy import stats

    print(f"\n  Fisher combination validity ({pop_name}):")

    # Collect p-values for channels with calibration
    p_columns = []
    ch_names = []
    for ch in DETECTION_CHANNELS:
        p_vals = data[f"{ch}_p"]
        valid = ~np.isnan(p_vals)
        if np.sum(valid) > 10:
            p_columns.append(p_vals)
            ch_names.append(ch)

    if len(p_columns) < 2:
        print("  Insufficient calibrated channels for Fisher test")
        return

    # For each target, compute Fisher statistic using available channels
    n_targets = len(p_columns[0])
    fisher_stats = []
    dofs = []

    for i in range(n_targets):
        p_vals = []
        for col in p_columns:
            if not np.isnan(col[i]) and col[i] > 0:
                p_vals.append(col[i])
        if len(p_vals) >= 2:
            # Fisher statistic: -2 * sum(log(p))
            fisher_stat = -2.0 * sum(np.log(max(p, 1e-300)) for p in p_vals)
            fisher_stats.append(fisher_stat)
            dofs.append(2 * len(p_vals))

    if len(fisher_stats) < 10:
        print("  Too few targets with multiple calibrated channels")
        return

    fisher_stats = np.array(fisher_stats)
    dofs = np.array(dofs)

    # Expected: chi2(dof) with mean=dof, var=2*dof
    # If inflated, channels are correlated (Fisher is anti-conservative)
    # If deflated, channels are redundant
    unique_dofs = np.unique(dofs)
    for dof in unique_dofs:
        mask = dofs == dof
        n_this = int(np.sum(mask))
        if n_this < 5:
            continue

        stats_this = fisher_stats[mask]
        expected_mean = float(dof)
        expected_std = np.sqrt(2.0 * dof)
        observed_mean = float(np.mean(stats_this))
        observed_std = float(np.std(stats_this))

        # KS test against chi2(dof)
        ks_stat, ks_p = stats.kstest(stats_this, 'chi2', args=(dof,))

        inflation = observed_mean / expected_mean if expected_mean > 0 else np.inf

        print(f"\n  DoF={dof} (k={dof//2} channels), N={n_this} targets:")
        print(f"    Expected mean={expected_mean:.1f}, std={expected_std:.2f}")
        print(f"    Observed mean={observed_mean:.1f}, std={observed_std:.2f}")
        print(f"    Inflation factor: {inflation:.3f}")
        print(f"    KS test vs chi2({dof}): D={ks_stat:.3f}, p={ks_p:.4f}")

        if inflation > 1.5:
            print(f"    WARNING: Fisher statistic inflated {inflation:.1f}x -- channels likely CORRELATED")
        elif inflation < 0.7:
            print(f"    NOTE: Fisher statistic deflated -- channels may be redundant")
        elif ks_p < 0.01:
            print(f"    WARNING: KS rejects chi2 null (p={ks_p:.4f}) -- deviation from independence")
        else:
            print(f"    PASS: Consistent with independent channels")


def analyze_expected_behavior(data: Dict[str, np.ndarray], pop_name: str, n_targets: int):
    """Check whether observed channel activations match expectations."""
    expected = EXPECTED.get(pop_name, {})
    if not expected:
        return

    print(f"\n  Expected vs observed behavior ({pop_name}):")

    for category, channels in [
        ("SHOULD fire", expected.get("should_fire", [])),
        ("MAY fire", expected.get("may_fire", [])),
        ("Should NOT fire", expected.get("should_not_fire", [])),
    ]:
        for ch in channels:
            has_data = data.get(f"{ch}_has_data", np.zeros(n_targets, dtype=bool))
            active = data.get(f"{ch}_active", np.zeros(n_targets, dtype=bool))
            n_data = int(np.sum(has_data))
            n_active = int(np.sum(active))
            rate = n_active / n_data * 100 if n_data > 0 else 0.0

            status = ""
            if category == "SHOULD fire" and rate < 20:
                status = " [LOW -- expected higher]"
            elif category == "Should NOT fire" and rate > 20:
                status = " [HIGH -- expected lower]"
            elif category == "Should NOT fire" and rate < 5:
                status = " [OK]"
            elif category == "SHOULD fire" and rate > 50:
                status = " [OK]"

            print(f"    {category}: {ch:<30s} = {rate:5.1f}% ({n_active}/{n_data}){status}")


def compute_false_positive_rate(data: Dict[str, np.ndarray], pop_name: str, n_targets: int):
    """Compute false positive rate: what % of targets trigger anomalies?"""
    total_score = data["total_score"]
    n_anomaly = int(np.sum(total_score > 0))
    n_fdr = int(np.sum(data["stouffer_p"] < 0.05))

    print(f"\n  False positive rates ({pop_name}, N={n_targets}):")
    print(f"    Any anomaly (score > 0): {n_anomaly}/{n_targets} ({n_anomaly/n_targets*100:.1f}%)")
    print(f"    FDR significant (stouffer_p < 0.05): {n_fdr}/{n_targets} ({n_fdr/n_targets*100:.1f}%)")
    print(f"    Score distribution: mean={np.mean(total_score):.3f}, "
          f"median={np.median(total_score):.3f}, max={np.max(total_score):.3f}")

    # Score percentiles
    for pct in [50, 90, 95, 99]:
        val = np.percentile(total_score, pct)
        print(f"    {pct}th percentile score: {val:.3f}")


# ── Per-channel FPR table: known population reports ──────
# Maps population label → report file path (relative to project root).
# These are the definitive calibration runs from sessions 20-25.
FPR_REPORT_MAP: Dict[str, str] = {
    # Report paths: set via environment variables or CLI --report-map.
    # No defaults — must be configured per installation.
    "Binary (500)": os.environ.get("EXODUS_REPORT_BINARY", ""),
    "Disk (500)": os.environ.get("EXODUS_REPORT_DISK", ""),
    "Lower-RUWE (500)": os.environ.get("EXODUS_REPORT_LRUWE", ""),
    "Control FGK (100)": os.environ.get("EXODUS_REPORT_CONTROL", ""),
    "Science Target": os.environ.get("EXODUS_REPORT_SCIENCE", ""),
}

# Channels to include in the FPR table, ordered by expected information content
FPR_CHANNELS = [
    "radio_emission",
    "ir_excess",
    "uv_anomaly",
    "hr_anomaly",
    "ir_variability",
    "proper_motion_anomaly",
]

# Short display labels for channels
FPR_CHANNEL_LABELS = {
    "radio_emission": "Radio emission",
    "ir_excess": "IR excess",
    "uv_anomaly": "UV anomaly",
    "hr_anomaly": "HR anomaly",
    "ir_variability": "IR variability",
    "proper_motion_anomaly": "Proper motion",
}


def generate_fpr_table(
    report_dir: str = "data/reports/",
    report_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Dict]]:
    """Generate a per-channel FPR (activation rate) table across all populations.

    Returns a nested dict: {channel: {population: {n_data, n_active, rate}}}
    """
    if report_map is None:
        report_map = FPR_REPORT_MAP

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent

    # Load all population data
    pop_data: Dict[str, Dict[str, np.ndarray]] = {}
    pop_n: Dict[str, int] = {}

    for pop_label, report_path in report_map.items():
        full_path = project_root / report_path
        if not full_path.exists():
            print(f"  WARNING: Report not found: {full_path}")
            continue

        report = load_report(str(full_path))
        if report is None:
            continue

        targets = report.get("all_scored", report.get("top_targets", []))
        if not targets:
            print(f"  WARNING: No targets in {report_path}")
            continue

        data = extract_channel_data(targets)
        pop_data[pop_label] = data
        pop_n[pop_label] = len(targets)

    if not pop_data:
        print("  ERROR: No populations loaded for FPR table")
        return {}

    # Build the table
    table: Dict[str, Dict[str, Dict]] = {}

    for ch in FPR_CHANNELS:
        table[ch] = {}
        for pop_label in pop_data:
            data = pop_data[pop_label]
            has_data = data.get(f"{ch}_has_data", np.zeros(pop_n[pop_label], dtype=bool))
            active = data.get(f"{ch}_active", np.zeros(pop_n[pop_label], dtype=bool))
            n_data = int(np.sum(has_data))
            n_active = int(np.sum(active))
            rate = n_active / n_data if n_data > 0 else None

            table[ch][pop_label] = {
                "n_data": n_data,
                "n_active": n_active,
                "rate": rate,
                "n_total": pop_n[pop_label],
            }

    return table


def compute_information_content(table: Dict[str, Dict[str, Dict]]) -> Dict[str, float]:
    """Compute effective information content per channel: I_ch = -log2(median FPR).

    Higher values = more informative (lower background rate).
    Channels with FPR=0 or no data get NaN.
    """
    info = {}
    for ch in FPR_CHANNELS:
        rates = []
        for pop_label, stats in table.get(ch, {}).items():
            if stats["rate"] is not None and stats["n_data"] >= 10:
                rates.append(stats["rate"])

        if not rates:
            info[ch] = float("nan")
        else:
            median_rate = float(np.median(rates))
            if median_rate > 0:
                info[ch] = -math.log2(median_rate)
            else:
                info[ch] = float("inf")  # Never fires → infinite information

    return info


def print_fpr_table(table: Dict[str, Dict[str, Dict]], info: Dict[str, float]):
    """Print a formatted per-channel FPR table to stdout."""
    populations = list(next(iter(table.values())).keys()) if table else []

    print(f"\n{'='*90}")
    print(f"  PER-CHANNEL ACTIVATION RATES (FPR) ACROSS POPULATIONS")
    print(f"{'='*90}")

    # Header
    pop_short = [p.split(" (")[0][:12] for p in populations]
    header = f"  {'Channel':<20s}"
    for ps in pop_short:
        header += f" {ps:>12s}"
    header += f" {'I(bits)':>10s}"
    print(header)
    print(f"  {'-'*(20 + 13*len(populations) + 11)}")

    # Data rows
    for ch in FPR_CHANNELS:
        label = FPR_CHANNEL_LABELS.get(ch, ch)[:20]
        row = f"  {label:<20s}"
        for pop in populations:
            stats = table[ch].get(pop, {})
            rate = stats.get("rate")
            n_data = stats.get("n_data", 0)
            if rate is None or n_data < 5:
                row += f" {'--':>12s}"
            else:
                row += f" {rate*100:>10.1f}%"
                if n_data < 20:
                    row = row[:-1] + "?"  # Uncertain — small sample
        # Information content
        i_val = info.get(ch, float("nan"))
        if math.isnan(i_val):
            row += f" {'--':>10s}"
        elif math.isinf(i_val):
            row += f" {'INF':>10s}"
        else:
            row += f" {i_val:>10.2f}"
        print(row)

    # Summary
    print()
    print(f"  Population sizes:")
    for pop in populations:
        # Get n_total from any channel
        for ch in FPR_CHANNELS:
            stats = table[ch].get(pop, {})
            if "n_total" in stats:
                print(f"    {pop}: N={stats['n_total']}")
                break

    print(f"\n  Information content I = -log2(median FPR across populations)")
    print(f"  Higher I = lower background rate = more informative per activation")
    print(f"  Radio emission: each activation is a genuine rarity")
    print(f"  Proper motion: activations are near-universal (dominated by systematics)")


def format_fpr_latex(table: Dict[str, Dict[str, Dict]], info: Dict[str, float]) -> str:
    """Generate a LaTeX-ready table for the paper.

    Returns a string containing the full LaTeX tabular environment.
    """
    populations = list(next(iter(table.values())).keys()) if table else []
    n_pop = len(populations)

    lines = []
    lines.append("% Per-channel activation rates across calibration populations")
    lines.append("% Generated by analyze_calibration.py --fpr-table")
    lines.append(r"\begin{table*}")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Per-channel activation rates (false positive rates) "
        r"across calibration populations. $I$ is the information content "
        r"$-\log_2(\mathrm{median\;FPR})$; higher values indicate channels "
        r"whose activations carry more diagnostic weight.}"
    )
    lines.append(r"  \label{tab:per_channel_fpr}")

    # Column spec
    col_spec = "l" + "r" * n_pop + "r"
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \hline")

    # Header row
    pop_headers = [p.replace("_", r"\_").replace("(", r"(").replace(")", r")") for p in populations]
    header = "    Channel"
    for ph in pop_headers:
        header += f" & {ph}"
    header += r" & $I$ (bits) \\"
    lines.append(header)
    lines.append(r"    \hline")

    # Data rows
    for ch in FPR_CHANNELS:
        label = FPR_CHANNEL_LABELS.get(ch, ch)
        row = f"    {label}"
        for pop in populations:
            stats = table[ch].get(pop, {})
            rate = stats.get("rate")
            n_data = stats.get("n_data", 0)
            if rate is None or n_data < 5:
                row += " & --"
            else:
                pct = rate * 100
                if pct < 0.05:
                    row += r" & $<$0.1\%"
                else:
                    row += f" & {pct:.1f}\\%"
        # Information content
        i_val = info.get(ch, float("nan"))
        if math.isnan(i_val):
            row += " & --"
        elif math.isinf(i_val):
            row += r" & $\infty$"
        else:
            row += f" & {i_val:.1f}"
        row += r" \\"
        lines.append(row)

    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")

    # Table note
    lines.append(
        r"  \tablecomments{Activation rate = fraction of targets where "
        r"channel score $\geq 0.3$ (universal threshold). "
        r"Populations: Binary = spectroscopic binaries from SB9; "
        r"Disk = debris disk hosts; Lower-RUWE = Gaia RUWE 1.4--3.0; "
        r"Control FGK = matched field FGK stars; "
        r"Science = IR-selected FGK target pool.}"
    )
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze calibration results for Fisher independence")
    parser.add_argument("--report-dir", default="data/reports/",
                        help="Directory containing quick_run report JSONs")
    parser.add_argument("--reports", nargs="+",
                        help="Explicit report paths: binary disk yso giant")
    parser.add_argument("--output", default="data/reports/calibration_analysis.json",
                        help="Output analysis JSON")
    parser.add_argument("--fpr-table", action="store_true",
                        help="Generate per-channel FPR table across all populations")
    parser.add_argument("--latex", action="store_true",
                        help="Output LaTeX-formatted FPR table (use with --fpr-table)")
    args = parser.parse_args()

    print("=" * 70)
    print("  EXODUS Calibration Analysis — Fisher Channel Independence")
    print("=" * 70)

    all_results = {}

    # ── FPR-only mode: skip Fisher analysis if only --fpr-table requested ──
    if args.fpr_table and not args.reports:
        # Run FPR table generation directly, skip Fisher analysis
        print(f"\n  Mode: Per-channel FPR table generation (--fpr-table)")
        fpr_table = generate_fpr_table(args.report_dir)
        if fpr_table:
            info_content = compute_information_content(fpr_table)
            print_fpr_table(fpr_table, info_content)

            all_results["fpr_table"] = {}
            for ch in FPR_CHANNELS:
                all_results["fpr_table"][ch] = {
                    pop: {
                        "rate": stats["rate"],
                        "n_data": stats["n_data"],
                        "n_active": stats["n_active"],
                    }
                    for pop, stats in fpr_table[ch].items()
                }
            all_results["information_content"] = {
                ch: (val if not math.isnan(val) and not math.isinf(val) else str(val))
                for ch, val in info_content.items()
            }

            if args.latex:
                latex_str = format_fpr_latex(fpr_table, info_content)
                latex_path = Path(args.report_dir) / "fpr_table.tex"
                with open(latex_path, "w") as f:
                    f.write(latex_str)
                print(f"\n  LaTeX table saved to: {latex_path}")
                print(f"\n  LaTeX output:")
                print(latex_str)

        # Save and exit
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Analysis saved to: {args.output}")
        print(f"\n{'='*70}")
        print(f"  CALIBRATION ANALYSIS COMPLETE")
        print(f"{'='*70}")
        return

    # ── Find reports for Fisher analysis ──
    if args.reports and len(args.reports) >= 4:
        report_map = {
            "binary": args.reports[0],
            "disk": args.reports[1],
            "yso": args.reports[2],
            "giant": args.reports[3],
        }
    else:
        print(f"\n  Scanning {args.report_dir} for calibration reports ...")
        report_map = identify_calibration_reports(args.report_dir)

    if not report_map:
        print("\n  ERROR: No calibration reports found!")
        print("  Run calibration campaigns first:")
        print("    nohup bash scripts/run_calibration_full.sh > data/reports/calibration_full_log.txt 2>&1 &")
        sys.exit(1)

    print(f"\n  Found {len(report_map)} calibration report(s):")
    for pop, path in report_map.items():
        print(f"    {pop}: {path}")

    missing = [p for p in ["binary", "disk", "yso", "giant"] if p not in report_map]
    if missing:
        print(f"\n  WARNING: Missing populations: {missing}")
        print("  Analysis will be partial.")

    for pop_name, report_path in report_map.items():
        print(f"\n{'='*70}")
        print(f"  POPULATION: {pop_name.upper()}")
        print(f"{'='*70}")

        report = load_report(report_path)
        if report is None:
            continue

        # Get target list — prefer all_scored, fall back to top_targets
        targets = report.get("all_scored", report.get("top_targets", []))
        n_targets = len(targets)
        n_total = report.get("n_targets", n_targets)

        print(f"\n  Report: {report_path}")
        print(f"  Targets in report: {n_targets} (of {n_total} total)")
        if n_targets < n_total:
            print(f"  WARNING: Only {n_targets}/{n_total} targets in report (top_targets only)")
            print(f"  Re-run with expanded reports for full population analysis")

        has_all = "all_scored" in report
        print(f"  Has all_scored: {has_all}")

        # Extract channel data
        data = extract_channel_data(targets)

        # 1. Activation rates
        analyze_activation_rates(data, pop_name, n_targets)

        # 2. Expected vs observed
        analyze_expected_behavior(data, pop_name, n_targets)

        # 3. False positive rates
        compute_false_positive_rate(data, pop_name, n_targets)

        # 4. Inter-channel correlations
        corr_result = analyze_correlations(data, pop_name)

        # 5. Fisher validity
        analyze_fisher_validity(data, pop_name)

        all_results[pop_name] = {
            "n_targets": n_targets,
            "n_total": n_total,
            "has_all_scored": has_all,
            "correlations": corr_result,
        }

    # ── Cross-population comparison ──
    if len(report_map) >= 2:
        print(f"\n{'='*70}")
        print(f"  CROSS-POPULATION COMPARISON")
        print(f"{'='*70}")

        print(f"\n  {'Population':<15s} {'N':>6s} {'Anomaly%':>10s} {'FDR%':>8s} {'MedScore':>10s} {'MaxScore':>10s}")
        print(f"  {'-'*60}")

        for pop_name, report_path in report_map.items():
            report = load_report(report_path)
            if report is None:
                continue
            targets = report.get("all_scored", report.get("top_targets", []))
            n = len(targets)
            scores = np.array([t.get("total_score", 0.0) for t in targets])
            stouffer = np.array([
                t.get("stouffer_p", 1.0) if t.get("stouffer_p") is not None else 1.0
                for t in targets
            ])
            n_anom = int(np.sum(scores > 0))
            n_fdr = int(np.sum(stouffer < 0.05))

            print(f"  {pop_name:<15s} {n:>6d} {n_anom/n*100:>9.1f}% {n_fdr/n*100:>7.1f}% "
                  f"{np.median(scores):>10.3f} {np.max(scores):>10.3f}")

        # Key question: do binaries trigger IR at the same rate as random stars?
        print(f"\n  KEY INDEPENDENCE QUESTIONS:")
        print(f"  1. Do binaries (PM-triggered) also trigger IR at elevated rates?")
        print(f"     → If yes: IR and PM are correlated for binaries")
        print(f"  2. Do disk hosts (IR-triggered) also trigger PM at elevated rates?")
        print(f"     → If yes: IR and PM are correlated for disk hosts")
        print(f"  3. Do giants (negative control) trigger anything?")
        print(f"     → Sets the population false positive rate baseline")

    # ── Save results ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Analysis saved to: {args.output}")

    print(f"\n{'='*70}")
    print(f"  CALIBRATION ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
