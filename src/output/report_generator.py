"""
Comprehensive Reporting & Publication Pipeline for Project EXODUS.

After each research iteration, generates:
  1. Dashboard HTML  — Sky map, top targets, hypothesis status, evolution log
  2. Iteration Summary — What was tested, found, and changed
  3. Candidate Reports — For targets reaching Breakthrough Level 3+, full
     multi-wavelength report suitable for a paper
  4. Methodology Paper Draft — Auto-updating document describing system,
     datasets, methods, and results (null results are publishable)

Report includes:
  - All hypotheses tested with results
  - EXODUS score distribution histogram (ASCII)
  - Top 20 targets with full multi-modal breakdown
  - All Breakthrough Engine escalations and outcomes
  - Evolver evolution log
  - Self-diagnosis results
  - Data coverage map (targets × datasets)

Public API
----------
generate_full_report(iteration, scores, breakthroughs, ...)
    Generate all report artefacts for one iteration.

generate_dashboard_html(iteration, scores, ...)
    Render the interactive HTML dashboard.

generate_candidate_report(target_id, score, ...)
    Detailed multi-wavelength candidate report (Markdown).

generate_methodology_draft(iteration, ...)
    Auto-updating methodology paper draft (Markdown).
"""

from __future__ import annotations

import sys
import json
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone

_UTC = timezone.utc
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger, PROJECT_ROOT

log = get_logger("output.report_generator")

REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
#  Data classes
# =====================================================================

@dataclass
class IterationContext:
    """Everything needed to generate reports for one iteration."""
    iteration: int
    timestamp: str = ""
    n_targets_scored: int = 0
    scores: List[Dict[str, Any]] = field(default_factory=list)
    breakthroughs: List[Dict[str, Any]] = field(default_factory=list)
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    evolver_log: List[Dict[str, Any]] = field(default_factory=list)
    diagnosis: Dict[str, Any] = field(default_factory=dict)
    imagination_results: List[Dict[str, Any]] = field(default_factory=list)
    data_coverage: Dict[str, List[str]] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(tz=_UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


@dataclass
class ReportBundle:
    """All generated report artefacts from one iteration."""
    dashboard_path: str = ""
    summary_path: str = ""
    methodology_path: str = ""
    candidate_paths: List[str] = field(default_factory=list)


# =====================================================================
#  ASCII histogram helper
# =====================================================================

def _ascii_histogram(values: List[float], bins: int = 20, width: int = 50,
                     title: str = "") -> str:
    """Render an ASCII histogram of values."""
    if not values:
        return "  (no data)\n"

    arr = np.array(values, dtype=float)
    counts, edges = np.histogram(arr, bins=bins)
    max_count = max(counts) if max(counts) > 0 else 1

    lines = []
    if title:
        lines.append(f"  {title}")
        lines.append(f"  {'─' * (width + 20)}")

    for i, count in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        bar_len = int(count / max_count * width)
        bar = "█" * bar_len
        lines.append(f"  {lo:8.3f}-{hi:8.3f} │{bar} {count}")

    lines.append(f"  {'─' * (width + 20)}")
    lines.append(f"  n={len(arr)}, min={arr.min():.4f}, "
                 f"max={arr.max():.4f}, median={np.median(arr):.4f}")
    return "\n".join(lines) + "\n"


def _data_coverage_table(coverage: Dict[str, List[str]],
                         max_targets: int = 30) -> str:
    """Render a target × dataset coverage table."""
    if not coverage:
        return "  (no coverage data)\n"

    all_datasets = sorted({ds for dsets in coverage.values() for ds in dsets})
    if not all_datasets:
        return "  (no datasets)\n"

    # Header
    col_w = 6
    header = f"  {'Target':<24s} │ " + " ".join(f"{d[:col_w]:<{col_w}s}" for d in all_datasets)
    lines = [header, "  " + "─" * len(header)]

    targets = sorted(coverage.keys())[:max_targets]
    for tid in targets:
        dsets = set(coverage[tid])
        cols = " ".join(
            f"{'  ✓   ' if d in dsets else '  ·   '}" for d in all_datasets
        )
        lines.append(f"  {tid:<24s} │ {cols}")

    if len(coverage) > max_targets:
        lines.append(f"  ... and {len(coverage) - max_targets} more targets")

    return "\n".join(lines) + "\n"


# =====================================================================
#  Iteration summary (Markdown)
# =====================================================================

def generate_iteration_summary(ctx: IterationContext) -> str:
    """Generate a Markdown iteration summary."""

    scores = ctx.scores
    total_scores = [s.get("total_score", 0) for s in scores]

    # Top 20 targets
    ranked = sorted(scores, key=lambda s: s.get("total_score", 0), reverse=True)[:20]

    top_table = []
    for i, s in enumerate(ranked, 1):
        tid = s.get("target_id", "?")
        ts = s.get("total_score", 0)
        n_active = s.get("n_active_channels", 0)
        channels = s.get("channel_scores", {})
        active_names = [
            ch for ch, cs in channels.items()
            if (cs.get("is_active") if isinstance(cs, dict) else getattr(cs, "is_active", False))
        ]
        top_table.append(
            f"| {i:>3} | {tid:<26s} | {ts:8.3f} | {n_active}/10+1prior | "
            f"{', '.join(active_names):<50s} |"
        )

    # Breakthrough escalations
    bt_lines = []
    for b in ctx.breakthroughs:
        name = b.get("target_id", b.get("target", "?"))
        level = b.get("level", "?")
        bt_lines.append(f"  - **{name}** → Level {level}")
    if not bt_lines:
        bt_lines.append("  - (none)")

    # Hypothesis results
    hyp_lines = []
    for h in ctx.hypotheses:
        hid = h.get("id", h.get("hypothesis_id", "?"))
        status = h.get("status", "untested")
        hyp_lines.append(f"  - `{hid}`: {status}")
    if not hyp_lines:
        hyp_lines.append("  - (none)")

    # Diagnosis summary
    diag_lines = []
    checks = ctx.diagnosis.get("checks", [])
    for ch in checks:
        name = ch.get("name", "?")
        status = ch.get("status", "?")
        emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(status, "⚪")
        diag_lines.append(f"  - {emoji} **{name}**: {status}")
    if not diag_lines:
        diag_lines.append("  - (no diagnosis data)")

    # Evolver changes
    evo_lines = []
    for e in ctx.evolver_log:
        desc = e.get("description", e.get("change", "?"))
        evo_lines.append(f"  - {desc}")
    if not evo_lines:
        evo_lines.append("  - (no evolution this iteration)")

    histogram = _ascii_histogram(total_scores, title="EXODUS Score Distribution")
    coverage = _data_coverage_table(ctx.data_coverage)

    md = textwrap.dedent(f"""\
    # Project EXODUS — Iteration {ctx.iteration} Summary

    **Generated:** {ctx.timestamp}
    **Targets scored:** {ctx.n_targets_scored}

    ---

    ## EXODUS Score Distribution

    ```
    {histogram}
    ```

    ## Top 20 Targets

    | Rank | Target                     |   Score  | Active  | Channels                                             |
    |------|----------------------------|----------|---------|------------------------------------------------------|
    """) + "\n".join(top_table) + "\n"

    md += textwrap.dedent(f"""
    ## Breakthrough Engine Escalations

    {chr(10).join(bt_lines)}

    ## Hypothesis Testing

    {chr(10).join(hyp_lines)}

    ## Self-Diagnosis

    {chr(10).join(diag_lines)}

    ## Evolver Changes

    {chr(10).join(evo_lines)}

    ## Data Coverage

    ```
    {coverage}
    ```

    ---
    *Report generated by Project EXODUS Reporting Pipeline v1.0*
    """)

    return md


# =====================================================================
#  Candidate report (Markdown)
# =====================================================================

def generate_candidate_report(
    target_id: str,
    score: Dict[str, Any],
    breakthrough: Optional[Dict[str, Any]] = None,
    ir_data: Optional[Dict[str, Any]] = None,
    transit_data: Optional[Dict[str, Any]] = None,
    radio_data: Optional[Dict[str, Any]] = None,
    cross_band: Optional[Dict[str, Any]] = None,
    stellar_anomaly: Optional[Dict[str, Any]] = None,
    frb_match: Optional[Dict[str, Any]] = None,
    gamma_match: Optional[Dict[str, Any]] = None,
    neutrino_match: Optional[Dict[str, Any]] = None,
    gw_match: Optional[Dict[str, Any]] = None,
    pulsar_match: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a detailed multi-wavelength candidate report."""

    ts = score.get("total_score", 0)
    n_active = score.get("n_active_channels", 0)
    ra = score.get("ra", 0)
    dec = score.get("dec", 0)

    channels = score.get("channel_scores", {})
    channel_detail = []
    for ch_name, cs in channels.items():
        if isinstance(cs, dict):
            sc = cs.get("score", 0)
            active = cs.get("is_active", False)
            details = cs.get("details", {})
        else:
            sc = getattr(cs, "score", 0)
            active = getattr(cs, "is_active", False)
            details = getattr(cs, "details", {})

        status = "ACTIVE" if active else "inactive"
        channel_detail.append(f"### {ch_name} — {sc:.3f} ({status})\n")
        if details:
            for k, v in details.items():
                channel_detail.append(f"  - {k}: {v}\n")
        channel_detail.append("")

    # Multi-messenger sections
    mm_sections = []

    if ir_data:
        mm_sections.append("### Mid-IR Excess\n")
        for k, v in ir_data.items():
            mm_sections.append(f"  - {k}: {v}\n")

    if transit_data:
        mm_sections.append("### Transit Anomaly\n")
        for k, v in transit_data.items():
            mm_sections.append(f"  - {k}: {v}\n")

    if radio_data:
        mm_sections.append("### Radio Analysis\n")
        for k, v in radio_data.items():
            mm_sections.append(f"  - {k}: {v}\n")

    if cross_band:
        mm_sections.append("### Cross-Band Temporal Correlation\n")
        for k, v in cross_band.items():
            mm_sections.append(f"  - {k}: {v}\n")

    if stellar_anomaly:
        mm_sections.append("### Stellar Anomaly (HR Diagram)\n")
        for k, v in stellar_anomaly.items():
            mm_sections.append(f"  - {k}: {v}\n")

    if gamma_match:
        mm_sections.append("### Gamma-Ray Cross-Match\n")
        for k, v in gamma_match.items():
            mm_sections.append(f"  - {k}: {v}\n")

    if neutrino_match:
        mm_sections.append("### Neutrino Cross-Match\n")
        for k, v in neutrino_match.items():
            mm_sections.append(f"  - {k}: {v}\n")

    if gw_match:
        mm_sections.append("### Gravitational Wave Cross-Match\n")
        for k, v in gw_match.items():
            mm_sections.append(f"  - {k}: {v}\n")

    if pulsar_match:
        mm_sections.append("### Pulsar Timing\n")
        for k, v in pulsar_match.items():
            mm_sections.append(f"  - {k}: {v}\n")

    if frb_match:
        mm_sections.append("### FRB-Orbital Correlation\n")
        for k, v in frb_match.items():
            mm_sections.append(f"  - {k}: {v}\n")

    bt_section = ""
    if breakthrough:
        level = breakthrough.get("level", "?")
        reason = breakthrough.get("reason", "N/A")
        bt_section = textwrap.dedent(f"""
        ## Breakthrough Engine Evaluation

        - **Level:** {level}
        - **Reason:** {reason}
        - **Status:** Escalated for detailed follow-up
        """)

    md = textwrap.dedent(f"""\
    # EXODUS Candidate Report: {target_id}

    **Generated:** {datetime.now(tz=_UTC).strftime("%Y-%m-%d %H:%M:%S UTC")}

    ## Summary

    | Property         | Value         |
    |------------------|---------------|
    | Target ID        | {target_id}   |
    | RA (deg)         | {ra:.6f}      |
    | Dec (deg)        | {dec:.6f}     |
    | EXODUS Score     | {ts:.4f}      |
    | Active Channels  | {n_active}/10+1prior  |

    ## Channel Scores

    """) + "\n".join(channel_detail)

    if mm_sections:
        md += "\n## Multi-Messenger Data\n\n" + "\n".join(mm_sections)

    if bt_section:
        md += bt_section

    md += textwrap.dedent("""
    ---

    ## Assessment

    This candidate was flagged by the EXODUS multi-modal convergence scoring
    system. The combination of active channels and multi-messenger data
    warrants further investigation. Natural explanations should be
    systematically ruled out before considering technosignature hypotheses.

    ---
    *Generated by Project EXODUS Reporting Pipeline v1.0*
    """)

    return md


# =====================================================================
#  Data provenance helper
# =====================================================================

def _generate_provenance_section(ctx: IterationContext) -> str:
    """Generate a data-provenance disclaimer from actual run metadata.

    Inspects which channels had real vs simulated data across all scored
    targets, and lists limitations accordingly.
    """
    lines = []

    # Collect channel data sources from scores
    channel_sources: Dict[str, set] = {}
    for s in ctx.scores:
        channels = s.get("channel_scores", {})
        for ch_name, ch_data in channels.items():
            if isinstance(ch_data, dict):
                src = ch_data.get("data_source") or ch_data.get("details", {}).get("data_source")
                if src:
                    channel_sources.setdefault(ch_name, set()).add(src)

    # Determine which channels were evaluated
    all_channels = {
        "ir_excess", "transit_anomaly", "radio_anomaly",
        "gaia_photometric_anomaly", "habitable_zone_planet",
        "proper_motion_anomaly", "ir_variability", "uv_anomaly",
        "radio_emission", "hr_anomaly", "abundance_anomaly",
    }
    evaluated = set(channel_sources.keys())
    not_evaluated = all_channels - evaluated

    # Report
    lines.append("**Note:** The dataset and channel lists above describe "
                 "the full EXODUS architecture. In the current run:")
    lines.append("")

    if evaluated:
        for ch in sorted(evaluated):
            sources = channel_sources[ch]
            if sources == {"simulated"}:
                lines.append(f"- **{ch}**: SIMULATED data only — results "
                             "are illustrative, not publication-grade.")
            elif "simulated" in sources:
                lines.append(f"- **{ch}**: mixed real/simulated data — "
                             "provenance varies by target.")
            else:
                lines.append(f"- **{ch}**: real observational data.")

    if not_evaluated:
        lines.append(f"- **Not evaluated in this run:** "
                     f"{', '.join(sorted(not_evaluated))}")

    # Calibrated channels
    cal = ctx.config.get("channels_calibrated", [])
    if cal:
        lines.append(f"\nFDR significance is computed only for calibrated "
                     f"channels ({', '.join(cal)}). Uncalibrated channels "
                     "are excluded from the Fisher combination.")

    return "\n    ".join(lines)


# =====================================================================
#  Methodology paper draft (Markdown)
# =====================================================================

def generate_methodology_draft(ctx: IterationContext) -> str:
    """Generate an auto-updating methodology paper draft."""

    n_scores = len(ctx.scores)
    total_scores = [s.get("total_score", 0) for s in ctx.scores]
    n_active_3plus = sum(
        1 for s in ctx.scores
        if s.get("n_active_channels", 0) >= 3
    )
    n_breakthroughs = len(ctx.breakthroughs)

    # Dataset list — ALL potentially available datasets.
    # The Limitations section below clarifies which were actually active.
    datasets = [
        ("Gaia DR3", "Astrometry, photometry, epoch variability"),
        ("2MASS + AllWISE", "Near/mid-infrared photometry"),
        ("NEOWISE Single Exposure", "Time-series infrared photometry (W1, W2)"),
        ("Kepler / TESS", "Optical light curves, transit detection"),
        ("VLASS / NVSS", "Radio continuum imaging at 3 GHz / 1.4 GHz"),
        ("Fermi 4FGL-DR4", "Gamma-ray source catalog"),
        ("IceCube", "High-energy neutrino events"),
        ("GWTC", "Gravitational wave transient catalog"),
        ("NANOGrav 15-year", "Pulsar timing residuals"),
        ("CHIME/FRB", "Fast Radio Burst catalog"),
        ("NASA Exoplanet Archive", "Confirmed exoplanets and host properties"),
    ]

    dataset_table = "\n".join(
        f"| {name:<30s} | {desc:<55s} |" for name, desc in datasets
    )

    # Scoring channels
    channels = [
        ("ir_excess", "Mid-IR excess above photospheric model", "Dyson sphere waste heat"),
        ("transit_anomaly", "BLS + asymmetry + variability", "Megastructure transits"),
        ("radio_anomaly", "Narrowband + Doppler drift search", "Artificial radio emission"),
        ("gaia_photometric_anomaly", "Epoch photometry scatter", "Irregular variability"),
        ("habitable_zone_planet", "HZ planet confirmation", "Habitability prerequisite"),
        ("proper_motion_anomaly", "RUWE / astrometric excess", "Unseen massive companion"),
        ("ir_variability", "NEOWISE 10yr secular trend + scatter", "Construction / variable dust"),
        ("uv_anomaly", "GALEX FUV/NUV excess or deficit", "Accretion / chromospheric activity"),
        ("radio_emission", "VLASS/NVSS continuum detection", "Non-thermal radio source"),
        ("hr_anomaly", "HR diagram outlier (BP-RP vs M_G)", "Anomalous luminosity / composite"),
        ("abundance_anomaly", "Chemical abundance outliers", "Unusual elemental ratios"),
    ]
    channel_table = "\n".join(
        f"| {n:<28s} | {m:<40s} | {s:<30s} |" for n, m, s in channels
    )

    md = textwrap.dedent(f"""\
    # Project EXODUS: An AI-Directed Multi-Messenger Technosignature Search

    **Auto-generated methodology draft — Iteration {ctx.iteration}**
    **Last updated:** {ctx.timestamp}

    ---

    ## Abstract

    We present Project EXODUS (EXoplanet Observations Detecting Unusual Signals),
    an automated, AI-directed system for the systematic search of technosignatures
    across multiple observational channels. EXODUS combines data from {len(datasets)}
    independent astronomical datasets — spanning radio, infrared, optical, gamma-ray,
    neutrino, and gravitational wave observations — to identify stellar targets
    exhibiting anomalous behaviour in multiple channels simultaneously. The system
    employs a novel multi-modal convergence score that rewards targets flagged
    independently by multiple detectors, an imagination engine that generates
    non-anthropocentric hypotheses, and a self-improving evolutionary architecture.
    After {ctx.iteration} iteration(s), EXODUS has scored {n_scores} targets,
    identified {n_active_3plus} with 3+ active anomaly channels, and escalated
    {n_breakthroughs} to its Breakthrough Engine for detailed analysis.
    {"Even null results constrain the prevalence of detectable technosignatures " +
     "in the solar neighbourhood." if n_breakthroughs == 0 else ""}

    ## 1. Introduction

    The search for extraterrestrial intelligence (SETI) has traditionally focused
    on narrow observational channels — primarily targeted radio surveys and, more
    recently, optical laser searches. However, a sufficiently advanced civilisation
    might be detectable across multiple wavelengths simultaneously: waste heat in
    the infrared, transit anomalies in optical photometry, artificial radio
    emissions, and potentially even neutrino or gravitational wave signatures.

    EXODUS adopts a multi-messenger approach to technosignature detection, drawing
    on the principle that natural astrophysical phenomena rarely produce correlated
    anomalies across many independent channels, while engineered structures or
    activities might.

    ## 2. Datasets

    | Dataset                        | Description                                             |
    |--------------------------------|---------------------------------------------------------|
    {dataset_table}

    ## 3. Methods

    ### 3.1 Scoring Channels

    Each target is evaluated across eleven independent anomaly detection channels:

    | Channel                      | Method                                   | Signature                      |
    |------------------------------|------------------------------------------|--------------------------------|
    {channel_table}

    ### 3.2 EXODUS Score

    The EXODUS score combines active channels using a geometric mean with an
    exponential convergence bonus:

    ```
    geo_mean = exp( mean( log(active_scores) ) )
    convergence_bonus = 2^(n_active - 1)
    EXODUS_score = geo_mean × convergence_bonus
    ```

    This formulation superlinearly rewards multi-channel convergence: a target
    with three moderately anomalous channels outranks one with a single highly
    anomalous channel.

    ### 3.3 Breakthrough Engine

    Targets exceeding configurable EXODUS score thresholds are escalated through
    four verification levels:
    - **Level 1:** Automated data quality checks
    - **Level 2:** Cross-catalogue consistency verification
    - **Level 3:** Detailed multi-wavelength analysis
    - **Level 4:** Publication-grade candidate report

    ### 3.4 Multi-Messenger Extensions

    Beyond the core detection channels, EXODUS cross-matches targets against:
    - Fermi 4FGL-DR4 unidentified gamma-ray sources
    - IceCube high-energy neutrino events
    - GWTC gravitational wave sky localisations
    - NANOGrav pulsar timing residuals (Shapiro delay search)
    - CHIME/FRB repeating FRB orbital period correlations
    - NEOWISE time-series infrared photometry (cross-band temporal correlation)

    ### 3.4.1 Data Provenance & Limitations

    {_generate_provenance_section(ctx)}

    ### 3.5 Self-Improvement

    The Evolver module analyses detection patterns after each iteration and
    adjusts thresholds, weights, and search strategies. The Self-Diagnosis
    Monitor runs seven automated checks for discovery-blocking failure modes.

    ### 3.6 Imagination Engine

    A bank of non-anthropocentric hypotheses generates novel observable
    predictions that conventional SETI would miss. Each hypothesis produces
    specific, testable predictions evaluated against available data.

    ## 4. Results

    After {ctx.iteration} iteration(s):

    - **Targets scored:** {n_scores}
    - **Targets with 3+ active channels:** {n_active_3plus}
    - **Breakthrough escalations:** {n_breakthroughs}
    - **Score range:** {f"{min(total_scores):.4f} – {max(total_scores):.4f} (median {float(np.median(total_scores)):.4f})" if total_scores else "(no scores)"}

    ## 5. Discussion

    {"The current results represent a null detection, constraining the prevalence " +
     "of multi-channel technosignatures among the surveyed targets. This null " +
     "result is itself valuable: it sets upper limits on the fraction of nearby " +
     "stars hosting detectable megastructures or artificial emissions at the " +
     "sensitivity of current surveys." if n_breakthroughs == 0 else
     f"EXODUS has identified {n_breakthroughs} candidate(s) warranting detailed " +
     "follow-up. Natural explanations must be systematically evaluated before " +
     "any technosignature claim."}

    ## 6. Conclusions

    Project EXODUS demonstrates the feasibility of automated, multi-messenger
    technosignature surveys using publicly available astronomical data. The
    system's self-improving architecture and non-anthropocentric hypothesis
    generation represent novel contributions to SETI methodology.

    ---

    ## References

    - Gaia Collaboration (2022). Gaia DR3. A&A.
    - Wright et al. (2016). The Ĝ Search for Extraterrestrial Civilizations
      with Large Energy Supplies. ApJ.
    - Price et al. (2020). The Breakthrough Listen Search. AJ.
    - CHIME/FRB Collaboration (2021). The First CHIME/FRB Catalog. ApJS.
    - Fermi-LAT Collaboration (2022). 4FGL-DR4 Catalog.
    - IceCube Collaboration (2023). 10-year point source catalog.
    - LIGO/Virgo/KAGRA (2023). GWTC-3. Phys. Rev. X.
    - NANOGrav Collaboration (2023). The 15-year data set. ApJL.

    ---
    *Auto-generated by Project EXODUS Reporting Pipeline v1.0*
    """)

    return md


# =====================================================================
#  Dashboard HTML
# =====================================================================

def generate_dashboard_html(ctx: IterationContext) -> str:
    """Generate an interactive HTML dashboard."""

    scores = ctx.scores
    total_scores = [s.get("total_score", 0) for s in scores]
    ranked = sorted(scores, key=lambda s: s.get("total_score", 0), reverse=True)

    # Top targets for display
    top_20 = ranked[:20]

    # Build target rows
    target_rows_html = []
    for i, s in enumerate(top_20, 1):
        tid = s.get("target_id", "?")
        ts = s.get("total_score", 0)
        n_active = s.get("n_active_channels", 0)
        ra = s.get("ra", 0)
        dec = s.get("dec", 0)
        channels = s.get("channel_scores", {})
        active_names = [
            ch for ch, cs in channels.items()
            if (cs.get("is_active") if isinstance(cs, dict) else getattr(cs, "is_active", False))
        ]
        max_score = max(total_scores) if total_scores else 0.001
        bar_w = min(int(ts / max(max_score, 0.001) * 200), 200)

        target_rows_html.append(f"""
        <tr>
          <td class="rank">{i}</td>
          <td class="target-id">{_html_esc(tid)}</td>
          <td class="coords">{ra:.4f}, {dec:.4f}</td>
          <td class="score">
            <div class="score-bar" style="width:{bar_w}px">{ts:.3f}</div>
          </td>
          <td class="active">{n_active}/10+1prior</td>
          <td class="channels">{', '.join(active_names)}</td>
        </tr>""")

    # Stats
    n_total = len(scores)
    n_3plus = sum(1 for s in scores if s.get("n_active_channels", 0) >= 3)
    n_bt = len(ctx.breakthroughs)
    median_score = float(np.median(total_scores)) if total_scores else 0

    # Diagnosis checks for sidebar
    diag_html = []
    for ch in ctx.diagnosis.get("checks", []):
        name = ch.get("name", "?")
        status = ch.get("status", "?")
        color = {"GREEN": "#00e676", "YELLOW": "#ffd600", "RED": "#ff1744"}.get(status, "#999")
        diag_html.append(
            f'<div class="diag-item" style="border-left:3px solid {color}">'
            f'<span class="diag-status" style="color:{color}">{status}</span> '
            f'{_html_esc(name)}</div>'
        )

    # Hypothesis status
    hyp_html = []
    for h in ctx.hypotheses:
        hid = h.get("id", h.get("hypothesis_id", "?"))
        status = h.get("status", "untested")
        hyp_html.append(
            f'<div class="hyp-item"><code>{_html_esc(hid)}</code>: {_html_esc(status)}</div>'
        )

    # Evolver log
    evo_html = []
    for e in ctx.evolver_log:
        desc = e.get("description", e.get("change", "?"))
        evo_html.append(f'<div class="evo-item">{_html_esc(desc)}</div>')

    # Sky map data points (JSON for JS)
    sky_data = json.dumps([
        {"ra": s.get("ra", 0), "dec": s.get("dec", 0),
         "score": s.get("total_score", 0),
         "id": s.get("target_id", "?")}
        for s in ranked[:200]
    ])

    html = textwrap.dedent(f"""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project EXODUS Dashboard &mdash; Iteration {ctx.iteration}</title>
    <style>

    /* ── Reset & base ─────────────────────────────────────────────── */
    *, *::before, *::after {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}

    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     Oxygen, Ubuntu, Cantarell, "Fira Sans", "Droid Sans",
                     "Helvetica Neue", sans-serif;
        background: #0a0a1a;
        color: #e0e0e0;
        line-height: 1.6;
        min-height: 100vh;
    }}

    /* ── Header ───────────────────────────────────────────────────── */
    .page-header {{
        background: linear-gradient(135deg, #1a1a2e 0%, #0a0a1a 100%);
        border-bottom: 1px solid #2a2a4e;
        padding: 1.5rem 2rem;
    }}

    .header-content {{
        max-width: 1400px;
        margin: 0 auto;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }}

    .logo {{
        display: flex;
        flex-direction: column;
    }}

    .logo-exodus {{
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 0.3em;
        background: linear-gradient(90deg, #7c4dff, #00e5ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .logo-sub {{
        font-size: 0.8rem;
        color: #888;
        letter-spacing: 0.15em;
    }}

    .iter-badge {{
        background: #1a1a3e;
        border: 1px solid #3a3a6e;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-size: 0.9rem;
    }}

    .iter-badge strong {{
        color: #7c4dff;
    }}

    /* ── Layout ───────────────────────────────────────────────────── */
    .main-grid {{
        max-width: 1400px;
        margin: 2rem auto;
        padding: 0 2rem;
        display: grid;
        grid-template-columns: 1fr 320px;
        gap: 2rem;
    }}

    /* ── Cards ────────────────────────────────────────────────────── */
    .card {{
        background: #12122a;
        border: 1px solid #2a2a4e;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }}

    .card h2 {{
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #b0b0d0;
        border-bottom: 1px solid #2a2a4e;
        padding-bottom: 0.5rem;
    }}

    /* ── Stats row ────────────────────────────────────────────────── */
    .stats-row {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }}

    .stat-card {{
        background: #1a1a3e;
        border: 1px solid #2a2a5e;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }}

    .stat-value {{
        font-size: 2rem;
        font-weight: 700;
        color: #7c4dff;
    }}

    .stat-label {{
        font-size: 0.8rem;
        color: #888;
        margin-top: 0.3rem;
    }}

    /* ── Table ────────────────────────────────────────────────────── */
    .target-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }}

    .target-table th {{
        text-align: left;
        padding: 0.6rem 0.8rem;
        border-bottom: 2px solid #3a3a6e;
        color: #a0a0c0;
        font-weight: 600;
    }}

    .target-table td {{
        padding: 0.5rem 0.8rem;
        border-bottom: 1px solid #1e1e3e;
    }}

    .target-table tr:hover {{
        background: #1a1a3e;
    }}

    .rank {{ color: #666; text-align: center; }}
    .target-id {{ font-family: monospace; color: #00e5ff; }}
    .coords {{ color: #666; font-size: 0.8rem; }}
    .active {{ text-align: center; font-weight: 600; }}
    .channels {{ color: #888; font-size: 0.8rem; }}

    .score-bar {{
        background: linear-gradient(90deg, #7c4dff, #00e5ff);
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        min-width: 50px;
    }}

    /* ── Sky map ──────────────────────────────────────────────────── */
    .sky-map {{
        width: 100%;
        height: 300px;
        background: #0d0d20;
        border-radius: 8px;
        border: 1px solid #2a2a4e;
        position: relative;
        overflow: hidden;
    }}

    .sky-dot {{
        position: absolute;
        border-radius: 50%;
        cursor: pointer;
    }}

    .sky-dot:hover {{
        outline: 2px solid white;
    }}

    .sky-grid {{
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        pointer-events: none;
    }}

    /* ── Sidebar ──────────────────────────────────────────────────── */
    .sidebar .card {{ font-size: 0.85rem; }}

    .diag-item {{
        padding: 0.4rem 0.6rem;
        margin-bottom: 0.4rem;
        background: #1a1a3e;
        border-radius: 4px;
    }}

    .diag-status {{ font-weight: 700; font-size: 0.75rem; }}

    .hyp-item, .evo-item {{
        padding: 0.3rem 0;
        border-bottom: 1px solid #1e1e3e;
        font-size: 0.82rem;
    }}

    .hyp-item code {{
        color: #00e5ff;
        font-size: 0.78rem;
    }}

    /* ── Footer ───────────────────────────────────────────────────── */
    .page-footer {{
        text-align: center;
        padding: 2rem;
        color: #555;
        font-size: 0.8rem;
        border-top: 1px solid #1a1a3e;
        margin-top: 2rem;
    }}

    </style>
    </head>
    <body>

    <header class="page-header">
      <div class="header-content">
        <div class="logo">
          <span class="logo-exodus">EXODUS</span>
          <span class="logo-sub">EXoplanet Observations Detecting Unusual Signals</span>
        </div>
        <div class="iter-badge">
          Iteration <strong>{ctx.iteration}</strong> &mdash; {ctx.timestamp}
        </div>
      </div>
    </header>

    <div class="main-grid">
      <div class="content">

        <!-- Stats -->
        <div class="stats-row">
          <div class="stat-card">
            <div class="stat-value">{n_total}</div>
            <div class="stat-label">Targets Scored</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{n_3plus}</div>
            <div class="stat-label">3+ Active Channels</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{n_bt}</div>
            <div class="stat-label">Breakthrough Escalations</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{median_score:.3f}</div>
            <div class="stat-label">Median Score</div>
          </div>
        </div>

        <!-- Sky Map -->
        <div class="card">
          <h2>Sky Map &mdash; Top Targets</h2>
          <div class="sky-map" id="skyMap"></div>
        </div>

        <!-- Top Targets Table -->
        <div class="card">
          <h2>Top 20 Targets</h2>
          <table class="target-table">
            <thead>
              <tr>
                <th>#</th><th>Target</th><th>Coords</th>
                <th>Score</th><th>Active</th><th>Channels</th>
              </tr>
            </thead>
            <tbody>
              {"".join(target_rows_html)}
            </tbody>
          </table>
        </div>

      </div>

      <!-- Sidebar -->
      <div class="sidebar">

        <div class="card">
          <h2>Self-Diagnosis</h2>
          {"".join(diag_html) if diag_html else "<p style='color:#666'>No diagnosis data</p>"}
        </div>

        <div class="card">
          <h2>Hypotheses</h2>
          {"".join(hyp_html) if hyp_html else "<p style='color:#666'>No hypotheses tested</p>"}
        </div>

        <div class="card">
          <h2>Evolution Log</h2>
          {"".join(evo_html) if evo_html else "<p style='color:#666'>No changes this iteration</p>"}
        </div>

      </div>
    </div>

    <footer class="page-footer">
      Project EXODUS Reporting Pipeline v1.0 &mdash; Generated {ctx.timestamp}
    </footer>

    <script>
    // ── Sky map rendering ──────────────────────────────────────────
    (function() {{
      var data = {sky_data};
      var map = document.getElementById('skyMap');
      var w = map.clientWidth;
      var h = map.clientHeight;

      // Draw grid lines
      var svg = '<svg class="sky-grid" viewBox="0 0 ' + w + ' ' + h + '">';
      svg += '<line x1="' + w/2 + '" y1="0" x2="' + w/2 + '" y2="' + h + '" stroke="#1a1a3e" />';
      svg += '<line x1="0" y1="' + h/2 + '" x2="' + w + '" y2="' + h/2 + '" stroke="#1a1a3e" />';
      for (var ra=0; ra<=360; ra+=60) {{
        var x = ra / 360 * w;
        svg += '<line x1="' + x + '" y1="0" x2="' + x + '" y2="' + h + '" stroke="#111133" stroke-dasharray="4" />';
      }}
      for (var dec=-60; dec<=60; dec+=30) {{
        var y = (90 - dec) / 180 * h;
        svg += '<line x1="0" y1="' + y + '" x2="' + w + '" y2="' + y + '" stroke="#111133" stroke-dasharray="4" />';
      }}
      svg += '</svg>';
      map.innerHTML = svg;

      // Find score range
      var maxScore = 0.001;
      data.forEach(function(d) {{ if (d.score > maxScore) maxScore = d.score; }});

      // Place dots
      data.forEach(function(d) {{
        var x = d.ra / 360 * w;
        var y = (90 - d.dec) / 180 * h;
        var r = Math.max(3, Math.min(10, d.score / maxScore * 10));
        var alpha = Math.max(0.3, d.score / maxScore);
        var dot = document.createElement('div');
        dot.className = 'sky-dot';
        dot.style.left = (x - r) + 'px';
        dot.style.top = (y - r) + 'px';
        dot.style.width = (r * 2) + 'px';
        dot.style.height = (r * 2) + 'px';
        dot.style.background = 'rgba(124,77,255,' + alpha + ')';
        dot.title = d.id + ' (' + d.score.toFixed(3) + ')';
        map.appendChild(dot);
      }});
    }})();
    </script>

    </body>
    </html>
    """)

    return html


def _html_esc(text: str) -> str:
    """Minimal HTML escaping."""
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


# =====================================================================
#  Public API — Generate full report bundle
# =====================================================================

def generate_full_report(ctx: IterationContext) -> ReportBundle:
    """Generate all report artefacts for one iteration.

    Parameters
    ----------
    ctx : IterationContext
        All data needed for report generation.

    Returns
    -------
    ReportBundle
        Paths to all generated artefact files.
    """
    log.info("Generating full report for iteration %d", ctx.iteration)
    bundle = ReportBundle()
    iteration_tag = f"iter{ctx.iteration:04d}"

    # 1. Dashboard HTML
    html = generate_dashboard_html(ctx)
    dashboard_path = REPORTS_DIR / f"exodus_dashboard_{iteration_tag}.html"
    dashboard_path.write_text(html, encoding="utf-8")
    bundle.dashboard_path = str(dashboard_path)
    log.info("Dashboard: %s", dashboard_path)

    # 2. Iteration Summary
    summary = generate_iteration_summary(ctx)
    summary_path = REPORTS_DIR / f"exodus_summary_{iteration_tag}.md"
    summary_path.write_text(summary, encoding="utf-8")
    bundle.summary_path = str(summary_path)
    log.info("Summary: %s", summary_path)

    # 3. Methodology Draft
    methodology = generate_methodology_draft(ctx)
    methodology_path = REPORTS_DIR / f"exodus_methodology_{iteration_tag}.md"
    methodology_path.write_text(methodology, encoding="utf-8")
    bundle.methodology_path = str(methodology_path)
    log.info("Methodology: %s", methodology_path)

    # 4. Candidate Reports (Breakthrough Level 3+)
    for bt in ctx.breakthroughs:
        level = bt.get("level", 0)
        if level >= 3:
            target_id = bt.get("target_id", bt.get("target", "unknown"))
            # Find the score for this target
            target_score = next(
                (s for s in ctx.scores if s.get("target_id") == target_id),
                {"target_id": target_id, "total_score": 0,
                 "n_active_channels": 0, "channel_scores": {},
                 "ra": 0, "dec": 0},
            )
            report = generate_candidate_report(target_id, target_score, bt)
            cand_path = REPORTS_DIR / f"exodus_candidate_{target_id}_{iteration_tag}.md"
            cand_path.write_text(report, encoding="utf-8")
            bundle.candidate_paths.append(str(cand_path))
            log.info("Candidate report: %s", cand_path)

    log.info(
        "Report generation complete: %d artefacts",
        3 + len(bundle.candidate_paths),
    )
    return bundle


# =====================================================================
#  CLI demo
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Project EXODUS -- Report Generator Demo")
    print("=" * 70)

    rng = np.random.default_rng(seed=42)

    # Generate simulated scores
    n_targets = 100
    sim_scores = []
    for i in range(n_targets):
        ra = rng.uniform(0, 360)
        dec = np.degrees(np.arcsin(rng.uniform(-1, 1)))
        n_active = int(rng.choice([0, 1, 1, 1, 2, 2, 3, 4], p=[0.1, 0.3, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05]))
        channels = {}
        for ch_name in ["ir_excess", "transit_anomaly", "radio_anomaly",
                        "gaia_photometric_anomaly", "habitable_zone_planet",
                        "proper_motion_anomaly"]:
            sc = float(rng.uniform(0.3, 0.95)) if n_active > 0 else float(rng.uniform(0, 0.2))
            n_active -= 1 if sc > 0.3 else 0
            channels[ch_name] = {
                "channel_name": ch_name,
                "score": sc,
                "is_active": sc > 0.3,
                "details": {},
            }
        active_count = sum(1 for c in channels.values() if c["is_active"])
        active_scores = [c["score"] for c in channels.values() if c["is_active"]]
        geo = float(np.exp(np.mean(np.log(active_scores)))) if active_scores else 0
        bonus = 2 ** max(active_count - 1, 0)
        total = geo * bonus

        sim_scores.append({
            "target_id": f"Gaia_DR3_{1000000000 + i}",
            "ra": float(ra),
            "dec": float(dec),
            "total_score": total,
            "n_active_channels": active_count,
            "channel_scores": channels,
            "convergence_bonus": bonus,
            "geo_mean": geo,
        })

    # Simulated breakthrough
    sim_breakthroughs = [
        {"target_id": sim_scores[0]["target_id"], "level": 3,
         "reason": "3+ active channels with high convergence"},
    ]

    # Simulated diagnosis
    sim_diagnosis = {
        "checks": [
            {"name": "Threshold sensitivity", "status": "GREEN"},
            {"name": "RFI over-correction", "status": "GREEN"},
            {"name": "Anthropocentric bias", "status": "YELLOW"},
            {"name": "Temporal resolution", "status": "GREEN"},
            {"name": "Catalog completeness", "status": "GREEN"},
            {"name": "Frequency coverage", "status": "GREEN"},
            {"name": "Slow changes", "status": "GREEN"},
        ]
    }

    ctx = IterationContext(
        iteration=1,
        n_targets_scored=n_targets,
        scores=sim_scores,
        breakthroughs=sim_breakthroughs,
        hypotheses=[
            {"id": "NAH001", "status": "tested — null result"},
            {"id": "NAH002", "status": "pending"},
            {"id": "NAH003", "status": "tested — interesting"},
        ],
        evolver_log=[
            {"description": "Raised ir_excess threshold 0.30 -> 0.35 (too many false positives)"},
            {"description": "Lowered radio_anomaly minimum SNR 5.0 -> 4.0"},
        ],
        diagnosis=sim_diagnosis,
        imagination_results=[],
        data_coverage={
            sim_scores[0]["target_id"]: ["Gaia", "WISE", "NEOWISE", "TESS"],
            sim_scores[1]["target_id"]: ["Gaia", "WISE"],
            sim_scores[2]["target_id"]: ["Gaia", "WISE", "VLASS", "Fermi"],
        },
    )

    bundle = generate_full_report(ctx)

    print(f"\n  Generated artefacts:")
    print(f"    Dashboard:   {bundle.dashboard_path}")
    print(f"    Summary:     {bundle.summary_path}")
    print(f"    Methodology: {bundle.methodology_path}")
    for cp in bundle.candidate_paths:
        print(f"    Candidate:   {cp}")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
