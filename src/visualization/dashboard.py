"""
Dashboard report generator for Project EXODUS.

Produces a self-contained HTML report with:
- Sky map of all targets, colored by EXODUS score (Aitoff projection)
- Top 20 targets ranked by score with channel breakdown
- Hypothesis status board with counts and color coding
- Evolution log showing threshold changes and strategy adjustments
- Breakthrough candidates with escalation level details
- Summary statistics panel

The HTML is fully self-contained: CSS is inline and matplotlib figures
are base64-encoded PNGs embedded directly in ``<img>`` tags.
"""

from __future__ import annotations

import base64
import io
import sys
import webbrowser
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- must precede pyplot import
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import get_logger, get_config, PROJECT_ROOT

log = get_logger("visualization.dashboard")

# ── Design constants ─────────────────────────────────────────────────
_BG_DARK = "#0a0a1a"
_BG_PANEL = "#1a1a2e"
_BG_CARD = "#16213e"
_ACCENT_CYAN = "#00d4ff"
_ACCENT_AMBER = "#ffa500"
_ACCENT_GREEN = "#00ff88"
_ACCENT_RED = "#ff4444"
_ACCENT_PURPLE = "#b088f9"
_TEXT_PRIMARY = "#e0e0e0"
_TEXT_SECONDARY = "#a0a0b0"
_BORDER_SUBTLE = "#2a2a4e"

# Hypothesis status -> color mapping
_STATUS_COLORS = {
    "pending":   _ACCENT_AMBER,
    "tested":    _ACCENT_CYAN,
    "confirmed": _ACCENT_GREEN,
    "rejected":  _ACCENT_RED,
}

# Channel display names for readability
_CHANNEL_DISPLAY = {
    "ir_excess":                "IR Excess",
    "transit_anomaly":          "Transit Anomaly",
    "radio_anomaly":            "Radio Anomaly",
    "gaia_photometric_anomaly": "Gaia Photometric",
    "habitable_zone_planet":    "Habitable Zone",
    "proper_motion_anomaly":    "Proper Motion",
}


# =====================================================================
#  Dashboard Generator
# =====================================================================

class DashboardGenerator:
    """Generate an HTML dashboard report for EXODUS results.

    Parameters
    ----------
    output_dir : str or Path or None
        Directory where reports are written.  Defaults to
        ``<PROJECT_ROOT>/data/reports/``.
    """

    def __init__(self, output_dir: Optional[str] = None):
        if output_dir is None:
            self.output_dir = PROJECT_ROOT / "data" / "reports"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info("DashboardGenerator initialised  output_dir=%s", self.output_dir)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def generate(self, results_data: Dict[str, Any]) -> Path:
        """Create the full HTML dashboard and write it to disk.

        Parameters
        ----------
        results_data : dict
            Expected keys:

            - ``scores``  : list of EXODUSScore-like dicts
            - ``hypotheses`` : list of hypothesis dicts
            - ``evolution_log`` : list of evolution record dicts
            - ``breakthrough_candidates`` : list of breakthrough dicts
            - ``iteration`` : int
            - ``timestamp`` : str (ISO-8601)

        Returns
        -------
        pathlib.Path
            Absolute path to the generated HTML file.
        """
        scores = results_data.get("scores", [])
        hypotheses = results_data.get("hypotheses", [])
        evolution_log = results_data.get("evolution_log", [])
        breakthroughs = results_data.get("breakthrough_candidates", [])
        iteration = results_data.get("iteration", 0)
        timestamp = results_data.get("timestamp", datetime.now(timezone.utc).isoformat())

        log.info(
            "Generating dashboard  iteration=%d  scores=%d  hypotheses=%d  "
            "evolution_records=%d  breakthroughs=%d",
            iteration, len(scores), len(hypotheses),
            len(evolution_log), len(breakthroughs),
        )

        # Build each section
        summary_html = self._generate_summary_stats(results_data)
        sky_map_html = self._generate_sky_map(scores)
        score_table_html = self._generate_score_table(scores)
        hypothesis_html = self._generate_hypothesis_board(hypotheses)
        evolution_html = self._generate_evolution_log(evolution_log)
        breakthrough_html = self._generate_breakthrough_section(breakthroughs)

        # Assemble the full page
        html = self._assemble_page(
            iteration=iteration,
            timestamp=timestamp,
            summary_html=summary_html,
            sky_map_html=sky_map_html,
            score_table_html=score_table_html,
            hypothesis_html=hypothesis_html,
            evolution_html=evolution_html,
            breakthrough_html=breakthrough_html,
        )

        # Write to disk
        filename = f"exodus_dashboard_iter{iteration:04d}.html"
        filepath = self.output_dir / filename
        filepath.write_text(html, encoding="utf-8")

        log.info("Dashboard written to %s", filepath)
        return filepath

    # ------------------------------------------------------------------
    #  Sky map
    # ------------------------------------------------------------------

    def _generate_sky_map(self, scores: List[Dict[str, Any]]) -> str:
        """Create an Aitoff projection sky map colored by EXODUS score.

        Parameters
        ----------
        scores : list of dict
            Each dict must have ``ra``, ``dec``, and ``total_score``.

        Returns
        -------
        str
            HTML ``<img>`` tag with the figure as a base64-encoded PNG.
        """
        if not scores:
            return '<p style="color:{};text-align:center;">No scored targets to map.</p>'.format(
                _TEXT_SECONDARY
            )

        ra_vals = []
        dec_vals = []
        score_vals = []

        for s in scores:
            try:
                ra = float(s.get("ra", 0))
                dec = float(s.get("dec", 0))
                sc = float(s.get("total_score", 0))
            except (TypeError, ValueError):
                continue
            ra_vals.append(ra)
            dec_vals.append(dec)
            score_vals.append(sc)

        if not ra_vals:
            return '<p style="color:{};text-align:center;">No valid coordinates in scores.</p>'.format(
                _TEXT_SECONDARY
            )

        ra_arr = np.array(ra_vals)
        dec_arr = np.array(dec_vals)
        score_arr = np.array(score_vals)

        # Convert RA/Dec to radians for Aitoff projection.
        # Aitoff expects longitude in [-pi, pi] and latitude in [-pi/2, pi/2].
        ra_rad = np.deg2rad(ra_arr)
        ra_rad = np.where(ra_rad > np.pi, ra_rad - 2 * np.pi, ra_rad)
        dec_rad = np.deg2rad(dec_arr)

        # Create the figure
        fig = plt.figure(figsize=(12, 6), facecolor=_BG_DARK)
        ax = fig.add_subplot(111, projection="aitoff")
        ax.set_facecolor(_BG_DARK)

        # Custom colormap: dark blue -> cyan -> amber for score
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "exodus_score",
            ["#0a0a3a", "#005577", _ACCENT_CYAN, _ACCENT_AMBER, _ACCENT_RED],
        )

        # Determine point sizes proportional to score
        sizes = 8 + score_arr * 80  # range: 8..88 pts

        scatter = ax.scatter(
            ra_rad, dec_rad,
            c=score_arr,
            cmap=cmap,
            s=sizes,
            alpha=0.85,
            edgecolors="none",
            zorder=5,
        )

        # Colorbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.08)
        cbar.set_label("EXODUS Score", color=_TEXT_PRIMARY, fontsize=10)
        cbar.ax.yaxis.set_tick_params(color=_TEXT_PRIMARY)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_TEXT_PRIMARY, fontsize=8)

        # Style the grid and labels
        ax.grid(True, color=_BORDER_SUBTLE, alpha=0.4, linewidth=0.5)
        ax.tick_params(colors=_TEXT_SECONDARY, labelsize=8)
        ax.set_title(
            "EXODUS Target Sky Map",
            color=_ACCENT_CYAN,
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Render to base64 PNG
        buf = io.BytesIO()
        fig.savefig(
            buf, format="png", dpi=150,
            bbox_inches="tight", facecolor=_BG_DARK,
            edgecolor="none",
        )
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")

        return (
            '<div style="text-align:center;">'
            '<img src="data:image/png;base64,{}" '
            'style="max-width:100%;border-radius:8px;border:1px solid {};" '
            'alt="EXODUS Sky Map"/>'
            '</div>'
        ).format(b64, _BORDER_SUBTLE)

    # ------------------------------------------------------------------
    #  Top-20 score table
    # ------------------------------------------------------------------

    def _generate_score_table(self, scores: List[Dict[str, Any]]) -> str:
        """Generate an HTML table of the top 20 targets by EXODUS score.

        Parameters
        ----------
        scores : list of dict
            EXODUSScore-like dicts with ``target_id``, ``total_score``,
            ``n_active_channels``, ``geo_mean``, ``convergence_bonus``,
            ``channel_scores``, ``rank``.

        Returns
        -------
        str
            Complete HTML table markup.
        """
        if not scores:
            return '<p style="color:{};">No scored targets available.</p>'.format(
                _TEXT_SECONDARY
            )

        # Sort by total_score descending, take top 20
        sorted_scores = sorted(
            scores,
            key=lambda s: float(s.get("total_score", 0)),
            reverse=True,
        )[:20]

        channel_names = [
            "ir_excess", "transit_anomaly", "radio_anomaly",
            "gaia_photometric_anomaly", "habitable_zone_planet",
            "proper_motion_anomaly",
        ]

        # Build table header
        header_cells = (
            "<th>Rank</th>"
            "<th>Target ID</th>"
            "<th>EXODUS Score</th>"
            "<th>Active</th>"
            "<th>Geo Mean</th>"
            "<th>Bonus</th>"
        )
        for ch in channel_names:
            display = _CHANNEL_DISPLAY.get(ch, ch)
            header_cells += f"<th>{escape(display)}</th>"

        rows = []
        for idx, s in enumerate(sorted_scores, start=1):
            rank = s.get("rank", idx)
            target_id = escape(str(s.get("target_id", "?")))
            total = float(s.get("total_score", 0))
            n_active = int(s.get("n_active_channels", 0))
            geo_mean = float(s.get("geo_mean", 0))
            bonus = float(s.get("convergence_bonus", 0))

            # Score color intensity
            score_color = _ACCENT_GREEN if total > 2.0 else (
                _ACCENT_CYAN if total > 0.5 else _TEXT_SECONDARY
            )

            row = (
                f'<td style="text-align:center;">{rank}</td>'
                f'<td style="font-family:monospace;font-size:0.85em;">{target_id}</td>'
                f'<td style="color:{score_color};font-weight:bold;text-align:right;">'
                f'{total:.3f}</td>'
                f'<td style="text-align:center;">{n_active}/6</td>'
                f'<td style="text-align:right;">{geo_mean:.3f}</td>'
                f'<td style="text-align:right;">{bonus:.0f}x</td>'
            )

            # Per-channel cells
            ch_scores = s.get("channel_scores", {})
            for ch in channel_names:
                ch_data = ch_scores.get(ch, {})
                ch_score = float(ch_data.get("score", 0))
                is_active = ch_data.get("is_active", False)

                if is_active:
                    cell_bg = "rgba(0,212,255,0.15)"
                    cell_color = _ACCENT_CYAN
                    indicator = "&#9679; "  # filled circle
                else:
                    cell_bg = "transparent"
                    cell_color = _TEXT_SECONDARY
                    indicator = ""

                row += (
                    f'<td style="text-align:center;background:{cell_bg};'
                    f'color:{cell_color};font-size:0.85em;">'
                    f'{indicator}{ch_score:.2f}</td>'
                )

            rows.append(f"<tr>{row}</tr>")

        return (
            '<div style="overflow-x:auto;">'
            '<table class="exodus-table">'
            f"<thead><tr>{header_cells}</tr></thead>"
            f'<tbody>{"".join(rows)}</tbody>'
            "</table>"
            "</div>"
        )

    # ------------------------------------------------------------------
    #  Hypothesis status board
    # ------------------------------------------------------------------

    def _generate_hypothesis_board(self, hypotheses: List[Dict[str, Any]]) -> str:
        """Generate a hypothesis status board with counts and detail cards.

        Parameters
        ----------
        hypotheses : list of dict
            Each dict should have ``hypothesis_id``, ``claim``, ``status``,
            ``dataset``, ``method``, ``kardashev``, ``timestamp``.

        Returns
        -------
        str
            HTML markup for the hypothesis section.
        """
        if not hypotheses:
            return '<p style="color:{};">No hypotheses recorded yet.</p>'.format(
                _TEXT_SECONDARY
            )

        # Count by status
        status_counts: Dict[str, int] = {}
        for h in hypotheses:
            st = h.get("status", "pending")
            status_counts[st] = status_counts.get(st, 0) + 1

        # Status summary cards
        summary_cards = ""
        for status in ["pending", "tested", "confirmed", "rejected"]:
            count = status_counts.get(status, 0)
            color = _STATUS_COLORS.get(status, _TEXT_SECONDARY)
            summary_cards += (
                f'<div class="stat-card" style="border-top:3px solid {color};">'
                f'<div class="stat-number" style="color:{color};">{count}</div>'
                f'<div class="stat-label">{status.capitalize()}</div>'
                f'</div>'
            )

        summary_row = f'<div class="stat-row">{summary_cards}</div>'

        # Detail cards for each hypothesis
        detail_cards = ""
        for h in hypotheses:
            hid = escape(str(h.get("hypothesis_id", "?"))[:16])
            claim = escape(str(h.get("claim", "No claim")))
            status = h.get("status", "pending")
            dataset = escape(str(h.get("dataset", "?")))
            method = escape(str(h.get("method", "?")))
            kardashev = escape(str(h.get("kardashev", "?")))
            ts = escape(str(h.get("timestamp", "")))
            color = _STATUS_COLORS.get(status, _TEXT_SECONDARY)

            detail_cards += (
                f'<div class="hypothesis-card" style="border-left:4px solid {color};">'
                f'<div class="hypothesis-header">'
                f'<span class="hypothesis-id">{hid}</span>'
                f'<span class="hypothesis-status" style="color:{color};">'
                f'{escape(status.upper())}</span>'
                f'</div>'
                f'<div class="hypothesis-claim">{claim}</div>'
                f'<div class="hypothesis-meta">'
                f'<span>Dataset: {dataset}</span>'
                f'<span>Method: {method}</span>'
                f'<span>Kardashev: {kardashev}</span>'
                f'</div>'
                f'<div class="hypothesis-timestamp">{ts}</div>'
                f'</div>'
            )

        return summary_row + '<div class="hypothesis-list">' + detail_cards + '</div>'

    # ------------------------------------------------------------------
    #  Evolution log
    # ------------------------------------------------------------------

    def _generate_evolution_log(self, evolution_records: List[Dict[str, Any]]) -> str:
        """Generate an evolution timeline showing what the system has learned.

        Parameters
        ----------
        evolution_records : list of dict
            Each dict mirrors the ``EvolutionRecord`` dataclass with keys
            like ``iteration``, ``timestamp``, ``threshold_changes``,
            ``new_hypotheses_generated``, ``strategies_deprioritized``,
            ``strategies_promoted``, ``false_positive_rate``,
            ``true_positive_rate``, ``recommendations``.

        Returns
        -------
        str
            HTML markup for the evolution timeline.
        """
        if not evolution_records:
            return '<p style="color:{};">No evolution records yet.</p>'.format(
                _TEXT_SECONDARY
            )

        entries = ""
        for rec in evolution_records:
            iteration = rec.get("iteration", "?")
            ts = escape(str(rec.get("timestamp", "")))
            fpr = rec.get("false_positive_rate", 0)
            tpr = rec.get("true_positive_rate", 0)

            # Threshold changes
            threshold_items = ""
            threshold_changes = rec.get("threshold_changes", {})
            if threshold_changes:
                for name, change in threshold_changes.items():
                    if isinstance(change, dict):
                        old = change.get("old", "?")
                        new = change.get("new", "?")
                        reason = escape(str(change.get("reason", "")))
                        direction = "up" if float(str(new)) > float(str(old)) else "down"
                        arrow = "&#9650;" if direction == "up" else "&#9660;"
                        arrow_color = _ACCENT_RED if direction == "up" else _ACCENT_GREEN
                        threshold_items += (
                            f'<div class="evo-change">'
                            f'<span style="color:{arrow_color};">{arrow}</span> '
                            f'<strong>{escape(str(name))}</strong>: '
                            f'{old} &rarr; {new}'
                            f'<span class="evo-reason">{reason}</span>'
                            f'</div>'
                        )

            # Strategy changes
            strategy_items = ""
            for dep in rec.get("strategies_deprioritized", []):
                method = escape(str(dep.get("method", "?")))
                reason = escape(str(dep.get("reason", "")))
                strategy_items += (
                    f'<div class="evo-change">'
                    f'<span style="color:{_ACCENT_RED};">&#9660;</span> '
                    f'Deprioritized <strong>{method}</strong>'
                    f'<span class="evo-reason">{reason}</span>'
                    f'</div>'
                )
            for prom in rec.get("strategies_promoted", []):
                method = escape(str(prom.get("method", "?")))
                reason = escape(str(prom.get("reason", "")))
                strategy_items += (
                    f'<div class="evo-change">'
                    f'<span style="color:{_ACCENT_GREEN};">&#9650;</span> '
                    f'Promoted <strong>{method}</strong>'
                    f'<span class="evo-reason">{reason}</span>'
                    f'</div>'
                )

            # New hypotheses generated
            new_hyp = rec.get("new_hypotheses_generated", [])
            hyp_count = len(new_hyp)

            # Recommendations
            recommendations = rec.get("recommendations", [])
            rec_items = ""
            for r in recommendations:
                rec_items += f'<li>{escape(str(r))}</li>'
            rec_html = f'<ul class="evo-recommendations">{rec_items}</ul>' if rec_items else ""

            entries += (
                f'<div class="evo-entry">'
                f'<div class="evo-header">'
                f'<span class="evo-iteration">Iteration {iteration}</span>'
                f'<span class="evo-timestamp">{ts}</span>'
                f'</div>'
                f'<div class="evo-metrics">'
                f'<span>FP Rate: <strong style="color:{_ACCENT_AMBER};">'
                f'{fpr:.1%}</strong></span>'
                f'<span>TP Rate: <strong style="color:{_ACCENT_GREEN};">'
                f'{tpr:.1%}</strong></span>'
                f'<span>New Hypotheses: <strong style="color:{_ACCENT_CYAN};">'
                f'{hyp_count}</strong></span>'
                f'</div>'
                f'{threshold_items}'
                f'{strategy_items}'
                f'{rec_html}'
                f'</div>'
            )

        return f'<div class="evo-timeline">{entries}</div>'

    # ------------------------------------------------------------------
    #  Breakthrough candidates
    # ------------------------------------------------------------------

    def _generate_breakthrough_section(
        self, candidates: List[Dict[str, Any]]
    ) -> str:
        """Generate breakthrough candidate cards with escalation details.

        Parameters
        ----------
        candidates : list of dict
            Each dict mirrors ``BreakthroughCandidate`` with keys like
            ``candidate_id``, ``target_info``, ``current_level``,
            ``status``, ``level_results``, ``timestamp``,
            ``resolved_explanation``.

        Returns
        -------
        str
            HTML markup for the breakthrough section.
        """
        if not candidates:
            return (
                '<div class="no-breakthroughs">'
                '<div class="no-bt-icon">&#9673;</div>'
                '<p>No breakthrough candidates at this time.</p>'
                '<p style="font-size:0.85em;color:{};">Candidates appear when '
                'anomalies survive the full analysis pipeline without a '
                'natural explanation.</p>'
                '</div>'
            ).format(_TEXT_SECONDARY)

        # Escalation levels for progress display
        escalation_levels = [
            "VERIFY", "REPRODUCE", "CHARACTERIZE",
            "EXHAUST_NATURAL", "REPORT", "PROPOSE",
        ]

        cards = ""
        for cand in candidates:
            cid = escape(str(cand.get("candidate_id", "?"))[:16])
            target_info = cand.get("target_info", {})
            target_id = escape(str(target_info.get("target_id", "Unknown")))
            ra = target_info.get("ra", "?")
            dec = target_info.get("dec", "?")
            current_level = str(cand.get("current_level", "VERIFY")).upper()
            status = str(cand.get("status", "active"))
            ts = escape(str(cand.get("timestamp", "")))
            resolved = cand.get("resolved_explanation")
            level_results = cand.get("level_results", {})

            # Status styling
            if status == "active":
                status_color = _ACCENT_AMBER
                status_icon = "&#9888;"  # warning sign
            elif status == "resolved_natural":
                status_color = _TEXT_SECONDARY
                status_icon = "&#10003;"  # checkmark
            elif status == "unresolved":
                status_color = _ACCENT_RED
                status_icon = "&#9733;"  # star
            else:
                status_color = _TEXT_SECONDARY
                status_icon = "&#9679;"

            # Build escalation progress bar
            progress_steps = ""
            current_idx = -1
            for i, lvl in enumerate(escalation_levels):
                if lvl == current_level:
                    current_idx = i

            for i, lvl in enumerate(escalation_levels):
                if i < current_idx:
                    step_class = "step-complete"
                elif i == current_idx:
                    step_class = "step-current"
                else:
                    step_class = "step-pending"

                # Check if this level has results
                has_result = lvl in level_results or lvl.lower() in level_results
                result_indicator = " &#10003;" if has_result and i < current_idx else ""

                progress_steps += (
                    f'<div class="escalation-step {step_class}">'
                    f'{lvl.replace("_", " ")}{result_indicator}'
                    f'</div>'
                )

            escalation_bar = f'<div class="escalation-bar">{progress_steps}</div>'

            # Resolved explanation
            resolved_html = ""
            if resolved:
                resolved_html = (
                    f'<div class="resolved-explanation">'
                    f'Resolved: {escape(str(resolved))}'
                    f'</div>'
                )

            cards += (
                f'<div class="breakthrough-card">'
                f'<div class="bt-header">'
                f'<span class="bt-id">{cid}</span>'
                f'<span class="bt-status" style="color:{status_color};">'
                f'{status_icon} {escape(status.upper())}</span>'
                f'</div>'
                f'<div class="bt-target">'
                f'<strong>{target_id}</strong>'
                f'<span class="bt-coords">RA={ra}, Dec={dec}</span>'
                f'</div>'
                f'{escalation_bar}'
                f'{resolved_html}'
                f'<div class="bt-timestamp">{ts}</div>'
                f'</div>'
            )

        return f'<div class="breakthrough-list">{cards}</div>'

    # ------------------------------------------------------------------
    #  Summary statistics
    # ------------------------------------------------------------------

    def _generate_summary_stats(self, results_data: Dict[str, Any]) -> str:
        """Generate key-number summary cards for the dashboard header.

        Parameters
        ----------
        results_data : dict
            The full results_data dict passed to :meth:`generate`.

        Returns
        -------
        str
            HTML markup for the summary statistics row.
        """
        scores = results_data.get("scores", [])
        hypotheses = results_data.get("hypotheses", [])
        breakthroughs = results_data.get("breakthrough_candidates", [])
        iteration = results_data.get("iteration", 0)

        n_targets = len(scores)
        n_active_targets = sum(
            1 for s in scores
            if int(s.get("n_active_channels", 0)) > 0
        )

        # Max EXODUS score
        max_score = 0.0
        if scores:
            max_score = max(float(s.get("total_score", 0)) for s in scores)

        # Average active channels among active targets
        avg_channels = 0.0
        if n_active_targets > 0:
            total_channels = sum(
                int(s.get("n_active_channels", 0))
                for s in scores
                if int(s.get("n_active_channels", 0)) > 0
            )
            avg_channels = total_channels / n_active_targets

        # Hypothesis counts
        n_hypotheses = len(hypotheses)
        n_confirmed = sum(1 for h in hypotheses if h.get("status") == "confirmed")

        # Active breakthroughs
        n_active_bt = sum(1 for b in breakthroughs if b.get("status") == "active")
        n_unresolved_bt = sum(1 for b in breakthroughs if b.get("status") == "unresolved")

        stats = [
            ("Iteration", str(iteration), _ACCENT_CYAN),
            ("Targets Scored", str(n_targets), _ACCENT_CYAN),
            ("Active Targets", str(n_active_targets), _ACCENT_GREEN),
            ("Peak Score", f"{max_score:.2f}", _ACCENT_AMBER),
            ("Avg Channels", f"{avg_channels:.1f}", _ACCENT_CYAN),
            ("Hypotheses", str(n_hypotheses), _ACCENT_PURPLE),
            ("Confirmed", str(n_confirmed), _ACCENT_GREEN),
            ("Breakthroughs", str(n_active_bt + n_unresolved_bt), _ACCENT_RED),
        ]

        cards = ""
        for label, value, color in stats:
            cards += (
                f'<div class="stat-card">'
                f'<div class="stat-number" style="color:{color};">{value}</div>'
                f'<div class="stat-label">{label}</div>'
                f'</div>'
            )

        return f'<div class="stat-row">{cards}</div>'

    # ------------------------------------------------------------------
    #  Browser opener
    # ------------------------------------------------------------------

    @staticmethod
    def open_in_browser(filepath: str | Path) -> None:
        """Open the generated report in the default web browser.

        Parameters
        ----------
        filepath : str or Path
            Path to the HTML file.
        """
        filepath = Path(filepath).resolve()
        url = filepath.as_uri()
        log.info("Opening dashboard in browser: %s", url)
        webbrowser.open(url)

    # ------------------------------------------------------------------
    #  HTML assembly (private)
    # ------------------------------------------------------------------

    def _assemble_page(
        self,
        iteration: int,
        timestamp: str,
        summary_html: str,
        sky_map_html: str,
        score_table_html: str,
        hypothesis_html: str,
        evolution_html: str,
        breakthrough_html: str,
    ) -> str:
        """Assemble all sections into a complete HTML page."""

        css = self._get_css()

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Project EXODUS Dashboard &mdash; Iteration {iteration}</title>
<style>
{css}
</style>
</head>
<body>

<header class="page-header">
    <div class="header-content">
        <h1 class="logo">
            <span class="logo-exodus">EXODUS</span>
            <span class="logo-sub">Exo-civilisation Detection Using Stellar Observation &amp; Unified Search</span>
        </h1>
        <div class="header-meta">
            <span class="iter-badge">Iteration {iteration}</span>
            <span class="timestamp">{escape(str(timestamp))}</span>
        </div>
    </div>
</header>

<main class="container">

    <!-- Summary Statistics -->
    <section class="panel" id="summary">
        <h2 class="section-title">Mission Overview</h2>
        {summary_html}
    </section>

    <!-- Sky Map -->
    <section class="panel" id="sky-map">
        <h2 class="section-title">Sky Map</h2>
        {sky_map_html}
    </section>

    <!-- Top Targets -->
    <section class="panel" id="top-targets">
        <h2 class="section-title">Top 20 Targets by EXODUS Score</h2>
        {score_table_html}
    </section>

    <!-- Hypothesis Board -->
    <section class="panel" id="hypotheses">
        <h2 class="section-title">Hypothesis Status Board</h2>
        {hypothesis_html}
    </section>

    <!-- Evolution Log -->
    <section class="panel" id="evolution">
        <h2 class="section-title">Evolution Log</h2>
        {evolution_html}
    </section>

    <!-- Breakthrough Candidates -->
    <section class="panel" id="breakthroughs">
        <h2 class="section-title">Breakthrough Candidates</h2>
        {breakthrough_html}
    </section>

</main>

<footer class="page-footer">
    <p>Project EXODUS &mdash; Generated {escape(str(timestamp))}</p>
</footer>

</body>
</html>"""

    @staticmethod
    def _get_css() -> str:
        """Return the complete inline CSS for the dashboard."""
        return f"""
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
    background: {_BG_DARK};
    color: {_TEXT_PRIMARY};
    line-height: 1.6;
    min-height: 100vh;
}}

/* ── Header ───────────────────────────────────────────────────── */
.page-header {{
    background: linear-gradient(135deg, {_BG_PANEL} 0%, {_BG_DARK} 100%);
    border-bottom: 1px solid {_BORDER_SUBTLE};
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
    letter-spacing: 0.2em;
    color: {_ACCENT_CYAN};
    text-shadow: 0 0 20px rgba(0,212,255,0.3);
}}

.logo-sub {{
    font-size: 0.75rem;
    font-weight: 400;
    color: {_TEXT_SECONDARY};
    letter-spacing: 0.05em;
}}

.header-meta {{
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.25rem;
}}

.iter-badge {{
    background: {_ACCENT_CYAN};
    color: {_BG_DARK};
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
}}

.timestamp {{
    font-size: 0.8rem;
    color: {_TEXT_SECONDARY};
}}

/* ── Container & panels ───────────────────────────────────────── */
.container {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}}

.panel {{
    background: {_BG_PANEL};
    border: 1px solid {_BORDER_SUBTLE};
    border-radius: 12px;
    padding: 1.5rem;
    overflow: hidden;
}}

.section-title {{
    font-size: 1.2rem;
    font-weight: 700;
    color: {_ACCENT_CYAN};
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid {_BORDER_SUBTLE};
}}

/* ── Stat cards ───────────────────────────────────────────────── */
.stat-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    justify-content: center;
}}

.stat-card {{
    background: {_BG_CARD};
    border: 1px solid {_BORDER_SUBTLE};
    border-radius: 8px;
    padding: 1rem 1.5rem;
    text-align: center;
    min-width: 120px;
    flex: 1;
    max-width: 180px;
}}

.stat-number {{
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1.2;
}}

.stat-label {{
    font-size: 0.75rem;
    color: {_TEXT_SECONDARY};
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
}}

/* ── Score table ──────────────────────────────────────────────── */
.exodus-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}}

.exodus-table thead th {{
    background: {_BG_CARD};
    color: {_ACCENT_CYAN};
    font-weight: 600;
    padding: 0.6rem 0.5rem;
    text-align: center;
    border-bottom: 2px solid {_BORDER_SUBTLE};
    position: sticky;
    top: 0;
    white-space: nowrap;
    font-size: 0.78rem;
}}

.exodus-table tbody td {{
    padding: 0.5rem 0.5rem;
    border-bottom: 1px solid rgba(42,42,78,0.5);
    vertical-align: middle;
}}

.exodus-table tbody tr:hover {{
    background: rgba(0,212,255,0.05);
}}

/* ── Hypothesis cards ─────────────────────────────────────────── */
.hypothesis-list {{
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    margin-top: 1rem;
    max-height: 500px;
    overflow-y: auto;
}}

.hypothesis-card {{
    background: {_BG_CARD};
    border: 1px solid {_BORDER_SUBTLE};
    border-radius: 8px;
    padding: 1rem;
}}

.hypothesis-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}}

.hypothesis-id {{
    font-family: monospace;
    font-size: 0.75rem;
    color: {_TEXT_SECONDARY};
}}

.hypothesis-status {{
    font-weight: 700;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
}}

.hypothesis-claim {{
    font-size: 0.9rem;
    line-height: 1.4;
    margin-bottom: 0.5rem;
}}

.hypothesis-meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    font-size: 0.75rem;
    color: {_TEXT_SECONDARY};
}}

.hypothesis-timestamp {{
    font-size: 0.7rem;
    color: {_TEXT_SECONDARY};
    margin-top: 0.5rem;
    text-align: right;
}}

/* ── Evolution timeline ───────────────────────────────────────── */
.evo-timeline {{
    display: flex;
    flex-direction: column;
    gap: 1rem;
}}

.evo-entry {{
    background: {_BG_CARD};
    border: 1px solid {_BORDER_SUBTLE};
    border-radius: 8px;
    padding: 1rem;
}}

.evo-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
}}

.evo-iteration {{
    font-weight: 700;
    color: {_ACCENT_CYAN};
    font-size: 0.95rem;
}}

.evo-timestamp {{
    font-size: 0.75rem;
    color: {_TEXT_SECONDARY};
}}

.evo-metrics {{
    display: flex;
    gap: 1.5rem;
    font-size: 0.85rem;
    margin-bottom: 0.75rem;
    flex-wrap: wrap;
}}

.evo-change {{
    font-size: 0.85rem;
    padding: 0.25rem 0;
    line-height: 1.5;
}}

.evo-reason {{
    display: block;
    font-size: 0.75rem;
    color: {_TEXT_SECONDARY};
    margin-left: 1.5rem;
}}

.evo-recommendations {{
    margin-top: 0.5rem;
    padding-left: 1.5rem;
    font-size: 0.85rem;
    color: {_TEXT_SECONDARY};
}}

.evo-recommendations li {{
    margin-bottom: 0.25rem;
}}

/* ── Breakthrough cards ───────────────────────────────────────── */
.breakthrough-list {{
    display: flex;
    flex-direction: column;
    gap: 1rem;
}}

.breakthrough-card {{
    background: {_BG_CARD};
    border: 1px solid {_ACCENT_AMBER};
    border-radius: 8px;
    padding: 1.25rem;
}}

.bt-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}}

.bt-id {{
    font-family: monospace;
    font-size: 0.75rem;
    color: {_TEXT_SECONDARY};
}}

.bt-status {{
    font-weight: 700;
    font-size: 0.8rem;
}}

.bt-target {{
    margin-bottom: 0.75rem;
}}

.bt-target strong {{
    color: {_ACCENT_CYAN};
}}

.bt-coords {{
    display: inline-block;
    margin-left: 1rem;
    font-size: 0.8rem;
    color: {_TEXT_SECONDARY};
}}

.bt-timestamp {{
    font-size: 0.7rem;
    color: {_TEXT_SECONDARY};
    margin-top: 0.5rem;
    text-align: right;
}}

/* ── Escalation progress bar ──────────────────────────────────── */
.escalation-bar {{
    display: flex;
    gap: 4px;
    margin: 0.75rem 0;
}}

.escalation-step {{
    flex: 1;
    text-align: center;
    padding: 0.4rem 0.25rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    white-space: nowrap;
}}

.step-complete {{
    background: rgba(0,255,136,0.2);
    color: {_ACCENT_GREEN};
    border: 1px solid rgba(0,255,136,0.4);
}}

.step-current {{
    background: rgba(255,165,0,0.25);
    color: {_ACCENT_AMBER};
    border: 1px solid rgba(255,165,0,0.5);
    animation: pulse-amber 2s ease-in-out infinite;
}}

.step-pending {{
    background: rgba(42,42,78,0.5);
    color: {_TEXT_SECONDARY};
    border: 1px solid {_BORDER_SUBTLE};
}}

@keyframes pulse-amber {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.6; }}
}}

/* ── No-breakthroughs placeholder ─────────────────────────────── */
.no-breakthroughs {{
    text-align: center;
    padding: 2rem;
    color: {_TEXT_SECONDARY};
}}

.no-bt-icon {{
    font-size: 3rem;
    color: {_BORDER_SUBTLE};
    margin-bottom: 0.5rem;
}}

/* ── Resolved explanation ─────────────────────────────────────── */
.resolved-explanation {{
    background: rgba(160,160,176,0.1);
    border-left: 3px solid {_TEXT_SECONDARY};
    padding: 0.5rem 0.75rem;
    font-size: 0.8rem;
    color: {_TEXT_SECONDARY};
    margin-top: 0.5rem;
    border-radius: 0 4px 4px 0;
}}

/* ── Footer ───────────────────────────────────────────────────── */
.page-footer {{
    text-align: center;
    padding: 1.5rem;
    font-size: 0.75rem;
    color: {_TEXT_SECONDARY};
    border-top: 1px solid {_BORDER_SUBTLE};
    margin-top: 1rem;
}}

/* ── Responsive ───────────────────────────────────────────────── */
@media (max-width: 768px) {{
    .header-content {{
        flex-direction: column;
        align-items: flex-start;
    }}
    .header-meta {{
        align-items: flex-start;
    }}
    .stat-card {{
        min-width: 100px;
        max-width: 140px;
    }}
    .stat-number {{
        font-size: 1.4rem;
    }}
    .escalation-step {{
        font-size: 0.55rem;
        padding: 0.3rem 0.15rem;
    }}
    .hypothesis-meta {{
        flex-direction: column;
        gap: 0.25rem;
    }}
}}

/* ── Scrollbar styling ────────────────────────────────────────── */
::-webkit-scrollbar {{
    width: 8px;
    height: 8px;
}}
::-webkit-scrollbar-track {{
    background: {_BG_DARK};
}}
::-webkit-scrollbar-thumb {{
    background: {_BORDER_SUBTLE};
    border-radius: 4px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: {_ACCENT_CYAN};
}}
"""


# =====================================================================
#  Main -- generate a demo dashboard with mock data
# =====================================================================

if __name__ == "__main__":
    import random

    random.seed(42)
    np.random.seed(42)

    log.setLevel("INFO")

    print("=" * 70)
    print("  Project EXODUS -- Dashboard Demo Generator")
    print("=" * 70)

    # ------------------------------------------------------------------
    #  Mock scores (EXODUSScore-like dicts)
    # ------------------------------------------------------------------
    channel_names = [
        "ir_excess", "transit_anomaly", "radio_anomaly",
        "gaia_photometric_anomaly", "habitable_zone_planet",
        "proper_motion_anomaly",
    ]

    mock_scores: List[Dict[str, Any]] = []
    for i in range(150):
        ra = random.uniform(0, 360)
        dec = random.uniform(-90, 90)

        # Decide how many channels are active
        n_active = np.random.choice(
            [0, 1, 2, 3, 4, 5, 6],
            p=[0.30, 0.30, 0.18, 0.12, 0.06, 0.03, 0.01],
        )

        activated = set(random.sample(channel_names, k=min(int(n_active), 6)))

        ch_scores = {}
        for ch in channel_names:
            if ch in activated:
                sc = random.uniform(0.35, 0.95)
                ch_scores[ch] = {"score": sc, "is_active": True}
            else:
                sc = random.uniform(0.0, 0.25)
                ch_scores[ch] = {"score": sc, "is_active": False}

        # Compute aggregate
        active_vals = [c["score"] for c in ch_scores.values() if c["is_active"]]
        if active_vals:
            geo_mean = float(np.exp(np.mean(np.log(active_vals))))
            convergence_bonus = float(2 ** (len(active_vals) - 1))
            total_score = geo_mean * convergence_bonus
        else:
            geo_mean = 0.0
            convergence_bonus = 0.0
            total_score = 0.0

        mock_scores.append({
            "target_id": f"GAIA_DR3_{2000000000 + i}",
            "ra": ra,
            "dec": dec,
            "total_score": total_score,
            "n_active_channels": len(active_vals),
            "geo_mean": geo_mean,
            "convergence_bonus": convergence_bonus,
            "channel_scores": ch_scores,
            "rank": None,
        })

    # Assign ranks
    mock_scores.sort(key=lambda s: s["total_score"], reverse=True)
    for idx, s in enumerate(mock_scores, start=1):
        s["rank"] = idx

    # ------------------------------------------------------------------
    #  Mock hypotheses
    # ------------------------------------------------------------------
    mock_hypotheses: List[Dict[str, Any]] = [
        {
            "hypothesis_id": "hyp_ir_hz_001",
            "claim": "Stars with confirmed habitable-zone planets and mid-IR excess above 5-sigma are technosignature candidates.",
            "dataset": "WISE + NASA Exoplanet Archive",
            "method": "cross_correlation",
            "status": "confirmed",
            "kardashev": "Type II",
            "timestamp": "2026-02-20T14:30:00Z",
        },
        {
            "hypothesis_id": "hyp_radio_transit",
            "claim": "Targets with both radio narrowband candidates and anomalous transit shapes warrant priority follow-up.",
            "dataset": "Breakthrough Listen + Kepler",
            "method": "multi_channel_convergence",
            "status": "tested",
            "kardashev": "Type I",
            "timestamp": "2026-02-20T15:00:00Z",
        },
        {
            "hypothesis_id": "hyp_gaia_var_001",
            "claim": "Gaia DR3 epoch photometry variability above 3% in G-band correlates with unusual mid-IR color.",
            "dataset": "Gaia DR3 + AllWISE",
            "method": "statistical_correlation",
            "status": "pending",
            "kardashev": "Type II",
            "timestamp": "2026-02-20T16:00:00Z",
        },
        {
            "hypothesis_id": "hyp_ruwe_cluster",
            "claim": "Clustered high-RUWE sources near known exoplanet hosts may indicate unseen megastructures.",
            "dataset": "Gaia DR3 astrometry",
            "method": "spatial_clustering",
            "status": "rejected",
            "kardashev": "Type III",
            "timestamp": "2026-02-20T12:00:00Z",
        },
        {
            "hypothesis_id": "hyp_dimming_001",
            "claim": "Secular dimming trends in Kepler long-cadence data correlate with mid-IR excess in WISE W4.",
            "dataset": "Kepler + WISE",
            "method": "time_series_correlation",
            "status": "pending",
            "kardashev": "Type II",
            "timestamp": "2026-02-21T08:00:00Z",
        },
        {
            "hypothesis_id": "hyp_pm_anomaly",
            "claim": "Proper motion anomalies in high-RUWE targets with HZ planets suggest unseen massive companions.",
            "dataset": "Gaia DR3 + Exoplanet Archive",
            "method": "astrometric_analysis",
            "status": "tested",
            "kardashev": "Type I",
            "timestamp": "2026-02-21T09:30:00Z",
        },
    ]

    # ------------------------------------------------------------------
    #  Mock evolution log
    # ------------------------------------------------------------------
    mock_evolution: List[Dict[str, Any]] = [
        {
            "iteration": 1,
            "timestamp": "2026-02-20T10:00:00Z",
            "threshold_changes": {
                "anomaly_sigma": {
                    "old": 3.0,
                    "new": 3.5,
                    "reason": "False positive rate was 35%, above 30% threshold. Tightening to reduce noise.",
                },
            },
            "new_hypotheses_generated": [
                {"id": "hyp_ir_hz_001", "text": "IR+HZ cross-correlation", "priority": "high"},
            ],
            "strategies_deprioritized": [],
            "strategies_promoted": [
                {"method": "multi_channel_convergence", "old_weight": 0.5, "new_weight": 0.8,
                 "reason": "Multi-channel convergence produced 3 interesting results vs 0 from single-channel."},
            ],
            "false_positive_rate": 0.35,
            "true_positive_rate": 0.12,
            "recommendations": [
                "Focus on multi-channel convergence targets for next iteration.",
                "Consider adding UV channel data from GALEX for additional cross-validation.",
            ],
        },
        {
            "iteration": 2,
            "timestamp": "2026-02-21T10:00:00Z",
            "threshold_changes": {
                "anomaly_sigma": {
                    "old": 3.5,
                    "new": 3.25,
                    "reason": "False positive rate dropped to 8%. Loosening slightly to recover sensitivity.",
                },
                "min_convergence_channels": {
                    "old": 2,
                    "new": 2,
                    "reason": "No change needed. Current threshold performing well.",
                },
            },
            "new_hypotheses_generated": [
                {"id": "hyp_gaia_var_001", "text": "Gaia variability correlation", "priority": "medium"},
                {"id": "hyp_dimming_001", "text": "Secular dimming + IR", "priority": "medium"},
            ],
            "strategies_deprioritized": [
                {"method": "spatial_clustering", "old_weight": 0.6, "new_weight": 0.2,
                 "reason": "100% dead-end rate after 5 tests. Clustering near exoplanet hosts yields no signal."},
            ],
            "strategies_promoted": [],
            "false_positive_rate": 0.08,
            "true_positive_rate": 0.22,
            "recommendations": [
                "Pipeline is well-tuned. Maintain current thresholds.",
                "Spatial clustering approach deprioritized -- focus resources elsewhere.",
                "Gaia variability channel shows promise; increase coverage.",
            ],
        },
    ]

    # ------------------------------------------------------------------
    #  Mock breakthrough candidates
    # ------------------------------------------------------------------
    mock_breakthroughs: List[Dict[str, Any]] = [
        {
            "candidate_id": "bt_a3f7c201",
            "target_info": {
                "target_id": "GAIA_DR3_2000000042",
                "ra": 284.75,
                "dec": -12.33,
            },
            "initial_result": {"anomaly_type": "ir_excess + transit_anomaly"},
            "current_level": "CHARACTERIZE",
            "level_results": {
                "VERIFY": {"passed": True, "note": "Data re-downloaded and re-processed successfully."},
                "REPRODUCE": {"passed": True, "note": "Independent TESS data confirms transit anomaly."},
            },
            "natural_explanations_tested": [],
            "status": "active",
            "timestamp": "2026-02-21T11:00:00Z",
            "resolved_explanation": None,
        },
        {
            "candidate_id": "bt_e9d12b04",
            "target_info": {
                "target_id": "GAIA_DR3_2000000099",
                "ra": 56.12,
                "dec": 41.88,
            },
            "initial_result": {"anomaly_type": "radio_anomaly"},
            "current_level": "EXHAUST_NATURAL",
            "level_results": {
                "VERIFY": {"passed": True},
                "REPRODUCE": {"passed": True},
                "CHARACTERIZE": {"passed": True, "note": "Full multi-wavelength workup complete."},
            },
            "natural_explanations_tested": [
                {"id": "variable_star", "verdict": "ruled_out"},
                {"id": "galaxy_contamination", "verdict": "ruled_out"},
            ],
            "status": "active",
            "timestamp": "2026-02-21T09:00:00Z",
            "resolved_explanation": None,
        },
        {
            "candidate_id": "bt_1234abcd",
            "target_info": {
                "target_id": "GAIA_DR3_2000000007",
                "ra": 178.44,
                "dec": -5.22,
            },
            "initial_result": {"anomaly_type": "ir_excess"},
            "current_level": "REPRODUCE",
            "level_results": {
                "VERIFY": {"passed": True},
                "REPRODUCE": {"passed": False, "note": "IR excess explained by background galaxy."},
            },
            "natural_explanations_tested": [],
            "status": "resolved_natural",
            "timestamp": "2026-02-20T18:00:00Z",
            "resolved_explanation": "Background galaxy contamination in WISE beam (resolved by HST imaging).",
        },
    ]

    # ------------------------------------------------------------------
    #  Assemble results_data and generate
    # ------------------------------------------------------------------
    results_data = {
        "scores": mock_scores,
        "hypotheses": mock_hypotheses,
        "evolution_log": mock_evolution,
        "breakthrough_candidates": mock_breakthroughs,
        "iteration": 2,
        "timestamp": "2026-02-21T12:00:00Z",
    }

    generator = DashboardGenerator()
    filepath = generator.generate(results_data)

    print(f"\nDashboard generated: {filepath}")
    print(f"File size: {filepath.stat().st_size / 1024:.1f} KB")
    print("\nTo open in browser, run:")
    print(f"  open {filepath}")
    print("\nDone.")
