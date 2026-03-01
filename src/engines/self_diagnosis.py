"""
Self-Diagnosis and Completeness Monitor for Project EXODUS.

Continuously monitors for failure modes that could block discovery.
Runs automatically after each research iteration and checks for 7
specific ways the system could suppress real signals:

  Check 2.1: Threshold sensitivity — are thresholds too aggressive?
  Check 2.2: RFI over-correction — are we killing real signals?
  Check 2.3: Anthropocentric bias — are we only finding what we expect?
  Check 2.4: Temporal resolution — are we averaging out transients?
  Check 2.5: Catalog completeness — are we missing uncataloged sources?
  Check 2.6: Frequency coverage — are we only searching expected frequencies?
  Check 2.7: Slow changes — are we missing gradual changes?

Each check returns GREEN (passing), YELLOW (borderline), or RED
(discovery-blocking issue).  RED results trigger automatic threshold
adjustment or hypothesis generation.

Public API
----------
SelfDiagnostics()
    Main class.

.run_all_checks(iteration_results)
    Run all 7 diagnostics checks.

.generate_diagnosis_report()
    Produce human-readable report.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger, PROJECT_ROOT

log = get_logger("engines.self_diagnosis")

# Status constants
GREEN = "GREEN"
YELLOW = "YELLOW"
RED = "RED"


# =====================================================================
#  Data classes
# =====================================================================

@dataclass
class DiagnosticResult:
    """Result of a single diagnostic check."""
    check_id: str
    check_name: str
    status: str           # GREEN, YELLOW, RED
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""


@dataclass
class DiagnosisReport:
    """Full diagnosis report from all checks."""
    timestamp: str
    iteration: int
    checks: List[DiagnosticResult] = field(default_factory=list)
    n_green: int = 0
    n_yellow: int = 0
    n_red: int = 0
    overall_status: str = GREEN
    action_items: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "iteration": self.iteration,
            "n_green": self.n_green,
            "n_yellow": self.n_yellow,
            "n_red": self.n_red,
            "overall_status": self.overall_status,
            "checks": [
                {
                    "id": c.check_id,
                    "name": c.check_name,
                    "status": c.status,
                    "message": c.message,
                    "recommendation": c.recommendation,
                }
                for c in self.checks
            ],
            "action_items": self.action_items,
        }


# =====================================================================
#  Self-Diagnostics Engine
# =====================================================================

class SelfDiagnostics:
    """Self-diagnosis monitor for Project EXODUS.

    Runs 7 automated checks after each research iteration to detect
    failure modes that could block discovery.
    """

    def __init__(self) -> None:
        self._history: List[DiagnosisReport] = []
        log.info("SelfDiagnostics initialized")

    # ─── Main entry point ─────────────────────────────────────────────

    def run_all_checks(
        self,
        iteration_results: Dict[str, Any],
    ) -> DiagnosisReport:
        """Run all 7 diagnostic checks.

        Parameters
        ----------
        iteration_results : dict
            Results from the latest research iteration.  Expected keys:
              - iteration : int
              - anomaly_counts : dict of {sigma_threshold -> n_candidates}
              - radio_results : list of radio processing results
              - all_results : list of all target results
              - timeseries_results : list of multi-resolution results
              - crossmatch_results : list of crossmatch results
              - temporal_results : list of temporal archaeology results
              - current_thresholds : dict

        Returns
        -------
        DiagnosisReport
        """
        iteration = iteration_results.get("iteration", 0)
        log.info("Running self-diagnostics for iteration %d", iteration)

        checks = [
            self.check_threshold_sensitivity(iteration_results),
            self.check_rfi_overcorrection(iteration_results),
            self.check_anthropocentric_bias(iteration_results),
            self.check_temporal_resolution(iteration_results),
            self.check_catalog_completeness(iteration_results),
            self.check_frequency_coverage(iteration_results),
            self.check_slow_changes(iteration_results),
        ]

        n_green = sum(1 for c in checks if c.status == GREEN)
        n_yellow = sum(1 for c in checks if c.status == YELLOW)
        n_red = sum(1 for c in checks if c.status == RED)

        if n_red > 0:
            overall = RED
        elif n_yellow > 0:
            overall = YELLOW
        else:
            overall = GREEN

        action_items = [
            c.recommendation for c in checks
            if c.status == RED and c.recommendation
        ]

        report = DiagnosisReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            iteration=iteration,
            checks=checks,
            n_green=n_green,
            n_yellow=n_yellow,
            n_red=n_red,
            overall_status=overall,
            action_items=action_items,
        )

        self._history.append(report)

        log.info(
            "Diagnostics complete: %d GREEN, %d YELLOW, %d RED -> %s",
            n_green, n_yellow, n_red, overall,
        )

        return report

    # ─── Individual checks ────────────────────────────────────────────

    def check_threshold_sensitivity(
        self, results: Dict[str, Any],
    ) -> DiagnosticResult:
        """Check 2.1: Are thresholds too aggressive?

        Re-run candidate counts at 2sigma, 3sigma, 5sigma.  If >100x more
        candidates at 2sigma vs 3sigma, warn that we're threshold-sensitive.
        """
        anomaly_counts = results.get("anomaly_counts", {})

        if not anomaly_counts:
            # Simulate from available data
            n_at_current = results.get("n_anomalies", 10)
            current_sigma = results.get("current_thresholds", {}).get("anomaly_sigma", 3.0)

            # Estimate counts at different thresholds (Gaussian tail scaling)
            from scipy.stats import norm
            counts = {}
            for sigma in [2.0, 3.0, 5.0]:
                # Ratio of Gaussian tail probabilities
                ratio = norm.sf(sigma) / norm.sf(current_sigma)
                counts[sigma] = max(int(n_at_current * ratio / norm.sf(current_sigma) * norm.sf(sigma)), 0)
                if sigma == current_sigma:
                    counts[sigma] = n_at_current

            anomaly_counts = counts

        n_2sig = anomaly_counts.get(2.0, anomaly_counts.get("2.0", 0))
        n_3sig = anomaly_counts.get(3.0, anomaly_counts.get("3.0", 0))
        n_5sig = anomaly_counts.get(5.0, anomaly_counts.get("5.0", 0))

        details = {
            "n_at_2sigma": n_2sig,
            "n_at_3sigma": n_3sig,
            "n_at_5sigma": n_5sig,
        }

        if n_3sig > 0 and n_2sig > 0:
            ratio = n_2sig / max(n_3sig, 1)
            details["ratio_2sig_3sig"] = ratio

            if ratio > 100:
                return DiagnosticResult(
                    check_id="2.1",
                    check_name="Threshold Sensitivity",
                    status=RED,
                    message=(
                        f"THRESHOLD-SENSITIVE: {n_2sig} candidates at 2sigma vs "
                        f"{n_3sig} at 3sigma (ratio={ratio:.0f}x). Many signals "
                        f"exist just below the detection threshold."
                    ),
                    details=details,
                    recommendation=(
                        "Lower anomaly_sigma by 0.5 and inspect the boundary "
                        "region for missed signals."
                    ),
                )
            elif ratio > 20:
                return DiagnosticResult(
                    check_id="2.1",
                    check_name="Threshold Sensitivity",
                    status=YELLOW,
                    message=f"Moderate threshold sensitivity (ratio={ratio:.0f}x)",
                    details=details,
                )

        return DiagnosticResult(
            check_id="2.1",
            check_name="Threshold Sensitivity",
            status=GREEN,
            message="Threshold sensitivity within acceptable range",
            details=details,
        )

    def check_rfi_overcorrection(
        self, results: Dict[str, Any],
    ) -> DiagnosticResult:
        """Check 2.2: Are we killing real signals with RFI filtering?

        Count signals rejected as RFI.  Flag any that appeared in ON-target
        only (not in OFF-source observations).
        """
        radio_results = results.get("radio_results", [])

        n_rfi_flagged = 0
        n_on_target_only = 0
        n_total_candidates = 0

        for rr in radio_results:
            n_total_candidates += rr.get("n_candidates", 0)
            n_rfi_flagged += rr.get("n_rfi_flagged", 0)
            n_on_target_only += rr.get("n_on_target_only", 0)

        details = {
            "n_total_candidates": n_total_candidates,
            "n_rfi_flagged": n_rfi_flagged,
            "n_on_target_only": n_on_target_only,
        }

        if n_total_candidates == 0:
            return DiagnosticResult(
                check_id="2.2",
                check_name="RFI Over-correction",
                status=GREEN,
                message="No radio candidates to evaluate",
                details=details,
            )

        rfi_fraction = n_rfi_flagged / max(n_total_candidates, 1)
        details["rfi_fraction"] = rfi_fraction

        if rfi_fraction > 0.95:
            return DiagnosticResult(
                check_id="2.2",
                check_name="RFI Over-correction",
                status=RED,
                message=(
                    f"OVER-CORRECTION: {rfi_fraction:.0%} of candidates rejected "
                    f"as RFI. Almost everything is being filtered out."
                ),
                details=details,
                recommendation=(
                    "Review RFI filtering criteria. Create a "
                    "SINGLE_OBSERVATION_CANDIDATES list for signals that "
                    "appeared only in ON-target observations."
                ),
            )
        elif rfi_fraction > 0.80:
            return DiagnosticResult(
                check_id="2.2",
                check_name="RFI Over-correction",
                status=YELLOW,
                message=f"High RFI rejection rate: {rfi_fraction:.0%}",
                details=details,
            )

        if n_on_target_only > 0:
            return DiagnosticResult(
                check_id="2.2",
                check_name="RFI Over-correction",
                status=YELLOW,
                message=(
                    f"{n_on_target_only} signal(s) appeared in ON-target only — "
                    f"flagged for re-observation"
                ),
                details=details,
            )

        return DiagnosticResult(
            check_id="2.2",
            check_name="RFI Over-correction",
            status=GREEN,
            message=f"RFI rejection rate: {rfi_fraction:.0%} (acceptable)",
            details=details,
        )

    def check_anthropocentric_bias(
        self, results: Dict[str, Any],
    ) -> DiagnosticResult:
        """Check 2.3: Are we only finding what we expect?

        Compare named detector results vs unsupervised anomaly stacking.
        If unsupervised finds targets that NO named detector flagged,
        highlight them.
        """
        all_results = results.get("all_results", [])
        named_detections = set()
        unsupervised_detections = set()

        for r in all_results:
            target_id = r.get("target_id", "")
            if r.get("named_detector_flag"):
                named_detections.add(target_id)
            if r.get("unsupervised_flag"):
                unsupervised_detections.add(target_id)

        # Targets found by unsupervised but NOT by any named detector
        unbiased_discoveries = unsupervised_detections - named_detections
        n_unbiased = len(unbiased_discoveries)

        details = {
            "n_named": len(named_detections),
            "n_unsupervised": len(unsupervised_detections),
            "n_unbiased_discoveries": n_unbiased,
            "unbiased_targets": list(unbiased_discoveries)[:10],
        }

        if n_unbiased > 0:
            return DiagnosticResult(
                check_id="2.3",
                check_name="Anthropocentric Bias",
                status=YELLOW,
                message=(
                    f"FOUND {n_unbiased} target(s) flagged by unsupervised analysis "
                    f"but NOT by any named detector. These are the MOST INTERESTING "
                    f"outputs — they represent anomalies we didn't anticipate."
                ),
                details=details,
                recommendation=(
                    "Investigate unbiased discoveries: "
                    + ", ".join(list(unbiased_discoveries)[:5])
                ),
            )

        if not all_results:
            return DiagnosticResult(
                check_id="2.3",
                check_name="Anthropocentric Bias",
                status=GREEN,
                message="No results to evaluate for bias",
                details=details,
            )

        return DiagnosticResult(
            check_id="2.3",
            check_name="Anthropocentric Bias",
            status=GREEN,
            message="All detections accounted for by named detectors",
            details=details,
        )

    def check_temporal_resolution(
        self, results: Dict[str, Any],
    ) -> DiagnosticResult:
        """Check 2.4: Are we averaging out transients?

        Compare detections at different time resolutions.  Flag any signal
        found at short timescale but not long.
        """
        ts_results = results.get("timeseries_results", [])

        n_transient_only = 0
        transient_targets = []

        for tr in ts_results:
            if tr.get("resolution_discrepant"):
                transients = tr.get("transient_only", [])
                if transients:
                    n_transient_only += len(transients)
                    transient_targets.extend(
                        t.get("target_id", "?") for t in transients
                    )

        details = {
            "n_timeseries_analyzed": len(ts_results),
            "n_transient_only": n_transient_only,
            "transient_targets": transient_targets[:10],
        }

        if n_transient_only > 0:
            return DiagnosticResult(
                check_id="2.4",
                check_name="Temporal Resolution",
                status=RED,
                message=(
                    f"TRANSIENT SIGNALS BEING LOST: {n_transient_only} signal(s) "
                    f"detected at fine time resolution but VANISH when time-averaged. "
                    f"Standard analysis would miss these."
                ),
                details=details,
                recommendation=(
                    "Run multi-resolution analysis on ALL radio and lightcurve data. "
                    "Report transient-only signals separately."
                ),
            )

        return DiagnosticResult(
            check_id="2.4",
            check_name="Temporal Resolution",
            status=GREEN,
            message="No transient-only signals detected",
            details=details,
        )

    def check_catalog_completeness(
        self, results: Dict[str, Any],
    ) -> DiagnosticResult:
        """Check 2.5: Are we missing uncataloged sources?

        In temporal archaeology, count sources with no catalog match.
        Flag catalog orphans co-located with exoplanet hosts.
        """
        xm_results = results.get("crossmatch_results", [])

        n_orphans = 0
        n_orphans_near_hosts = 0
        total_sources = 0

        for xm in xm_results:
            total_sources += xm.get("n_sources", 0)
            n_orphans += xm.get("n_no_catalog_match", 0)
            n_orphans_near_hosts += xm.get("n_orphans_near_hosts", 0)

        details = {
            "total_sources": total_sources,
            "n_orphans": n_orphans,
            "n_orphans_near_hosts": n_orphans_near_hosts,
        }

        if n_orphans_near_hosts > 0:
            return DiagnosticResult(
                check_id="2.5",
                check_name="Catalog Completeness",
                status=YELLOW,
                message=(
                    f"{n_orphans_near_hosts} uncataloged source(s) found near "
                    f"exoplanet hosts. These may be new or transient sources."
                ),
                details=details,
                recommendation="Cross-reference orphan sources against deep survey data",
            )

        orphan_fraction = n_orphans / max(total_sources, 1)
        if orphan_fraction > 0.30:
            return DiagnosticResult(
                check_id="2.5",
                check_name="Catalog Completeness",
                status=YELLOW,
                message=f"High orphan fraction: {orphan_fraction:.0%} uncataloged",
                details=details,
            )

        return DiagnosticResult(
            check_id="2.5",
            check_name="Catalog Completeness",
            status=GREEN,
            message="Catalog coverage adequate",
            details=details,
        )

    def check_frequency_coverage(
        self, results: Dict[str, Any],
    ) -> DiagnosticResult:
        """Check 2.6: Are we only searching expected frequencies?

        Report what fraction of available bandwidth we actually searched.
        Flag any bands excluded by RFI filtering.
        """
        radio_results = results.get("radio_results", [])

        total_bandwidth_mhz = 0
        searched_bandwidth_mhz = 0
        excluded_bands = []

        for rr in radio_results:
            bw = rr.get("bandwidth_mhz", 0)
            searched = rr.get("searched_bandwidth_mhz", bw)
            excluded = rr.get("excluded_bands_mhz", [])
            total_bandwidth_mhz += bw
            searched_bandwidth_mhz += searched
            excluded_bands.extend(excluded)

        details = {
            "total_bandwidth_mhz": total_bandwidth_mhz,
            "searched_bandwidth_mhz": searched_bandwidth_mhz,
            "n_excluded_bands": len(excluded_bands),
        }

        if total_bandwidth_mhz > 0:
            coverage = searched_bandwidth_mhz / total_bandwidth_mhz
            details["coverage_fraction"] = coverage

            if coverage < 0.50:
                return DiagnosticResult(
                    check_id="2.6",
                    check_name="Frequency Coverage",
                    status=RED,
                    message=(
                        f"LOW COVERAGE: Only {coverage:.0%} of bandwidth searched. "
                        f"{len(excluded_bands)} band(s) excluded by RFI filtering."
                    ),
                    details=details,
                    recommendation=(
                        "Review RFI exclusion zones. Consider searching excluded "
                        "bands with relaxed thresholds."
                    ),
                )
            elif coverage < 0.80:
                return DiagnosticResult(
                    check_id="2.6",
                    check_name="Frequency Coverage",
                    status=YELLOW,
                    message=f"Moderate coverage: {coverage:.0%} of bandwidth",
                    details=details,
                )

        return DiagnosticResult(
            check_id="2.6",
            check_name="Frequency Coverage",
            status=GREEN,
            message="Frequency coverage adequate",
            details=details,
        )

    def check_slow_changes(
        self, results: Dict[str, Any],
    ) -> DiagnosticResult:
        """Check 2.7: Are we missing gradual changes?

        Compute flux ratios (not just present/absent).  Flag sources with
        >50% change between survey epochs.
        """
        temporal_results = results.get("temporal_results", [])

        n_slow_changes = 0
        slow_change_targets = []

        for tr in temporal_results:
            flux_ratios = tr.get("flux_ratios", [])
            for ratio_info in flux_ratios:
                ratio = ratio_info.get("ratio", 1.0)
                if abs(ratio - 1.0) > 0.50:  # >50% change
                    n_slow_changes += 1
                    slow_change_targets.append(ratio_info.get("source_id", "?"))

        details = {
            "n_temporal_sources": len(temporal_results),
            "n_slow_changes": n_slow_changes,
            "slow_change_targets": slow_change_targets[:10],
        }

        if n_slow_changes > 0:
            return DiagnosticResult(
                check_id="2.7",
                check_name="Slow Changes",
                status=YELLOW,
                message=(
                    f"{n_slow_changes} source(s) with >50% flux change between "
                    f"survey epochs. These gradual changes may be missed by "
                    f"present/absent detection alone."
                ),
                details=details,
                recommendation="Compute flux ratios for all temporal archaeology sources",
            )

        return DiagnosticResult(
            check_id="2.7",
            check_name="Slow Changes",
            status=GREEN,
            message="No significant slow changes detected",
            details=details,
        )

    # ─── Reporting ────────────────────────────────────────────────────

    def generate_diagnosis_report(self) -> str:
        """Produce human-readable report of all checks.

        Returns the most recent report as a formatted string.
        """
        if not self._history:
            return "No diagnostics have been run yet."

        report = self._history[-1]
        lines = [
            "=" * 60,
            "  EXODUS SELF-DIAGNOSIS REPORT",
            f"  Iteration {report.iteration}",
            f"  {report.timestamp}",
            "=" * 60,
            "",
            f"  Overall Status: {report.overall_status}",
            f"  GREEN: {report.n_green}  YELLOW: {report.n_yellow}  RED: {report.n_red}",
            "",
        ]

        for check in report.checks:
            icon = {"GREEN": "[OK]", "YELLOW": "[!!]", "RED": "[XX]"}[check.status]
            lines.append(f"  {icon} Check {check.check_id}: {check.check_name}")
            lines.append(f"      Status: {check.status}")
            lines.append(f"      {check.message}")
            if check.recommendation:
                lines.append(f"      -> {check.recommendation}")
            lines.append("")

        if report.action_items:
            lines.append("  ACTION ITEMS:")
            for i, action in enumerate(report.action_items, 1):
                lines.append(f"    {i}. {action}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def get_history(self) -> List[DiagnosisReport]:
        """Return all diagnosis reports."""
        return list(self._history)

    def save_report(self, path: Optional[Path] = None) -> Path:
        """Save the latest report to disk."""
        if not self._history:
            raise ValueError("No diagnostics have been run yet")

        if path is None:
            path = PROJECT_ROOT / "data" / "reports" / "self_diagnosis.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "reports": [r.to_dict() for r in self._history],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        log.info("Diagnosis report saved to %s", path)
        return path


# =====================================================================
#  CLI demo / self-test
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Project EXODUS -- Self-Diagnosis Monitor Demo")
    print("=" * 70)

    diag = SelfDiagnostics()

    # ── Test 1: Healthy system ────────────────────────────────────────
    print("\n[1] Healthy system (all checks should be GREEN)")
    print("-" * 50)

    healthy_results = {
        "iteration": 1,
        "n_anomalies": 15,
        "current_thresholds": {"anomaly_sigma": 3.0},
        "anomaly_counts": {2.0: 45, 3.0: 15, 5.0: 3},
        "radio_results": [
            {"n_candidates": 20, "n_rfi_flagged": 8, "n_on_target_only": 0,
             "bandwidth_mhz": 500, "searched_bandwidth_mhz": 450},
        ],
        "all_results": [],
        "timeseries_results": [],
        "crossmatch_results": [{"n_sources": 100, "n_no_catalog_match": 5, "n_orphans_near_hosts": 0}],
        "temporal_results": [],
    }

    report1 = diag.run_all_checks(healthy_results)
    print(diag.generate_diagnosis_report())

    # ── Test 2: Misconfigured system (should trigger RED warnings) ───
    print("\n[2] Misconfigured system (aggressive thresholds + RFI)")
    print("-" * 50)

    bad_results = {
        "iteration": 2,
        "n_anomalies": 2,
        "current_thresholds": {"anomaly_sigma": 10.0},
        "anomaly_counts": {2.0: 500, 3.0: 2, 5.0: 0},
        "radio_results": [
            {"n_candidates": 100, "n_rfi_flagged": 98, "n_on_target_only": 2,
             "bandwidth_mhz": 1000, "searched_bandwidth_mhz": 300,
             "excluded_bands_mhz": ["1200-1400", "2300-2500"]},
        ],
        "all_results": [],
        "timeseries_results": [
            {"resolution_discrepant": True,
             "transient_only": [{"target_id": "TRANSIENT_001"}]},
        ],
        "crossmatch_results": [],
        "temporal_results": [],
    }

    report2 = diag.run_all_checks(bad_results)
    print(diag.generate_diagnosis_report())

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  Test 1 (healthy):       {report1.overall_status} "
          f"(G={report1.n_green} Y={report1.n_yellow} R={report1.n_red})")
    print(f"  Test 2 (misconfigured): {report2.overall_status} "
          f"(G={report2.n_green} Y={report2.n_yellow} R={report2.n_red})")
    print(f"  >> {'PASS' if report1.overall_status == GREEN else 'FAIL'}: Healthy system green")
    print(f"  >> {'PASS' if report2.n_red >= 2 else 'FAIL'}: Misconfigured triggers RED")
    print("=" * 70)
    print("  Done.")
    print("=" * 70)
