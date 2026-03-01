#!/usr/bin/env python3
"""
Project EXODUS -- Calibration Campaign Runner
==============================================

Runs the pipeline on calibration targets (known positives + negatives)
and validates that detection channels produce expected results.

This is Phase 0 of the EXODUS Research Campaign. It MUST pass before
any science run proceeds.

Usage
-----
    python scripts/run_calibration.py
    python scripts/run_calibration.py --target-file data/targets/calibration_set.json
    python scripts/run_calibration.py --strict    # exit code 1 on any failure
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# -- Ensure project root is on sys.path --------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# -- Project utilities --------------------------------------------------------
from src.utils import get_logger, safe_json_dump, PROJECT_ROOT as _PR

log = get_logger("calibration_runner")

# -- Graceful imports ---------------------------------------------------------
try:
    from scripts.run_quick import QuickRunner
except ImportError:
    try:
        # Fallback: direct import when running from project root
        sys.path.insert(0, str(SCRIPT_DIR))
        from run_quick import QuickRunner
    except ImportError as e:
        log.error("Cannot import QuickRunner: %s", e)
        QuickRunner = None

try:
    from src.output.campaign_report import (
        generate_calibration_report,
        CalibrationReport,
    )
except ImportError as e:
    log.error("Cannot import campaign_report: %s", e)
    generate_calibration_report = None
    CalibrationReport = None

try:
    from src.ingestion.target_loader import load_target_file, CampaignTargets
except ImportError as e:
    log.error("Cannot import target_loader: %s", e)
    load_target_file = None
    CampaignTargets = None


# =====================================================================
#  Constants
# =====================================================================

DEFAULT_TARGET_FILE = PROJECT_ROOT / "data" / "targets" / "calibration_set.json"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"


# =====================================================================
#  Calibration runner
# =====================================================================

def run_calibration(
    target_file: str | Path | None = None,
    strict: bool = True,
    max_targets: int = 50,
) -> Dict[str, Any]:
    """Execute the calibration campaign.

    Parameters
    ----------
    target_file : str or Path, optional
        Path to calibration target JSON.  Defaults to
        ``data/targets/calibration_set.json``.
    strict : bool
        If True (default), the function returns ``exit_code=1`` in the result
        dict when any positive control fails or any negative control
        false-alarms.  Phase 0 calibration MUST pass before science runs.
    max_targets : int
        Maximum number of targets to process (default: 50).

    Returns
    -------
    dict
        Summary including calibration report, timing, and exit code.
    """
    start = time.time()

    # Resolve target file
    if target_file is None:
        target_file = DEFAULT_TARGET_FILE
    target_file = Path(target_file)

    log.info("=" * 60)
    log.info("  EXODUS Calibration Campaign Runner")
    log.info("=" * 60)
    log.info("Target file: %s", target_file)
    log.info("Strict mode: %s", strict)

    # ------------------------------------------------------------------
    # Step 1: Load campaign targets
    # ------------------------------------------------------------------
    if load_target_file is None:
        log.error("target_loader not available. Cannot proceed.")
        return {"error": "target_loader unavailable", "exit_code": 2}

    try:
        campaign_targets = load_target_file(target_file)
    except FileNotFoundError:
        log.error("Calibration target file not found: %s", target_file)
        return {"error": f"File not found: {target_file}", "exit_code": 2}
    except Exception:
        log.error("Failed to load target file:\n%s", traceback.format_exc())
        return {"error": "Failed to load targets", "exit_code": 2}

    log.info("Campaign: %s", campaign_targets.campaign)
    log.info("Targets:  %d", campaign_targets.n_targets)
    log.info("Controls: %d positive, %d negative",
             len(campaign_targets.positive_controls),
             len(campaign_targets.negative_controls))

    # ------------------------------------------------------------------
    # Step 2: Run pipeline via QuickRunner
    # ------------------------------------------------------------------
    if QuickRunner is None:
        log.error("QuickRunner not available. Cannot proceed.")
        return {"error": "QuickRunner unavailable", "exit_code": 2}

    log.info("Running EXODUS pipeline on calibration targets ...")

    runner = QuickRunner(
        max_targets=max_targets,
        run_hypotheses=False,
        target_file=str(target_file),
        tier=2,  # Calibration must evaluate ALL channels including radio
    )

    try:
        pipeline_summary = runner.run()
    except Exception:
        log.error("Pipeline run failed:\n%s", traceback.format_exc())
        return {"error": "Pipeline run failed", "exit_code": 2}

    if "error" in pipeline_summary:
        log.error("Pipeline returned error: %s", pipeline_summary["error"])
        return {"error": pipeline_summary["error"], "exit_code": 2}

    # ------------------------------------------------------------------
    # Step 3: Extract scored results from runner
    # ------------------------------------------------------------------
    # The QuickRunner stores targets internally; we retrieve them by
    # re-loading from the runner's campaign_metadata and checking the
    # pipeline results.  The targets list in the runner contains the
    # scored exodus_score dictionaries.
    results = _extract_scored_targets(runner)

    if not results:
        log.warning("No scored results to validate. Pipeline may have failed silently.")
        return {
            "error": "No scored results",
            "exit_code": 2,
            "pipeline_summary": pipeline_summary,
        }

    log.info("Extracted %d scored targets.", len(results))

    # ------------------------------------------------------------------
    # Step 4: Generate calibration report
    # ------------------------------------------------------------------
    if generate_calibration_report is None:
        log.error("campaign_report module not available.")
        return {"error": "campaign_report unavailable", "exit_code": 2}

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cal_report = generate_calibration_report(
        results, campaign_targets, output_dir=REPORTS_DIR,
    )

    # ------------------------------------------------------------------
    # Step 5: Print pass/fail matrix to stdout
    # ------------------------------------------------------------------
    _print_pass_fail_matrix(cal_report)

    # ------------------------------------------------------------------
    # Step 6: Save summary report
    # ------------------------------------------------------------------
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary_path = REPORTS_DIR / f"calibration_{ts}.json"
    elapsed = time.time() - start

    summary = {
        "project": "EXODUS",
        "mode": "calibration",
        "campaign": cal_report.campaign,
        "n_targets": cal_report.n_targets,
        "n_true_positive": cal_report.n_true_positive,
        "n_true_negative": cal_report.n_true_negative,
        "n_false_positive": cal_report.n_false_positive,
        "n_false_negative": cal_report.n_false_negative,
        "pass_rate": cal_report.pass_rate,
        "all_passed": cal_report.all_passed,
        "strict": strict,
        "exit_code": 0 if cal_report.all_passed or not strict else 1,
        "elapsed_sec": round(elapsed, 1),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "pipeline_summary": pipeline_summary,
    }

    with open(summary_path, "w") as f:
        safe_json_dump(summary, f, indent=2)
    log.info("Calibration summary saved: %s", summary_path)

    return summary


def _extract_scored_targets(runner: Any) -> List[Dict[str, Any]]:
    """Extract scored target dicts from a QuickRunner after execution.

    The QuickRunner stores scored targets on ``runner.scored_targets``
    after processing.  Each target dict contains ``exodus_score``,
    ``ir_excess``, and other channel results.
    """
    # Primary: access the scored_targets list stored during run()
    scored_targets = getattr(runner, "scored_targets", None)
    if scored_targets:
        return scored_targets

    # Fallback 1: try the scorer's top targets
    if runner.scorer is not None:
        try:
            all_scored = runner.scorer.get_top_targets(n=999)
            scored = []
            for s in all_scored:
                scored.append(s.to_dict() if hasattr(s, "to_dict") else s)
            if scored:
                return scored
        except Exception as exc:
            log.debug("Could not extract from scorer: %s", exc)

    # Fallback 2: try the runner's most recent report file
    try:
        reports_dir = runner.REPORTS_DIR
        latest = sorted(reports_dir.glob("quick_run_*.json"))
        if latest:
            with open(latest[-1]) as f:
                data = json.load(f)
            top = data.get("top_targets", [])
            if top:
                return top
    except Exception as exc:
        log.debug("Could not extract from report file: %s", exc)

    return []


def _print_pass_fail_matrix(report: CalibrationReport) -> None:
    """Print a clear pass/fail matrix to stdout."""
    print()
    print("=" * 70)
    print("  EXODUS CALIBRATION RESULTS")
    print("=" * 70)
    print()

    # Summary counts
    print(f"  Campaign:        {report.campaign}")
    print(f"  Targets:         {report.n_targets}")
    print(f"  True Positives:  {report.n_true_positive}")
    print(f"  True Negatives:  {report.n_true_negative}")
    print(f"  False Positives: {report.n_false_positive}")
    print(f"  False Negatives: {report.n_false_negative}")
    print(f"  Pass Rate:       {report.pass_rate:.1%}")
    print()

    # Per-target matrix
    print("  Per-Target Channel Matrix:")
    print("  " + "-" * 66)
    print(f"  {'Target':<24s} {'Channel':<30s} {'Exp':>4s} {'Score':>7s} {'Result'}")
    print("  " + "-" * 66)

    for entry in report.per_target:
        tid = entry["target_id"]
        first = True
        for ch in entry["channels"]:
            if ch["expected"] == "neutral" and ch["score"] is None:
                continue

            target_col = tid if first else ""
            first = False

            score_str = f"{ch['score']:.3f}" if ch['score'] is not None else "  N/A"
            outcome = ch["outcome"]

            # Mark failures
            marker = ""
            if outcome == "FALSE_POSITIVE":
                marker = "  << FALSE ALARM"
            elif outcome == "FALSE_NEGATIVE":
                marker = "  << MISSED"

            print(
                f"  {target_col:<24s} {ch['channel']:<30s} "
                f"{ch['expected']:>4s} {score_str:>7s} "
                f"{outcome}{marker}"
            )

    print("  " + "-" * 66)
    print()

    # Overall verdict
    if report.all_passed:
        print("  VERDICT: ALL CALIBRATION CHECKS PASSED")
    else:
        failures = report.n_false_positive + report.n_false_negative
        print(f"  VERDICT: CALIBRATION FAILED ({failures} failure(s))")
        if report.n_false_positive > 0:
            print(f"    - {report.n_false_positive} false positive(s): "
                  "negative controls triggered unexpectedly")
        if report.n_false_negative > 0:
            print(f"    - {report.n_false_negative} false negative(s): "
                  "positive controls failed to trigger")

    print()
    print("=" * 70)


# =====================================================================
#  CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project EXODUS -- Calibration Campaign Runner (Phase 0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs the EXODUS pipeline on calibration targets with known
expected behavior, then validates that each detection channel produces
the correct result.

Examples:
  python scripts/run_calibration.py
  python scripts/run_calibration.py --target-file data/targets/calibration_set.json
  python scripts/run_calibration.py --strict
  python scripts/run_calibration.py --strict --max-targets 10
        """,
    )
    parser.add_argument(
        "--target-file", "-f",
        type=str,
        default=None,
        help="Path to the calibration target JSON file.  "
             f"Default: {DEFAULT_TARGET_FILE}",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        default=False,
        help="Disable strict mode (allow calibration to pass despite failures). "
             "By default, calibration exits with code 1 on any failure.",
    )
    parser.add_argument(
        "--max-targets", "-n",
        type=int,
        default=50,
        help="Maximum number of targets to process (default: 50).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    summary = run_calibration(
        target_file=args.target_file,
        strict=not args.no_strict,
        max_targets=args.max_targets,
    )

    exit_code = summary.get("exit_code", 0)
    if exit_code != 0 and summary.get("error"):
        print(f"\n  ERROR: {summary['error']}")

    elapsed = summary.get("elapsed_sec", 0)
    if elapsed:
        print(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
