#!/usr/bin/env python3
"""
EXODUS Pipeline Watcher — monitors running chains, auto-triggers analysis.

Watches two pipeline chains:
  Chain 1 (calibration): binary→disk→YSO→giant→VASCO 28→VASCO 127→SIMBAD 391
  Chain 2 (rerun):       SmartTargets 500→EXODUS-500 500

On completion of each chain, automatically runs:
  - analyze_calibration.py (when calibration finishes)
  - escalate.py (when rerun finishes)
  - VASCO discovery scan (when VASCO steps finish)
  - Old vs new results comparison

Usage:
  nohup ./venv/bin/python scripts/watcher.py > data/reports/watcher_log.txt 2>&1 &
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PYTHON = str(Path(__file__).resolve().parent.parent / "venv" / "bin" / "python")
REPORT_DIR = Path("data/reports")
TARGET_DIR = Path("data/targets")

# ── Chain definitions ────────────────────────────────────────────
# PIDs are passed via CLI or auto-detected from running processes
CALIBRATION_PID = int(os.environ.get("EXODUS_CALIBRATION_PID", 0))
RERUN_PID = int(os.environ.get("EXODUS_RERUN_PID", 0))

# Expected target files in order for each chain
CALIBRATION_STEPS = [
    ("binary", "calibration_binary.json", 500),
    ("disk", "calibration_disk.json", 500),
    ("yso", "calibration_yso.json", 500),
    ("giant", "calibration_giant.json", 456),
]

# Novel target campaigns (signal hunting, NOT calibration)
# Run separately via scripts/run_novel_targets.sh
# VASCO_28/127: vanishing stars (USNO B1.0 missing from Pan-STARRS)
# SIMBAD_391: YSO/WD/X-ray anomalous objects
NOVEL_CAMPAIGNS = [
    ("vasco_28", "vasco_vanishing_28.json", 28),
    ("vasco_127", "vasco_vanishing_127.json", 127),
    ("simbad", "simbad_anomalous_objects.json", 391),
]

RERUN_STEPS = [
    ("smart_targets", "smart_targets.json", 500),
    ("exodus_500", None, 500),  # Uses --targets 500, no target file
]

# Poll interval (seconds)
POLL_INTERVAL = 120


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def pid_alive(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def find_latest_report(target_file: Optional[str] = None, min_targets: int = 10) -> Optional[Path]:
    """Find the most recent quick_run report matching criteria."""
    reports = sorted(REPORT_DIR.glob("quick_run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    for rpath in reports:
        try:
            with open(rpath) as f:
                data = json.load(f)
            tf = data.get("target_file", "")
            n = data.get("n_targets", 0)

            if target_file and target_file not in tf:
                continue
            if n < min_targets:
                continue

            return rpath
        except Exception:
            continue
    return None


def find_reports_since(since_mtime: float) -> List[Path]:
    """Find all quick_run reports created after a given timestamp."""
    reports = []
    for rpath in REPORT_DIR.glob("quick_run_*.json"):
        if rpath.stat().st_mtime > since_mtime:
            reports.append(rpath)
    return sorted(reports, key=lambda p: p.stat().st_mtime)


def scan_vasco_discovery(report_path: Path) -> List[Dict]:
    """Scan a VASCO report for discovery signals: IR/radio WITHOUT optical."""
    discoveries = []
    try:
        with open(report_path) as f:
            data = json.load(f)

        targets = data.get("all_scored", data.get("top_targets", []))
        for t in targets:
            target_id = t.get("target_id", "unknown")
            channels = t.get("channel_scores", {})

            ir = channels.get("ir_excess", {})
            ir_score = ir.get("score", 0.0)
            ir_var = channels.get("ir_variability", {})
            ir_var_score = ir_var.get("score", 0.0)

            # Check for IR detection
            has_ir = ir_score > 0.3 or ir_var_score > 0.3

            # Check unexplainability
            unex = t.get("unexplainability_score", 0.0)

            # Multi-channel convergence
            n_active = t.get("n_active_channels", 0)

            # FDR significance
            fdr_sig = t.get("fdr_significant", False)

            # Discovery criteria for VASCO:
            # IR or IR-variability detection at a vanished-star position
            if has_ir:
                discoveries.append({
                    "target_id": target_id,
                    "ir_score": ir_score,
                    "ir_var_score": ir_var_score,
                    "n_active": n_active,
                    "total_score": t.get("total_score", 0.0),
                    "unexplainability": unex,
                    "fdr_significant": fdr_sig,
                    "distance_pc": t.get("distance_pc"),
                    "ra": t.get("ra"),
                    "dec": t.get("dec"),
                })

    except Exception as e:
        log(f"  ERROR scanning VASCO report: {e}")

    return discoveries


def scan_for_candidates(report_path: Path) -> List[Dict]:
    """Scan any report for genuine candidate signals."""
    candidates = []
    try:
        with open(report_path) as f:
            data = json.load(f)

        targets = data.get("all_scored", data.get("top_targets", []))
        for t in targets:
            n_active = t.get("n_active_channels", 0)
            unex = t.get("unexplainability_score", 0.0)
            fdr_sig = t.get("fdr_significant", False)
            total_score = t.get("total_score", 0.0)

            # Genuine candidate: multi-channel + unexplained + FDR significant
            if n_active >= 2 and (unex > 0.5 or fdr_sig):
                candidates.append({
                    "target_id": t.get("target_id", "unknown"),
                    "total_score": total_score,
                    "n_active": n_active,
                    "unexplainability": unex,
                    "fdr_significant": fdr_sig,
                    "distance_pc": t.get("distance_pc"),
                })

    except Exception as e:
        log(f"  ERROR scanning report: {e}")

    return candidates


def compare_with_old_results(new_report: Path, population: str) -> Optional[Dict]:
    """Compare new results with the most recent older report for same population."""
    reports = sorted(REPORT_DIR.glob("quick_run_*.json"), key=lambda p: p.stat().st_mtime)

    old_report = None
    for rpath in reports:
        if rpath == new_report:
            continue
        try:
            with open(rpath) as f:
                data = json.load(f)
            tf = data.get("target_file", "")
            if population == "exodus_500" and "--targets 500" not in tf and "smart_targets" not in tf:
                # EXODUS-500 uses --targets, not a target file, so match by n_targets
                if data.get("n_targets", 0) == 500 and "calibration" not in tf:
                    old_report = rpath
            elif population != "exodus_500" and population in tf:
                old_report = rpath
        except Exception:
            continue

    if old_report is None:
        return None

    try:
        with open(old_report) as f:
            old_data = json.load(f)
        with open(new_report) as f:
            new_data = json.load(f)

        old_targets = {t.get("target_id"): t for t in old_data.get("all_scored", old_data.get("top_targets", []))}
        new_targets = {t.get("target_id"): t for t in new_data.get("all_scored", new_data.get("top_targets", []))}

        common = set(old_targets.keys()) & set(new_targets.keys())
        if not common:
            return {"note": "No overlapping targets found"}

        score_diffs = []
        for tid in common:
            old_score = old_targets[tid].get("total_score", 0.0)
            new_score = new_targets[tid].get("total_score", 0.0)
            score_diffs.append(new_score - old_score)

        import numpy as np
        diffs = np.array(score_diffs)

        return {
            "old_report": str(old_report),
            "new_report": str(new_report),
            "common_targets": len(common),
            "mean_score_change": float(np.mean(diffs)),
            "median_score_change": float(np.median(diffs)),
            "max_increase": float(np.max(diffs)),
            "max_decrease": float(np.min(diffs)),
            "n_improved": int(np.sum(diffs > 0.01)),
            "n_degraded": int(np.sum(diffs < -0.01)),
            "n_unchanged": int(np.sum(np.abs(diffs) <= 0.01)),
        }
    except Exception as e:
        return {"error": str(e)}


def run_analysis(analysis_type: str, **kwargs) -> bool:
    """Run an analysis script and return success status."""
    try:
        if analysis_type == "calibration":
            cmd = [PYTHON, "scripts/analyze_calibration.py"]
            log("  Running: analyze_calibration.py")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                log("  Calibration analysis COMPLETE")
                # Print key lines
                for line in result.stdout.split("\n"):
                    if any(kw in line for kw in ["WARNING", "PASS", "inflation", "rho", "FDR"]):
                        log(f"    {line.strip()}")
                return True
            else:
                log(f"  Calibration analysis FAILED: {result.stderr[:200]}")
                return False

        elif analysis_type == "escalate":
            report = kwargs.get("report")
            if not report:
                return False
            cmd = [PYTHON, "scripts/escalate.py", "--report", str(report), "--top", "50"]
            log(f"  Running: escalate.py --report {report}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                log("  Escalation COMPLETE")
                for line in result.stdout.split("\n")[-10:]:
                    if line.strip():
                        log(f"    {line.strip()}")
                return True
            else:
                log(f"  Escalation FAILED: {result.stderr[:200]}")
                return False

    except subprocess.TimeoutExpired:
        log(f"  {analysis_type} analysis TIMED OUT")
        return False
    except Exception as e:
        log(f"  {analysis_type} analysis ERROR: {e}")
        return False


def main():
    log("=" * 70)
    log("  EXODUS Pipeline Watcher")
    log(f"  Calibration chain PID: {CALIBRATION_PID}")
    log(f"  Rerun chain PID: {RERUN_PID}")
    log(f"  Poll interval: {POLL_INTERVAL}s")
    log("=" * 70)

    # Track state
    start_time = time.time()
    cal_alive = pid_alive(CALIBRATION_PID)
    rerun_alive = pid_alive(RERUN_PID)
    cal_done = not cal_alive
    rerun_done = not rerun_alive
    cal_analyzed = False
    rerun_analyzed = False

    # Track which reports we've already seen
    seen_reports = set(str(p) for p in REPORT_DIR.glob("quick_run_*.json"))

    if not cal_alive and not rerun_alive:
        log("  WARNING: Both PIDs already dead. Running post-mortem analysis.")

    # Main poll loop
    while True:
        # Check for new reports
        current_reports = set(str(p) for p in REPORT_DIR.glob("quick_run_*.json"))
        new_reports = current_reports - seen_reports

        if new_reports:
            for rpath_str in sorted(new_reports):
                rpath = Path(rpath_str)
                log(f"\n  NEW REPORT DETECTED: {rpath.name}")

                try:
                    with open(rpath) as f:
                        data = json.load(f)
                    tf = data.get("target_file", "")
                    n = data.get("n_targets", 0)
                    log(f"    Target file: {tf}")
                    log(f"    N targets: {n}")

                    # Identify which step this is
                    step_name = "unknown"
                    for name, tfile, expected_n in CALIBRATION_STEPS + NOVEL_CAMPAIGNS + RERUN_STEPS:
                        if tfile and tfile in tf:
                            step_name = name
                            break
                    if step_name == "unknown" and n == 500 and "calibration" not in tf:
                        step_name = "exodus_500"

                    log(f"    Step: {step_name}")

                    # Run step-specific analysis
                    if step_name.startswith("vasco"):
                        log(f"\n  VASCO DISCOVERY SCAN — {step_name}")
                        discoveries = scan_vasco_discovery(rpath)
                        if discoveries:
                            log(f"  *** {len(discoveries)} IR DETECTION(S) AT VANISHED-STAR POSITIONS ***")
                            for d in discoveries:
                                log(f"    {d['target_id']}: IR={d['ir_score']:.2f}, "
                                    f"IR_var={d['ir_var_score']:.2f}, "
                                    f"n_active={d['n_active']}, "
                                    f"unexplain={d['unexplainability']:.2f}")
                                if d["unexplainability"] > 0.5:
                                    log(f"    *** UNEXPLAINED VASCO TARGET — CANDIDATE FOR TIER 1 ***")
                        else:
                            log(f"  No IR detections at VASCO positions (expected for plate artifacts)")

                    elif step_name == "simbad":
                        log(f"\n  SIMBAD ANOMALY SCAN")
                        candidates = scan_for_candidates(rpath)
                        if candidates:
                            log(f"  *** {len(candidates)} MULTI-CHANNEL CANDIDATE(S) ***")
                            for c in candidates:
                                log(f"    {c['target_id']}: score={c['total_score']:.3f}, "
                                    f"n_active={c['n_active']}, "
                                    f"unexplain={c['unexplainability']:.2f}, "
                                    f"fdr_sig={c['fdr_significant']}")
                        else:
                            log(f"  No multi-channel candidates in SIMBAD sample")

                    # General candidate scan for any step
                    if not step_name.startswith("vasco") and step_name != "simbad":
                        candidates = scan_for_candidates(rpath)
                        if candidates:
                            log(f"  {len(candidates)} candidate(s) found in {step_name}")
                            for c in candidates[:5]:
                                log(f"    {c['target_id']}: score={c['total_score']:.3f}")

                    # Compare with old results (for rerun steps)
                    if step_name in ("smart_targets", "exodus_500"):
                        log(f"\n  COMPARING {step_name} with previous run ...")
                        comparison = compare_with_old_results(rpath, step_name)
                        if comparison and "error" not in comparison:
                            log(f"    Common targets: {comparison.get('common_targets', 0)}")
                            log(f"    Mean score change: {comparison.get('mean_score_change', 0):+.4f}")
                            log(f"    Improved: {comparison.get('n_improved', 0)}, "
                                f"Degraded: {comparison.get('n_degraded', 0)}, "
                                f"Unchanged: {comparison.get('n_unchanged', 0)}")
                        elif comparison:
                            log(f"    Comparison: {comparison}")

                except Exception as e:
                    log(f"    ERROR processing report: {e}")

            seen_reports = current_reports

        # Check chain completion
        if not cal_done and not pid_alive(CALIBRATION_PID):
            cal_done = True
            log("\n" + "=" * 70)
            log("  CALIBRATION CHAIN COMPLETED")
            log("=" * 70)

            if not cal_analyzed:
                # Run calibration analysis
                log("\n  Running calibration analysis ...")
                run_analysis("calibration")
                cal_analyzed = True

                # Check for VASCO discoveries across both VASCO steps
                for vasco_file in ["vasco_vanishing_28.json", "vasco_vanishing_127.json"]:
                    report = find_latest_report(vasco_file)
                    if report:
                        discoveries = scan_vasco_discovery(report)
                        if discoveries:
                            log(f"\n  *** VASCO {vasco_file}: {len(discoveries)} IR detection(s) ***")

        if not rerun_done and not pid_alive(RERUN_PID):
            rerun_done = True
            log("\n" + "=" * 70)
            log("  RERUN CHAIN COMPLETED")
            log("=" * 70)

            if not rerun_analyzed:
                # Find the SmartTargets and EXODUS-500 reports
                st_report = find_latest_report("smart_targets.json", min_targets=100)
                if st_report:
                    log(f"\n  SmartTargets report: {st_report.name}")
                    run_analysis("escalate", report=st_report)

                # EXODUS-500 report (no target_file, uses --targets)
                e500_reports = find_reports_since(start_time)
                for rp in reversed(e500_reports):
                    try:
                        with open(rp) as f:
                            d = json.load(f)
                        if d.get("n_targets", 0) >= 400 and "smart_targets" not in d.get("target_file", ""):
                            log(f"\n  EXODUS-500 report: {rp.name}")
                            run_analysis("escalate", report=rp)
                            break
                    except Exception:
                        continue

                rerun_analyzed = True

        # Both done
        if cal_done and rerun_done:
            if cal_analyzed and rerun_analyzed:
                log("\n" + "=" * 70)
                log("  ALL CHAINS COMPLETE — ALL ANALYSES RUN")
                log("=" * 70)

                # Final summary
                log("\n  FINAL SUMMARY:")
                log(f"  Total runtime: {(time.time() - start_time)/3600:.1f} hours")

                # Count total reports generated
                new_reports_final = set(str(p) for p in REPORT_DIR.glob("quick_run_*.json")) - set()
                log(f"  Reports generated: {len(new_reports_final)}")

                # Save watcher summary
                summary = {
                    "watcher_started": datetime.fromtimestamp(start_time).isoformat(),
                    "watcher_finished": datetime.now().isoformat(),
                    "calibration_pid": CALIBRATION_PID,
                    "rerun_pid": RERUN_PID,
                    "total_runtime_hours": (time.time() - start_time) / 3600,
                }
                summary_path = REPORT_DIR / "watcher_summary.json"
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2)
                log(f"  Summary saved: {summary_path}")

                break
            else:
                # Chains died but we haven't analyzed yet — try analysis anyway
                if not cal_analyzed:
                    run_analysis("calibration")
                    cal_analyzed = True
                if not rerun_analyzed:
                    rerun_analyzed = True
                continue

        # Status update
        elapsed = (time.time() - start_time) / 60
        cal_status = "RUNNING" if not cal_done else "DONE"
        rerun_status = "RUNNING" if not rerun_done else "DONE"
        log(f"  [{elapsed:.0f}m] Calibration: {cal_status} | Rerun: {rerun_status}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
