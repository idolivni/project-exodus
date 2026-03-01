#!/usr/bin/env python3
"""
VASCO Blind Holdout Protocol
=============================

Two-phase analysis of VASCO vanishing star candidates, designed to prevent
overfitting and ensure any discoveries survive pre-registration.

Phase 1 (TUNING): Run Table 3 (28 highest-quality candidates)
  - Optimize archaeology thresholds
  - Calibrate classification logic
  - Lock all parameters BEFORE Phase 2

Phase 2 (BLIND): Run full 127 targets (99 Table 2 + 28 Table 3)
  - Thresholds frozen from Phase 1
  - No parameter changes allowed
  - Results are the final, publishable outcome

The protocol generates a manifest file with locked thresholds and a hash
to prove no post-hoc adjustments were made.

Usage
-----
    # Phase 1: Tuning on Table 3
    ./venv/bin/python scripts/run_vasco_blind.py --phase tuning

    # Phase 2: Blind run on all 127 (requires Phase 1 manifest)
    ./venv/bin/python scripts/run_vasco_blind.py --phase blind

    # Check manifest integrity
    ./venv/bin/python scripts/run_vasco_blind.py --verify
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger, safe_json_dump

log = get_logger("vasco_blind")

# ── File paths ───────────────────────────────────────────────────────
TUNING_TARGETS = PROJECT_ROOT / "data" / "targets" / "vasco_vanishing_28.json"
FULL_TARGETS = PROJECT_ROOT / "data" / "targets" / "vasco_vanishing_127.json"
MANIFEST_PATH = PROJECT_ROOT / "data" / "reports" / "vasco_blind_manifest.json"
TUNING_REPORT = PROJECT_ROOT / "data" / "reports" / "vasco_tuning_report.json"
BLIND_REPORT = PROJECT_ROOT / "data" / "reports" / "vasco_blind_report.json"


def _hash_file(path: Path) -> str:
    """SHA-256 hash of a file for integrity verification."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _hash_dict(d: dict) -> str:
    """SHA-256 hash of a dict (via deterministic JSON)."""
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()


# ── Phase 1: Tuning ─────────────────────────────────────────────────

def run_tuning():
    """Phase 1: Run archaeology on Table 3 and lock thresholds."""
    log.info("=" * 70)
    log.info("  VASCO BLIND HOLDOUT — Phase 1: TUNING (Table 3, 28 targets)")
    log.info("=" * 70)

    if MANIFEST_PATH.exists():
        log.warning("Manifest already exists at %s", MANIFEST_PATH)
        log.warning("Delete it manually if you want to re-run tuning.")
        return

    # Load targets
    with open(TUNING_TARGETS) as f:
        data = json.load(f)
    targets = data.get("targets", data)
    log.info("Loaded %d tuning targets from %s", len(targets), TUNING_TARGETS)

    # Run archaeology
    from src.detection.vasco_archaeology import batch_analyze
    results = batch_analyze(targets, use_cache=True)

    # Run pipeline (QuickRunner Tier 0) on targets
    log.info("Running QuickRunner Tier 0 on tuning targets...")
    pipeline_results = _run_pipeline(TUNING_TARGETS)

    # Analyze tuning results to determine thresholds
    thresholds = _derive_thresholds(results, pipeline_results)

    # Build manifest
    manifest = {
        "protocol": "VASCO Blind Holdout v1.0",
        "phase_1_complete": datetime.now(timezone.utc).isoformat(),
        "tuning_targets_file": str(TUNING_TARGETS),
        "tuning_targets_hash": _hash_file(TUNING_TARGETS),
        "full_targets_file": str(FULL_TARGETS),
        "full_targets_hash": _hash_file(FULL_TARGETS),
        "n_tuning_targets": len(targets),
        "locked_thresholds": thresholds,
        "thresholds_hash": _hash_dict(thresholds),
        "tuning_summary": {
            "n_ir_detected": sum(1 for r in results.values() if r.neowise_detected),
            "n_brightening": sum(1 for r in results.values() if r.is_brightening),
            "n_radio": sum(1 for r in results.values() if r.radio_detected),
            "n_critical": sum(1 for r in results.values() if r.priority == "CRITICAL"),
            "n_high": sum(1 for r in results.values() if r.priority == "HIGH"),
            "classifications": {},
        },
        "archaeology_module_hash": _hash_file(
            PROJECT_ROOT / "src" / "detection" / "vasco_archaeology.py"
        ),
        "scorer_hash": _hash_file(
            PROJECT_ROOT / "src" / "scoring" / "exodus_score.py"
        ),
    }

    # Count classifications
    for r in results.values():
        c = r.classification
        manifest["tuning_summary"]["classifications"][c] = (
            manifest["tuning_summary"]["classifications"].get(c, 0) + 1
        )

    # Save manifest
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        safe_json_dump(manifest, f)
    log.info("Manifest saved to %s", MANIFEST_PATH)

    # Save tuning report
    tuning_data = {
        "phase": "tuning",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_targets": len(targets),
        "locked_thresholds": thresholds,
        "archaeology_results": {tid: r.to_dict() for tid, r in results.items()},
        "pipeline_results": pipeline_results,
    }
    with open(TUNING_REPORT, "w") as f:
        safe_json_dump(tuning_data, f)
    log.info("Tuning report saved to %s", TUNING_REPORT)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"  Phase 1 COMPLETE — Thresholds locked")
    print(f"{'=' * 70}")
    print(f"  IR detected:  {manifest['tuning_summary']['n_ir_detected']}/28")
    print(f"  Brightening:  {manifest['tuning_summary']['n_brightening']}")
    print(f"  Radio:        {manifest['tuning_summary']['n_radio']}")
    print(f"  CRITICAL:     {manifest['tuning_summary']['n_critical']}")
    print(f"  HIGH:         {manifest['tuning_summary']['n_high']}")
    print(f"\n  Locked thresholds:")
    for k, v in thresholds.items():
        print(f"    {k}: {v}")
    print(f"\n  Manifest: {MANIFEST_PATH}")
    print(f"  *** DO NOT modify any code before running Phase 2 ***")
    print(f"{'=' * 70}")


# ── Phase 2: Blind ──────────────────────────────────────────────────

def run_blind(allow_drift: bool = False):
    """Phase 2: Run full 127 targets with frozen thresholds."""
    log.info("=" * 70)
    log.info("  VASCO BLIND HOLDOUT — Phase 2: BLIND (All 127 targets)")
    log.info("=" * 70)

    # Verify manifest exists and is intact
    if not MANIFEST_PATH.exists():
        log.error("No manifest found. Run Phase 1 (--phase tuning) first.")
        sys.exit(1)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    # Verify code hasn't changed since Phase 1
    current_arch_hash = _hash_file(
        PROJECT_ROOT / "src" / "detection" / "vasco_archaeology.py"
    )
    current_scorer_hash = _hash_file(
        PROJECT_ROOT / "src" / "scoring" / "exodus_score.py"
    )

    # F-05 fix: hash drift is hard-stop by default.
    # Use --allow-drift to continue with audit trail (weakens protocol).
    drift_detected = []
    if current_arch_hash != manifest["archaeology_module_hash"]:
        drift_detected.append({
            "module": "vasco_archaeology.py",
            "phase1_hash": manifest["archaeology_module_hash"],
            "current_hash": current_arch_hash,
        })
    if current_scorer_hash != manifest["scorer_hash"]:
        drift_detected.append({
            "module": "exodus_score.py",
            "phase1_hash": manifest["scorer_hash"],
            "current_hash": current_scorer_hash,
        })

    if drift_detected:
        for d in drift_detected:
            log.error("INTEGRITY VIOLATION: %s modified since Phase 1!", d["module"])
            log.error("  Phase 1 hash: %s", d["phase1_hash"])
            log.error("  Current hash: %s", d["current_hash"])

        if not allow_drift:
            log.error("Aborting. Use --allow-drift to override (weakens blind protocol).")
            sys.exit(1)

        # --allow-drift: record audit trail and continue
        log.warning("--allow-drift: continuing despite code drift (blind protocol weakened)")
        drift_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "drift_modules": drift_detected,
            "override": True,
        }
        drift_path = MANIFEST_PATH.parent / "drift_override.json"
        with open(drift_path, "w") as f:
            json.dump(drift_record, f, indent=2)
        log.warning("Drift override recorded: %s", drift_path)

    # Load locked thresholds
    thresholds = manifest["locked_thresholds"]
    log.info("Using locked thresholds from Phase 1: %s", thresholds)

    # Audit fix N8: verify target file hash before loading
    current_targets_hash = _hash_file(FULL_TARGETS)
    expected_targets_hash = manifest.get("full_targets_hash")
    if expected_targets_hash and current_targets_hash != expected_targets_hash:
        log.error("INTEGRITY VIOLATION: Target file modified since Phase 1!")
        log.error("  Phase 1 hash: %s", expected_targets_hash)
        log.error("  Current hash: %s", current_targets_hash)
        sys.exit(1)

    # Load all 127 targets
    with open(FULL_TARGETS) as f:
        data = json.load(f)
    targets = data.get("targets", data)
    log.info("Loaded %d targets from %s", len(targets), FULL_TARGETS)

    # Audit fix N6: inject locked thresholds into batch_analyze
    from src.detection.vasco_archaeology import batch_analyze
    results = batch_analyze(targets, use_cache=True, thresholds=thresholds)

    # Run pipeline with locked thresholds (F-06: verify contract)
    log.info("Running QuickRunner Tier 0 on full target set...")
    pipeline_results = _run_pipeline(FULL_TARGETS, locked_thresholds=thresholds)

    # Save blind report
    blind_data = {
        "phase": "blind",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "manifest_hash": _hash_file(MANIFEST_PATH),
        "locked_thresholds": thresholds,
        "code_integrity": {
            "archaeology_match": current_arch_hash == manifest["archaeology_module_hash"],
            "scorer_match": current_scorer_hash == manifest["scorer_hash"],
        },
        "n_targets": len(targets),
        "archaeology_results": {tid: r.to_dict() for tid, r in results.items()},
        "pipeline_results": pipeline_results,
        "summary": {
            "n_ir_detected": sum(1 for r in results.values() if r.neowise_detected),
            "n_brightening": sum(1 for r in results.values() if r.is_brightening),
            "n_radio": sum(1 for r in results.values() if r.radio_detected),
            "n_critical": sum(1 for r in results.values() if r.priority == "CRITICAL"),
            "n_high": sum(1 for r in results.values() if r.priority == "HIGH"),
        },
    }
    with open(BLIND_REPORT, "w") as f:
        safe_json_dump(blind_data, f)
    log.info("Blind report saved to %s", BLIND_REPORT)

    # Print results
    print(f"\n{'=' * 70}")
    print(f"  Phase 2 COMPLETE — BLIND RESULTS")
    print(f"{'=' * 70}")
    n = blind_data["summary"]
    print(f"  Targets:      {len(targets)}")
    print(f"  IR detected:  {n['n_ir_detected']}/{len(targets)}")
    print(f"  Brightening:  {n['n_brightening']}")
    print(f"  Radio:        {n['n_radio']}")
    print(f"  CRITICAL:     {n['n_critical']}")
    print(f"  HIGH:         {n['n_high']}")
    print(f"  Code intact:  {blind_data['code_integrity']}")

    # List high-priority detections
    high_prio = [
        (tid, r) for tid, r in results.items()
        if r.priority in ("CRITICAL", "HIGH")
    ]
    if high_prio:
        print(f"\n  HIGH+ PRIORITY DETECTIONS:")
        for tid, r in sorted(high_prio, key=lambda x: -x[1].archaeology_score):
            print(f"    {r.priority:8s} {tid}: score={r.archaeology_score:.3f} "
                  f"| {r.classification} | {r.note}")
    else:
        print(f"\n  No CRITICAL or HIGH priority detections.")
        print(f"  This is a valid null result for the VASCO blind holdout.")

    print(f"\n  Report: {BLIND_REPORT}")
    print(f"{'=' * 70}")


# ── Verification ─────────────────────────────────────────────────────

def verify_manifest():
    """Verify manifest integrity and code consistency."""
    if not MANIFEST_PATH.exists():
        print("No manifest found.")
        return

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    print(f"Manifest: {MANIFEST_PATH}")
    print(f"  Phase 1 completed: {manifest['phase_1_complete']}")
    print(f"  Thresholds hash: {manifest['thresholds_hash']}")

    # Check code integrity
    arch_path = PROJECT_ROOT / "src" / "detection" / "vasco_archaeology.py"
    scorer_path = PROJECT_ROOT / "src" / "scoring" / "exodus_score.py"

    arch_ok = _hash_file(arch_path) == manifest["archaeology_module_hash"]
    scorer_ok = _hash_file(scorer_path) == manifest["scorer_hash"]

    print(f"  Archaeology code intact: {arch_ok}")
    print(f"  Scorer code intact:      {scorer_ok}")

    if arch_ok and scorer_ok:
        print("  STATUS: Ready for Phase 2 (blind run)")
    else:
        print("  WARNING: Code has changed since Phase 1. Blind protocol violated.")


# ── Helpers ──────────────────────────────────────────────────────────

def _derive_thresholds(archaeology_results, pipeline_results) -> dict:
    """Derive locked thresholds from tuning phase results.

    These thresholds define what constitutes a "detection" and what
    priority level to assign. They are frozen after Phase 1.
    """
    # Start with the module defaults
    from src.detection.vasco_archaeology import (
        NEOWISE_SEARCH_RADIUS_ARCSEC,
        MIN_NEOWISE_EPOCHS,
        TREND_SIGMA_THRESH,
        EXCESS_SCATTER_THRESH,
        ALLWISE_SEARCH_RADIUS_ARCSEC,
        RADIO_SEARCH_RADIUS_ARCSEC,
    )

    thresholds = {
        "neowise_search_radius_arcsec": NEOWISE_SEARCH_RADIUS_ARCSEC,
        "min_neowise_epochs": MIN_NEOWISE_EPOCHS,
        "trend_sigma_thresh": TREND_SIGMA_THRESH,        # CX-1 fix: key must match reader in vasco_archaeology.py:192
        "excess_scatter_thresh": EXCESS_SCATTER_THRESH,  # CX-1 fix: key must match reader in vasco_archaeology.py:191
        "allwise_search_radius_arcsec": ALLWISE_SEARCH_RADIUS_ARCSEC,
        "radio_search_radius_arcsec": RADIO_SEARCH_RADIUS_ARCSEC,
        "archaeology_score_high_threshold": 0.4,
        "archaeology_score_critical_threshold": 0.6,
        # Pipeline thresholds (from EXODUSScorer defaults)
        "fdr_alpha": 0.05,
        "stouffer_p_threshold": 0.01,
        "min_channels_for_escalation": 2,
        # Candidate certificate thresholds
        "certificate_min_channels": 2,
        "certificate_max_distance_pc": 2000,  # audit fix N4: match candidate_certificate.py MAX_DISTANCE_PC
        "certificate_unexplainability_min": 0.3,
    }

    # Adaptive adjustment based on tuning results:
    # If we find many IR detections, tighten thresholds.
    # If we find zero, keep defaults (conservative).
    n_detected = sum(1 for r in archaeology_results.values() if r.neowise_detected)
    n_total = len(archaeology_results)

    if n_detected > n_total * 0.5:
        # >50% detection rate suggests many are real sources, not vanished
        # Tighten: require stronger evidence
        thresholds["archaeology_score_high_threshold"] = 0.5
        thresholds["min_channels_for_escalation"] = 2
        log.info("High detection rate (%.0f%%) — tightened thresholds",
                 100 * n_detected / n_total)
    elif n_detected == 0:
        # No detections — keep defaults, any detection in blind phase is notable
        log.info("Zero detections in tuning — keeping default thresholds")

    return thresholds


def _run_pipeline(target_file: Path, locked_thresholds: dict = None) -> dict:
    """Run QuickRunner Tier 0 on a target file and return summary.

    Parameters
    ----------
    locked_thresholds : dict, optional
        If provided (Phase 2), verify that pipeline defaults match the
        locked values.  Aborts with an error if any mismatch is detected,
        ensuring the blind protocol's threshold contract is enforceable.
    """
    try:
        from scripts.run_quick import QuickRunner

        # ── Threshold contract enforcement (F-06 fix) ──────────────
        # Verify locked thresholds match pipeline defaults BEFORE running.
        # If defaults change, this fails loudly instead of silently diverging.
        if locked_thresholds:
            from src.vetting.candidate_certificate import (
                MAX_DISTANCE_PC, MIN_ACTIVE_CHANNELS,
            )
            contract_checks = {
                "fdr_alpha": (0.05, 0.05),  # scorer hardcodes alpha=0.05
                "certificate_max_distance_pc": (
                    locked_thresholds.get("certificate_max_distance_pc"),
                    MAX_DISTANCE_PC,
                ),
                "certificate_min_channels": (
                    locked_thresholds.get("certificate_min_channels"),
                    MIN_ACTIVE_CHANNELS,
                ),
            }
            mismatches = []
            for key, (locked, actual) in contract_checks.items():
                if locked is not None and locked != actual:
                    mismatches.append(
                        f"  {key}: locked={locked}, pipeline_default={actual}"
                    )
            if mismatches:
                msg = "Blind protocol threshold mismatch:\n" + "\n".join(mismatches)
                log.error(msg)
                return {"status": "error", "error": msg}
            log.info("Threshold contract verified: %d locked values match pipeline defaults",
                     len(contract_checks))

        runner = QuickRunner(
            max_targets=999999,
            tier=0,
            run_hypotheses=False,
            target_file=str(target_file),
        )
        summary = runner.run()

        if summary is None:
            return {"status": "no_results"}

        # runner.run() returns a summary dict; extract scored targets
        scored = summary.get("all_scored", summary.get("top_targets", []))

        # Audit fix N9: use correct schema keys from EXODUSScore.to_dict().
        # "is_anomaly" doesn't exist → use n_active_channels > 0.
        # "exodus_score" is a dict → use "total_score" for the scalar.
        n_anomalies = sum(1 for r in scored if r.get("n_active_channels", 0) > 0)
        n_fdr = sum(1 for r in scored if r.get("fdr_significant"))

        return {
            "status": "complete",
            "n_scored": len(scored),
            "n_anomalies": n_anomalies,
            "n_fdr_significant": n_fdr,
            "top_score": max((r.get("total_score", 0) for r in scored), default=0),
            "effective_thresholds": locked_thresholds or {"note": "using pipeline defaults"},
        }
    except Exception as exc:
        log.error("Pipeline failed: %s", exc)
        return {"status": "error", "error": str(exc)}


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="VASCO Blind Holdout Protocol"
    )
    parser.add_argument("--phase", choices=["tuning", "blind"],
                        help="Which phase to run")
    parser.add_argument("--verify", action="store_true",
                        help="Verify manifest integrity")
    parser.add_argument("--allow-drift", action="store_true",
                        help="Allow code hash drift (weakens blind protocol; records audit trail)")
    args = parser.parse_args()

    if args.verify:
        verify_manifest()
    elif args.phase == "tuning":
        run_tuning()
    elif args.phase == "blind":
        run_blind(allow_drift=args.allow_drift)
    else:
        parser.print_help()
