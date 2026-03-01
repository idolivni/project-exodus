#!/usr/bin/env bash
# ============================================================
# Full Calibration Campaign Runner (500 targets each)
# ============================================================
# Re-runs all 4 calibration populations with CatWISE PM fix
# and expanded report saving (all_scored).
#
# Usage:
#   nohup bash scripts/run_calibration_full.sh > data/reports/calibration_full_log.txt 2>&1 &
#
# Estimated time: ~8-12 hours total
#   - Binary:  ~2-3 hr (500 targets)
#   - Disk:    ~2-3 hr (500 targets)
#   - YSO:     ~2-3 hr (500 targets)
#   - Giant:   ~2-3 hr (456 targets)
# ============================================================

set -e
cd "$(dirname "$0")/.."
PYTHON=./venv/bin/python

echo "============================================================"
echo "  Full Calibration Campaign Runner"
echo "  CatWISE PM fix: YES"
echo "  Report expansion (all_scored): YES"
echo "  Started: $(date)"
echo "============================================================"

# ── 1. Calibration: Binaries ──────────────────────────────────
echo ""
echo "[1/4] Calibration: Binary stars — 500 targets"
echo "  Start: $(date)"
$PYTHON scripts/run_quick.py \
    --target-file data/targets/calibration_binary.json \
    --tier 0 --no-hypotheses
echo "  Done: $(date)"
echo ""

# ── 2. Calibration: Disk hosts ────────────────────────────────
echo "[2/4] Calibration: Disk hosts — 500 targets"
echo "  Start: $(date)"
$PYTHON scripts/run_quick.py \
    --target-file data/targets/calibration_disk.json \
    --tier 0 --no-hypotheses
echo "  Done: $(date)"
echo ""

# ── 3. Calibration: YSOs ──────────────────────────────────────
echo "[3/4] Calibration: Young Stellar Objects — 500 targets"
echo "  Start: $(date)"
$PYTHON scripts/run_quick.py \
    --target-file data/targets/calibration_yso.json \
    --tier 0 --no-hypotheses
echo "  Done: $(date)"
echo ""

# ── 4. Calibration: Giants ────────────────────────────────────
echo "[4/4] Calibration: Red Giants — 456 targets"
echo "  Start: $(date)"
$PYTHON scripts/run_quick.py \
    --target-file data/targets/calibration_giant.json \
    --tier 0 --no-hypotheses
echo "  Done: $(date)"
echo ""

echo "============================================================"
echo "  ALL 4 CALIBRATION CAMPAIGNS COMPLETE"
echo "  Finished: $(date)"
echo "============================================================"
echo ""
echo "  Chaining into Novel Target Campaign ..."
echo ""

# ── 5. NOVEL TARGETS — chained after calibration ────────────
# Strategic pivot: VASCO vanishing stars + SIMBAD anomalous objects
# These have NEVER been cross-matched before.

echo "============================================================"
echo "  NOVEL TARGET CAMPAIGN (chained after calibration)"
echo "  VASCO vanishing stars + SIMBAD anomalous objects"
echo "  Started: $(date)"
echo "============================================================"

# ── 5a. VASCO Table 3 — 28 highest-quality vanishing stars ──
echo ""
echo "[5/7] VASCO Table 3 — 28 highest-quality vanishing stars"
echo "  These are USNO B1.0 objects MISSING from Pan-STARRS."
echo "  If any show IR/radio emission with no optical = IMMEDIATE ESCALATION"
echo "  Start: $(date)"
$PYTHON scripts/run_quick.py \
    --target-file data/targets/vasco_vanishing_28.json \
    --tier 0 --no-hypotheses
echo "  Done: $(date)"
echo ""

# ── 5b. VASCO Full — all 127 vanishing star candidates ──────
echo "[6/7] VASCO Full — 127 vanishing star candidates"
echo "  Start: $(date)"
$PYTHON scripts/run_quick.py \
    --target-file data/targets/vasco_vanishing_127.json \
    --tier 0 --no-hypotheses
echo "  Done: $(date)"
echo ""

# ── 5c. SIMBAD anomalous objects — 391 targets ──────────────
echo "[7/7] SIMBAD anomalous objects — 391 targets"
echo "  250 YSO candidates + 139 WD candidates + 2 X-ray"
echo "  Start: $(date)"
$PYTHON scripts/run_quick.py \
    --target-file data/targets/simbad_anomalous_objects.json \
    --tier 0 --no-hypotheses
echo "  Done: $(date)"
echo ""

echo "============================================================"
echo "  ALL CAMPAIGNS COMPLETE (Calibration + Novel Targets)"
echo "  Finished: $(date)"
echo ""
echo "  CRITICAL CHECK:"
echo "  1. Calibration: compare Fisher p-value distributions across populations"
echo "  2. VASCO: Any target with IR detection but NO optical counterpart?"
echo "     → This is the Dyson sphere signature. IMMEDIATE Tier 1."
echo "  3. SIMBAD: Any multi-channel convergence (2+ channels)?"
echo "     → Cross-check with Red-Team. If unexplained → Tier 1."
echo "============================================================"
