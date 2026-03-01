#!/usr/bin/env bash
# ============================================================
# Novel Target Campaign Runner
# ============================================================
# Strategic pivot: run genuinely novel target populations through
# EXODUS that have NEVER been cross-matched before.
#
# This is the highest-probability path to finding a telescope
# candidate — targets pre-selected for genuine strangeness.
#
# Usage:
#   nohup bash scripts/run_novel_targets.sh > data/reports/novel_targets_log.txt 2>&1 &
#
# Estimated time: ~4-8 hours total
#   - VASCO Table 3 (28):    ~30 min  (highest quality vanishing stars)
#   - VASCO Full (127):      ~2-3 hr  (all vanishing star candidates)
#   - SIMBAD anomalous (391): ~4-6 hr (YSO candidates + WD candidates + X-ray)
# ============================================================

set -e
cd "$(dirname "$0")/.."
PYTHON=./venv/bin/python

echo "============================================================"
echo "  Novel Target Campaign Runner"
echo "  VASCO vanishing stars + SIMBAD anomalous objects"
echo "  CatWISE PM fix: YES"
echo "  ir_variability channel: YES (6+1 scorer)"
echo "  Expanded reports: YES (all_scored)"
echo "  Started: $(date)"
echo "============================================================"

# ── 1. VASCO Table 3 — 28 highest-quality vanishing stars ────
echo ""
echo "[1/3] VASCO Table 3 — 28 highest-quality vanishing stars"
echo "  These are USNO B1.0 objects MISSING from Pan-STARRS."
echo "  If any show IR/radio emission with no optical = IMMEDIATE ESCALATION"
echo "  Start: $(date)"
$PYTHON scripts/run_quick.py \
    --target-file data/targets/vasco_vanishing_28.json \
    --tier 0 --no-hypotheses
echo "  Done: $(date)"
echo ""

# ── 2. VASCO Full — all 127 vanishing star candidates ────────
echo "[2/3] VASCO Full — 127 vanishing star candidates"
echo "  Start: $(date)"
$PYTHON scripts/run_quick.py \
    --target-file data/targets/vasco_vanishing_127.json \
    --tier 0 --no-hypotheses
echo "  Done: $(date)"
echo ""

# ── 3. SIMBAD anomalous objects — 391 targets ────────────────
echo "[3/3] SIMBAD anomalous objects — 391 targets"
echo "  250 YSO candidates + 139 WD candidates + 2 X-ray"
echo "  Start: $(date)"
$PYTHON scripts/run_quick.py \
    --target-file data/targets/simbad_anomalous_objects.json \
    --tier 0 --no-hypotheses
echo "  Done: $(date)"
echo ""

echo "============================================================"
echo "  ALL NOVEL TARGET CAMPAIGNS COMPLETE"
echo "  Finished: $(date)"
echo ""
echo "  CRITICAL CHECK:"
echo "  1. VASCO: Any target with IR detection but NO optical counterpart?"
echo "     → This is the Dyson sphere signature. IMMEDIATE Tier 1."
echo "  2. SIMBAD: Any multi-channel convergence (2+ channels)?"
echo "     → Cross-check with Red-Team. If unexplained → Tier 1."
echo "  3. Compare: Does any novel target out-score ALL 1000+ previous targets?"
echo ""
echo "  Escalation command:"
echo "  \$PYTHON scripts/escalate.py --report <latest_report.json> --top 20"
echo "============================================================"
