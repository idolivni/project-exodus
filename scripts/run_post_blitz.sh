#!/usr/bin/env bash
# ============================================================
# Post-Blitz Campaign Runner
# ============================================================
# Run this AFTER the EXODUS-500 blitz completes.
# Chains: Hephaistos re-run (PM fix) → 4 calibration campaigns
#
# Usage:
#   nohup bash scripts/run_post_blitz.sh > data/reports/post_blitz_log.txt 2>&1 &
#
# Total estimated time: ~8-12 hours
#   - Hephaistos: ~50 min (7 targets + 50 controls)
#   - Binary:     ~2-3 hr (500 targets)
#   - Disk:       ~2-3 hr (500 targets)
#   - YSO:        ~2-3 hr (500 targets)
#   - Giant:      ~2-3 hr (456 targets)
# ============================================================

set -e
cd "$(dirname "$0")/.."
PYTHON=./venv/bin/python
REPORTS=data/reports

echo "============================================================"
echo "  Post-Blitz Campaign Runner"
echo "  Started: $(date)"
echo "============================================================"

# ── 1. Hephaistos re-run (with PM sigma fix) ──────────────────
echo ""
echo "[1/5] Hephaistos re-run (PM sigma fix) — 7 targets"
echo "  Start: $(date)"
$PYTHON scripts/run_quick.py \
    --target-file data/targets/hephaistos_candidates.json \
    --tier 0
echo "  Done: $(date)"
echo ""

# ── 2. Calibration: Binaries ──────────────────────────────────
echo "[2/5] Calibration: Binary stars — 500 targets"
echo "  Start: $(date)"
$PYTHON scripts/run_quick.py \
    --target-file data/targets/calibration_binary.json \
    --tier 0 --no-hypotheses
echo "  Done: $(date)"
echo ""

# ── 3. Calibration: Disk hosts ────────────────────────────────
echo "[3/5] Calibration: Disk hosts — 500 targets"
echo "  Start: $(date)"
$PYTHON scripts/run_quick.py \
    --target-file data/targets/calibration_disk.json \
    --tier 0 --no-hypotheses
echo "  Done: $(date)"
echo ""

# ── 4. Calibration: YSOs ──────────────────────────────────────
echo "[4/5] Calibration: Young Stellar Objects — 500 targets"
echo "  Start: $(date)"
$PYTHON scripts/run_quick.py \
    --target-file data/targets/calibration_yso.json \
    --tier 0 --no-hypotheses
echo "  Done: $(date)"
echo ""

# ── 5. Calibration: Giants ────────────────────────────────────
echo "[5/5] Calibration: Red Giants — 456 targets"
echo "  Start: $(date)"
$PYTHON scripts/run_quick.py \
    --target-file data/targets/calibration_giant.json \
    --tier 0 --no-hypotheses
echo "  Done: $(date)"
echo ""

echo "============================================================"
echo "  ALL CAMPAIGNS COMPLETE"
echo "  Finished: $(date)"
echo ""
echo "  Next steps:"
echo "  1. Escalate blitz: $PYTHON scripts/escalate.py --report $REPORTS/<blitz_report>.json --top 50"
echo "  2. Analyse calibration: compare Fisher p-value distributions across populations"
echo "  3. Update Paper 4 with clean Hephaistos PM sigma values"
echo "============================================================"
