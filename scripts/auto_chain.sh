#!/usr/bin/env bash
# ============================================================
# Auto-Chain Script — Waits for EXODUS-500 blitz to finish,
# then runs escalation + post-blitz campaigns automatically.
#
# Usage:
#   nohup bash scripts/auto_chain.sh > data/reports/auto_chain_log.txt 2>&1 &
#
# Monitors BLITZ_PID (set below). When it finishes:
#   1. Finds the latest report JSON
#   2. Runs escalation (--top 50)
#   3. Runs post-blitz campaigns (Hephaistos + calibration)
# ============================================================

set -e
cd "$(dirname "$0")/.."
PYTHON=./venv/bin/python
REPORTS=data/reports
BLITZ_PID=${1:?Usage: auto_chain.sh <blitz_pid>}

echo "============================================================"
echo "  EXODUS Auto-Chain Script"
echo "  Started: $(date)"
echo "  Waiting for blitz PID $BLITZ_PID to finish..."
echo "============================================================"

# ── Wait for blitz to complete ──────────────────────────────
while kill -0 $BLITZ_PID 2>/dev/null; do
    # Check progress every 5 minutes
    LAST_CHECKPOINT=$(grep "Control checkpoint" $REPORTS/blitz_log.txt 2>/dev/null | tail -1)
    echo "  [$(date +%H:%M)] Still running... $LAST_CHECKPOINT"
    sleep 300
done

echo ""
echo "  Blitz PID $BLITZ_PID finished at $(date)"

# Check if it succeeded (look for report file)
LATEST_REPORT=$(ls -t $REPORTS/quick_run_*.json 2>/dev/null | head -1)

if [ -z "$LATEST_REPORT" ]; then
    echo "  ERROR: No report file found! Blitz may have crashed."
    echo "  Check: tail -50 $REPORTS/blitz_log.txt"
    exit 1
fi

# Check if the report is newer than 1 hour (meaning it was just written)
REPORT_AGE=$(( $(date +%s) - $(stat -f %m "$LATEST_REPORT") ))
if [ $REPORT_AGE -gt 7200 ]; then
    echo "  WARNING: Latest report is ${REPORT_AGE}s old. May not be from this blitz run."
    echo "  Latest: $LATEST_REPORT"
    echo "  Checking blitz log for errors..."
    if grep -q "Traceback\|Error\|Exception" $REPORTS/blitz_log.txt; then
        echo "  ERRORS FOUND in blitz log. Aborting chain."
        grep "Traceback\|Error" $REPORTS/blitz_log.txt | tail -5
        exit 1
    fi
fi

echo ""
echo "  Using report: $LATEST_REPORT"

# ── 1. Escalation ──────────────────────────────────────────
echo ""
echo "============================================================"
echo "[1/6] Escalation — Top 50 candidates"
echo "  Start: $(date)"
echo "============================================================"
$PYTHON scripts/escalate.py --report "$LATEST_REPORT" --top 50
echo "  Done: $(date)"

# ── 2. Hephaistos re-run (with PM sigma fix) ──────────────
echo ""
echo "============================================================"
echo "[2/6] Hephaistos re-run (PM sigma fix) — 7 targets"
echo "  Start: $(date)"
echo "============================================================"
$PYTHON scripts/run_quick.py \
    --target-file data/targets/hephaistos_candidates.json \
    --tier 0
echo "  Done: $(date)"

# ── 3. Calibration: Binaries ──────────────────────────────
echo ""
echo "============================================================"
echo "[3/6] Calibration: Binary stars — 500 targets"
echo "  Start: $(date)"
echo "============================================================"
$PYTHON scripts/run_quick.py \
    --target-file data/targets/calibration_binary.json \
    --tier 0 --no-hypotheses
echo "  Done: $(date)"

# ── 4. Calibration: Disk hosts ────────────────────────────
echo ""
echo "============================================================"
echo "[4/6] Calibration: Disk hosts — 500 targets"
echo "  Start: $(date)"
echo "============================================================"
$PYTHON scripts/run_quick.py \
    --target-file data/targets/calibration_disk.json \
    --tier 0 --no-hypotheses
echo "  Done: $(date)"

# ── 5. Calibration: YSOs ──────────────────────────────────
echo ""
echo "============================================================"
echo "[5/6] Calibration: Young Stellar Objects — 500 targets"
echo "  Start: $(date)"
echo "============================================================"
$PYTHON scripts/run_quick.py \
    --target-file data/targets/calibration_yso.json \
    --tier 0 --no-hypotheses
echo "  Done: $(date)"

# ── 6. Calibration: Giants ────────────────────────────────
echo ""
echo "============================================================"
echo "[6/6] Calibration: Red Giants — 456 targets"
echo "  Start: $(date)"
echo "============================================================"
$PYTHON scripts/run_quick.py \
    --target-file data/targets/calibration_giant.json \
    --tier 0 --no-hypotheses
echo "  Done: $(date)"

# ── Summary ───────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  ALL CAMPAIGNS COMPLETE"
echo "  Finished: $(date)"
echo ""
echo "  Reports generated:"
ls -lt $REPORTS/quick_run_*.json | head -8
echo ""
echo "  Next steps:"
echo "  1. Analyse escalation output for Tier 1 candidates"
echo "  2. Compare calibration Fisher p-value distributions"
echo "  3. Update Paper 4 with clean Hephaistos PM sigma values"
echo "  4. Run IR-selected FGK science targets"
echo "============================================================"
