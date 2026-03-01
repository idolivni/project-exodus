#!/usr/bin/env bash
# ============================================================
# Fixed Re-Run: SmartTargets + EXODUS-500 (all fixes applied)
# ============================================================
# Re-runs both target sets with all pipeline improvements:
#   - CatWISE PM systematic floor fix (magnitude/PM-scaled)
#   - ir_variability scoring channel (NEOWISE 10yr timeseries)
#   - GALEX UV + radio continuum vetting data
#   - 6+1 channel scorer (was 5+1 in original runs)
#   - Expanded reports saving ALL scored targets
#
# Usage:
#   nohup bash scripts/run_rerun_fixed.sh > data/reports/rerun_fixed_log.txt 2>&1 &
#
# Estimated time: ~10-16 hours total
#   - SmartTargets:  ~5-8 hr (500 anomaly-driven targets)
#   - EXODUS-500:    ~5-8 hr (500 exoplanet-archive targets)
# ============================================================

set -e
cd "$(dirname "$0")/.."
PYTHON=./venv/bin/python

echo "============================================================"
echo "  Fixed Re-Run: SmartTargets + EXODUS-500"
echo "  CatWISE PM fix: YES"
echo "  ir_variability channel: YES (6+1 scorer)"
echo "  GALEX UV vetting: YES"
echo "  Radio continuum vetting: YES"
echo "  Report expansion (all_scored): YES"
echo "  Started: $(date)"
echo "============================================================"

# ── 1. SmartTargets Re-Run ──────────────────────────────────
echo ""
echo "[1/2] SmartTargets — 500 anomaly-driven targets (Tier 0)"
echo "  Target file: data/targets/smart_targets.json"
echo "  Start: $(date)"
$PYTHON scripts/run_quick.py \
    --target-file data/targets/smart_targets.json \
    --tier 0 --no-hypotheses
echo "  Done: $(date)"
echo ""

# ── 2. EXODUS-500 Re-Run ────────────────────────────────────
echo "[2/2] EXODUS-500 — 500 exoplanet-archive targets (Tier 0)"
echo "  Start: $(date)"
$PYTHON scripts/run_quick.py \
    --tier 0 --targets 500 --no-hypotheses
echo "  Done: $(date)"
echo ""

echo "============================================================"
echo "  BOTH RE-RUNS COMPLETE"
echo "  Finished: $(date)"
echo ""
echo "  Next steps:"
echo "  1. Compare: diff new vs old EXODUS-500 results (CatWISE fix impact)"
echo "  2. Compare: diff new vs old SmartTargets results (ir_variability impact)"
echo "  3. Escalate: run scripts/escalate.py --report <new_report> --top 50"
echo "  4. Check: how many FDR-significant targets survive with fixed PM floor?"
echo "  5. Check: does ir_variability channel add any new convergence?"
echo "============================================================"
