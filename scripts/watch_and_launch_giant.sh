#!/bin/bash
# Watcher: polls YSO calibration PID every 60s.
# When it finishes, automatically launches giant calibration (step 4).

YSO_PID=${1:?Usage: watch_and_launch_giant.sh <yso_pid>}
POLL_INTERVAL=60
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$PROJECT_DIR/venv/bin/python"
GIANT_LOG="/tmp/calibration_giant.log"

echo "[$(date)] Watcher started. Monitoring YSO PID $YSO_PID..."

while kill -0 "$YSO_PID" 2>/dev/null; do
    echo "[$(date)] YSO PID $YSO_PID still running. Checking again in ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
done

echo "[$(date)] YSO PID $YSO_PID finished!"
echo "[$(date)] Launching giant calibration (456 targets)..."

cd "$PROJECT_DIR"
nohup "$PYTHON" scripts/run_quick.py \
    --target-file data/targets/calibration_giant.json \
    --tier 0 \
    --no-hypotheses \
    > "$GIANT_LOG" 2>&1 &

GIANT_PID=$!
echo "[$(date)] Giant calibration launched as PID $GIANT_PID"
echo "[$(date)] Logs: $GIANT_LOG"
