#!/usr/bin/env bash
set +e
set +u
set +o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
SUITE="${SUITE:-all}"
GPUS="${GPUS:-0}"
DATA_ROOT="${DATA_ROOT:-$REPO/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/outputs/reference}"
mkdir -p "$OUTPUT_ROOT"
LOG="$OUTPUT_ROOT/launcher.log"
RC_FILE="$OUTPUT_ROOT/launcher.rc"
DONE_FILE="$OUTPUT_ROOT/launcher.done"
if [ -e "$LOG" ]; then
  echo "Existing launcher.log. Preserve it and choose a new output root, or rename it before resuming."
  exit 1
fi
nohup bash -c '
  "$1" "$2/reproduce.py" run --suite "$3" --gpus "$4" --data-root "$5" --output-root "$6"
  RUN_RC=$?
  echo "$RUN_RC" > "$7"
  echo DONE > "$8"
' _ "$PY" "$REPO" "$SUITE" "$GPUS" "$DATA_ROOT" "$OUTPUT_ROOT" "$RC_FILE" "$DONE_FILE" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUTPUT_ROOT/launcher.pid"
printf 'PID=%s\nLOG=%s\nRC=%s\nDONE=%s\n' "$PID" "$LOG" "$RC_FILE" "$DONE_FILE"
