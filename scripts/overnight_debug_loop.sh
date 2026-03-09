#!/usr/bin/env bash
# overnight_debug_loop.sh — Run debug_all_models until ALL flows pass, then exit.
#
# Iterates up to MAX_ATTEMPTS times. After each run, reads the summary JSON
# and checks for any FAILED flows. If all pass → exits 0. If still failing
# after MAX_ATTEMPTS → exits 1 with a report.
#
# Usage:
#   bash scripts/overnight_debug_loop.sh [--max-attempts N] [--sleep-between S]
#
# Defaults: max 8 attempts, 120s sleep between runs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MAX_ATTEMPTS=8
SLEEP_BETWEEN=120
EXPERIMENT="debug_all_models"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
    --sleep-between) SLEEP_BETWEEN="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

LOG_DIR="outputs/debug/logs"
SUMMARY_DIR="outputs/debug"
mkdir -p "$LOG_DIR" "$SUMMARY_DIR"

echo "════════════════════════════════════════════════════════"
echo "  Overnight debug loop: up to $MAX_ATTEMPTS attempts"
echo "  Experiment: $EXPERIMENT"
echo "  Sleep between runs: ${SLEEP_BETWEEN}s"
echo "════════════════════════════════════════════════════════"

attempt=0
while (( attempt < MAX_ATTEMPTS )); do
  attempt=$(( attempt + 1 ))
  TIMESTAMP=$(date -u +%Y%m%d-%H%M%SZ)
  RUN_LOG="$LOG_DIR/debug_all_models_${TIMESTAMP}.log"

  echo ""
  echo "┌─── Attempt $attempt / $MAX_ATTEMPTS  [$(date -u '+%Y-%m-%d %H:%M:%S UTC')]"
  echo "│    Log: $RUN_LOG"

  # Run the full debug pipeline (train → post_training → analyze)
  set +e
  bash scripts/run_debug.sh --experiment "$EXPERIMENT" > "$RUN_LOG" 2>&1
  RUN_EXIT=$?
  set -e

  # Find the summary written by this run (most recent summary_*.json)
  SUMMARY=$(ls -t "$SUMMARY_DIR"/summary_*.json 2>/dev/null | head -1)

  if [[ -z "$SUMMARY" ]]; then
    echo "│  ✗ No summary JSON found — run_debug.sh likely crashed (exit=$RUN_EXIT)"
    echo "│    Last 10 log lines:"
    tail -10 "$RUN_LOG" | sed 's/^/│      /'
    echo "└─────────────────────────────────────────"
    sleep "$SLEEP_BETWEEN"
    continue
  fi

  # Parse summary: count OK vs FAILED
  RESULT=$(python3 -c "
import json, sys
d = json.load(open('$SUMMARY'))
flows = d.get('flows', [])
ok = [f['name'] for f in flows if f['status'] == 'OK']
failed = [f['name'] for f in flows if f['status'] != 'OK']
print('OK:', ' '.join(ok) if ok else 'none')
print('FAILED:', ' '.join(failed) if failed else 'none')
print('ALL_PASS:', 'yes' if not failed else 'no')
")

  OK_LIST=$(echo "$RESULT" | grep "^OK:" | cut -d' ' -f2-)
  FAIL_LIST=$(echo "$RESULT" | grep "^FAILED:" | cut -d' ' -f2-)
  ALL_PASS=$(echo "$RESULT" | grep "^ALL_PASS:" | cut -d' ' -f2)

  echo "│  ✓ passed: $OK_LIST"
  if [[ "$FAIL_LIST" != "none" ]]; then
    echo "│  ✗ failed: $FAIL_LIST"
  fi
  echo "└─────────────────────────────────────────"

  if [[ "$ALL_PASS" == "yes" ]]; then
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  ALL FLOWS PASSED on attempt $attempt! 🎉"
    echo "  Summary: $SUMMARY"
    echo "════════════════════════════════════════════════════════"
    exit 0
  fi

  # Extract specific errors to guide next iteration
  echo "  Diagnosing failures from log..."
  if grep -q "OutOfMemoryError\|out of memory" "$RUN_LOG" 2>/dev/null; then
    OOM_INFO=$(grep "out of memory" "$RUN_LOG" | head -1 | grep -oP '\d+\.\d+ GiB.*?free' || true)
    echo "  → OOM detected: $OOM_INFO"
    echo "  → sam3_hybrid patch fix: patch_size=[32,32,3] (run_debug.sh override)"
  fi
  if grep -q "IndexError" "$RUN_LOG" 2>/dev/null; then
    echo "  → IndexError detected in post_training plugins"
  fi

  if (( attempt < MAX_ATTEMPTS )); then
    echo "  Sleeping ${SLEEP_BETWEEN}s before next attempt..."
    sleep "$SLEEP_BETWEEN"
  fi
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "  MAX ATTEMPTS ($MAX_ATTEMPTS) reached — still failing: $FAIL_LIST"
echo "  Manual investigation required."
echo "════════════════════════════════════════════════════════"
exit 1
