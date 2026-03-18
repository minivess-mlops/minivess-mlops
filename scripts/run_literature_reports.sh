#!/usr/bin/env bash
# Sequential literature report runner.
#
# NO screen, NO tmux, NO background processes.
# Each report runs as a blocking `claude -p` invocation.
# Output logged to individual .log files.
#
# WARNING: claude -p invocations have historically frozen
# (metalearning 2026-03-09, 2026-03-16). The RECOMMENDED
# approach is manual execution of each report in a fresh
# Claude Code session. This script is OPTIONAL.
#
# Usage:
#   bash scripts/run_literature_reports.sh
#   bash scripts/run_literature_reports.sh R2   # Start from R2

set -euo pipefail

REPORTS_DIR="docs/planning"
LOGS_DIR="outputs/literature-report-logs"
mkdir -p "$LOGS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Which report to start from (default: R1)
START_FROM="${1:-R1}"
STARTED=false

declare -A REPORTS=(
  ["R1"]="lit-report-post-training-methods"
  ["R2"]="lit-report-ensemble-uncertainty"
  ["R3"]="lit-report-federated-multisite"
  ["R4"]="lit-report-regulatory-postmarket"
)

# Ordered execution
for RID in R1 R2 R3 R4; do
  if [[ "$STARTED" == false && "$RID" != "$START_FROM" ]]; then
    echo "[SKIP] $RID (starting from $START_FROM)"
    continue
  fi
  STARTED=true

  REPORT_NAME="${REPORTS[$RID]}"
  XML_FILE="$REPORTS_DIR/${REPORT_NAME}.xml"
  LOG_FILE="$LOGS_DIR/${REPORT_NAME}_${TIMESTAMP}.log"

  if [[ ! -f "$XML_FILE" ]]; then
    echo "[ERROR] Missing plan: $XML_FILE"
    continue
  fi

  echo "=============================================="
  echo "[$RID] Starting: $REPORT_NAME"
  echo "  Plan: $XML_FILE"
  echo "  Log:  $LOG_FILE"
  echo "  Time: $(date)"
  echo "=============================================="

  # Extract the invocation prompt from the XML plan
  PROMPT=$(python3 -c "
import xml.etree.ElementTree as ET
tree = ET.parse('$XML_FILE')
prompt = tree.find('.//invocation-prompt')
if prompt is not None and prompt.text:
    print(prompt.text.strip())
else:
    print('create literature report')
")

  # Run claude with the prompt, logging output
  # The --timeout flag prevents infinite hangs
  if claude -p "$PROMPT" 2>&1 | tee "$LOG_FILE"; then
    echo "[$RID] COMPLETED at $(date)"
  else
    echo "[$RID] FAILED at $(date) (exit code: $?)"
    echo "  Check log: $LOG_FILE"
  fi

  echo ""
done

echo "=============================================="
echo "All reports complete at $(date)"
echo "Logs in: $LOGS_DIR/"
echo "=============================================="
