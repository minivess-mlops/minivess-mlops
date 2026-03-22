#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="/tmp/minivess-overnight"
TIMEOUT="${CHILD_TIMEOUT_SEC:-7200}"
SKIP_TO="${SKIP_TO:-1}"

mkdir -p "${LOG_DIR}"

# Kill any leftover claude processes from previous runs
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  MinIVess Overnight Executor — Flow 0"
echo "  Repo:    ${REPO}"
echo "  Logs:    ${LOG_DIR}/"
echo "  Timeout: ${TIMEOUT}s per child"
echo "  SKIP_TO: ${SKIP_TO}"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════════"
echo ""

echo "[$(date +%H:%M:%S)] Killing leftover claude processes..."
pkill -f 'claude.*dangerous' 2>/dev/null && echo "  Killed old processes. Waiting 2s..." && sleep 2 || echo "  None found."
echo ""

PLANS=(
  "docs/planning/overnight-child-01-acquisition.xml|Flow 0: Data Acquisition"
  "docs/planning/overnight-child-02-annotation.xml|Data Annotation"
  "docs/planning/overnight-child-03-dashboard.xml|Unified Dashboard"
)

FAILED=()
CHILD_NUM=0

for entry in "${PLANS[@]}"; do
  CHILD_NUM=$(( CHILD_NUM + 1 ))
  plan="${entry%%|*}"
  desc="${entry##*|}"
  plan_abs="${REPO}/${plan}"
  log_file="${LOG_DIR}/child-${CHILD_NUM}.log"

  if [ "${CHILD_NUM}" -lt "${SKIP_TO}" ]; then
    echo "[$(date +%H:%M:%S)] SKIP child-${CHILD_NUM}: ${desc} (SKIP_TO=${SKIP_TO})"
    continue
  fi

  echo "════════════════════════════════════════════════════════════"
  echo "  STARTING child-${CHILD_NUM}: ${desc}"
  echo "  Plan: ${plan}"
  echo "  Log:  ${log_file}"
  echo "  Time: $(date)"
  echo "════════════════════════════════════════════════════════════"
  echo ""

  if [ ! -f "${plan_abs}" ]; then
    echo "[$(date +%H:%M:%S)] ERROR: plan file not found: ${plan_abs}"
    FAILED+=("child-${CHILD_NUM}: ${desc} (missing plan)")
    continue
  fi

  cd "${REPO}"

  exit_code=0
  timeout "${TIMEOUT}" claude \
    --dangerously-skip-permissions \
    --output-format stream-json \
    --verbose \
    --include-partial-messages \
    -p "You are executing a TDD implementation plan for the MinIVess MLOps repo.

REPO ROOT: ${REPO}
PLAN FILE: ${plan_abs}

Read the plan file first (use the Read tool), then execute every phase in order.

CRITICAL RULES (from CLAUDE.md):
- TDD: write failing tests first, then implement
- Library-First: use MONAI/Prefect/existing code before custom logic
- GitHub Actions EXPLICITLY DISABLED — never add automatic CI triggers
- Every test failure must be fixed or reported immediately

COMPLETION PROTOCOL:
1. uv run pytest tests/ -x -q — loop until exit 0, never use -k/--ignore/xfail
2. uv run pre-commit run --all-files — loop until exit 0, never use --no-verify/SKIP=
3. Only after both pass: git push -u origin HEAD && gh pr create" \
    2>&1 | tee "${log_file}" \
         | jq -rj 'select(.type=="stream_event" and (.event.delta.type?=="text_delta")) | .event.delta.text' \
    || exit_code=$?

  echo ""
  log_bytes=$(wc -c < "${log_file}" 2>/dev/null || echo 0)

  if [ "${exit_code}" -eq 124 ]; then
    echo "[$(date +%H:%M:%S)] TIMEOUT child-${CHILD_NUM}: ${desc} (hit ${TIMEOUT}s limit)"
    FAILED+=("child-${CHILD_NUM}: ${desc} (TIMEOUT)")
  elif [ "${exit_code}" -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] FAILED child-${CHILD_NUM}: ${desc} (exit ${exit_code}, ${log_bytes} bytes)"
    FAILED+=("child-${CHILD_NUM}: ${desc} (exit ${exit_code})")
  else
    echo "[$(date +%H:%M:%S)] DONE child-${CHILD_NUM}: ${desc} (${log_bytes} bytes)"
  fi

  echo ""
  echo "  Pausing 30s before next child..."
  echo ""
  sleep 30
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ALL CHILDREN FINISHED — $(date)"
echo ""
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "  FAILURES (${#FAILED[@]}):"
  for f in "${FAILED[@]}"; do
    echo "    - ${f}"
  done
else
  echo "  ALL SUCCEEDED"
fi
echo ""
echo "  Logs: ${LOG_DIR}/"
echo "  Open PRs:"
gh pr list --state open 2>/dev/null || true
echo "════════════════════════════════════════════════════════════"
