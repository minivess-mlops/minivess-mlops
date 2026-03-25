#!/usr/bin/env bash
# MinIVess MLOps — Fire-and-Forget Factorial Launcher
#
# Wraps run_factorial.sh with an outer retry loop for true fire-and-forget execution.
# User runs ONE command, comes back in 1 hour or 1 week — results are there.
#
# Usage:
#   nohup bash scripts/run_factorial_resilient.sh configs/factorial/debug.yaml &
#   # Or with custom settings:
#   MAX_WAIT_HOURS=48 RETRY_INTERVAL=600 bash scripts/run_factorial_resilient.sh configs/factorial/debug.yaml
#
# Behavior:
#   1. Calls run_factorial.sh --resume every RETRY_INTERVAL seconds
#   2. run_factorial.sh --resume skips already-submitted conditions (idempotent)
#   3. Stops when ALL conditions are terminal (SUCCEEDED/FAILED/CANCELLED) or MAX_WAIT_HOURS exceeded
#   4. Logs to outputs/resilient_<timestamp>.log
#
# Exit codes:
#   0 = all conditions eventually submitted
#   1 = max wait time exceeded with conditions still pending
#   2 = config file not found
#
# See: docs/planning/retries-for-skypilot-spots-and-autoresume-plan.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${1:?Usage: $0 <factorial-config.yaml>}"

# Configurable via environment (read from factorial config infrastructure section if available)
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-168}"  # Default: 1 week
RETRY_INTERVAL="${RETRY_INTERVAL:-300}"  # Default: 5 minutes between retries
MAX_OUTER_RETRIES="${MAX_OUTER_RETRIES:-2016}"  # 168 hrs / 5 min = 2016 iterations

# Try to read from config (override defaults if present)
if command -v python3 &>/dev/null; then
    _CONFIG_WAIT=$(python3 -c "
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path('${REPO_ROOT}/${CONFIG_FILE}').read_text(encoding='utf-8'))
infra = cfg.get('infrastructure', {})
print(infra.get('max_wait_hours', '${MAX_WAIT_HOURS}'))
" 2>/dev/null || echo "${MAX_WAIT_HOURS}")
    MAX_WAIT_HOURS="${_CONFIG_WAIT}"
fi

# Validate
if [ ! -f "${REPO_ROOT}/${CONFIG_FILE}" ]; then
    echo "FATAL: Config file not found: ${CONFIG_FILE}"
    exit 2
fi

# Setup logging
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG_DIR="${REPO_ROOT}/outputs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/resilient_${TIMESTAMP}.log"

echo "╔══════════════════════════════════════════════════════════════╗" | tee "${LOG_FILE}"
echo "║  MinIVess Resilient Factorial Launcher                      ║" | tee -a "${LOG_FILE}"
echo "╠══════════════════════════════════════════════════════════════╣" | tee -a "${LOG_FILE}"
echo "║  Config:         ${CONFIG_FILE}" | tee -a "${LOG_FILE}"
echo "║  Max wait:       ${MAX_WAIT_HOURS} hours" | tee -a "${LOG_FILE}"
echo "║  Retry interval: ${RETRY_INTERVAL} seconds" | tee -a "${LOG_FILE}"
echo "║  Log:            ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "║  PID:            $$" | tee -a "${LOG_FILE}"
echo "║  Started:        $(date -u)" | tee -a "${LOG_FILE}"
echo "╚══════════════════════════════════════════════════════════════╝" | tee -a "${LOG_FILE}"

# Write PID file for status checks
echo "$$" > "${LOG_DIR}/resilient_factorial.pid"

# Cleanup PID file on exit
cleanup() {
    rm -f "${LOG_DIR}/resilient_factorial.pid"
    echo "[$(date -u)] Resilient launcher stopped (PID $$)" | tee -a "${LOG_FILE}"
}
trap cleanup EXIT

START_EPOCH=$(date +%s)
MAX_WAIT_SECONDS=$((MAX_WAIT_HOURS * 3600))
ATTEMPT=0

# ─── Permanent failure tracking (Issue #947) ─────────────────────────────────
# Track per-condition failure counts. After MAX_CONDITION_FAILURES consecutive
# failures for the same condition, mark it as PERMANENTLY_FAILED and stop retrying.
MAX_CONDITION_FAILURES="${MAX_CONDITION_FAILURES:-3}"
FAIL_COUNT_FILE="${LOG_DIR}/resilient_${TIMESTAMP}_fail_counts.txt"
PERM_FAIL_FILE="${LOG_DIR}/resilient_${TIMESTAMP}_permanently_failed.txt"
touch "${FAIL_COUNT_FILE}" "${PERM_FAIL_FILE}"

# count_condition_failures: read the fail count for a given condition from the
# tracking file. Returns 0 if no entry found.
count_condition_failures() {
    local condition="$1"
    local count
    count=$(awk -v cond="${condition}" '$1 == cond {print $2}' "${FAIL_COUNT_FILE}" 2>/dev/null)
    echo "${count:-0}"
}

# increment_condition_failures: increment the fail count for a condition.
# If count reaches MAX_CONDITION_FAILURES, append to permanently failed list.
increment_condition_failures() {
    local condition="$1"
    local current_count
    current_count=$(count_condition_failures "${condition}")
    local new_count=$((current_count + 1))

    # Update or append the count (rewrite file without the old entry, then append)
    local tmp_file="${FAIL_COUNT_FILE}.tmp"
    awk -v cond="${condition}" '$1 != cond' "${FAIL_COUNT_FILE}" > "${tmp_file}" 2>/dev/null || true
    echo "${condition} ${new_count}" >> "${tmp_file}"
    mv "${tmp_file}" "${FAIL_COUNT_FILE}"

    if [ "${new_count}" -ge "${MAX_CONDITION_FAILURES}" ]; then
        echo "${condition}" >> "${PERM_FAIL_FILE}"
        echo "[$(date -u)] PERMANENTLY_FAILED: ${condition} (failed ${new_count} times)" | tee -a "${LOG_FILE}"
    fi
}

# check_permanently_failed: scan job logs for LAUNCH_FAILED entries and update counts
check_for_permanent_failures() {
    local latest_log
    latest_log=$(ls -t "${LOG_DIR}/"*_factorial_job_ids.txt 2>/dev/null | head -1)
    if [ -z "${latest_log}" ]; then
        return
    fi

    # Extract LAUNCH_FAILED conditions from the latest job log
    while IFS='|' read -r _cond_id model loss aux_calib fold status; do
        status=$(echo "${status}" | tr -d ' ')
        if [ "${status}" = "LAUNCH_FAILED" ]; then
            model=$(echo "${model}" | tr -d ' ')
            loss=$(echo "${loss}" | tr -d ' ')
            aux_calib=$(echo "${aux_calib}" | tr -d ' ')
            fold=$(echo "${fold}" | tr -d ' ')
            local condition_name="${model}-${loss}-calib${aux_calib}-f${fold}"
            increment_condition_failures "${condition_name}"
        fi
    done < <(grep "LAUNCH_FAILED" "${latest_log}" 2>/dev/null || true)
}

while true; do
    ATTEMPT=$((ATTEMPT + 1))
    ELAPSED=$(( $(date +%s) - START_EPOCH ))

    # Check timeout
    if [ "${ELAPSED}" -ge "${MAX_WAIT_SECONDS}" ]; then
        echo "[$(date -u)] MAX_WAIT_HOURS (${MAX_WAIT_HOURS}h) exceeded. Stopping." | tee -a "${LOG_FILE}"
        # Report permanent failures
        if [ -s "${PERM_FAIL_FILE}" ]; then
            echo "[$(date -u)] Permanently failed conditions:" | tee -a "${LOG_FILE}"
            cat "${PERM_FAIL_FILE}" | tee -a "${LOG_FILE}"
        fi
        exit 1
    fi

    REMAINING_HOURS=$(( (MAX_WAIT_SECONDS - ELAPSED) / 3600 ))
    echo "" | tee -a "${LOG_FILE}"
    echo "[$(date -u)] Attempt ${ATTEMPT} (elapsed: $((ELAPSED / 3600))h, remaining: ${REMAINING_HOURS}h)" | tee -a "${LOG_FILE}"

    # Report permanent failure count
    PERM_FAIL_COUNT=0
    if [ -s "${PERM_FAIL_FILE}" ]; then
        PERM_FAIL_COUNT=$(wc -l < "${PERM_FAIL_FILE}")
        echo "[$(date -u)] Permanently failed conditions: ${PERM_FAIL_COUNT}" | tee -a "${LOG_FILE}"
    fi

    # Run factorial with --resume (idempotent — skips already-submitted conditions)
    EXIT_CODE=0
    bash "${REPO_ROOT}/scripts/run_factorial.sh" --resume "${CONFIG_FILE}" >> "${LOG_FILE}" 2>&1 || EXIT_CODE=$?

    case ${EXIT_CODE} in
        0)
            echo "[$(date -u)] All conditions submitted successfully!" | tee -a "${LOG_FILE}"
            echo "[$(date -u)] Monitor with: uv run sky jobs queue" | tee -a "${LOG_FILE}"
            if [ -s "${PERM_FAIL_FILE}" ]; then
                echo "[$(date -u)] WARNING: ${PERM_FAIL_COUNT} conditions were permanently failed:" | tee -a "${LOG_FILE}"
                cat "${PERM_FAIL_FILE}" | tee -a "${LOG_FILE}"
            fi
            exit 0
            ;;
        2)
            echo "[$(date -u)] Partial failure — some conditions failed to launch." | tee -a "${LOG_FILE}"
            check_for_permanent_failures
            echo "[$(date -u)] Retrying in ${RETRY_INTERVAL}s..." | tee -a "${LOG_FILE}"
            ;;
        1)
            echo "[$(date -u)] Total failure — no conditions launched." | tee -a "${LOG_FILE}"
            check_for_permanent_failures
            echo "[$(date -u)] Retrying in ${RETRY_INTERVAL}s..." | tee -a "${LOG_FILE}"
            ;;
        *)
            echo "[$(date -u)] Unexpected exit code ${EXIT_CODE}. Retrying in ${RETRY_INTERVAL}s..." | tee -a "${LOG_FILE}"
            ;;
    esac

    # Sleep before retry
    echo "[$(date -u)] Sleeping ${RETRY_INTERVAL}s before next attempt..." | tee -a "${LOG_FILE}"
    sleep "${RETRY_INTERVAL}"
done
