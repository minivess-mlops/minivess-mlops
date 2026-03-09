#!/usr/bin/env bash
# =============================================================================
# MinIVess MLOps — Overnight Master Executor
# Closes: #328, #330, #331, #342, #139, #150, #176
#
# ── HOW TO RUN ────────────────────────────────────────────────────────────────
#   ALWAYS use screen or tmux — NEVER nohup (nohup hides output).
#
#   screen -S overnight
#   ./docs/planning/overnight-master-flow0-annotation-dashboard.sh
#   # Ctrl-A D  →  detach      screen -r overnight  →  reattach
#
# ── RESUME AFTER CRASH ────────────────────────────────────────────────────────
#   SKIP_TO=2 ./docs/planning/overnight-master-flow0-annotation-dashboard.sh
#
# ── OVERRIDE TIMEOUT ─────────────────────────────────────────────────────────
#   CHILD_TIMEOUT_SEC=14400 ./docs/planning/overnight-master-flow0-annotation-dashboard.sh
#
# ── WHAT YOU WILL SEE ─────────────────────────────────────────────────────────
#   Claude token stream live in terminal via jq filter (real-time, not buffered)
#   ♥ heartbeat line every 60s — proves master script is alive
#   ⚠ STALL WARNING within 10 min if a child session freezes
#   Full stream-json session saved to /tmp/minivess-overnight/child-N.log
#   Replay: jq -rj '...' /tmp/minivess-overnight/child-1.log
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="/tmp/minivess-overnight"
SKIP_TO="${SKIP_TO:-1}"
CHILD_TIMEOUT_SEC="${CHILD_TIMEOUT_SEC:-7200}"   # 2h default; override for longer plans
BASE_BRANCH="main"

mkdir -p "${LOG_DIR}"

# ── Logging ───────────────────────────────────────────────────────────────────

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "${LOG_DIR}/master.log"; }
log_sep() { log "═══════════════════════════════════════════════════════"; }

# ── Heartbeat: log a timestamp every 60s while a child runs ──────────────────

start_heartbeat() {
  local child_num="$1"
  ( local n=0
    while true; do
      sleep 60
      n=$(( n + 1 ))
      log "♥ heartbeat: child-${child_num} running ${n}m"
    done
  ) &
  echo $!
}

# ── Watchdog: STALL WARNING if log is 0 bytes 10 min after start ─────────────

start_watchdog() {
  local log_file="$1"
  local child_num="$2"
  ( sleep 600
    while true; do
      local bytes
      bytes=$(wc -c < "${log_file}" 2>/dev/null || echo 0)
      if [ "${bytes}" -lt 1 ]; then
        log "⚠ STALL WARNING: child-${child_num} log still empty — claude may be frozen!"
        log "  Claude PIDs: $(pgrep -f 'claude.*dangerously' | tr '\n' ' ' || echo none)"
        log "  Kill & resume: SKIP_TO=${child_num} $0"
      else
        log "♥ watchdog: child-${child_num} log ${bytes} bytes — alive"
      fi
      sleep 60
    done
  ) &
  echo $!
}

# ── Core: run one child plan ──────────────────────────────────────────────────

run_child() {
  local child_num="$1"
  local plan_rel="$2"
  local description="$3"
  local plan_abs="${REPO_ROOT}/${plan_rel}"
  local log_file="${LOG_DIR}/child-${child_num}.log"

  if [ "${child_num}" -lt "${SKIP_TO}" ]; then
    log "SKIP: Child ${child_num} (${description}) — SKIP_TO=${SKIP_TO}"
    return 0
  fi

  log_sep
  log "START: Child ${child_num} — ${description}"
  log "Plan:  ${plan_abs}"
  log "Log:   ${log_file}"
  log "Timeout: ${CHILD_TIMEOUT_SEC}s ($(( CHILD_TIMEOUT_SEC / 60 ))m)"
  log_sep

  [ -f "${plan_abs}" ] || { log "ERROR: plan not found: ${plan_abs}"; exit 1; }

  local cur_branch
  cur_branch=$(git -C "${REPO_ROOT}" branch --show-current)
  if [ "${cur_branch}" != "${BASE_BRANCH}" ]; then
    log "Switching from '${cur_branch}' to '${BASE_BRANCH}'..."
    git -C "${REPO_ROOT}" checkout "${BASE_BRANCH}"
    git -C "${REPO_ROOT}" pull origin "${BASE_BRANCH}"
  fi

  : > "${log_file}"   # truncate so watchdog baseline is 0 bytes

  local start_time
  start_time=$(date +%s)

  local heartbeat_pid watchdog_pid
  heartbeat_pid=$(start_heartbeat "${child_num}")
  watchdog_pid=$(start_watchdog "${log_file}" "${child_num}")

  local exit_code=0

  # ── THREE FIXES vs. the broken pattern that caused 14h silent freeze ───────
  # Fix 1: pass plan FILE PATH (not $(cat 19KB)) — claude streams immediately
  # Fix 2: --output-format stream-json --verbose --include-partial-messages
  #        → real-time token streaming, zero buffering delay
  # Fix 3: timeout → hard-kill frozen sessions (exit code 124 = timeout)
  # jq filter: human-readable text in terminal; full JSON saved in log_file
  (
    cd "${REPO_ROOT}"
    timeout "${CHILD_TIMEOUT_SEC}" claude \
      --dangerously-skip-permissions \
      --output-format stream-json \
      --verbose \
      --include-partial-messages \
      -p "You are executing a TDD implementation plan for the MinIVess MLOps repo.

REPO ROOT: ${REPO_ROOT}
PLAN FILE: ${plan_abs}

Read the plan file first (use the Read tool), then execute every phase in order.

CRITICAL RULES (from CLAUDE.md — non-negotiable):
- TDD: write failing tests first (RED), then minimum implementation (GREEN)
- Library-First: use MONAI, Prefect, existing code before writing custom logic
- No regex for structured data — use json.loads(), yaml.safe_load(), pathlib.Path
- All flows via Prefect in Docker — no standalone scripts as run path
- /tmp FORBIDDEN for artifacts — use pytest tmp_path in tests only
- .env.example is single source of truth for all config values
- GitHub Actions EXPLICITLY DISABLED — never add automatic CI triggers
- Every test failure must be fixed or reported immediately

COMPLETION PROTOCOL — mandatory, no shortcuts, no exceptions:

STEP 1 — FULL TEST SUITE (loop until zero failures):
  uv run pytest tests/ -x -q
  - If ANY test fails: diagnose root cause, fix code, run full suite again from scratch
  - NEVER use -k, --ignore, or -m to reduce scope
  - NEVER add xfail/skip markers to hide failures — fix the root cause
  - Repeat until pytest exits 0 with zero failures

STEP 2 — PRE-COMMIT ON ALL FILES (loop until fully clean):
  uv run pre-commit run --all-files
  - If ANY hook fails: fix the issue, run --all-files again
  - NEVER use --no-verify, SKIP= env var, or bypass any hook
  - Repeat until all hooks pass with exit 0

STEP 3 — ONLY after Step 1 AND Step 2 are both exit 0:
  git push -u origin HEAD
  gh pr create with descriptive title and body
  Report completion status.

FORBIDDEN (each is a violation — do not do any of these):
  pytest -k 'subset'      never filter tests
  pytest --ignore=...     never ignore test directories
  git commit --no-verify  never bypass pre-commit
  SKIP=hook pre-commit    never skip individual hooks"
  ) 2>&1 \
    | tee "${log_file}" \
    | jq -rj 'select(.type=="stream_event" and (.event.delta.type?=="text_delta")) | .event.delta.text' \
    || exit_code=$?

  kill "${heartbeat_pid}" "${watchdog_pid}" 2>/dev/null || true
  wait "${heartbeat_pid}" "${watchdog_pid}" 2>/dev/null || true

  local end_time duration log_bytes
  end_time=$(date +%s)
  duration=$(( end_time - start_time ))
  log_bytes=$(wc -c < "${log_file}" 2>/dev/null || echo 0)

  if [ "${exit_code}" -eq 0 ]; then
    log "DONE: Child ${child_num} — ${description}"
    log "  Duration: ${duration}s ($(( duration / 60 ))m)  Log: ${log_bytes} bytes"
  elif [ "${exit_code}" -eq 124 ]; then
    log "TIMEOUT: Child ${child_num} — hit ${CHILD_TIMEOUT_SEC}s limit"
    log "  Partial log: ${log_file} (${log_bytes} bytes)"
    log "  Resume: SKIP_TO=${child_num} $0"
    exit 1
  else
    log "FAILED: Child ${child_num} — exit=${exit_code} (${duration}s)"
    log "  Log: ${log_file} (${log_bytes} bytes)"
    log "  Resume: SKIP_TO=${child_num} $0"
    exit "${exit_code}"
  fi

  log "Pausing 30s between sessions (Node.js GC)..."
  sleep 30
}

# =============================================================================
# Prerequisites
# =============================================================================

log_sep
log "MinIVess Overnight Executor — Flow 0 / Annotation / Dashboard"
log "SKIP_TO=${SKIP_TO}  TIMEOUT=${CHILD_TIMEOUT_SEC}s"
log "Logs: ${LOG_DIR}/"
log_sep

command -v claude  >/dev/null 2>&1 || { log "ERROR: claude CLI not found";          exit 1; }
command -v gh      >/dev/null 2>&1 || { log "ERROR: gh CLI not found";              exit 1; }
command -v uv      >/dev/null 2>&1 || { log "ERROR: uv not found";                  exit 1; }
command -v timeout >/dev/null 2>&1 || { log "ERROR: timeout not found (coreutils)"; exit 1; }
command -v jq      >/dev/null 2>&1 || { log "ERROR: jq not found (apt install jq)"; exit 1; }
command -v docker  >/dev/null 2>&1 || log "WARNING: docker not found — integration tests skipped"

if [ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]; then
  log "ERROR: Uncommitted changes — stash or commit first."
  git -C "${REPO_ROOT}" status --short
  exit 1
fi

log "Prerequisites OK."
log ""

# =============================================================================
# Child 1: Acquisition Flow (#328 #139 #150 #176)
# =============================================================================
run_child 1 \
  "docs/planning/overnight-child-01-acquisition.xml" \
  "Flow 0: Data Acquisition — MiniVess + external datasets + DVC (#328 #139 #150 #176)"

# =============================================================================
# Child 2: Annotation Flow (#330)
# =============================================================================
run_child 2 \
  "docs/planning/overnight-child-02-annotation.xml" \
  "Data Annotation — MONAI Label + 3D Slicer + BentoML champion (#330)"

# =============================================================================
# Child 3: Dashboard + QA (#331 #342)
# =============================================================================
run_child 3 \
  "docs/planning/overnight-child-03-dashboard.xml" \
  "Unified Dashboard — React+Vite + FastAPI + QA monitoring (#331 #342)"

# =============================================================================
# Post-run summary
# =============================================================================
log_sep
log "ALL CHILDREN COMPLETE"
log_sep

log "Open PRs:"
gh pr list --repo minivess-mlops/minivess-mlops --state open 2>/dev/null \
  | tee -a "${LOG_DIR}/master.log" || true

log "Issue states:"
for issue in 328 330 331 342 139 150 176; do
  state=$(gh issue view "${issue}" --repo minivess-mlops/minivess-mlops \
    --json state -q '.state' 2>/dev/null || echo "unknown")
  log "  #${issue}: ${state}"
done

log ""
log "Done. Review PRs before merging."
log "Logs: ${LOG_DIR}/"
log ""
log "Replay any child session as text:"
log "  jq -rj 'select(.type==\"stream_event\" and (.event.delta.type?==\"text_delta\")) | .event.delta.text' \\"
log "    ${LOG_DIR}/child-1.log"
