#!/usr/bin/env bash
# =============================================================================
# MinIVess MLOps — Overnight Master Executor
# Closes: #328, #330, #331, #342, #139, #150, #176
#
# Runs 3 child plans sequentially via Claude Code headless sessions.
# Each child plan is a self-contained TDD branch.
#
# Usage:
#   chmod +x docs/planning/overnight-master-flow0-annotation-dashboard.sh
#   nohup ./docs/planning/overnight-master-flow0-annotation-dashboard.sh \
#     > /tmp/overnight-master.log 2>&1 &
#   tail -f /tmp/overnight-master.log
#
# RAM note: each Claude Code session accumulates context (10-20 GB Node.js heap
# over hundreds of tool calls). Sessions are SEQUENTIAL — one at a time — so
# max concurrent RAM is 1× session overhead + your normal workstation usage.
# If a session crashes, restart from the failed child (SKIP_TO variable below).
#
# SKIP_TO: set to 2 or 3 to resume from a specific child plan after a crash.
# Example: SKIP_TO=2 ./docs/planning/overnight-master-flow0-annotation-dashboard.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLANS_DIR="${REPO_ROOT}/docs/planning"
LOG_DIR="/tmp/minivess-overnight"
SKIP_TO="${SKIP_TO:-1}"  # Start from child 1 by default; override to resume

mkdir -p "${LOG_DIR}"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

log() {
  echo "[$(timestamp)] $*" | tee -a "${LOG_DIR}/master.log"
}

run_child() {
  local child_num="$1"
  local plan_file="$2"
  local description="$3"
  local log_file="${LOG_DIR}/child-${child_num}.log"

  if [ "${child_num}" -lt "${SKIP_TO}" ]; then
    log "SKIP: Child ${child_num} (${description}) — SKIP_TO=${SKIP_TO}"
    return 0
  fi

  log "═══════════════════════════════════════════════════════"
  log "START: Child ${child_num} — ${description}"
  log "Plan:  ${plan_file}"
  log "Log:   ${log_file}"
  log "═══════════════════════════════════════════════════════"

  # Verify plan file exists
  if [ ! -f "${REPO_ROOT}/${plan_file}" ]; then
    log "ERROR: Plan file not found: ${REPO_ROOT}/${plan_file}"
    exit 1
  fi

  # Verify we're on main (or the expected base branch)
  local current_branch
  current_branch=$(git -C "${REPO_ROOT}" branch --show-current)
  if [ "${current_branch}" != "main" ]; then
    log "WARNING: Not on main branch (on '${current_branch}'). Checking out main..."
    git -C "${REPO_ROOT}" checkout main
    git -C "${REPO_ROOT}" pull origin main
  fi

  # Run Claude Code with the child plan XML as the prompt
  local start_time
  start_time=$(date +%s)

  claude \
    --dangerously-skip-permissions \
    -p "$(cat << EOF
You are executing a detailed TDD implementation plan. Read and follow ALL phases in order.

CRITICAL RULES (non-negotiable, from CLAUDE.md):
- TDD MANDATORY: write failing tests first (RED), then minimum implementation (GREEN)
- Library-First: use MONAI, Prefect, existing project code before writing custom logic
- No regex for structured data — use json.loads(), yaml.safe_load(), pathlib.Path
- No standalone scripts as run path — all flows via Prefect in Docker
- /tmp FORBIDDEN for artifacts — use pytest tmp_path for tests
- .env.example is single source of truth for all config values
- GitHub Actions EXPLICITLY DISABLED — never add automatic CI triggers
- Zero Tolerance: every test failure must be fixed or reported immediately

Read the plan file and execute every phase in order:
$(cat "${REPO_ROOT}/${plan_file}")

After completing all phases:
1. Verify all tests pass: uv run pytest (relevant test files) -q
2. Verify pre-commit passes: uv run pre-commit run --all-files
3. Push branch and create PR with gh pr create
4. Report completion status

Start now — read the plan and begin Phase 0.
EOF
)" \
    --cwd "${REPO_ROOT}" \
    2>&1 | tee "${log_file}"

  local exit_code=$?
  local end_time
  end_time=$(date +%s)
  local duration=$(( end_time - start_time ))

  if [ ${exit_code} -eq 0 ]; then
    log "DONE: Child ${child_num} — ${description} (${duration}s)"
  else
    log "FAILED: Child ${child_num} — ${description} (exit=${exit_code}, ${duration}s)"
    log "Check log: ${log_file}"
    log "To resume: SKIP_TO=${child_num} $0"
    exit ${exit_code}
  fi

  # Brief pause between sessions to let Node.js GC clear
  log "Pausing 30s between sessions (Node.js GC)..."
  sleep 30
}

# =============================================================================
# Verification: check prerequisites
# =============================================================================
log "Checking prerequisites..."

command -v claude >/dev/null 2>&1 || { log "ERROR: claude CLI not found. Install: npm install -g @anthropic-ai/claude-code"; exit 1; }
command -v gh >/dev/null 2>&1 || { log "ERROR: gh CLI not found. Install: https://cli.github.com"; exit 1; }
command -v uv >/dev/null 2>&1 || { log "ERROR: uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
command -v docker >/dev/null 2>&1 || { log "WARNING: docker not found. Some integration tests will be skipped."; }

# Verify repo is clean (no uncommitted work that could be clobbered)
if [ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]; then
  log "ERROR: Uncommitted changes in repo. Stash or commit before running overnight plan."
  git -C "${REPO_ROOT}" status --short
  exit 1
fi

log "Prerequisites OK. Starting overnight execution..."
log "Plans directory: ${PLANS_DIR}"
log "Log directory: ${LOG_DIR}"
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
# Child 3: Dashboard + QA merged (#331 #342)
# =============================================================================
run_child 3 \
  "docs/planning/overnight-child-03-dashboard.xml" \
  "Unified Dashboard — React+Vite + FastAPI + QA monitoring (#331 #342)"

# =============================================================================
# Post-run verification
# =============================================================================
log "═══════════════════════════════════════════════════════"
log "ALL CHILDREN COMPLETE"
log "═══════════════════════════════════════════════════════"

log "Checking open PRs..."
gh pr list --repo minivess-mlops/minivess-mlops --state open 2>/dev/null | tee -a "${LOG_DIR}/master.log" || true

log "Checking closed issues..."
for issue in 328 330 331 342 139 150 176; do
  state=$(gh issue view "${issue}" --repo minivess-mlops/minivess-mlops --json state -q '.state' 2>/dev/null || echo "unknown")
  log "  Issue #${issue}: ${state}"
done

log ""
log "Overnight run complete. Review PRs before merging."
log "Logs: ${LOG_DIR}/"
