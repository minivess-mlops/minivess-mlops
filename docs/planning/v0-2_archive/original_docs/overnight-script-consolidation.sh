#!/usr/bin/env bash
# overnight-script-consolidation.sh — sequential execution of 2 child plans
#
# Branch 1: feat/hydra-bridge    (script-consolidation Phases 0-1)
# Branch 2: feat/debug-configs   (script-consolidation Phases 2-5)
#
# Usage: bash docs/planning/overnight-script-consolidation.sh
#
# IMPORTANT: This script MUST run BEFORE overnight-prefect-docker-monai.sh
# because the Hydra bridge (Phase 0) is a prerequisite for all downstream work.
#
# Each child plan runs in its own claude session with --dangerously-skip-permissions.
# Logs go to /tmp/minivess-overnight/ with live tee output.
#
# Plan: docs/planning/script-consolidation.xml

set -euo pipefail

REPO="/home/petteri/Dropbox/github-personal/minivess-mlops"
PLANS_DIR="$REPO/docs/planning"
LOG_DIR="/tmp/minivess-overnight"
CHILD_TIMEOUT_SEC="${CHILD_TIMEOUT_SEC:-7200}"
mkdir -p "$LOG_DIR"

CHILDREN=(hydra-bridge debug-configs)
LABELS=(
  "hydra-zen-bridge-to-train-flow"
  "debug-experiment-configs-shell-wrapper"
)
BRANCHES=(
  "feat/hydra-bridge"
  "feat/debug-configs"
)

print_banner() {
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  $1"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

START_ALL=$(date +%s)
print_banner "MinIVess overnight: Script Consolidation — $(date)"
echo "  Logs     → $LOG_DIR"
echo "  Plans    → $PLANS_DIR"
echo "  Children : ${#CHILDREN[@]}"
echo ""
echo "  IMPORTANT: This runs BEFORE overnight-prefect-docker-monai.sh"
echo "  The Hydra bridge is a prerequisite for all downstream work."
echo ""
for idx in "${!CHILDREN[@]}"; do
  echo "    [$((idx+1))] ${LABELS[$idx]} — branch: ${BRANCHES[$idx]}"
done
echo ""

FAILED=()

for idx in "${!CHILDREN[@]}"; do
  i="${CHILDREN[$idx]}"
  label="${LABELS[$idx]}"
  plan="$PLANS_DIR/overnight-child-${i}.xml"
  log="$LOG_DIR/child-${i}-${label}.log"

  print_banner "Child $((idx+1)) / ${#CHILDREN[@]}: $label"
  echo "  Branch : ${BRANCHES[$idx]}"
  echo "  Plan   : $plan"
  echo "  Log    : $log"
  echo "  Started: $(date)"
  echo ""

  START=$(date +%s)

  prompt="Read and execute the plan at: ${plan}

Use the Read tool to read the plan file, then execute every phase in order.
Use the self-learning-iterative-coder TDD skill for every task.

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

FORBIDDEN (each is a violation — do not do any of these):
  pytest -k 'subset'      never filter tests
  pytest --ignore=...     never ignore test directories
  git commit --no-verify  never bypass pre-commit
  SKIP=hook pre-commit    never skip individual hooks

Then stop."

  exit_code=0
  if timeout "${CHILD_TIMEOUT_SEC}" claude \
    --dangerously-skip-permissions \
    --output-format stream-json \
    --verbose \
    --include-partial-messages \
    -p "${prompt}" \
    2>&1 | tee "$log" | jq -rj 'select(.type=="stream_event" and (.event.delta.type?=="text_delta")) | .event.delta.text'; then

    END=$(date +%s)
    echo ""
    echo "  ✓ Child $((idx+1)) ($label) done in $(( (END - START) / 60 )) min"
    echo "    Log: $log"
  else
    END=$(date +%s)
    echo ""
    echo "  ✗ Child $((idx+1)) ($label) FAILED after $(( (END - START) / 60 )) min"
    echo "    Log: $log"
    FAILED+=("$i:$label")
  fi
done

END_ALL=$(date +%s)
TOTAL=$(( (END_ALL - START_ALL) / 60 ))

print_banner "Script consolidation complete — $(date)"
echo "  Total time: ${TOTAL} min"
echo ""

if [ ${#FAILED[@]} -eq 0 ]; then
  echo "  All ${#CHILDREN[@]} child plans succeeded."
  echo ""
  echo "  Next steps:"
  echo "    1. Review PRs on GitHub"
  echo "    2. Merge feat/hydra-bridge FIRST (feat/debug-configs depends on it)"
  echo "    3. Then run: bash docs/planning/overnight-prefect-docker-monai.sh"
  echo "    4. Smoke test: docker compose run --rm -e EXPERIMENT=debug_single_model train"
else
  echo "  Failed plans (${#FAILED[@]}):"
  for f in "${FAILED[@]}"; do
    echo "    - $f"
    child_id="${f%%:*}"
    echo "      Log: $LOG_DIR/child-${child_id}-*.log"
    echo "      Resume: claude --dangerously-skip-permissions -p 'Resume from where you left off. Read the log at $LOG_DIR/child-${child_id}-*.log first.'"
  done
  exit 1
fi
