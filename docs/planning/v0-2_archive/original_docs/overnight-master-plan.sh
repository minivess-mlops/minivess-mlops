#!/usr/bin/env bash
# overnight-master-plan.sh — runs all 4 child plans sequentially with live logging
# Usage: bash docs/planning/overnight-master-plan.sh
# Each child plan runs in its own claude session with --dangerously-skip-permissions.
# Logs go to /tmp/minivess-overnight/ with live tail output.

set -euo pipefail

REPO="/home/petteri/Dropbox/github-personal/minivess-mlops"
PLANS_DIR="$REPO/docs/planning"
LOG_DIR="/tmp/minivess-overnight"
CHILD_TIMEOUT_SEC="${CHILD_TIMEOUT_SEC:-7200}"
mkdir -p "$LOG_DIR"

CHILDREN=(01 02 03 04)
LABELS=(
  "infrastructure-hardening"
  "monai-ecosystem-audit"
  "test-tiers"
  "training-resume-hpo"
)

print_banner() {
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  $1"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

START_ALL=$(date +%s)
print_banner "MinIVess overnight execution started — $(date)"
echo "  Logs → $LOG_DIR"
echo "  Plans: ${#CHILDREN[@]} child plans"
echo ""

FAILED=()

for idx in "${!CHILDREN[@]}"; do
  i="${CHILDREN[$idx]}"
  label="${LABELS[$idx]}"
  plan="$PLANS_DIR/overnight-child-$i.xml"
  log="$LOG_DIR/child-$i-${label}.log"

  print_banner "Child $i / ${#CHILDREN[@]}: $label"
  echo "  Plan: $plan"
  echo "  Log:  $log"
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
  Close issues when done.

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
    echo "  ✓ Child $i done in $(( (END - START) / 60 )) min"
  else
    END=$(date +%s)
    echo ""
    echo "  ✗ Child $i FAILED after $(( (END - START) / 60 )) min — continuing with next"
    FAILED+=("$i:$label")
  fi
done

END_ALL=$(date +%s)
TOTAL=$(( (END_ALL - START_ALL) / 60 ))

print_banner "Overnight run complete — $(date)"
echo "  Total time: ${TOTAL} min"
echo ""

if [ ${#FAILED[@]} -eq 0 ]; then
  echo "  All ${#CHILDREN[@]} child plans succeeded."
else
  echo "  Failed plans (${#FAILED[@]}):"
  for f in "${FAILED[@]}"; do
    echo "    - $f"
    child_id="${f%%:*}"
    echo "      Log: $LOG_DIR/child-$child_id-*.log"
  done
  exit 1
fi
