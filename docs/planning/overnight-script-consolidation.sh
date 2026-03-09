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

  if stdbuf -o0 claude --dangerously-skip-permissions -p \
    "Read and execute $plan autonomously. Follow ALL instructions exactly. \
     Use the self-learning-iterative-coder TDD skill for every task. \
     Commit after each phase. Create PR when branch is complete. \
     Then stop." \
    2>&1 | tee "$log"; then

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
