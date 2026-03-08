#!/usr/bin/env bash
# overnight-master-plan.sh — runs all 4 child plans sequentially with live logging
# Usage: bash docs/planning/overnight-master-plan.sh
# Each child plan runs in its own claude session with --dangerously-skip-permissions.
# Logs go to /tmp/minivess-overnight/ with live tail output.

set -euo pipefail

REPO="/home/petteri/Dropbox/github-personal/minivess-mlops"
PLANS_DIR="$REPO/docs/planning"
LOG_DIR="/tmp/minivess-overnight"
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

  if claude --dangerously-skip-permissions -p \
    "Read and execute $plan autonomously. Follow ALL instructions exactly. Commit after each phase. Push, create PR, close issues when done. Then stop." \
    2>&1 | tee "$log"; then

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
