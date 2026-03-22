#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG="/tmp/minivess-overnight/child-1-fresh.log"
PLAN="${REPO}/docs/planning/overnight-child-01-acquisition.xml"

mkdir -p /tmp/minivess-overnight
pkill -f 'claude.*dangerous' 2>/dev/null || true
sleep 2

cd "${REPO}"

timeout 7200 claude \
  --dangerously-skip-permissions \
  --output-format stream-json \
  --verbose \
  --include-partial-messages \
  -p "You are executing a TDD implementation plan for the MinIVess MLOps repo.

REPO ROOT: ${REPO}
PLAN FILE: ${PLAN}

Read the plan file first (use the Read tool), then execute every phase in order.

CRITICAL RULES (from CLAUDE.md — non-negotiable):
- TDD: write failing tests first (RED), then minimum implementation (GREEN)
- Library-First: use MONAI, Prefect, existing code before writing custom logic
- GitHub Actions EXPLICITLY DISABLED — never add automatic CI triggers
- Every test failure must be fixed or reported immediately

COMPLETION PROTOCOL — mandatory, no shortcuts, no exceptions:

STEP 1 — FULL TEST SUITE (loop until zero failures):
  uv run pytest tests/ -x -q
  Repeat until exit 0. Never use -k, --ignore, or xfail.

STEP 2 — PRE-COMMIT ON ALL FILES (loop until fully clean):
  uv run pre-commit run --all-files
  Repeat until exit 0. Never use --no-verify or SKIP=.

STEP 3 — ONLY after Step 1 AND Step 2 are both exit 0:
  git push -u origin HEAD
  gh pr create with descriptive title and body." \
  2>&1 | tee "${LOG}" \
       | jq -rj 'select(.type=="stream_event" and (.event.delta.type?=="text_delta")) | .event.delta.text'
