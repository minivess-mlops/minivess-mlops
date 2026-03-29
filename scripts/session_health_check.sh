#!/usr/bin/env bash
# Session Health Check — Layer 2 deterministic defense.
#
# Run at the start of every Claude Code session to surface ALL issues
# before any new work begins. Uses --maxfail=200 (NOT -x) to show
# the full scope of problems, preventing whac-a-mole serial fixing.
#
# Usage:
#   bash scripts/session_health_check.sh
#   make session-health-check
#
# See: docs/planning/v0-2_archive/critical-failure-fixing-and-silent-failure-fix.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="$REPO_ROOT/tests/health_baseline.json"

echo "================================================================"
echo "  SESSION HEALTH CHECK — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"
echo ""

# ── 1. Ruff lint check ──────────────────────────────────────────────
echo "[1/5] Ruff lint check..."
RUFF_ERRORS=$(uv run ruff check src/ tests/ scripts/ --output-format json 2>/dev/null | python3 -c "import json,sys; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "-1")
if [ "$RUFF_ERRORS" = "0" ]; then
    echo "  ✓ Ruff: 0 errors"
else
    echo "  ✗ Ruff: $RUFF_ERRORS errors — FIX BEFORE STARTING WORK"
    uv run ruff check src/ tests/ scripts/ 2>&1 | head -20
fi
echo ""

# ── 2. Test staging (full scope, no -x) ─────────────────────────────
echo "[2/5] Staging test suite (--maxfail=200, NOT -x)..."
# Delegates to Makefile target (host escape hatch handled there, not in .sh scripts)
TEST_OUTPUT=$(make -C "$REPO_ROOT" test-staging-no-stop 2>&1)
PASSED=$(echo "$TEST_OUTPUT" | grep -oP '\d+ passed' | grep -oP '\d+' || echo "0")
FAILED=$(echo "$TEST_OUTPUT" | grep -oP '\d+ failed' | grep -oP '\d+' || echo "0")
SKIPPED=$(echo "$TEST_OUTPUT" | grep -oP '\d+ skipped' | grep -oP '\d+' || echo "0")
echo "  Tests: $PASSED passed, $FAILED failed, $SKIPPED skipped"
if [ "$FAILED" != "0" ]; then
    echo "  ✗ FAILURES DETECTED — FIX BEFORE STARTING WORK:"
    echo "$TEST_OUTPUT" | grep "^FAILED" | head -10
fi
if [ "$SKIPPED" != "0" ]; then
    echo "  ✗ SKIPS DETECTED — investigate per Rule 28:"
    echo "$TEST_OUTPUT" | grep "SKIPPED" | head -5
fi
if [ "$FAILED" = "0" ] && [ "$SKIPPED" = "0" ]; then
    echo "  ✓ All tests green"
fi
echo ""

# ── 3. Deployment staleness check ───────────────────────────────────
echo "[3/5] Deployment staleness check..."
if [ -f "$BASELINE" ]; then
    BASELINE_PYPROJECT_HASH=$(python3 -c "import json; print(json.load(open('$BASELINE'))['deployment_state']['pyproject_toml_sha256'])" 2>/dev/null || echo "unknown")
    CURRENT_PYPROJECT_HASH=$(sha256sum "$REPO_ROOT/pyproject.toml" | cut -d' ' -f1)
    if [ "$CURRENT_PYPROJECT_HASH" != "$BASELINE_PYPROJECT_HASH" ]; then
        echo "  ✗ pyproject.toml CHANGED since last Docker build — REBUILD NEEDED"
        echo "    Run: make build-base-gpu && docker push to GAR"
        echo "    Then: uv run python scripts/update_health_baseline.py --docker-pushed"
    else
        echo "  ✓ pyproject.toml matches last Docker build"
    fi

    BASELINE_LOCK_HASH=$(python3 -c "import json; print(json.load(open('$BASELINE'))['deployment_state']['uv_lock_sha256'])" 2>/dev/null || echo "unknown")
    CURRENT_LOCK_HASH=$(sha256sum "$REPO_ROOT/uv.lock" | cut -d' ' -f1)
    if [ "$CURRENT_LOCK_HASH" != "$BASELINE_LOCK_HASH" ]; then
        echo "  ✗ uv.lock CHANGED since last Docker build — REBUILD NEEDED"
    else
        echo "  ✓ uv.lock matches last Docker build"
    fi

    PULUMI_PENDING=$(python3 -c "import json; print(json.load(open('$BASELINE'))['deployment_state'].get('pulumi_pending_changes', False))" 2>/dev/null || echo "False")
    if [ "$PULUMI_PENDING" = "True" ]; then
        echo "  ✗ Pulumi has PENDING changes — deploy before launching experiments"
        echo "    Run: cd deployment/pulumi/gcp && pulumi up"
    else
        echo "  ✓ Pulumi: no pending changes"
    fi
else
    echo "  ⚠ No baseline file — run: uv run python scripts/update_health_baseline.py"
fi
echo ""

# ── 4. Region consistency ───────────────────────────────────────────
echo "[4/5] Region consistency check..."
STALE_REFS=$(grep -rl "europe-north1" "$REPO_ROOT/deployment/skypilot/" "$REPO_ROOT/configs/registry/" "$REPO_ROOT/configs/cloud/gcp_spot.yaml" "$REPO_ROOT/configs/factorial/debug.yaml" "$REPO_ROOT/.sky.yaml" "$REPO_ROOT/scripts/preflight_gcp.py" 2>/dev/null | grep -v __pycache__ | wc -l)
if [ "$STALE_REFS" != "0" ]; then
    echo "  ✗ $STALE_REFS files still reference europe-north1 — MIGRATION INCOMPLETE"
else
    echo "  ✓ No stale europe-north1 references in active config"
fi
echo ""

# ── 5. Health baseline comparison ───────────────────────────────────
echo "[5/5] Baseline comparison..."
if [ -f "$BASELINE" ]; then
    BASELINE_PASSED=$(python3 -c "import json; print(json.load(open('$BASELINE'))['test_staging']['passed'])" 2>/dev/null || echo "0")
    BASELINE_RUFF=$(python3 -c "import json; print(json.load(open('$BASELINE'))['ruff']['error_count'])" 2>/dev/null || echo "0")
    echo "  Baseline: $BASELINE_PASSED tests, $BASELINE_RUFF ruff errors"
    echo "  Current:  $PASSED tests, $RUFF_ERRORS ruff errors"
    if [ "$PASSED" -lt "$BASELINE_PASSED" ] 2>/dev/null; then
        echo "  ✗ TEST COUNT DECREASED: $PASSED < $BASELINE_PASSED"
    fi
else
    echo "  ⚠ No baseline to compare against"
fi

echo ""
echo "================================================================"
echo "  HEALTH CHECK COMPLETE"
echo "================================================================"
