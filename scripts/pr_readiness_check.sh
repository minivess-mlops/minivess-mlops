#!/usr/bin/env bash
# PR Readiness Check — run BEFORE creating a pull request.
#
# This is NOT a git hook. Run it explicitly:
#   bash scripts/pr_readiness_check.sh
#
# Runs the full validation suite locally:
#   1. Ruff lint + format
#   2. mypy typecheck
#   3. Unit tests (tests/unit/)
#   4. Test collection gate (all test files importable)
#
# Exit code 0 = ready to push. Non-zero = fix issues first.

set -euo pipefail

echo "=== PR Readiness Check ==="
echo ""

echo "[1/4] Ruff lint..."
uv run ruff check src/minivess/ tests/

echo "[2/4] mypy typecheck..."
uv run mypy src/minivess/

if [[ "${1:-}" == "--full" ]]; then
  echo "[3/4] Full test suite (prod tier)..."
  uv run pytest tests/ -x --tb=short -q
else
  echo "[3/4] Staging tests (fast unit tests only)..."
  uv run pytest -c pytest-staging.ini
fi

echo "[4/4] Test collection gate (import check)..."
uv run pytest --collect-only -q tests/unit/ tests/v2/unit/

echo ""
echo "=== ALL CHECKS PASSED — ready to push ==="
