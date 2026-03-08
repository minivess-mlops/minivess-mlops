#!/usr/bin/env bash
# Prod tests: full suite including slow, integration, and GPU tests.
#
# Runs ALL tests — no test left behind.
# Use for nightly runs or before merging to main.
#
# Usage: bash scripts/run_prod_tests.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Prod Tests (full suite) ==="
uv run pytest tests/ -x --tb=short "$@"
