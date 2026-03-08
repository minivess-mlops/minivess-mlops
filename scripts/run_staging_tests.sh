#!/usr/bin/env bash
# Staging tests: fast unit tests only (target: under 2 minutes)
#
# Excludes: slow, integration, e2e, gpu tests.
# Use for pre-PR local validation.
#
# Usage: bash scripts/run_staging_tests.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Staging Tests (fast unit tests only) ==="
uv run pytest -c pytest-staging.ini "$@"
