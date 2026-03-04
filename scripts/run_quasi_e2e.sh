#!/usr/bin/env bash
# run_quasi_e2e.sh — Run all quasi-E2E tests for model×loss combinations.
#
# This script:
# 1. Validates the capability schema (--check)
# 2. Runs unit tests for discovery + runner infrastructure
# 3. Runs quasi-E2E forward-backward tests for all practical combos
# 4. Reports results
#
# Usage:
#   ./scripts/run_quasi_e2e.sh          # Full quasi-E2E run
#   ./scripts/run_quasi_e2e.sh --quick  # Unit tests only (no fwd-bwd)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

QUICK_MODE=false
if [[ "${1:-}" == "--quick" ]]; then
    QUICK_MODE=true
fi

echo "============================================"
echo "  Quasi-E2E Method Testing Pipeline"
echo "============================================"
echo ""

# Step 1: Capability schema check
echo -e "${YELLOW}[1/4] Checking capability schema consistency...${NC}"
if uv run python -m minivess.testing.capability_discovery --check 2>/dev/null; then
    echo -e "${GREEN}  ✓ Schema consistent${NC}"
else
    echo -e "${RED}  ✗ Schema inconsistent — fix method_capabilities.yaml${NC}"
    exit 1
fi
echo ""

# Step 2: Print discovery summary
echo -e "${YELLOW}[2/4] Discovery summary:${NC}"
uv run python -m minivess.testing.capability_discovery 2>/dev/null
echo ""

# Step 3: Unit tests (discovery + runner infrastructure)
echo -e "${YELLOW}[3/4] Running unit tests...${NC}"
uv run pytest tests/v2/unit/test_capability_discovery.py \
    tests/v2/unit/test_harmonized_output.py \
    tests/v2/unit/test_debug_data_config.py \
    tests/v2/unit/test_quasi_e2e_runner.py \
    tests/v2/unit/test_capability_check.py \
    -x -q 2>&1
UNIT_STATUS=$?
if [[ $UNIT_STATUS -eq 0 ]]; then
    echo -e "${GREEN}  ✓ Unit tests passed${NC}"
else
    echo -e "${RED}  ✗ Unit tests failed${NC}"
    exit 1
fi
echo ""

if [[ "$QUICK_MODE" == true ]]; then
    echo -e "${GREEN}Quick mode: skipping quasi-E2E forward-backward tests.${NC}"
    echo ""
    echo "============================================"
    echo -e "${GREEN}  All quick checks passed!${NC}"
    echo "============================================"
    exit 0
fi

# Step 4: Quasi-E2E forward-backward tests
echo -e "${YELLOW}[4/4] Running quasi-E2E forward-backward tests...${NC}"
echo "  (This tests all model×loss practical combinations)"
echo ""
uv run pytest tests/v2/quasi_e2e/ -v --tb=short 2>&1
E2E_STATUS=$?
echo ""
if [[ $E2E_STATUS -eq 0 ]]; then
    echo -e "${GREEN}  ✓ All quasi-E2E tests passed!${NC}"
else
    echo -e "${RED}  ✗ Some quasi-E2E tests failed${NC}"
    exit 1
fi

echo ""
echo "============================================"
echo -e "${GREEN}  All quasi-E2E tests passed!${NC}"
echo "============================================"
