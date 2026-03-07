#!/bin/bash
################################################################################
# train_mamba_variants.sh
#
# Full training pipeline for Mamba model variants on MiniVess dataset:
#   - comma_mamba:  Coordinate Mamba 3D segmentation (Shi et al., 2025)
#                   ~6 GB VRAM, O(n) state-space complexity
#   - ulike_mamba:  U-Like Mamba, lightweight ~2M params, ~4 GB VRAM
#
# Both variants run 3 folds (from configs/splits/3fold_seed42.json) with
# the default cbdice_cldice loss.
#
# Prerequisites:
#   - uv installed and project dependencies synced
#   - GPU with 8GB VRAM (gpu_low profile)
#   - MiniVess dataset available in configured data path
#
# Usage:
#   ./scripts/train_mamba_variants.sh                    # 100 epochs, gpu_low
#   ./scripts/train_mamba_variants.sh --epochs 50        # 50 epochs
#   ./scripts/train_mamba_variants.sh --compute gpu_high # higher-VRAM profile
#   ./scripts/train_mamba_variants.sh --debug            # smoke test (1 epoch)
#   ./scripts/train_mamba_variants.sh --resume           # resume checkpoints
#
# Environment:
#   MINIVESS_LOG_DIR  Override log directory (default: ./logs/mamba_variants)
#   MINIVESS_COMPUTE  Override compute profile (default: gpu_low)
#   MINIVESS_EPOCHS   Override epochs (default: 100)
#
# Exit codes:
#   0  All variants trained successfully
#   1  At least one variant failed
#   2  Missing prerequisites
################################################################################

set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EPOCHS="${MINIVESS_EPOCHS:-100}"
COMPUTE="${MINIVESS_COMPUTE:-gpu_low}"
LOG_BASE="${MINIVESS_LOG_DIR:-${PROJECT_ROOT}/logs/mamba_variants}"
DEBUG_MODE=0
RESUME_MODE=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==============================================================================
# Functions
# ==============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[✗]${NC} $*"
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Train both Mamba variants for MiniVess segmentation (3 folds each).

Variants:
    comma_mamba  Coordinate Mamba 3D (Shi et al., 2025), ~6 GB VRAM
    ulike_mamba  U-Like Mamba, lightweight ~2M params, ~4 GB VRAM

Options:
    --epochs N              Number of max epochs (default: 100)
    --compute PROFILE       Compute profile: cpu, gpu_low, gpu_high
                            (default: gpu_low)
    --log-dir DIR           Output directory for logs and checkpoints
                            (default: ./logs/mamba_variants)
    --debug                 Smoke test mode: 1 epoch per variant
    --resume                Resume from existing checkpoints (if available)
    -h, --help              Show this help message

VRAM guide:
    ulike_mamba  ~4 GB  → gpu_low safe
    comma_mamba  ~6 GB  → gpu_low safe

Examples:
    # Full 100-epoch training
    $0

    # Quick smoke test (1 epoch each variant)
    $0 --debug

    # Resume interrupted run
    $0 --resume
EOF
    exit 0
}

# ==============================================================================
# Argument Parsing
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --compute)
            COMPUTE="$2"
            shift 2
            ;;
        --log-dir)
            LOG_BASE="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE=1
            EPOCHS=1
            shift
            ;;
        --resume)
            RESUME_MODE=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# ==============================================================================
# Validation
# ==============================================================================

if ! command -v uv &> /dev/null; then
    log_error "uv not found. Install uv or add to PATH."
    exit 2
fi

if [ ! -f "${PROJECT_ROOT}/pyproject.toml" ]; then
    log_error "pyproject.toml not found in ${PROJECT_ROOT}"
    exit 2
fi

cd "${PROJECT_ROOT}" || exit 2

log_info "Syncing dependencies..."
uv sync --quiet || { log_error "Failed to sync dependencies"; exit 2; }

# ==============================================================================
# Setup
# ==============================================================================

VARIANTS=("comma_mamba" "ulike_mamba")

mkdir -p "${LOG_BASE}"

log_info "Mamba Variants Training Pipeline"
log_info "=========================================="
log_info "Variants: ${VARIANTS[*]}"
log_info "Epochs: ${EPOCHS}"
log_info "Compute: ${COMPUTE}"
log_info "Log dir: ${LOG_BASE}"
log_info "Debug mode: $([ $DEBUG_MODE -eq 1 ] && echo 'YES' || echo 'NO')"
log_info "Resume mode: $([ $RESUME_MODE -eq 1 ] && echo 'YES' || echo 'NO')"
log_info ""

# ==============================================================================
# Training Loop
# ==============================================================================

FAILED=()
PASSED=()

for VARIANT in "${VARIANTS[@]}"; do
    LOG_DIR="${LOG_BASE}/${VARIANT}"

    echo ""
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "Training: ${VARIANT}"
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    CMD=(
        "uv" "run" "python" "scripts/train_monitored.py"
        "--model-family" "${VARIANT}"
        "--max-epochs" "${EPOCHS}"
        "--compute" "${COMPUTE}"
        "--log-dir" "${LOG_DIR}"
    )

    # Model-specific memory overrides for gpu_low (8 GB VRAM)
    # COMMA Mamba flattens full spatial volume to sequence → more memory intensive.
    # Use patch=48x48x24 + batch=1 on gpu_low to stay within 7.5 GB.
    # Z=24 matches the DynUNet pipeline convention; XY reduced from 96→48 for memory.
    if [[ "${VARIANT}" == "comma_mamba" && "${COMPUTE}" == "gpu_low" ]]; then
        CMD+=("--patch-size" "48x48x24" "--batch-size" "1")
        log_info "  comma_mamba gpu_low override: patch=48x48x24 batch=1"
    fi

    if [ $DEBUG_MODE -eq 1 ]; then
        CMD+=("--debug")
    fi

    if [ $RESUME_MODE -eq 1 ]; then
        CMD+=("--resume")
    fi

    log_info "Command: ${CMD[*]}"
    log_info "Logging to: ${LOG_DIR}"

    if "${CMD[@]}"; then
        log_success "${VARIANT} completed"
        PASSED+=("${VARIANT}")
    else
        log_error "${VARIANT} failed (exit code: $?)"
        FAILED+=("${VARIANT}")
    fi
done

# ==============================================================================
# Summary
# ==============================================================================

echo ""
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "Training Complete"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ${#PASSED[@]} -gt 0 ]; then
    log_success "Passed (${#PASSED[@]}/${#VARIANTS[@]}): ${PASSED[*]}"
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    log_error "Failed (${#FAILED[@]}/${#VARIANTS[@]}): ${FAILED[*]}"
    echo ""
    log_warn "Check logs for details:"
    for variant in "${FAILED[@]}"; do
        echo "  ${LOG_BASE}/${variant}/"
    done
    exit 1
else
    log_success "All Mamba variants trained successfully!"
    echo ""
    log_info "Results saved to: ${LOG_BASE}"
    log_info "Next steps:"
    log_info "  1. Inspect logs: ls -lh ${LOG_BASE}/*/monitor/"
    log_info "  2. Check metrics: uv run python scripts/analyze_mlflow_runs.py"
    log_info "  3. Compare vs DynUNet baseline: uv run python scripts/compare_models.py"
    exit 0
fi
