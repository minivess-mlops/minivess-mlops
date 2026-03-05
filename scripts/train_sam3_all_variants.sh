#!/bin/bash
################################################################################
# train_sam3_all_variants.sh
#
# Full training pipeline for all 3 SAMv3 variants on MiniVess dataset
# - sam3_vanilla:   Frozen SAM3 + trainable decoder
# - sam3_topolora:  SAM3 + LoRA + topology-aware loss (cbdice_cldice)
# - sam3_hybrid:    Frozen SAM3 + DynUNet 3D + gated fusion
#
# Prerequisites:
#   - uv installed and project dependencies in sync
#   - GPU with 8GB+ VRAM (or adjust --compute accordingly)
#   - MiniVess dataset available in configured data path
#
# Usage:
#   ./scripts/train_sam3_all_variants.sh                    # 100 epochs, auto compute
#   ./scripts/train_sam3_all_variants.sh --epochs 50        # 50 epochs
#   ./scripts/train_sam3_all_variants.sh --compute gpu_low  # explicit compute profile
#   ./scripts/train_sam3_all_variants.sh --debug            # smoke test (1 epoch)
#
# Environment:
#   MINIVESS_LOG_DIR  Override log directory (default: ./logs/sam3_variants)
#   MINIVESS_COMPUTE  Override compute profile (default: auto)
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
COMPUTE="${MINIVESS_COMPUTE:-auto}"
LOG_BASE="${MINIVESS_LOG_DIR:-${PROJECT_ROOT}/logs/sam3_variants}"
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

Train all 3 SAMv3 variants for MiniVess segmentation.

Options:
    --epochs N              Number of max epochs (default: 100)
    --compute PROFILE       Compute profile: auto, gpu_low, gpu_mid, gpu_full, cpu
                           (default: auto)
    --log-dir DIR          Output directory for logs and checkpoints
                           (default: ./logs/sam3_variants)
    --debug                Smoke test mode: 1 epoch per variant (debug loss)
    --resume               Resume from existing checkpoints (if available)
    -h, --help            Show this help message

Examples:
    # Full 100-epoch training on GPU
    $0

    # Quick smoke test (1 epoch each)
    $0 --debug

    # 50 epochs with explicit GPU profile
    $0 --epochs 50 --compute gpu_low

    # Resume interrupted training
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

# Verify uv sync is up to date
log_info "Syncing dependencies..."
uv sync --quiet || { log_error "Failed to sync dependencies"; exit 2; }

# ==============================================================================
# Setup
# ==============================================================================

mkdir -p "${LOG_BASE}"

log_info "SAMv3 Full Training Pipeline"
log_info "=========================================="
log_info "Variants: sam3_vanilla, sam3_topolora, sam3_hybrid"
log_info "Epochs: ${EPOCHS}"
log_info "Compute: ${COMPUTE}"
log_info "Log dir: ${LOG_BASE}"
log_info "Debug mode: $([ $DEBUG_MODE -eq 1 ] && echo 'YES' || echo 'NO')"
log_info "Resume mode: $([ $RESUME_MODE -eq 1 ] && echo 'YES' || echo 'NO')"
log_info ""

# ==============================================================================
# Training Loop
# ==============================================================================

VARIANTS=("sam3_vanilla" "sam3_topolora" "sam3_hybrid")
FAILED=()
PASSED=()

for VARIANT in "${VARIANTS[@]}"; do
    LOG_DIR="${LOG_BASE}/${VARIANT}"

    echo ""
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "Training: ${VARIANT}"
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Build command
    CMD=(
        "uv" "run" "python" "scripts/train_monitored.py"
        "--model-family" "${VARIANT}"
        "--max-epochs" "${EPOCHS}"
        "--compute" "${COMPUTE}"
        "--log-dir" "${LOG_DIR}"
    )

    # Add optional flags
    if [ $DEBUG_MODE -eq 1 ]; then
        CMD+=("--debug")
    fi

    if [ $RESUME_MODE -eq 1 ]; then
        CMD+=("--resume")
    fi

    # Log the command
    log_info "Command: ${CMD[*]}"
    log_info "Logging to: ${LOG_DIR}"

    # Run training
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
    log_success "All variants trained successfully!"
    echo ""
    log_info "Results saved to: ${LOG_BASE}"
    log_info "Next steps:"
    log_info "  1. Inspect logs: ls -lh ${LOG_BASE}/*/logs/"
    log_info "  2. Check metrics: uv run python scripts/analyze_mlflow_runs.py"
    log_info "  3. Compare SAMv3 vs DynUNet baseline: ./scripts/compare_models.sh"
    exit 0
fi
