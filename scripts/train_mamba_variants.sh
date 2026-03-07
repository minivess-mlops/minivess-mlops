#!/bin/bash
################################################################################
# train_mamba_variants.sh
#
# Full training pipeline for Mamba model variants on MiniVess dataset.
# Runs inside Docker via docker compose (CLAUDE.md Rules #17, #18).
#
# Variants:
#   - comma_mamba:  Coordinate Mamba 3D segmentation (Shi et al., 2025)
#                   ~6 GB VRAM, O(n) state-space complexity
#   - ulike_mamba:  U-Like Mamba, lightweight ~2M params, ~4 GB VRAM
#
# Both variants run 3 folds with the default cbdice_cldice loss.
#
# Volume mounts defined in docker-compose.flows.yml:
#   data_cache   -> /app/data
#   configs      -> /app/configs/splits
#   checkpoints  -> /app/checkpoints
#   mlruns       -> /app/mlruns
#   logs         -> /app/logs
#
# Prerequisites:
#   - Docker with GPU support (nvidia-container-toolkit)
#   - Infrastructure stack running (docker compose -f deployment/docker-compose.yml up)
#   - MiniVess dataset available in the data_cache volume
#
# Usage:
#   ./scripts/train_mamba_variants.sh                    # 100 epochs
#   ./scripts/train_mamba_variants.sh --epochs 50        # 50 epochs
#   ./scripts/train_mamba_variants.sh --debug            # smoke test (1 epoch)
#
# Exit codes:
#   0  All variants trained successfully
#   1  At least one variant failed
#   2  Missing prerequisites
################################################################################

set -euo pipefail
cd "$(dirname "$0")/.."

# ==============================================================================
# Configuration
# ==============================================================================

EPOCHS=100
LOSS="cbdice_cldice"
NUM_FOLDS=3
BATCH_SIZE=1
DEBUG_MODE=0

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
    echo -e "${GREEN}[OK]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $*"
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Train both Mamba variants for MiniVess segmentation (3 folds each) via Docker Compose.

Variants:
    comma_mamba  Coordinate Mamba 3D (Shi et al., 2025), ~6 GB VRAM
    ulike_mamba  U-Like Mamba, lightweight ~2M params, ~4 GB VRAM

Options:
    --epochs N              Number of max epochs (default: 100)
    --debug                 Smoke test mode: 1 epoch per variant
    -h, --help              Show this help message

VRAM guide:
    ulike_mamba  ~4 GB  -> safe on 8GB
    comma_mamba  ~6 GB  -> safe on 8GB

Examples:
    # Full 100-epoch training
    $0

    # Quick smoke test (1 epoch each variant)
    $0 --debug
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
        --debug)
            DEBUG_MODE=1
            EPOCHS=1
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

if ! command -v docker &> /dev/null; then
    log_error "docker not found. Install Docker with GPU support."
    exit 2
fi

if [ ! -f "deployment/docker-compose.flows.yml" ]; then
    log_error "deployment/docker-compose.flows.yml not found."
    exit 2
fi

# ==============================================================================
# Setup
# ==============================================================================

VARIANTS=("comma_mamba" "ulike_mamba")

echo ""
log_info "=== Mamba Variants Training (Docker) ==="
log_info "Start:    $(date)"
log_info "GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
log_info "Variants: ${VARIANTS[*]}"
log_info "Loss:     ${LOSS}"
log_info "Epochs:   ${EPOCHS}"
log_info "Folds:    ${NUM_FOLDS}"
log_info "Debug:    $([ $DEBUG_MODE -eq 1 ] && echo 'YES' || echo 'NO')"
log_info "========================================="
echo ""

# ==============================================================================
# Training Loop
# ==============================================================================

FAILED=()
PASSED=()

for VARIANT in "${VARIANTS[@]}"; do
    echo ""
    log_info "--------------------------------------------"
    log_info "Training: ${VARIANT}"
    log_info "--------------------------------------------"

    if docker compose \
        -f deployment/docker-compose.flows.yml \
        run \
        --rm \
        -e MODEL_FAMILY="${VARIANT}" \
        -e LOSS_NAME="${LOSS}" \
        -e MAX_EPOCHS="${EPOCHS}" \
        -e NUM_FOLDS="${NUM_FOLDS}" \
        -e BATCH_SIZE="${BATCH_SIZE}" \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -e ORT_LOGGING_LEVEL=3 \
        train; then
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
log_info "========================================="
log_info "Training Complete: $(date)"
log_info "========================================="

if [ ${#PASSED[@]} -gt 0 ]; then
    log_success "Passed (${#PASSED[@]}/${#VARIANTS[@]}): ${PASSED[*]}"
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    log_error "Failed (${#FAILED[@]}/${#VARIANTS[@]}): ${FAILED[*]}"
    exit 1
else
    log_success "All Mamba variants trained successfully!"
    exit 0
fi
