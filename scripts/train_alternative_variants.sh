#!/bin/bash
################################################################################
# train_alternative_variants.sh
#
# Full training pipeline for non-DynUNet, non-SAM3 model variants:
#   - vesselfm:    VesselFM foundation model (Wittmann et al., CVPR 2025)
#                  DynUNet pre-trained on 17 vessel datasets. ~30M params.
#                  REQUIRES ~10GB VRAM — may OOM on 8GB GPUs (see --skip-heavy)
#   - comma_mamba: Coordinate Mamba 3D segmentation (Shi et al., 2025)
#   - ulike_mamba: U-Like Mamba, lightweight ~2M params, O(n) complexity
#
# Prerequisites:
#   - uv installed, project dependencies synced
#   - GPU with 8GB+ VRAM (16GB+ recommended for vesselfm without --skip-heavy)
#   - MiniVess dataset available in configured data path
#
# Usage:
#   ./scripts/train_alternative_variants.sh                      # 100 epochs, gpu_low
#   ./scripts/train_alternative_variants.sh --epochs 50          # 50 epochs
#   ./scripts/train_alternative_variants.sh --compute gpu_high   # higher-VRAM profile
#   ./scripts/train_alternative_variants.sh --debug              # smoke test (1 epoch)
#   ./scripts/train_alternative_variants.sh --skip-heavy         # skip vesselfm (8GB GPU)
#
# Environment:
#   MINIVESS_LOG_DIR  Override log directory (default: ./logs/alternative_variants)
#   MINIVESS_COMPUTE  Override compute profile (default: gpu_low)
#   MINIVESS_EPOCHS   Override epochs (default: 100)
#
# Exit codes:
#   0  All requested variants trained successfully
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
LOG_BASE="${MINIVESS_LOG_DIR:-${PROJECT_ROOT}/logs/alternative_variants}"
DEBUG_MODE=0
RESUME_MODE=0
SKIP_HEAVY=0

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

Train non-DynUNet, non-SAM3 model variants for MiniVess segmentation.

Variants:
    vesselfm     VesselFM foundation model (~30M params, ~10GB VRAM)
    comma_mamba  Coordinate Mamba 3D architecture
    ulike_mamba  Lightweight U-Like Mamba (~2M params, ~4GB VRAM)

Options:
    --epochs N              Number of max epochs (default: 100)
    --compute PROFILE       Compute profile: cpu, gpu_low, gpu_high, dgx_spark,
                           cloud_single, cloud_multi (default: gpu_low)
    --log-dir DIR          Output directory for logs and checkpoints
                           (default: ./logs/alternative_variants)
    --skip-heavy           Skip vesselfm (use on 8GB GPUs to avoid OOM)
    --debug                Smoke test mode: 1 epoch per variant
    --resume               Resume from existing checkpoints (if available)
    -h, --help            Show this help message

VRAM guide:
    ulike_mamba  ~4 GB   → gpu_low safe
    comma_mamba  ~6 GB   → gpu_low safe
    vesselfm     ~10 GB  → requires gpu_high (16GB+); use --skip-heavy on 8GB

Examples:
    # Full training on 16GB+ GPU
    $0 --compute gpu_high

    # 8GB GPU: skip VesselFM
    $0 --skip-heavy

    # Quick smoke test
    $0 --debug --skip-heavy
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
        --skip-heavy)
            SKIP_HEAVY=1
            shift
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

# Build variant list; optionally skip VesselFM on 8GB GPUs
if [ $SKIP_HEAVY -eq 1 ]; then
    VARIANTS=("comma_mamba" "ulike_mamba")
    log_warn "--skip-heavy: skipping vesselfm (requires ~10GB VRAM)"
else
    VARIANTS=("vesselfm" "comma_mamba" "ulike_mamba")
    if [[ "$COMPUTE" == "gpu_low" ]]; then
        log_warn "vesselfm requires ~10GB VRAM but gpu_low targets 8GB GPUs."
        log_warn "If you hit OOM, re-run with --skip-heavy or --compute gpu_high."
    fi
fi

mkdir -p "${LOG_BASE}"

log_info "Alternative Variants Training Pipeline"
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

    # Build command — calls training_flow() directly (Prefect-aware, no orphan run)
    CMD=(
        "uv" "run" "python" "scripts/run_training_flow.py"
        "--model-family" "${VARIANT}"
        "--max-epochs" "${EPOCHS}"
        "--compute" "${COMPUTE}"
        "--trigger-source" "train_alternative_variants.sh"
    )

    if [ $DEBUG_MODE -eq 1 ]; then
        CMD+=("--debug")
    fi

    log_info "Command: ${CMD[*]}"

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
    log_info "  3. Compare vs DynUNet baseline: uv run python scripts/compare_models.py"
    exit 0
fi
