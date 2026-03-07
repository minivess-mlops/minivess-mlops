#!/bin/bash
################################################################################
# train_all_best.sh
#
# Train the best-validated configuration of EACH model family on MiniVess
# via Docker (CLAUDE.md Rules #17-19).
#
# Best settings (validated by experiments):
#   dynunet        cbdice_cldice   Standard width ~5M params
#   sam3_vanilla   cbdice_cldice   Most stable SAM3 variant (frozen encoder)
#   comma_mamba    cbdice_cldice   Best Mamba-based model (COMMA-3D)
#
# These produce the baseline paper figures used in the comparison table.
# Run this after any major codebase change to verify nothing regressed.
#
# Usage:
#   ./scripts/train_all_best.sh              # all 3 families, 100 epochs
#   ./scripts/train_all_best.sh --debug      # smoke test, 1 epoch each
#   ./scripts/train_all_best.sh --epochs 50  # quick check
#   ./scripts/train_all_best.sh --families dynunet,comma_mamba
#
# Environment:
#   MINIVESS_EPOCHS    Override epochs (default: 100)
#   MINIVESS_FAMILIES  Override families as comma-separated list
################################################################################

set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${MINIVESS_EPOCHS:-100}"
DEBUG_FLAG=""

# Best config per family: (model_family, loss_name)
declare -A BEST_LOSS=(
    ["dynunet"]="cbdice_cldice"
    ["sam3_vanilla"]="cbdice_cldice"
    ["comma_mamba"]="cbdice_cldice"
)

# Default: run all families
FAMILIES_CSV="${MINIVESS_FAMILIES:-dynunet,sam3_vanilla,comma_mamba}"

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)   EPOCHS="$2";        shift 2 ;;
        --families) FAMILIES_CSV="$2";  shift 2 ;;
        --debug)    DEBUG_FLAG="true";  shift   ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

IFS=',' read -ra FAMILIES <<< "${FAMILIES_CSV}"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

for FAM in "${FAMILIES[@]}"; do
    if [[ -z "${BEST_LOSS[$FAM]+x}" ]]; then
        echo "[ERROR] Unknown model family '${FAM}'." >&2
        echo "        Known: ${!BEST_LOSS[*]}" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Run each family via Docker
# ---------------------------------------------------------------------------

echo "[INFO] Training best configuration for each model family (Docker)"
echo "[INFO]   families : ${FAMILIES[*]}"
echo "[INFO]   epochs   : ${EPOCHS}"
echo "[INFO]   debug    : ${DEBUG_FLAG:-false}"

FAILED=()
PASSED=()

for MODEL in "${FAMILIES[@]}"; do
    LOSS="${BEST_LOSS[$MODEL]}"

    echo ""
    echo "[INFO] Training: ${MODEL} (loss=${LOSS})"

    BATCH=2
    HF_ARGS=""
    if [[ "$MODEL" == sam3_* ]]; then
        BATCH=1
        HF_ARGS="-e HF_TOKEN=${HF_TOKEN:-}"
    fi

    if docker compose \
        -f deployment/docker-compose.flows.yml \
        run \
        --rm \
        -e MODEL_FAMILY="${MODEL}" \
        -e LOSS_NAME="${LOSS}" \
        -e MAX_EPOCHS="${EPOCHS}" \
        -e BATCH_SIZE="${BATCH}" \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -e ORT_LOGGING_LEVEL=3 \
        ${HF_ARGS} \
        ${DEBUG_FLAG:+-e DEBUG=true} \
        train; then
        echo "[OK] ${MODEL} complete"
        PASSED+=("${MODEL}")
    else
        echo "[ERROR] ${MODEL} failed" >&2
        FAILED+=("${MODEL}")
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "[INFO] Training Complete"

if [[ ${#PASSED[@]} -gt 0 ]]; then
    echo "[OK] Passed (${#PASSED[@]}/${#FAMILIES[@]}): ${PASSED[*]}"
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "[ERROR] Failed (${#FAILED[@]}/${#FAMILIES[@]}): ${FAILED[*]}" >&2
    exit 1
fi

exit 0
