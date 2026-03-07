#!/bin/bash
################################################################################
# train_all_best.sh
#
# Train the best-validated configuration of EACH model family on MiniVess.
# One run per family — the production "reference" set for comparison tables.
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
#                                            # subset of families
#
# Environment:
#   MINIVESS_EPOCHS    Override epochs (default: 100)
#   MINIVESS_COMPUTE   Override compute profile (default: gpu_low)
#   MINIVESS_FAMILIES  Override families as comma-separated list
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

EPOCHS="${MINIVESS_EPOCHS:-100}"
COMPUTE="${MINIVESS_COMPUTE:-gpu_low}"
DEBUG_MODE=0

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
        --epochs)   EPOCHS="$2";      shift 2 ;;
        --compute)  COMPUTE="$2";     shift 2 ;;
        --families) FAMILIES_CSV="$2"; shift 2 ;;
        --debug)    DEBUG_MODE=1;      shift   ;;
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
# Run
# ---------------------------------------------------------------------------

echo "[INFO] Training best configuration for each model family"
echo "[INFO]   families : ${FAMILIES[*]}"
echo "[INFO]   epochs   : ${EPOCHS}"
echo "[INFO]   compute  : ${COMPUTE}"
echo "[INFO]   debug    : $([ $DEBUG_MODE -eq 1 ] && echo YES || echo NO)"

FAILED=()
PASSED=()

for MODEL in "${FAMILIES[@]}"; do
    LOSS="${BEST_LOSS[$MODEL]}"

    echo ""
    echo "[INFO] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[INFO] Training: ${MODEL} (loss=${LOSS})"
    echo "[INFO] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    CMD=(
        "uv" "run" "python" "scripts/run_training_flow.py"
        "--model-family"   "${MODEL}"
        "--loss-name"      "${LOSS}"
        "--compute"        "${COMPUTE}"
        "--max-epochs"     "${EPOCHS}"
        "--trigger-source" "train_all_best.sh"
    )

    if [ $DEBUG_MODE -eq 1 ]; then
        CMD+=("--debug")
    fi

    echo "[INFO] Command: ${CMD[*]}"

    if "${CMD[@]}"; then
        echo "[OK] ${MODEL} complete"
        PASSED+=("${MODEL}")
    else
        echo "[ERROR] ${MODEL} failed (exit code: $?)" >&2
        FAILED+=("${MODEL}")
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "[INFO] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[INFO] Training Complete"
echo "[INFO] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ ${#PASSED[@]} -gt 0 ]]; then
    echo "[OK] Passed (${#PASSED[@]}/${#FAMILIES[@]}): ${PASSED[*]}"
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "[ERROR] Failed (${#FAILED[@]}/${#FAMILIES[@]}): ${FAILED[*]}" >&2
    exit 1
fi

exit 0
