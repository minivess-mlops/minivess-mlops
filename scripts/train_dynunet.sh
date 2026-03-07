#!/bin/bash
################################################################################
# train_dynunet.sh
#
# Train the standard DynUNet model on MiniVess.
# One variant: standard width (~5M params) + cbdice_cldice loss.
#
# This is the validated production configuration from dynunet_loss_variation_v2:
#   - cbdice_cldice achieves 0.906 clDice (best topology)
#   - Standard width avoids OOM on 8 GB VRAM with gpu_low profile
#
# Usage:
#   ./scripts/train_dynunet.sh                     # 100 epochs, gpu_low
#   ./scripts/train_dynunet.sh --epochs 50         # 50 epochs
#   ./scripts/train_dynunet.sh --compute auto      # let system auto-detect
#   ./scripts/train_dynunet.sh --debug             # smoke test (1 epoch)
#   ./scripts/train_dynunet.sh --num-folds 1       # single fold, fast check
#
# See CLAUDE.md §Default Loss Function for rationale.
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

EPOCHS="${MINIVESS_EPOCHS:-100}"
COMPUTE="${MINIVESS_COMPUTE:-gpu_low}"
NUM_FOLDS="${MINIVESS_NUM_FOLDS:-3}"
LOSS="cbdice_cldice"
MODEL="dynunet"
DEBUG_MODE=0

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)    EPOCHS="$2";    shift 2 ;;
        --compute)   COMPUTE="$2";   shift 2 ;;
        --num-folds) NUM_FOLDS="$2"; shift 2 ;;
        --debug)     DEBUG_MODE=1;   shift   ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

echo "[INFO] DynUNet training"
echo "[INFO]   model   : ${MODEL}"
echo "[INFO]   loss    : ${LOSS}"
echo "[INFO]   compute : ${COMPUTE}"
echo "[INFO]   epochs  : ${EPOCHS}"
echo "[INFO]   folds   : ${NUM_FOLDS}"
echo "[INFO]   debug   : $([ $DEBUG_MODE -eq 1 ] && echo YES || echo NO)"

CMD=(
    "uv" "run" "python" "scripts/run_training_flow.py"
    "--model-family" "${MODEL}"
    "--loss-name"    "${LOSS}"
    "--compute"      "${COMPUTE}"
    "--max-epochs"   "${EPOCHS}"
    "--num-folds"    "${NUM_FOLDS}"
    "--trigger-source" "train_dynunet.sh"
)

if [ $DEBUG_MODE -eq 1 ]; then
    CMD+=("--debug")
fi

echo "[INFO] Command: ${CMD[*]}"

if "${CMD[@]}"; then
    echo "[OK] DynUNet training complete"
    exit 0
else
    echo "[ERROR] DynUNet training failed (exit code: $?)" >&2
    exit 1
fi
