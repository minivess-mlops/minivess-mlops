#!/bin/bash
################################################################################
# train_dynunet.sh
#
# Train the standard DynUNet model on MiniVess via Docker (CLAUDE.md Rules #17-19).
# One variant: standard width (~5M params) + cbdice_cldice loss.
#
# This is the validated production configuration from dynunet_loss_variation_v2:
#   - cbdice_cldice achieves 0.906 clDice (best topology)
#   - Standard width avoids OOM on 8 GB VRAM with gpu_low profile
#
# Usage:
#   ./scripts/train_dynunet.sh                     # 100 epochs
#   ./scripts/train_dynunet.sh --epochs 50         # 50 epochs
#   ./scripts/train_dynunet.sh --debug             # smoke test (1 epoch)
#   ./scripts/train_dynunet.sh --num-folds 1       # single fold, fast check
#
# See CLAUDE.md Default Loss Function for rationale.
################################################################################

set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${MINIVESS_EPOCHS:-100}"
NUM_FOLDS="${MINIVESS_NUM_FOLDS:-3}"
LOSS="cbdice_cldice"
MODEL="dynunet"
DEBUG_FLAG=""

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)    EPOCHS="$2";    shift 2 ;;
        --num-folds) NUM_FOLDS="$2"; shift 2 ;;
        --debug)     DEBUG_FLAG="true"; shift ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Run via Docker
# ---------------------------------------------------------------------------

echo "[INFO] DynUNet training (Docker)"
echo "[INFO]   model  : ${MODEL}"
echo "[INFO]   loss   : ${LOSS}"
echo "[INFO]   epochs : ${EPOCHS}"
echo "[INFO]   folds  : ${NUM_FOLDS}"
echo "[INFO]   debug  : ${DEBUG_FLAG:-false}"

docker compose \
    -f deployment/docker-compose.flows.yml \
    run \
    --rm \
    -e MODEL_FAMILY="${MODEL}" \
    -e LOSS_NAME="${LOSS}" \
    -e MAX_EPOCHS="${EPOCHS}" \
    -e NUM_FOLDS="${NUM_FOLDS}" \
    -e BATCH_SIZE=2 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e ORT_LOGGING_LEVEL=3 \
    ${DEBUG_FLAG:+-e DEBUG=true} \
    train

echo "[OK] DynUNet training complete"
