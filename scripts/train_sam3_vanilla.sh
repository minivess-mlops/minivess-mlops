#!/usr/bin/env bash
# Train SAM3 Vanilla (V1) — frozen encoder + trainable decoder.
# Run this and go to sleep. Logs to mlruns/ and stdout.
#
# Usage:
#   chmod +x scripts/train_sam3_vanilla.sh
#   nohup ./scripts/train_sam3_vanilla.sh > sam3_training.log 2>&1 &
#
# Or just run directly:
#   ./scripts/train_sam3_vanilla.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export SPLITS_DIR=configs/splits
export CHECKPOINT_DIR=checkpoints
export MLFLOW_TRACKING_URI=mlruns
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Local dev run: disable Prefect orchestration (no server required).
# Training still logs to MLflow. Set PREFECT_DISABLED=0 + start a Prefect
# server if you want full orchestration (retries, UI tracking).
export PREFECT_DISABLED=1

echo "=== SAM3 Vanilla Training ==="
echo "Start:  $(date)"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Model:  sam3_vanilla"
echo "Loss:   cbdice_cldice"
echo "Epochs: 50"
echo "Folds:  3"
echo "Prefect: DISABLED (local dev run — MLflow logging only)"
echo "============================"

uv run python -c "
from minivess.orchestration.flows.train_flow import training_flow
training_flow(
    model_family='sam3_vanilla',
    loss_name='cbdice_cldice',
    debug=False,
    max_epochs=50,
    num_folds=3,
    batch_size=1,
)
"

echo "=== Training complete: $(date) ==="
