#!/usr/bin/env bash
# Train SAM3 Vanilla (V1) — frozen encoder + trainable decoder.
# Run this and go to sleep. Logs to mlruns/ and stdout.
#
# Architecture: Frozen SAM3 ViT-32L (454M params, SDPA attention) + trainable
# lightweight Conv decoder (66K params). SDPA reduces encoder peak VRAM from
# ~7 GB to ~1.1 GB, making training feasible on 8 GB consumer GPUs.
#
# Expected timing on RTX 2070 Super (8 GB):
#   - Training: ~8 min/epoch (40 batches × 3 slices × encoder forward)
#   - Validation: ~35 min/epoch (sliding window on full 512×512×Z volumes)
#   - val_interval=10: validate every 10 epochs → ~16h total for 50 epochs × 3 folds
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
export HF_HUB_DISABLE_PROGRESS_BARS=1

# Suppress non-actionable warnings (CLAUDE.md Core Principle #7)
export ORT_LOGGING_LEVEL=3

# Local dev run: disable Prefect orchestration (no server required).
# Training still logs to MLflow. Set PREFECT_DISABLED=0 + start a Prefect
# server if you want full orchestration (retries, UI tracking).
export PREFECT_DISABLED=1

echo "=== SAM3 Vanilla Training ==="
echo "Start:  $(date)"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Model:  sam3_vanilla (frozen ViT-32L + trainable Conv decoder)"
echo "Loss:   cbdice_cldice"
echo "Epochs: 50"
echo "Folds:  3"
echo "SDPA:   enabled (peak ~2.9 GB vs ~7 GB eager)"
echo "ValInt: 10 (validate every 10 epochs)"
echo "Prefect: DISABLED (local dev run — MLflow logging only)"
echo "============================"

uv run python -c "
import warnings, os
os.environ.setdefault('ORT_LOGGING_LEVEL', '3')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*cuda.cudart.*')

from minivess.orchestration.flows.train_flow import training_flow
result = training_flow(
    model_family='sam3_vanilla',
    loss_name='cbdice_cldice',
    debug=False,
    max_epochs=50,
    num_folds=3,
    batch_size=1,
)
print(f'=== Result: status={result.status}, folds={result.n_folds}, run_id={result.mlflow_run_id} ===')
"

echo "=== Training complete: $(date) ==="
