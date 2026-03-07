#!/usr/bin/env bash
# Train SAM3 Vanilla (V1) — frozen encoder + trainable decoder.
# Runs inside Docker via docker compose (CLAUDE.md Rules #17, #18, #19).
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

echo "=== SAM3 Vanilla Training (Docker) ==="
echo "Start:  $(date)"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Model:  sam3_vanilla (frozen ViT-32L + trainable Conv decoder)"
echo "Loss:   cbdice_cldice"
echo "Epochs: 50"
echo "Folds:  3"
echo "SDPA:   enabled (peak ~2.9 GB vs ~7 GB eager)"
echo "ValInt: 10 (validate every 10 epochs)"
echo "============================="

# ---------------------------------------------------------------------------
# Docker Compose run — training happens inside the container.
# Volume mounts defined in docker-compose.flows.yml:
#   data_cache   → /app/data
#   configs      → /app/configs
#   checkpoints  → /app/checkpoints
#   mlruns       → /app/mlruns
#   logs         → /app/logs
# ---------------------------------------------------------------------------
docker compose \
    -f deployment/docker-compose.flows.yml \
    run \
    --rm \
    -e MODEL_FAMILY=sam3_vanilla \
    -e LOSS_NAME=cbdice_cldice \
    -e MAX_EPOCHS=50 \
    -e NUM_FOLDS=3 \
    -e BATCH_SIZE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
    -e ORT_LOGGING_LEVEL=3 \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    train

echo "=== Training complete: $(date) ==="
