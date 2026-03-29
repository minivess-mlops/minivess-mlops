#!/usr/bin/env bash
# run_smoke_mini.sh — Reproducible local mini-experiment launcher.
#
# Runs the smoke_mini factorial experiment through the Docker+Prefect pipeline.
# 2 losses (dice_ce, cbdice_cldice) × 3 folds × 20 epochs on local GPU.
#
# This is the ONLY supported way to run the mini-experiment.
# DO NOT bypass Docker+Prefect with escape hatches or direct Python invocation.
#
# Prerequisites:
#   - .env file with DagsHub credentials (MLFLOW_TRACKING_URI, DAGSHUB_TOKEN)
#   - Docker image built: make build-base && docker compose -f deployment/docker-compose.flows.yml build train
#   - Data in Docker volumes: see deployment/CLAUDE.md "Data Population"
#   - NVIDIA GPU with CDI support (Docker 25+)
#
# Usage:
#   ./scripts/run_smoke_mini.sh                    # Run all 6 training jobs
#   ./scripts/run_smoke_mini.sh --dry-run          # Validate config without training
#   ./scripts/run_smoke_mini.sh --loss dice_ce     # Run one loss only
#   # Note: folds are handled internally by the training flow (num_folds=3)
#
# Config:
#   Factorial:     configs/factorial/smoke_mini.yaml
#   Biostatistics: configs/biostatistics/smoke_mini.yaml
#   Environment:   .env (DagsHub credentials, paths)
#
# See: docs/planning/v0-2_archive/original_docs/biostatistics-polishing-plan.xml Phase 3

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"
COMPOSE_FILE="${REPO_ROOT}/deployment/docker-compose.flows.yml"
FACTORIAL_YAML="configs/factorial/smoke_mini.yaml"

# ─── Parse arguments ──────────────────────────────────────────────────
DRY_RUN=false
LOSS_FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=true; shift ;;
        --loss)       LOSS_FILTER="$2"; shift 2 ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ─── Preflight checks ────────────────────────────────────────────────
echo "=== Smoke Mini Preflight ==="

# Check .env exists
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: .env not found at ${ENV_FILE}. Copy .env.example → .env and configure."
    exit 1
fi

# Check DagsHub MLflow is configured
MLFLOW_URI=$(grep "^MLFLOW_TRACKING_URI=" "${ENV_FILE}" | head -1 | cut -d= -f2-)
if [[ -z "${MLFLOW_URI}" ]]; then
    echo "ERROR: MLFLOW_TRACKING_URI not set in .env"
    exit 1
fi
echo "  MLflow URI: ${MLFLOW_URI}"

# Check Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker daemon not running"
    exit 1
fi

# Check GPU is available
if ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: NVIDIA GPU not available (nvidia-smi failed)"
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "  GPU: ${GPU_NAME}"

# Check train image exists
if ! docker image inspect minivess-train:latest >/dev/null 2>&1; then
    echo "ERROR: minivess-train:latest not built. Run:"
    echo "  make build-base"
    echo "  docker compose --env-file ${ENV_FILE} -f ${COMPOSE_FILE} build train"
    exit 1
fi
echo "  Docker image: minivess-train:latest ✓"

# Check factorial config exists
if [[ ! -f "${REPO_ROOT}/${FACTORIAL_YAML}" ]]; then
    echo "ERROR: ${FACTORIAL_YAML} not found"
    exit 1
fi
echo "  Factorial config: ${FACTORIAL_YAML} ✓"

echo "=== Preflight PASSED ==="

if [[ "${DRY_RUN}" == true ]]; then
    echo ""
    echo "DRY RUN — would launch these jobs:"
    echo "  Losses: dice_ce, cbdice_cldice"
    echo "  Folds: 0, 1, 2"
    echo "  Epochs: 20"
    echo "  Total: 6 training runs"
    echo "  Estimated time: 2-4 hours on ${GPU_NAME}"
    exit 0
fi

# ─── Launch training jobs ─────────────────────────────────────────────
# The training flow handles all folds internally (num_folds=3 → trains fold 0,1,2).
# Each loss function has its own Hydra experiment config in configs/experiment/.
# We loop over experiment names — one Docker invocation per loss function.
EXPERIMENTS=("mini_experiment_dice_ce" "mini_experiment_cbdice_cldice")

# Apply filters
if [[ -n "${LOSS_FILTER}" ]]; then
    EXPERIMENTS=("mini_experiment_${LOSS_FILTER}")
fi

TOTAL=${#EXPERIMENTS[@]}
CURRENT=0

echo ""
echo "=== Launching ${TOTAL} training jobs (each trains 3 folds internally) ==="
echo "  Config: ${FACTORIAL_YAML}"
echo "  MLflow: ${MLFLOW_URI}"
echo ""

for experiment in "${EXPERIMENTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo "[${CURRENT}/${TOTAL}] Training: experiment=${experiment}, 3 folds × 20 epochs"

    docker compose --env-file "${ENV_FILE}" \
        -f "${COMPOSE_FILE}" \
        run --rm \
        -e EXPERIMENT="${experiment}" \
        train

    echo "[${CURRENT}/${TOTAL}] ✓ Completed: ${experiment} (all 3 folds)"
    echo ""
done

echo "=== All ${TOTAL} loss functions trained (${TOTAL}×3 = $((TOTAL*3)) fold runs) ==="
echo "Results logged to: ${MLFLOW_URI}"
echo ""
echo "Next steps:"
echo "  1. Run post-training: docker compose --env-file .env -f ${COMPOSE_FILE} run --rm post_training"
echo "  2. Run analysis:      docker compose --env-file .env -f ${COMPOSE_FILE} run --rm analyze"
echo "  3. Run biostatistics: docker compose --env-file .env -f ${COMPOSE_FILE} run --rm biostatistics"
