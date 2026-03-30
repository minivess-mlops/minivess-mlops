#!/usr/bin/env bash
# run_local_dynunet_mechanics.sh — Reproducible local mechanics test.
#
# Full Train → Analysis → Biostatistics pipeline with DynUNet.
# 2 losses × 2 aux_calib × 2 post-training = 8 conditions × 2 folds × 15 epochs.
# DeepVess external test included (OBLIGATORY — CLAUDE.md).
# All execution through Docker+Prefect (Rule #33).
#
# Prerequisites:
#   - .env with DagsHub credentials
#   - Docker images built: make build-base && docker compose build train analyze biostatistics
#   - Data volumes populated (MiniVess 70 vols + DeepVess 1 vol)
#   - NVIDIA GPU with CDI (Docker 25+)
#
# Usage:
#   ./scripts/run_local_dynunet_mechanics.sh              # Full pipeline
#   ./scripts/run_local_dynunet_mechanics.sh --dry-run    # Preflight only
#   ./scripts/run_local_dynunet_mechanics.sh --phase 3    # Start from Phase 3 (training)
#
# Config:
#   Factorial:     configs/factorial/local_dynunet_mechanics.yaml
#   Biostatistics: configs/biostatistics/local_dynunet.yaml
#   Environment:   .env

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"
COMPOSE_FILE="${REPO_ROOT}/deployment/docker-compose.flows.yml"

DRY_RUN=false
START_PHASE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --phase)   START_PHASE="$2"; shift 2 ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ─── Preflight ────────────────────────────────────────────────────────
echo "=== Local DynUNet Mechanics Test — Preflight ==="

[[ -f "${ENV_FILE}" ]] || { echo "ERROR: .env not found"; exit 1; }
docker info >/dev/null 2>&1 || { echo "ERROR: Docker not running"; exit 1; }
nvidia-smi >/dev/null 2>&1 || { echo "ERROR: No GPU"; exit 1; }

MLFLOW_URI=$(grep "^MLFLOW_TRACKING_URI=" "${ENV_FILE}" | head -1 | cut -d= -f2-)
echo "  MLflow: ${MLFLOW_URI}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  MiniVess: $(docker run --rm -v deployment_data_cache:/app/data:ro ubuntu:22.04 ls /app/data/raw/minivess/imagesTr/ 2>/dev/null | wc -l) volumes"
echo "  DeepVess: $(docker run --rm -v deployment_data_cache:/app/data:ro ubuntu:22.04 ls /app/data/raw/deepvess/imagesTr/ 2>/dev/null | wc -l) volumes"
echo "=== Preflight PASSED ==="

if [[ "${DRY_RUN}" == true ]]; then
    echo ""
    echo "DRY RUN — would execute:"
    echo "  Phase 3: 4 training jobs (2 losses × 2 aux_calib), each 2 folds × 15 epochs + SWAG post-training"
    echo "  Phase 4: Analysis with 2 ensemble strategies + DeepVess external test"
    echo "  Phase 5: Biostatistics with R/ggplot2 figures + LaTeX tables"
    echo "  Estimated: ~4 hours on RTX 2070 Super"
    exit 0
fi

# ─── Phase 3: Training ───────────────────────────────────────────────
if [[ ${START_PHASE} -le 3 ]]; then
    echo ""
    echo "=== PHASE 3: Training (4 conditions × 2 folds × 15 epochs) ==="

    EXPERIMENTS=(
        "local_dynunet_dice_ce"
        "local_dynunet_dice_ce_auxcalib"
        "local_dynunet_cbdice_cldice"
        "local_dynunet_cbdice_cldice_auxcalib"
    )

    TOTAL=${#EXPERIMENTS[@]}
    CURRENT=0

    for experiment in "${EXPERIMENTS[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo "[${CURRENT}/${TOTAL}] Training: ${experiment} (2 folds × 15 epochs + SWAG)"

        docker compose --env-file "${ENV_FILE}" \
            -f "${COMPOSE_FILE}" \
            run --rm \
            -e EXPERIMENT="${experiment}" \
            train

        echo "[${CURRENT}/${TOTAL}] ✓ Completed: ${experiment}"
        echo ""
    done

    echo "=== Phase 3 COMPLETE: All training runs logged to ${MLFLOW_URI} ==="
fi

# ─── Phase 4: Analysis ───────────────────────────────────────────────
if [[ ${START_PHASE} -le 4 ]]; then
    echo ""
    echo "=== PHASE 4: Analysis (ensembles + DeepVess evaluation) ==="

    docker compose --env-file "${ENV_FILE}" \
        -f "${COMPOSE_FILE}" \
        run --rm \
        -e UPSTREAM_EXPERIMENT=local_dynunet_mechanics_training \
        -e EXTERNAL_DATA_DIR=/app/data/raw/deepvess \
        analyze

    echo "=== Phase 4 COMPLETE ==="
fi

# ─── Phase 5: Biostatistics ──────────────────────────────────────────
if [[ ${START_PHASE} -le 5 ]]; then
    echo ""
    echo "=== PHASE 5: Biostatistics (DuckDB + R/ggplot2 figures + LaTeX tables) ==="

    docker compose --env-file "${ENV_FILE}" \
        -f "${COMPOSE_FILE}" \
        run --rm \
        -e BIOSTATISTICS_CONFIG=/app/configs/biostatistics/local_dynunet.yaml \
        biostatistics

    echo "=== Phase 5 COMPLETE ==="
fi

echo ""
echo "=== ALL PHASES COMPLETE ==="
echo "Results: ${MLFLOW_URI}"
echo "Biostatistics: outputs/biostatistics_local_dynunet/"
echo "Figures: outputs/biostatistics_local_dynunet/figures/"
echo "Tables: outputs/biostatistics_local_dynunet/tables/"
