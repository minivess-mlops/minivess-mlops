#!/usr/bin/env bash
# run_factorial.sh — Deterministic launch script for the factorial experiment.
#
# Reads a factorial YAML config and launches ALL conditions via SkyPilot on GCP.
# Same script for debug AND production — only the YAML changes.
#
# Usage:
#   # Debug run (2 epochs, half data, 1 fold, 24+2 conditions):
#   ./scripts/run_factorial.sh configs/experiment/debug_factorial.yaml
#
#   # Production run (50 epochs, full data, 3 folds, 72+6 conditions):
#   ./scripts/run_factorial.sh configs/hpo/paper_factorial.yaml
#
# Requirements:
#   - .env file with HF_TOKEN, GCP credentials
#   - SkyPilot configured for GCP: sky check gcp
#   - Docker image pushed to GAR: europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest
#   - DVC data on GCS: gs://minivess-mlops-dvc-data
#
# This script is PURE sky jobs launch calls in a loop.
# NO claude -p, NO screen, NO nohup, NO pipe chains.
# See: .claude/metalearning/2026-03-09-overnight-script-silent-freeze.md
# See: .claude/metalearning/2026-03-16-overnight-runner-script-freeze-v2.md
#
# Source of truth: knowledge-graph/decisions/L3-technology/paper_model_comparison.yaml
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKYPILOT_YAML="${REPO_ROOT}/deployment/skypilot/train_factorial.yaml"
ENV_FILE="${REPO_ROOT}/.env"

# ─── Validate inputs ─────────────────────────────────────────────────────────
if [ $# -lt 1 ]; then
    echo "Usage: $0 <factorial-config.yaml>"
    echo ""
    echo "Examples:"
    echo "  $0 configs/experiment/debug_factorial.yaml   # Debug (2 epochs, half data)"
    echo "  $0 configs/hpo/paper_factorial.yaml          # Production (50 epochs, full data)"
    exit 1
fi

CONFIG_FILE="$1"
if [ ! -f "${REPO_ROOT}/${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

if [ ! -f "${ENV_FILE}" ]; then
    echo "ERROR: .env file not found at ${ENV_FILE}"
    echo "Copy .env.example to .env and fill in credentials."
    exit 1
fi

if [ ! -f "${SKYPILOT_YAML}" ]; then
    echo "ERROR: SkyPilot YAML not found: ${SKYPILOT_YAML}"
    echo "Create deployment/skypilot/train_factorial.yaml first."
    exit 1
fi

# ─── Parse factorial config ──────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  MinIVess Factorial Experiment Launcher                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Config: ${CONFIG_FILE}"
echo "║  SkyPilot: ${SKYPILOT_YAML}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Extract factorial factors from YAML using Python (no regex — CLAUDE.md Rule 16)
read -r MODELS LOSSES AUX_CALIBS MAX_EPOCHS NUM_FOLDS MAX_TRAIN MAX_VAL EXPERIMENT_NAME < <(
    python3 -c "
import yaml, sys, pathlib
cfg = yaml.safe_load(pathlib.Path('${REPO_ROOT}/${CONFIG_FILE}').read_text(encoding='utf-8'))

# Handle both factorial config (paper_factorial.yaml) and experiment config (debug_factorial.yaml)
if 'factors' in cfg:
    models = cfg['factors']['model_family']
    losses = cfg['factors']['loss_name']
    aux_calibs = [str(x).lower() for x in cfg['factors'].get('aux_calibration', [False])]
    fixed = cfg.get('fixed', {})
    max_epochs = fixed.get('max_epochs', 50)
    num_folds = fixed.get('num_folds', 3)
    max_train = fixed.get('max_train_volumes', 0)  # 0 = full dataset
    max_val = fixed.get('max_val_volumes', 0)
else:
    models = [cfg.get('model', 'dynunet')]
    losses = cfg.get('losses', ['cbdice_cldice'])
    aux_calibs = ['false', 'true'] if 'with_aux_calib' in cfg else ['false']
    max_epochs = cfg.get('max_epochs', 50)
    num_folds = cfg.get('num_folds', 3)
    max_train = cfg.get('max_train_volumes', 0)
    max_val = cfg.get('max_val_volumes', 0)

experiment_name = cfg.get('experiment_name', 'factorial')
print(','.join(models), ','.join(losses), ','.join(aux_calibs),
      max_epochs, num_folds, max_train, max_val, experiment_name)
"
)

IFS=',' read -ra MODEL_ARRAY <<< "${MODELS}"
IFS=',' read -ra LOSS_ARRAY <<< "${LOSSES}"
IFS=',' read -ra AUX_CALIB_ARRAY <<< "${AUX_CALIBS}"

TOTAL_TRAINABLE=$(( ${#MODEL_ARRAY[@]} * ${#LOSS_ARRAY[@]} * ${#AUX_CALIB_ARRAY[@]} * NUM_FOLDS ))

echo "Factorial design:"
echo "  Models:        ${MODEL_ARRAY[*]} (${#MODEL_ARRAY[@]})"
echo "  Losses:        ${LOSS_ARRAY[*]} (${#LOSS_ARRAY[@]})"
echo "  Aux calibs:    ${AUX_CALIB_ARRAY[*]} (${#AUX_CALIB_ARRAY[@]})"
echo "  Epochs:        ${MAX_EPOCHS}"
echo "  Folds:         ${NUM_FOLDS}"
echo "  Max train vol: ${MAX_TRAIN} (0=full)"
echo "  Max val vol:   ${MAX_VAL} (0=full)"
echo "  Experiment:    ${EXPERIMENT_NAME}"
echo "  Total runs:    ${TOTAL_TRAINABLE} trainable"
echo ""

# ─── Launch trainable conditions ──────────────────────────────────────────────
CONDITION=0
for model in "${MODEL_ARRAY[@]}"; do
    for loss in "${LOSS_ARRAY[@]}"; do
        for aux_calib in "${AUX_CALIB_ARRAY[@]}"; do
            for fold in $(seq 0 $((NUM_FOLDS - 1))); do
                CONDITION=$((CONDITION + 1))
                echo "[${CONDITION}/${TOTAL_TRAINABLE}] ${model} × ${loss} × aux_calib=${aux_calib} × fold=${fold}"

                sky jobs launch "${SKYPILOT_YAML}" \
                    --env MODEL_FAMILY="${model}" \
                    --env LOSS_NAME="${loss}" \
                    --env FOLD_ID="${fold}" \
                    --env WITH_AUX_CALIB="${aux_calib}" \
                    --env MAX_EPOCHS="${MAX_EPOCHS}" \
                    --env MAX_TRAIN_VOLUMES="${MAX_TRAIN}" \
                    --env MAX_VAL_VOLUMES="${MAX_VAL}" \
                    --env EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
                    --env-file "${ENV_FILE}" \
                    -y
            done
        done
    done
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "All ${TOTAL_TRAINABLE} trainable conditions launched."
echo "Monitor: sky jobs queue"
echo "═══════════════════════════════════════════════════════════════"

# ─── Zero-shot baselines ─────────────────────────────────────────────────────
# Parse zero-shot baselines from YAML (if present)
ZERO_SHOT_COUNT=$(python3 -c "
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path('${REPO_ROOT}/${CONFIG_FILE}').read_text(encoding='utf-8'))
baselines = cfg.get('zero_shot_baselines', [])
print(len(baselines))
")

if [ "${ZERO_SHOT_COUNT}" -gt 0 ]; then
    echo ""
    echo "Launching ${ZERO_SHOT_COUNT} zero-shot baselines..."

    python3 -c "
import yaml, pathlib, subprocess, os
cfg = yaml.safe_load(pathlib.Path('${REPO_ROOT}/${CONFIG_FILE}').read_text(encoding='utf-8'))
baselines = cfg.get('zero_shot_baselines', [])
fixed = cfg.get('fixed', {})
num_folds = fixed.get('num_folds', cfg.get('num_folds', 3))
experiment = cfg.get('experiment_name', 'factorial')

for b in baselines:
    model = b['model']
    strategy = b.get('strategy', 'zero_shot')
    folds = b.get('folds', num_folds)
    dataset = b.get('dataset', 'minivess')
    if isinstance(dataset, list):
        dataset = ','.join(dataset)

    for fold in range(folds):
        print(f'  [zero-shot] {model} × fold={fold} × dataset={dataset}')
        subprocess.run([
            'sky', 'jobs', 'launch',
            '${SKYPILOT_YAML}',
            '--env', f'MODEL_FAMILY={model}',
            '--env', f'LOSS_NAME=none',
            '--env', f'FOLD_ID={fold}',
            '--env', f'WITH_AUX_CALIB=false',
            '--env', f'MAX_EPOCHS=0',
            '--env', f'EXPERIMENT_NAME={experiment}',
            '--env', f'ZERO_SHOT=true',
            '--env', f'EVAL_DATASET={dataset}',
            '--env-file', '${ENV_FILE}',
            '-y',
        ], check=True)
"
    echo ""
    echo "Zero-shot baselines launched."
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ALL CONDITIONS LAUNCHED                                    ║"
echo "║  Monitor: sky jobs queue                                    ║"
echo "║  Logs:    sky jobs logs <JOB_ID>                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
