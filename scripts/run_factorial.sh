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
#   # Dry run (print commands without launching):
#   ./scripts/run_factorial.sh --dry-run configs/experiment/debug_factorial.yaml
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

# ─── Find sky binary (venv or system) ───────────────────────────────────
SKY_BIN=""
if command -v sky &>/dev/null; then
    SKY_BIN="sky"
elif [ -x "${REPO_ROOT}/.venv/bin/sky" ]; then
    SKY_BIN="${REPO_ROOT}/.venv/bin/sky"
else
    echo "ERROR: sky binary not found. Install: uv sync --extra infra"
    exit 1
fi

# ─── Parse flags ─────────────────────────────────────────────────────────────
DRY_RUN=false
CONFIG_FILE=""

for arg in "$@"; do
    case "${arg}" in
        --dry-run)
            DRY_RUN=true
            ;;
        -*)
            echo "ERROR: Unknown flag: ${arg}"
            exit 1
            ;;
        *)
            CONFIG_FILE="${arg}"
            ;;
    esac
done

# ─── Validate inputs ─────────────────────────────────────────────────────────
if [ -z "${CONFIG_FILE}" ]; then
    echo "Usage: $0 [--dry-run] <factorial-config.yaml>"
    echo ""
    echo "Examples:"
    echo "  $0 configs/experiment/debug_factorial.yaml          # Debug (2 epochs, half data)"
    echo "  $0 configs/hpo/paper_factorial.yaml                 # Production (50 epochs, full data)"
    echo "  $0 --dry-run configs/experiment/debug_factorial.yaml # Print commands only"
    exit 1
fi

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

# ─── Output directory for job tracking ───────────────────────────────────────
OUTPUT_DIR="${REPO_ROOT}/outputs"
mkdir -p "${OUTPUT_DIR}"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
JOB_LOG="${OUTPUT_DIR}/${TIMESTAMP}_factorial_job_ids.txt"

# ─── Parse factorial config ──────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  MinIVess Factorial Experiment Launcher                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Config:   ${CONFIG_FILE}"
echo "║  SkyPilot: ${SKYPILOT_YAML}"
echo "║  Dry run:  ${DRY_RUN}"
echo "║  Job log:  ${JOB_LOG}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Extract factorial factors from YAML using Python (no regex — CLAUDE.md Rule 16)
read -r MODELS LOSSES AUX_CALIBS MAX_EPOCHS NUM_FOLDS MAX_TRAIN MAX_VAL EXPERIMENT_NAME PT_METHODS < <(
    python3 -c "
import yaml, sys, pathlib
cfg = yaml.safe_load(pathlib.Path('${REPO_ROOT}/${CONFIG_FILE}').read_text(encoding='utf-8'))

# Parse training factors — supports two config layouts:
#   Layered:  factors.training.model_family  (configs/factorial/*.yaml)
#   Flat:     factors.model_family           (configs/experiment/*.yaml, legacy)
if 'factors' in cfg:
    factors = cfg['factors']
    # Prefer layered structure (factors.training.*)
    if 'training' in factors and isinstance(factors['training'], dict):
        training = factors['training']
    else:
        training = factors
    models = training['model_family']
    losses = training['loss_name']
    aux_calibs = [str(x).lower() for x in training.get('aux_calibration', [False])]
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

# Post-training methods run IN THE SAME SkyPilot job as training (not separate jobs).
# The parent flow iterates over methods internally: sub-flow 1 (training = 'none' cell),
# sub-flow 2 (SWAG or other methods). Decided: only 'none' and 'swag'.
# See: docs/planning/training-and-post-training-into-two-subflows-under-one-flow.md
# Check layered structure first (factors.post_training.method), then flat (post_training.methods)
if 'factors' in cfg and 'post_training' in cfg['factors']:
    pt_methods = cfg['factors']['post_training'].get('method', ['none', 'swag'])
else:
    pt_methods = cfg.get('post_training', {}).get('methods', ['none', 'swag'])

experiment_name = cfg.get('experiment_name', 'factorial')
print(','.join(models), ','.join(losses), ','.join(aux_calibs),
      max_epochs, num_folds, max_train, max_val, experiment_name,
      ','.join(pt_methods))
"
)

IFS=',' read -ra MODEL_ARRAY <<< "${MODELS}"
IFS=',' read -ra LOSS_ARRAY <<< "${LOSSES}"
IFS=',' read -ra AUX_CALIB_ARRAY <<< "${AUX_CALIBS}"

# Layer A (training) determines GPU job count. Post-training runs in the SAME job
# (parent flow iterates over methods internally — no extra jobs).
# Full factorial: pre-gcp-master-plan.xml line 16. Only Layer A launches SkyPilot jobs.
TOTAL_GPU_JOBS=$(( ${#MODEL_ARRAY[@]} * ${#LOSS_ARRAY[@]} * ${#AUX_CALIB_ARRAY[@]} * NUM_FOLDS ))

echo "Factorial design:"
echo "  Layer A (training, cloud GPU):"
echo "    Models:        ${MODEL_ARRAY[*]} (${#MODEL_ARRAY[@]})"
echo "    Losses:        ${LOSS_ARRAY[*]} (${#LOSS_ARRAY[@]})"
echo "    Aux calibs:    ${AUX_CALIB_ARRAY[*]} (${#AUX_CALIB_ARRAY[@]})"
echo "    Epochs:        ${MAX_EPOCHS}"
echo "    Folds:         ${NUM_FOLDS}"
echo "    Max train vol: ${MAX_TRAIN} (0=full)"
echo "    Max val vol:   ${MAX_VAL} (0=full)"
echo "  Layer B (post-training, same GPU job):"
echo "    Methods:       ${PT_METHODS} (iterated internally by parent flow)"
echo "  Experiment:      ${EXPERIMENT_NAME}"
echo "  GPU jobs:        ${TOTAL_GPU_JOBS} (each runs training + post-training)"
echo ""

# Write header to job log
echo "# MinIVess Factorial Experiment — ${TIMESTAMP}" > "${JOB_LOG}"
echo "# Config: ${CONFIG_FILE}" >> "${JOB_LOG}"
echo "# GPU jobs: ${TOTAL_GPU_JOBS} (each runs training + post-training internally)" >> "${JOB_LOG}"
echo "# Post-training methods: ${PT_METHODS} (iterated inside each job)" >> "${JOB_LOG}"
echo "#" >> "${JOB_LOG}"
echo "# FORMAT: condition_id | model | loss | aux_calib | fold | status" >> "${JOB_LOG}"

# ─── Launch GPU jobs (each runs training + post-training in same session) ─────
# Post-training methods (e.g., none,swag) are passed as POST_TRAINING_METHODS env var.
# The parent flow iterates over methods internally — no extra SkyPilot jobs needed.
# "none" cell comes for free from training sub-flow. SWAG runs in post-training sub-flow.
CONDITION=0
FAILED=0
LAUNCHED=0
for model in "${MODEL_ARRAY[@]}"; do
    for loss in "${LOSS_ARRAY[@]}"; do
        for aux_calib in "${AUX_CALIB_ARRAY[@]}"; do
            for fold in $(seq 0 $((NUM_FOLDS - 1))); do
                CONDITION=$((CONDITION + 1))
                CONDITION_NAME="${model}-${loss}-calib${aux_calib}-f${fold}"
                echo "[${CONDITION}/${TOTAL_GPU_JOBS}] ${model} × ${loss} × aux_calib=${aux_calib} × fold=${fold} (pt: ${PT_METHODS})"

                if [ "${DRY_RUN}" = true ]; then
                    echo "  [DRY RUN] sky jobs launch ${SKYPILOT_YAML} --name ${CONDITION_NAME} --env MODEL_FAMILY=${model} ..."
                    echo "${CONDITION} | ${model} | ${loss} | ${aux_calib} | ${fold} | DRY_RUN" >> "${JOB_LOG}"
                else
                    if "${SKY_BIN}" jobs launch "${SKYPILOT_YAML}" \
                        --name "${CONDITION_NAME}" \
                        --env MODEL_FAMILY="${model}" \
                        --env LOSS_NAME="${loss}" \
                        --env FOLD_ID="${fold}" \
                        --env WITH_AUX_CALIB="${aux_calib}" \
                        --env MAX_EPOCHS="${MAX_EPOCHS}" \
                        --env MAX_TRAIN_VOLUMES="${MAX_TRAIN}" \
                        --env MAX_VAL_VOLUMES="${MAX_VAL}" \
                        --env EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
                        --env POST_TRAINING_METHODS="${PT_METHODS}" \
                        --env-file "${ENV_FILE}" \
                        -y; then
                        LAUNCHED=$((LAUNCHED + 1))
                        echo "${CONDITION} | ${model} | ${loss} | ${aux_calib} | ${fold} | LAUNCHED" >> "${JOB_LOG}"
                    else
                        FAILED=$((FAILED + 1))
                        echo "${CONDITION} | ${model} | ${loss} | ${aux_calib} | ${fold} | LAUNCH_FAILED" >> "${JOB_LOG}"
                        echo "  WARNING: Launch failed for ${CONDITION_NAME} — continuing with remaining conditions"
                    fi

                    # Rate limiting: prevent SkyPilot API quota issues
                    sleep 5
                fi
            done
        done
    done
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Trainable conditions: ${LAUNCHED} launched, ${FAILED} failed (of ${TOTAL_GPU_JOBS})"
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
import yaml, pathlib, subprocess, os, sys, time

cfg = yaml.safe_load(pathlib.Path('${REPO_ROOT}/${CONFIG_FILE}').read_text(encoding='utf-8'))
baselines = cfg.get('zero_shot_baselines', [])
fixed = cfg.get('fixed', {})
num_folds = fixed.get('num_folds', cfg.get('num_folds', 3))
experiment = cfg.get('experiment_name', 'factorial')
dry_run = '${DRY_RUN}' == 'true'
job_log = '${JOB_LOG}'

zs_launched = 0
zs_failed = 0

for b in baselines:
    model = b['model']
    strategy = b.get('strategy', 'zero_shot')
    folds = b.get('folds', num_folds)
    dataset = b.get('dataset', 'minivess')
    if isinstance(dataset, list):
        dataset = ','.join(dataset)

    for fold in range(folds):
        condition_name = f'{model}-zeroshot-{dataset}-f{fold}'
        print(f'  [zero-shot] {model} x fold={fold} x dataset={dataset}')

        if dry_run:
            print(f'  [DRY RUN] sky jobs launch --name {condition_name} ...')
            with open(job_log, 'a', encoding='utf-8') as f:
                f.write(f'ZS | {model} | zero_shot | false | {fold} | DRY_RUN\n')
        else:
            result = subprocess.run([
                '${SKY_BIN}', 'jobs', 'launch',
                '${SKYPILOT_YAML}',
                '--name', condition_name,
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
            ])
            if result.returncode == 0:
                zs_launched += 1
                with open(job_log, 'a', encoding='utf-8') as f:
                    f.write(f'ZS | {model} | zero_shot | false | {fold} | LAUNCHED\n')
            else:
                zs_failed += 1
                with open(job_log, 'a', encoding='utf-8') as f:
                    f.write(f'ZS | {model} | zero_shot | false | {fold} | LAUNCH_FAILED\n')
                print(f'  WARNING: Launch failed for {condition_name}')

            # Rate limiting
            time.sleep(5)

print(f'Zero-shot baselines: {zs_launched} launched, {zs_failed} failed')
"
    echo ""
    echo "Zero-shot baselines launched."
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ALL CONDITIONS LAUNCHED                                    ║"
echo "║  Job log: ${JOB_LOG}"
echo "║  Monitor: sky jobs queue                                    ║"
echo "║  Logs:    sky jobs logs <JOB_ID>                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"

if [ "${FAILED}" -gt 0 ]; then
    echo ""
    echo "WARNING: ${FAILED} trainable condition(s) failed to launch."
    echo "Review ${JOB_LOG} for details and re-launch failed conditions manually."
    exit 1
fi
