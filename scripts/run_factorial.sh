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

# ─── Resilience parameters ────────────────────────────────────────────────
MAX_RETRIES=3              # Retry failed launches up to 3 times
RETRY_DELAY=2              # Initial backoff delay in seconds (doubles each retry)

# ─── Signal handling (Gap #6) ─────────────────────────────────────────────
# Kill background launch subshells on Ctrl+C or TERM signal.
cleanup() {
    echo ""
    echo "Signal received — killing background launch jobs..."
    # Kill all background children of this script
    kill $(jobs -p) 2>/dev/null || true
    wait 2>/dev/null || true
    echo "Cleanup complete. Check job log for partial results."
    exit 130  # Standard exit code for SIGINT
}
trap cleanup INT TERM

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
RESUME=false
CONFIG_FILE=""

for arg in "$@"; do
    case "${arg}" in
        --dry-run)
            DRY_RUN=true
            ;;
        --resume)
            RESUME=true
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

# ─── Preflight checks (skip for dry-run) ────────────────────────────────────
if [ "${DRY_RUN}" = false ] && [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
    echo "Running GCP preflight checks..."
    if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
        "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/preflight_gcp.py" || {
            echo ""
            echo "FATAL: Preflight checks failed. Fix issues above before launching."
            echo "To bypass (DANGEROUS): SKIP_PREFLIGHT=1 ./scripts/run_factorial.sh ..."
            exit 1
        }
        echo ""
    else
        echo "FATAL: .venv/bin/python not found — cannot run preflight checks."
        echo "Run: uv sync --all-extras"
        exit 1
    fi
fi

# ─── Load cloud infrastructure config ──────────────────────────────────────
# Read infrastructure.cloud_config from factorial YAML → load cloud config → extract params.
# NEVER hardcode parallel_submissions or rate_limit_seconds in this script.
CLOUD_CONFIG_NAME=$(python3 -c "
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path('${REPO_ROOT}/${CONFIG_FILE}').read_text(encoding='utf-8'))
infra = cfg.get('infrastructure', {})
print(infra.get('cloud_config', 'local'))
")

CLOUD_CONFIG_PATH="${REPO_ROOT}/configs/cloud/${CLOUD_CONFIG_NAME}.yaml"
if [ -f "${CLOUD_CONFIG_PATH}" ]; then
    read -r PARALLEL_SUBMISSIONS RATE_LIMIT_SECONDS < <(
        python3 -c "
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path('${CLOUD_CONFIG_PATH}').read_text(encoding='utf-8'))
infra = cfg.get('infrastructure', {})
print(infra.get('parallel_submissions', 1), infra.get('rate_limit_seconds', 5))
"
    )
    echo "Cloud config: ${CLOUD_CONFIG_NAME} (parallel=${PARALLEL_SUBMISSIONS}, rate_limit=${RATE_LIMIT_SECONDS}s)"
else
    PARALLEL_SUBMISSIONS=1
    RATE_LIMIT_SECONDS=5
    echo "WARNING: Cloud config not found: ${CLOUD_CONFIG_PATH} — using defaults (parallel=1, rate_limit=5s)"
fi

# Sync SkyPilot controller placement to match job cloud
if [ "${DRY_RUN}" = false ] && [ -f "${CLOUD_CONFIG_PATH}" ]; then
    if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
        "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/sync_sky_config.py" "${CLOUD_CONFIG_PATH}" || {
            echo "FATAL: sync_sky_config.py failed — controller placement failed."
            echo "Cross-cloud SSH adds ~30 min/submission (5th pass root cause)."
            exit 1
        }
    fi
fi

# ─── Dynamic region injection (Phase 2: composable regions) ──────────────────
# Read infrastructure.region_config from factorial YAML → generate temp SkyPilot
# YAML with ordered: block injected. If no region_config, use static YAML as-is.
# See: docs/planning/cold-start-prompt-composable-regions-phase2.md
REGION_CONFIG_NAME=$(python3 -c "
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path('${REPO_ROOT}/${CONFIG_FILE}').read_text(encoding='utf-8'))
print(cfg.get('infrastructure', {}).get('region_config', ''))
")

LAUNCH_YAML="${SKYPILOT_YAML}"
GENERATED_YAML=""
if [ -n "${REGION_CONFIG_NAME}" ]; then
    REGION_CONFIG_PATH="${REPO_ROOT}/configs/cloud/regions/${REGION_CONFIG_NAME}.yaml"
    if [ -f "${REGION_CONFIG_PATH}" ]; then
        GENERATED_YAML=$("${REPO_ROOT}/.venv/bin/python" -m minivess.cloud.region_injection \
            --base "${SKYPILOT_YAML}" \
            --region-config "${REGION_CONFIG_PATH}" \
            --output-dir "${OUTPUT_DIR}")
        LAUNCH_YAML="${GENERATED_YAML}"
        echo "Region config: ${REGION_CONFIG_NAME} → generated ${LAUNCH_YAML}"
    else
        echo "WARNING: Region config not found: ${REGION_CONFIG_PATH} — using static YAML"
    fi
else
    echo "No region_config set — using static SkyPilot YAML (no ordered: block)"
fi

# ─── Parse factorial config ──────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  MinIVess Factorial Experiment Launcher                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Config:   ${CONFIG_FILE}"
echo "║  SkyPilot: ${LAUNCH_YAML}"
echo "║  Dry run:  ${DRY_RUN}"
echo "║  Job log:  ${JOB_LOG}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Extract factorial factors from YAML using Python (no regex — CLAUDE.md Rule 16)
# Gap #7: Python parsing wrapped in try/except for clear error diagnostics.
read -r MODELS LOSSES AUX_CALIBS MAX_EPOCHS NUM_FOLDS MAX_TRAIN MAX_VAL EXPERIMENT_NAME PT_METHODS < <(
    python3 -c "
import yaml, sys, pathlib, traceback
try:
    cfg = yaml.safe_load(pathlib.Path('${REPO_ROOT}/${CONFIG_FILE}').read_text(encoding='utf-8'))

    # Parse training factors — supports two config layouts:
    #   Layered:  factors.training.model_family  (configs/factorial/*.yaml)
    #   Flat:     factors.model_family           (configs/experiment/*.yaml, legacy)
    if 'factors' in cfg:
        factors = cfg['factors']
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
        max_train = fixed.get('max_train_volumes', 0)
        max_val = fixed.get('max_val_volumes', 0)
    else:
        models = [cfg.get('model', 'dynunet')]
        losses = cfg.get('losses', ['cbdice_cldice'])
        aux_calibs = ['false', 'true'] if 'with_aux_calib' in cfg else ['false']
        max_epochs = cfg.get('max_epochs', 50)
        num_folds = cfg.get('num_folds', 3)
        max_train = cfg.get('max_train_volumes', 0)
        max_val = cfg.get('max_val_volumes', 0)

    # Post-training methods
    if 'factors' in cfg and 'post_training' in cfg['factors']:
        pt_methods = cfg['factors']['post_training'].get('method', ['none', 'swag'])
    else:
        pt_methods = cfg.get('post_training', {}).get('methods', ['none', 'swag'])

    experiment_name = cfg.get('experiment_name', 'factorial')
    print(','.join(models), ','.join(losses), ','.join(aux_calibs),
          max_epochs, num_folds, max_train, max_val, experiment_name,
          ','.join(pt_methods))
except Exception:
    traceback.print_exc()
    print('FATAL: Failed to parse factorial config', file=sys.stderr)
    sys.exit(1)
"
)

IFS=',' read -ra MODEL_ARRAY <<< "${MODELS}"
IFS=',' read -ra LOSS_ARRAY <<< "${LOSSES}"
IFS=',' read -ra AUX_CALIB_ARRAY <<< "${AUX_CALIBS}"

# ─── Parse model_overrides for per-model batch_size / grad accum ────────────
# Reads model_overrides section from factorial YAML. Models not listed use
# fixed.batch_size with gradient_accumulation_steps=1 (default).
# See: docs/planning/sam3-batch-size-1-and-robustification-plan.xml Task 1.2
GLOBAL_BATCH_SIZE=$(python3 -c "
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path('${REPO_ROOT}/${CONFIG_FILE}').read_text(encoding='utf-8'))
print(cfg.get('fixed', {}).get('batch_size', 2))
")

declare -A MODEL_BATCH_SIZE
declare -A MODEL_GRAD_ACCUM

# Populate per-model overrides from YAML (if model_overrides section exists)
while IFS=' ' read -r m_name m_bs m_accum; do
    MODEL_BATCH_SIZE["${m_name}"]="${m_bs}"
    MODEL_GRAD_ACCUM["${m_name}"]="${m_accum}"
done < <(
    python3 -c "
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path('${REPO_ROOT}/${CONFIG_FILE}').read_text(encoding='utf-8'))
overrides = cfg.get('model_overrides', {})
for model_name, settings in overrides.items():
    bs = settings.get('batch_size', cfg.get('fixed', {}).get('batch_size', 2))
    accum = settings.get('gradient_accumulation_steps', 1)
    print(f'{model_name} {bs} {accum}')
"
)

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

# ─── Resume: check existing jobs (Gap #3) ─────────────────────────────────────
# Extract NAME (column 3) and STATUS (column 9) from sky jobs queue.
# Only skip jobs that are PENDING, STARTING, RUNNING, or SUCCEEDED.
# FAILED and CANCELLED jobs should be RE-SUBMITTED (that's the point of resume).
EXISTING_ACTIVE_JOBS=""
if [ "${RESUME}" = true ] && [ "${DRY_RUN}" = false ]; then
    echo "Resume mode: checking for already-submitted conditions..."
    # Column 3 = NAME, Column 9 = STATUS (PENDING/STARTING/RUNNING/SUCCEEDED/FAILED/CANCELLED)
    # Only skip active/succeeded jobs — FAILED/CANCELLED should be retried.
    EXISTING_ACTIVE_JOBS=$("${SKY_BIN}" jobs queue 2>/dev/null \
        | grep -E "PENDING|STARTING|RUNNING|SUCCEEDED|RECOVERING" \
        | awk '{print $3}' || echo "")
    if [ -n "${EXISTING_ACTIVE_JOBS}" ]; then
        SKIP_COUNT=$(echo "${EXISTING_ACTIVE_JOBS}" | wc -l)
    else
        SKIP_COUNT=0
    fi
    echo "Found ${SKIP_COUNT} active/succeeded jobs in queue (FAILED/CANCELLED will be retried)"
fi

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

                # Resolve per-model batch_size and gradient_accumulation_steps
                BATCH_SIZE="${MODEL_BATCH_SIZE[${model}]:-${GLOBAL_BATCH_SIZE}}"
                GRAD_ACCUM="${MODEL_GRAD_ACCUM[${model}]:-1}"
                EFFECTIVE_BS=$((BATCH_SIZE * GRAD_ACCUM))

                echo "[${CONDITION}/${TOTAL_GPU_JOBS}] ${model} × ${loss} × aux_calib=${aux_calib} × fold=${fold} (bs=${BATCH_SIZE}, accum=${GRAD_ACCUM}, eff_bs=${EFFECTIVE_BS}, pt: ${PT_METHODS})"

                if [ "${DRY_RUN}" = true ]; then
                    echo "  [DRY RUN] sky jobs launch ${LAUNCH_YAML} --name ${CONDITION_NAME} --env MODEL_FAMILY=${model} --env BATCH_SIZE=${BATCH_SIZE} --env GRAD_ACCUM_STEPS=${GRAD_ACCUM} ..."
                    echo "${CONDITION} | ${model} | ${loss} | ${aux_calib} | ${fold} | bs=${BATCH_SIZE} | accum=${GRAD_ACCUM} | DRY_RUN" >> "${JOB_LOG}"
                else
                    # Resume: skip already-submitted conditions (Gap #3)
                    if [ "${RESUME}" = true ] && echo "${EXISTING_ACTIVE_JOBS}" | grep -qF "${CONDITION_NAME}"; then
                        echo "  [SKIP] ${CONDITION_NAME} already in queue"
                        echo "${CONDITION} | ${model} | ${loss} | ${aux_calib} | ${fold} | SKIPPED_RESUME" >> "${JOB_LOG}"
                        continue
                    fi

                    # Launch in background subshell with retry + exponential backoff (Gap #1).
                    # Each subshell logs its own result to JOB_LOG (atomic appends).
                    (
                        BACKOFF=${RETRY_DELAY}
                        for attempt in $(seq 1 "${MAX_RETRIES}"); do
                            if "${SKY_BIN}" jobs launch "${LAUNCH_YAML}" \
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
                                --env BATCH_SIZE="${BATCH_SIZE}" \
                                --env GRAD_ACCUM_STEPS="${GRAD_ACCUM}" \
                                --env-file "${ENV_FILE}" \
                                -y; then
                                echo "${CONDITION} | ${model} | ${loss} | ${aux_calib} | ${fold} | LAUNCHED" >> "${JOB_LOG}"
                                exit 0
                            fi
                            if [ "${attempt}" -lt "${MAX_RETRIES}" ]; then
                                echo "  Retry ${attempt}/${MAX_RETRIES} for ${CONDITION_NAME} (backoff ${BACKOFF}s)"
                                sleep "${BACKOFF}"
                                BACKOFF=$((BACKOFF * 2))
                            fi
                        done
                        echo "${CONDITION} | ${model} | ${loss} | ${aux_calib} | ${fold} | LAUNCH_FAILED" >> "${JOB_LOG}"
                        echo "  FAILED: ${CONDITION_NAME} after ${MAX_RETRIES} attempts"
                    ) &

                    # Rate limiting: prevent SkyPilot API quota issues (from cloud config)
                    sleep "${RATE_LIMIT_SECONDS}"

                    # Parallel launch: if we've hit the parallel_submissions limit, wait for a slot
                    RUNNING_JOBS=$(jobs -rp | wc -l)
                    if [ "${RUNNING_JOBS}" -ge "${PARALLEL_SUBMISSIONS}" ]; then
                        wait -n 2>/dev/null || true
                    fi
                fi
            done
        done
    done
done

# Wait for all background launch subshells to complete
wait

# Count results from job log (background subshells wrote directly to log)
# Count results from job log. grep -c exits 1 when no matches, which triggers
# || echo 0, producing "0\n0" (two zeros). Use { grep || true; } to suppress exit code.
LAUNCHED=$(grep -c "| LAUNCHED$" "${JOB_LOG}" 2>/dev/null; true)
LAUNCHED="${LAUNCHED%%[^0-9]*}"  # Strip non-digits (newlines, spaces)
LAUNCHED="${LAUNCHED:-0}"
FAILED=$(grep -c "| LAUNCH_FAILED$" "${JOB_LOG}" 2>/dev/null; true)
FAILED="${FAILED%%[^0-9]*}"
FAILED="${FAILED:-0}"

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

        # Resolve per-model overrides for zero-shot baselines
        overrides = cfg.get('model_overrides', {})
        model_cfg = overrides.get(model, {})
        zs_batch_size = model_cfg.get('batch_size', fixed.get('batch_size', 2))
        zs_grad_accum = model_cfg.get('gradient_accumulation_steps', 1)

        if dry_run:
            print(f'  [DRY RUN] sky jobs launch --name {condition_name} --env MODEL_FAMILY={model} --env BATCH_SIZE={zs_batch_size} --env GRAD_ACCUM_STEPS={zs_grad_accum} ...')
            with open(job_log, 'a', encoding='utf-8') as f:
                f.write(f'ZS | {model} | zero_shot | false | {fold} | bs={zs_batch_size} | accum={zs_grad_accum} | DRY_RUN\n')
        else:
            result = subprocess.run([
                '${SKY_BIN}', 'jobs', 'launch',
                '${LAUNCH_YAML}',
                '--name', condition_name,
                '--env', f'MODEL_FAMILY={model}',
                '--env', f'LOSS_NAME=none',
                '--env', f'FOLD_ID={fold}',
                '--env', f'WITH_AUX_CALIB=false',
                '--env', f'MAX_EPOCHS=0',
                '--env', f'MAX_TRAIN_VOLUMES=${MAX_TRAIN}',
                '--env', f'MAX_VAL_VOLUMES=${MAX_VAL}',
                '--env', f'POST_TRAINING_METHODS=none',
                '--env', f'EXPERIMENT_NAME={experiment}',
                '--env', f'ZERO_SHOT=true',
                '--env', f'EVAL_DATASET={dataset}',
                '--env', f'BATCH_SIZE={zs_batch_size}',
                '--env', f'GRAD_ACCUM_STEPS={zs_grad_accum}',
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

            # Rate limiting (from cloud config)
            time.sleep(${RATE_LIMIT_SECONDS})

print(f'Zero-shot baselines: {zs_launched} launched, {zs_failed} failed')
"
    echo ""
    echo "Zero-shot baselines launched."
fi

# ─── Clean up generated YAML (if any) ──────────────────────────────────────
if [ -n "${GENERATED_YAML}" ] && [ -f "${GENERATED_YAML}" ]; then
    rm -f "${GENERATED_YAML}"
    echo "Cleaned up generated YAML: ${GENERATED_YAML}"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ALL CONDITIONS LAUNCHED                                    ║"
echo "║  Job log: ${JOB_LOG}"
echo "║  Monitor: sky jobs queue                                    ║"
echo "║  Logs:    sky jobs logs <JOB_ID>                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Exit codes (Gap #10): 0=all launched, 1=all failed, 2=partial failure
if [ "${FAILED}" -gt 0 ]; then
    echo ""
    echo "WARNING: ${FAILED} trainable condition(s) failed to launch."
    echo "Review ${JOB_LOG} for details."
    echo "Resume with: $0 --resume ${CONFIG_FILE}"
    if [ "${LAUNCHED}" -gt 0 ]; then
        exit 2  # Partial failure: some launched, some failed
    else
        exit 1  # Total failure: nothing launched
    fi
fi

exit 0  # All conditions launched successfully
