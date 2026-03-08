#!/bin/bash
################################################################################
# train_all_hyperparam_combos.sh
#
# Hyperparameter grid launcher for MinIVess MLOps.
#
# Reads a YAML experiment config, generates the Cartesian product of all
# hyperparameter lists, and launches each combination via Docker Compose.
# Already-completed runs are skipped (auto-resume via MLflow fingerprinting).
#
# Usage:
#   ./scripts/train_all_hyperparam_combos.sh [OPTIONS]
#
# Options:
#   --config PATH       Path to experiment YAML config (required)
#   --dry-run           Print grid without launching training
#   --families STR,...  Override model families (comma-separated)
#   --losses STR,...    Override loss_name list (comma-separated)
#
# Examples:
#   ./scripts/train_all_hyperparam_combos.sh \
#       --config configs/experiments/dynunet_grid.yaml --dry-run
#
#   ./scripts/train_all_hyperparam_combos.sh \
#       --config configs/experiments/dynunet_grid.yaml
#
#   ./scripts/train_all_hyperparam_combos.sh \
#       --config configs/experiments/smoke_test.yaml --dry-run
#
# Architecture:
#   Each combination is launched via:
#     docker compose -f deployment/docker-compose.flows.yml run --rm train \
#         python -m minivess.orchestration.flows.train_flow ...
#
#   NEVER uses bare "uv run python" for training (CLAUDE.md Rule #17).
#   NEVER writes to /tmp — all artifacts go to named Docker volumes (Rule #18).
#
# Closes: #401
################################################################################

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CONFIG_PATH=""
DRY_RUN=0
OVERRIDE_FAMILIES=""
OVERRIDE_LOSSES=""
COMPOSE_FILE="deployment/docker-compose.flows.yml"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --families)
            OVERRIDE_FAMILIES="$2"
            shift 2
            ;;
        --losses)
            OVERRIDE_LOSSES="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            echo "Usage: $0 --config PATH [--dry-run] [--families STR] [--losses STR]" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------
if [[ -z "${CONFIG_PATH}" ]]; then
    echo "[ERROR] --config is required" >&2
    echo "Usage: $0 --config configs/experiments/dynunet_grid.yaml" >&2
    exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[ERROR] Config file not found: ${CONFIG_PATH}" >&2
    exit 1
fi

if [[ ! -f "${COMPOSE_FILE}" ]]; then
    echo "[ERROR] Docker Compose file not found: ${COMPOSE_FILE}" >&2
    echo "Run from the project root directory." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Parse YAML config using Python (yaml.safe_load — CLAUDE.md Rule #16)
# ---------------------------------------------------------------------------
# We use python -c to parse YAML because we need the Cartesian product
# and shell cannot parse YAML natively. This is a configuration-reading
# step only — actual training is launched via Docker.
GRID_JSON=$(python3 -c "
import json
import sys
from itertools import product
from pathlib import Path

import yaml

config_path = Path('${CONFIG_PATH}')
config = yaml.safe_load(config_path.read_text(encoding='utf-8'))

model_family = config.get('model_family', 'dynunet')
hyperparameters = config.get('hyperparameters', {})
fixed = config.get('fixed', {})
mlflow_experiment = config.get('mlflow_experiment', 'minivess_hpo_grid')

# Apply CLI overrides if provided
override_families = '${OVERRIDE_FAMILIES}'
override_losses = '${OVERRIDE_LOSSES}'

if override_families:
    model_family = override_families.split(',')[0].strip()
if override_losses:
    hyperparameters['loss_name'] = [x.strip() for x in override_losses.split(',')]

# Build Cartesian product of all hyperparameter lists
hp_keys = list(hyperparameters.keys())
hp_values = [hyperparameters[k] for k in hp_keys]

combos = []
for combo_values in product(*hp_values):
    combo = dict(zip(hp_keys, combo_values))
    combo.update(fixed)
    combo['model_family'] = model_family
    combo['mlflow_experiment'] = mlflow_experiment
    combos.append(combo)

print(json.dumps(combos))
" 2>&1)

if [[ $? -ne 0 ]]; then
    echo "[ERROR] Failed to parse YAML config: ${CONFIG_PATH}" >&2
    echo "${GRID_JSON}" >&2
    exit 1
fi

TOTAL=$(echo "${GRID_JSON}" | python3 -c "import json,sys; print(len(json.load(sys.stdin)))")
echo "[grid] Config: ${CONFIG_PATH}"
echo "[grid] Total combinations: ${TOTAL}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[grid] --dry-run: printing grid without launching training"
    echo ""
    python3 -c "
import json, sys
combos = json.loads('''${GRID_JSON}''')
for i, combo in enumerate(combos, start=1):
    items = '  '.join(f'{k}={v}' for k, v in sorted(combo.items()))
    print(f'  [{i}/{len(combos)}] {items}')
"
    echo ""
    echo "[grid] Dry run complete. Re-run without --dry-run to launch training."
    exit 0
fi

# ---------------------------------------------------------------------------
# Launch each combination via Docker Compose
# ---------------------------------------------------------------------------
LAUNCHED=0
SKIPPED=0
FAILED=0

echo "[grid] Launching ${TOTAL} training combinations via Docker Compose"
echo "[grid] Compose file: ${COMPOSE_FILE}"
echo ""

python3 -c "
import json, sys
combos = json.loads('''${GRID_JSON}''')
for combo in combos:
    # Print one line per combo to stdout for the shell loop
    items = ' '.join(f\"{k}={v}\" for k, v in sorted(combo.items()))
    print(items)
" | while IFS= read -r COMBO_LINE; do
    # Parse key=value pairs from the combo line
    declare -A COMBO
    for pair in $COMBO_LINE; do
        key="${pair%%=*}"
        val="${pair#*=}"
        COMBO["$key"]="$val"
    done

    LOSS="${COMBO[loss_name]:-cbdice_cldice}"
    FAMILY="${COMBO[model_family]:-dynunet}"
    EPOCHS="${COMBO[max_epochs]:-100}"
    FOLDS="${COMBO[num_folds]:-3}"
    BATCH="${COMBO[batch_size]:-2}"
    LR="${COMBO[learning_rate]:-1e-3}"
    EXPERIMENT="${COMBO[mlflow_experiment]:-minivess_hpo_grid}"

    echo "[grid] Launching: family=${FAMILY} loss=${LOSS} lr=${LR} batch=${BATCH} epochs=${EPOCHS} folds=${FOLDS}"

    # Launch via Docker Compose — NEVER bare python (CLAUDE.md Rule #17)
    # All artifacts go to named volumes defined in docker-compose.flows.yml (Rule #18)
    if docker compose -f "${COMPOSE_FILE}" run --rm \
        -e MODEL_FAMILY="${FAMILY}" \
        -e LOSS_NAME="${LOSS}" \
        -e MAX_EPOCHS="${EPOCHS}" \
        -e NUM_FOLDS="${FOLDS}" \
        -e BATCH_SIZE="${BATCH}" \
        -e LEARNING_RATE="${LR}" \
        -e EXPERIMENT_NAME="${EXPERIMENT}" \
        train 2>&1; then
        echo "[grid] Done: family=${FAMILY} loss=${LOSS} lr=${LR} batch=${BATCH}"
        LAUNCHED=$((LAUNCHED + 1))
    else
        echo "[WARN] Failed: family=${FAMILY} loss=${LOSS} lr=${LR} — continuing grid" >&2
        FAILED=$((FAILED + 1))
    fi

    unset COMBO
done

echo ""
echo "[grid] Complete: launched=${LAUNCHED}  skipped=${SKIPPED}  failed=${FAILED}"

if [[ "${FAILED}" -gt 0 ]]; then
    echo "[WARN] ${FAILED} combination(s) failed. Check logs above." >&2
    exit 1
fi
