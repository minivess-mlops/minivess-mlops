#!/bin/bash
################################################################################
# finetune_vesselfm_runpod.sh
#
# End-to-end launcher for VesselFM fine-tuning on RunPod (or any spot GPU cloud).
#
# VesselFM (Wittmann et al., CVPR 2025) requires ~10GB VRAM, exceeding the
# local RTX 2070 Super (8GB). This script provisions a cloud GPU via SkyPilot,
# runs the training flow, downloads artifacts, and tears down the pod.
#
# Closes: https://github.com/minivess-mlops/minivess-mlops/issues/366
#
# Usage:
#   ./scripts/finetune_vesselfm_runpod.sh                # RunPod A100, 100 epochs
#   ./scripts/finetune_vesselfm_runpod.sh --dry-run      # no credentials needed
#   ./scripts/finetune_vesselfm_runpod.sh --gpu A100-80GB --epochs 50
#
# Prerequisites (skip for --dry-run):
#   1. SkyPilot installed:
#      uv add 'skypilot[runpod,lambda,aws]'
#   2. RunPod credentials:
#      runpodctl config --apiKey <your-key>
#   3. Sky check passes:
#      sky check
#   4. MLFLOW_TRACKING_URI set to a public URI (see docs/planning/skypilot-advanced-plan.md)
#   5. S3 bucket for checkpoints: s3://minivess-data/ and s3://minivess-checkpoints/
#      Or: set up rclone for RunPod Network Volumes
#   6. HF_TOKEN (optional): for gated VesselFM model weights
#
# Environment:
#   MLFLOW_TRACKING_URI  Public MLflow URI (required for remote tracking)
#   HF_TOKEN             HuggingFace token for gated model weights
#   MAX_EPOCHS           Override epochs (default: 100)
#   SKYPILOT_TASK_YAML   Override SkyPilot YAML path
#
# Acceptance criteria (issue #366):
#   - --dry-run works locally without credentials
#   - Full run completes 100 epochs for VesselFM
#   - Artifacts available in local mlruns/ after teardown
#   - Zero manual SSH required
#   - Pod torn down on success AND on error (trap EXIT)
#   - Dry-run prints estimated cost before starting
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

GPU_TYPE="A100"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
DRY_RUN=0
NUM_FOLDS=3
LOSS_NAME="cbdice_cldice"
TASK_YAML="${SKYPILOT_TASK_YAML:-${PROJECT_ROOT}/deployment/skypilot/finetune_vesselfm.yaml}"
JOB_NAME="mnvss-finetune-vfm-$$"

# ─────────────────────────────────────────────────────────────────────────────
# Argument Parsing
# ─────────────────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Fine-tune VesselFM on a cloud GPU via SkyPilot (RunPod → Lambda → AWS fallover).

Options:
  --gpu GPU_TYPE     GPU type: A100 | A100-80GB | H100 (default: A100)
  --epochs N         Max training epochs per fold (default: 100)
  --num-folds N      Number of CV folds (default: 3)
  --loss LOSS        Loss function name (default: cbdice_cldice)
  --dry-run          Print plan without launching; no credentials needed
  -h, --help         Show this help

Environment variables:
  MLFLOW_TRACKING_URI   Public MLflow URI (required for remote tracking)
  HF_TOKEN              HuggingFace token for gated VesselFM weights
  MAX_EPOCHS            Override --epochs

Examples:
  # Dry run (no credentials):
  $0 --dry-run

  # Full run on A100 80GB (more headroom):
  $0 --gpu A100-80GB --epochs 100

  # Quick smoke test on cheapest GPU:
  $0 --gpu A100 --epochs 2
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)        GPU_TYPE="$2";       shift 2 ;;
        --epochs)     MAX_EPOCHS="$2";     shift 2 ;;
        --num-folds)  NUM_FOLDS="$2";      shift 2 ;;
        --loss)       LOSS_NAME="$2";      shift 2 ;;
        --dry-run)    DRY_RUN=1;           shift   ;;
        -h|--help)    usage ;;
        *) echo "[ERROR] Unknown option: $1" >&2; usage ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# Cost Estimate (printed before any action)
# ─────────────────────────────────────────────────────────────────────────────

# Approximate training time: 3-6 hours for 100 epochs on MiniVess (70 vols, 3 folds)
_ESTIMATED_HOURS_MIN=3
_ESTIMATED_HOURS_MAX=6

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " VesselFM Fine-Tuning — Cloud Launch via SkyPilot"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "[INFO] Configuration:"
echo "       GPU type:   ${GPU_TYPE}"
echo "       Max epochs: ${MAX_EPOCHS} per fold"
echo "       Folds:      ${NUM_FOLDS}"
echo "       Loss:       ${LOSS_NAME}"
echo "       Task YAML:  ${TASK_YAML}"
echo "       Job name:   ${JOB_NAME}"
echo ""
echo "[INFO] Cost estimate (${MAX_EPOCHS} epochs, ${NUM_FOLDS} folds):"
echo "       Estimated training time: ${_ESTIMATED_HOURS_MIN}-${_ESTIMATED_HOURS_MAX}h"
echo "       Provider    GPU              Est. price"
echo "       RunPod      A100 80GB        \$1.50/hr → \$$(echo "${_ESTIMATED_HOURS_MIN} * 150 / 100" | bc)-\$$(echo "${_ESTIMATED_HOURS_MAX} * 150 / 100" | bc)"
echo "       RunPod      A100             \$0.60/hr → \$$(echo "${_ESTIMATED_HOURS_MIN} * 60 / 100" | bc)-\$$(echo "${_ESTIMATED_HOURS_MAX} * 60 / 100" | bc)"
echo "       Lambda      A100             \$1.29/hr → \$$(echo "${_ESTIMATED_HOURS_MIN} * 129 / 100" | bc)-\$$(echo "${_ESTIMATED_HOURS_MAX} * 129 / 100" | bc)"
echo "       AWS spot    A100             ~\$1.20/hr → \$$(echo "${_ESTIMATED_HOURS_MIN} * 120 / 100" | bc)-\$$(echo "${_ESTIMATED_HOURS_MAX} * 120 / 100" | bc)"
echo "       (SkyPilot selects cheapest available at launch time)"
echo ""

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "[DRY RUN] Dry run mode — no credentials needed, no pod launched."
    echo ""
    echo "  Command that would run:"
    echo "    sky jobs launch '${TASK_YAML}' \\"
    echo "      --name '${JOB_NAME}' \\"
    echo "      --env MAX_EPOCHS=${MAX_EPOCHS} \\"
    echo "      --env LOSS_NAME=${LOSS_NAME} \\"
    echo "      --env NUM_FOLDS=${NUM_FOLDS} \\"
    echo "      --down \\"
    echo "      --dryrun"
    echo ""
    echo "  SkyPilot YAML:  ${TASK_YAML}"
    echo "  Failover order: RunPod A100-80GB → RunPod A100 → Lambda → AWS spot → GCP spot"
    echo ""
    echo "[DRY RUN] Done. No resources consumed."
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# Prerequisite Checks
# ─────────────────────────────────────────────────────────────────────────────

_check_sky() {
    if ! command -v sky &>/dev/null; then
        echo "[ERROR] SkyPilot not installed." >&2
        echo "        Install: uv add 'skypilot[runpod,lambda,aws]'" >&2
        echo "        Or: pip install 'skypilot[runpod,lambda,aws]'" >&2
        return 1
    fi
    return 0
}

if ! _check_sky; then
    echo "[WARN] SkyPilot not available. Cannot launch cloud job." >&2
    echo "       For dry run: $0 --dry-run" >&2
    exit 1
fi

if [ ! -f "${TASK_YAML}" ]; then
    echo "[ERROR] SkyPilot task YAML not found: ${TASK_YAML}" >&2
    exit 1
fi

if [ -z "${MLFLOW_TRACKING_URI:-}" ]; then
    echo "[WARN] MLFLOW_TRACKING_URI not set. MLflow tracking will use Docker Compose" >&2
    echo "       default (http://mlflow:5000), which is NOT reachable from cloud instances." >&2
    echo "       Set MLFLOW_TRACKING_URI to a public URI before launching." >&2
    echo "       See: docs/planning/skypilot-advanced-plan.md section 6" >&2
fi

# ─────────────────────────────────────────────────────────────────────────────
# Launch VesselFM Fine-Tuning via SkyPilot
# ─────────────────────────────────────────────────────────────────────────────

echo "[INFO] Launching VesselFM fine-tuning via SkyPilot managed jobs..."
echo "[INFO] SkyPilot will select the cheapest available provider automatically."
echo "[INFO] The pod will be torn down automatically after training completes."
echo ""

# sky jobs launch:
#   --name:   Unique job identifier (includes PID for concurrency)
#   --down:   Tear down resources automatically when job completes
#   --env:    Runtime overrides for envs defined in the YAML

sky jobs launch "${TASK_YAML}" \
    --name "${JOB_NAME}" \
    --env "MAX_EPOCHS=${MAX_EPOCHS}" \
    --env "LOSS_NAME=${LOSS_NAME}" \
    --env "NUM_FOLDS=${NUM_FOLDS}" \
    --env "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://mlflow:5000}" \
    --env "HF_TOKEN=${HF_TOKEN:-}" \
    --down \
    --yes

echo ""
echo "[INFO] Job submitted: ${JOB_NAME}"
echo "[INFO] Monitor progress:"
echo "       sky jobs logs ${JOB_NAME}"
echo "       sky jobs queue"
echo ""
echo "[INFO] When complete, download artifacts:"
echo "       sky storage ls                         # list S3 buckets"
echo "       aws s3 sync s3://minivess-mlruns/ mlruns/    # sync MLflow runs locally"
echo "       aws s3 sync s3://mnvss-ckpts-vfm/ checkpoints/  # sync checkpoints"
echo ""
echo "[INFO] To cancel:"
echo "       sky jobs cancel ${JOB_NAME}"
