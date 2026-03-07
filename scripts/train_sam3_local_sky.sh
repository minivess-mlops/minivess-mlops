#!/bin/bash
################################################################################
# train_sam3_local_sky.sh
#
# Run all SAM3 variants that fit in 8 GB VRAM (RTX 2070 Super).
#
# Execution modes (automatic detection):
#   1. SkyPilot SSH node pool (if configured and sky check ssh passes):
#        sky launch deployment/skypilot/train_sam3_8gb.yaml \
#          --infra ssh/<pool_name> --env MODEL_FAMILY=<variant>
#   2. Direct local execution (fallback, no SkyPilot required):
#        uv run python scripts/run_training_flow.py --model-family <variant>
#
# 8 GB-compatible SAM3 variants:
#   sam3_vanilla   — frozen encoder + trainable decoder  (~5-6 GB)
#   sam3_topolora  — LoRA rank=16 + trainable decoder    (~5.5-6.5 GB)
#   sam3_hybrid    — SAM3 features + DynUNet3D + gating  (~7.5 GB, MARGINAL)
#
# Prerequisites for SkyPilot SSH mode:
#   1. SSH server enabled:  sudo systemctl enable --now ssh
#   2. SkyPilot installed:  uv add 'skypilot[ssh]' (or pip install)
#   3. Node pool registered: sky ssh up  (one-time; needs kubectl)
#   4. Verify:               sky show-gpus --infra ssh/<pool_name>
#
# See: docs/planning/skypilot-advanced-plan-execution.xml
#      https://docs.skypilot.co/en/stable/reservations/existing-machines.html
#
# Usage:
#   ./scripts/train_sam3_local_sky.sh                          # all 3 variants
#   ./scripts/train_sam3_local_sky.sh --epochs 50              # shorter run
#   ./scripts/train_sam3_local_sky.sh --sky-pool local-gpu     # name the pool
#   ./scripts/train_sam3_local_sky.sh --skip-hybrid            # skip 7.5GB variant
#   ./scripts/train_sam3_local_sky.sh --debug                  # 1 epoch smoke test
#   ./scripts/train_sam3_local_sky.sh --force-direct           # bypass SkyPilot
#
# Environment:
#   MINIVESS_EPOCHS      Override max epochs (default: 100)
#   MINIVESS_SKY_POOL    SSH node pool name (default: local-gpu)
#   MLFLOW_TRACKING_URI  MLflow URI (default: local file store)
#   HF_TOKEN             HuggingFace token for SAM3 weights
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

MAX_EPOCHS="${MINIVESS_EPOCHS:-100}"
SKY_POOL="${MINIVESS_SKY_POOL:-local-gpu}"
SKIP_HYBRID=0
DEBUG_MODE=0
FORCE_DIRECT=0
LOSS_NAME="cbdice_cldice"
TASK_YAML="${PROJECT_ROOT}/deployment/skypilot/train_sam3_8gb.yaml"

# ─────────────────────────────────────────────────────────────────────────────
# Argument Parsing
# ─────────────────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Train SAM3 variants compatible with 8 GB VRAM on local GPU via SkyPilot SSH
or directly without SkyPilot (automatic fallback).

Options:
  --epochs N         Max training epochs (default: 100)
  --sky-pool NAME    SkyPilot SSH pool name (default: local-gpu)
  --skip-hybrid      Skip sam3_hybrid (~7.5 GB marginal variant)
  --debug            Smoke test: 1 epoch per variant
  --force-direct     Bypass SkyPilot; run with uv run directly
  --loss LOSS        Loss function (default: cbdice_cldice)
  -h, --help         Show this help

SkyPilot SSH setup (one-time):
  sudo systemctl enable --now ssh
  sky ssh up    # registers localhost as SSH node pool
  sky show-gpus --infra ssh/${SKY_POOL:-local-gpu}

Variants trained (8 GB):
  sam3_vanilla    ~5-6 GB   — frozen encoder + decoder
  sam3_topolora   ~5.5-6.5 GB — LoRA rank=16 + topology loss
  sam3_hybrid     ~7.5 GB   — SAM3 + DynUNet3D + gated fusion (--skip-hybrid to exclude)
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)      MAX_EPOCHS="$2";    shift 2 ;;
        --sky-pool)    SKY_POOL="$2";      shift 2 ;;
        --loss)        LOSS_NAME="$2";     shift 2 ;;
        --skip-hybrid) SKIP_HYBRID=1;      shift   ;;
        --debug)       DEBUG_MODE=1;       shift   ;;
        --force-direct) FORCE_DIRECT=1;   shift   ;;
        -h|--help)     usage ;;
        *) echo "[ERROR] Unknown option: $1" >&2; usage ;;
    esac
done

if [ "${DEBUG_MODE}" -eq 1 ]; then
    MAX_EPOCHS=1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Variant List
# ─────────────────────────────────────────────────────────────────────────────

if [ "${SKIP_HYBRID}" -eq 1 ]; then
    VARIANTS=("sam3_vanilla" "sam3_topolora")
    echo "[WARN] --skip-hybrid: skipping sam3_hybrid (~7.5 GB, marginal for 8 GB GPU)"
else
    VARIANTS=("sam3_vanilla" "sam3_topolora" "sam3_hybrid")
    echo "[INFO] sam3_hybrid requires ~7.5 GB VRAM (marginal on 8 GB GPU)."
    echo "[INFO] If it OOMs, re-run with --skip-hybrid."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Execution Mode Detection
# ─────────────────────────────────────────────────────────────────────────────

_USE_SKYPILOT=0

if [ "${FORCE_DIRECT}" -eq 0 ]; then
    if command -v sky &>/dev/null; then
        # Check if SSH pool is configured and reachable
        if sky check 2>/dev/null | grep -q "ssh"; then
            _USE_SKYPILOT=1
            echo "[INFO] SkyPilot SSH node pool detected: using sky launch --infra ssh/${SKY_POOL}"
        else
            echo "[INFO] SkyPilot installed but SSH node pool '${SKY_POOL}' not configured."
            echo "[INFO] To enable SkyPilot SSH mode:"
            echo "       1. sudo systemctl enable --now ssh"
            echo "       2. sky ssh up"
            echo "       3. sky show-gpus --infra ssh/${SKY_POOL}"
            echo "[INFO] Falling back to direct execution."
        fi
    else
        echo "[INFO] SkyPilot not installed. Using direct uv run execution."
        echo "[INFO] To install: uv add 'skypilot[ssh]'"
    fi
else
    echo "[INFO] --force-direct: skipping SkyPilot check."
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " SAM3 Training — Local 8 GB GPU"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[INFO] Mode:     $([ "${_USE_SKYPILOT}" -eq 1 ] && echo "SkyPilot SSH (${SKY_POOL})" || echo "Direct (uv run)")"
echo "[INFO] Variants: ${VARIANTS[*]}"
echo "[INFO] Epochs:   ${MAX_EPOCHS}"
echo "[INFO] Loss:     ${LOSS_NAME}"
echo "[INFO] Debug:    $([ "${DEBUG_MODE}" -eq 1 ] && echo "YES (1 epoch)" || echo "NO")"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

FAILED=()
PASSED=()

for VARIANT in "${VARIANTS[@]}"; do
    echo ""
    echo "[INFO] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[INFO] Training: ${VARIANT}"
    echo "[INFO] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ "${_USE_SKYPILOT}" -eq 1 ]; then
        # ─── SkyPilot SSH mode ─────────────────────────────────────────────
        # sky launch (NOT sky jobs launch — managed jobs unsupported on SSH pools)
        # --down: tear down after job completes (frees SkyPilot resources)
        CMD=(
            sky launch "${TASK_YAML}"
            --infra "ssh/${SKY_POOL}"
            --name "mnvss-sam3-${VARIANT}-$$"
            --env "MODEL_FAMILY=${VARIANT}"
            --env "LOSS_NAME=${LOSS_NAME}"
            --env "MAX_EPOCHS=${MAX_EPOCHS}"
            --env "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-file:///$(pwd)/mlruns}"
            --env "HF_TOKEN=${HF_TOKEN:-}"
            --down
            --yes
        )
    else
        # ─── Direct execution fallback ─────────────────────────────────────
        CMD=(
            uv run python scripts/run_training_flow.py
            --model-family "${VARIANT}"
            --loss-name "${LOSS_NAME}"
            --compute "gpu_low"
            --max-epochs "${MAX_EPOCHS}"
            --trigger-source "train_sam3_local_sky.sh"
        )
        if [ "${DEBUG_MODE}" -eq 1 ]; then
            CMD+=("--debug")
        fi
    fi

    echo "[INFO] Command: ${CMD[*]}"
    echo ""

    if "${CMD[@]}"; then
        echo "[OK] ${VARIANT} completed"
        PASSED+=("${VARIANT}")
    else
        echo "[ERROR] ${VARIANT} failed (exit code: $?)" >&2
        FAILED+=("${VARIANT}")
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "[INFO] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[INFO] Training Complete"
echo "[INFO] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "${#PASSED[@]}" -gt 0 ]; then
    echo "[OK] Passed (${#PASSED[@]}/${#VARIANTS[@]}): ${PASSED[*]}"
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "[ERROR] Failed (${#FAILED[@]}/${#VARIANTS[@]}): ${FAILED[*]}" >&2

    if [ "${_USE_SKYPILOT}" -eq 1 ]; then
        echo ""
        echo "[HINT] SkyPilot SSH mode failures — try:"
        echo "       1. sky show-gpus --infra ssh/${SKY_POOL}  (verify GPU visible)"
        echo "       2. $0 --force-direct  (bypass SkyPilot, run directly)"
        echo "       3. $0 --skip-hybrid   (skip the 7.5 GB sam3_hybrid variant)"
    fi
    exit 1
fi

echo "[OK] All SAM3 variants trained on local 8 GB GPU."
exit 0
