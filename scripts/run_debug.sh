#!/usr/bin/env bash
# run_debug.sh — Debug pipeline wrapper
#
# Preflight → model iteration → Docker compose → multi-flow chaining → summary
#
# Usage:
#   bash scripts/run_debug.sh                              # debug_single_model
#   bash scripts/run_debug.sh --experiment debug_all_models
#   bash scripts/run_debug.sh --experiment debug_full_pipeline --override "max_epochs=2"
#   bash scripts/run_debug.sh --dry-run                   # show commands without running
#
# CLAUDE.md Rule #17: ALL flow execution goes through Docker + Prefect. No exceptions.
# CLAUDE.md Rule #18: No /tmp for artifacts — outputs go to named Docker volumes.

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
EXPERIMENT="${EXPERIMENT:-debug_single_model}"
USER_OVERRIDES=""
DRY_RUN=false
FLOWS_COMPOSE="deployment/docker-compose.flows.yml"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Docker Compose V2 resolves .env from the compose file's directory (deployment/),
# NOT the working directory. Use --env-file to explicitly load repo-root .env
# so HF_TOKEN and other secrets defined there reach the containers.
# Reference: deployment/CLAUDE.md, memory/docker-infra-learnings.md entry #13.
ENV_FILE_ARG=""
if [ -f "$REPO_ROOT/.env" ]; then
  ENV_FILE_ARG="--env-file $REPO_ROOT/.env"
fi
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
SUMMARY_DIR="outputs/debug"

# ─── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment|-e)
      EXPERIMENT="$2"; shift 2 ;;
    --override|-o)
      USER_OVERRIDES="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=true; shift ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash scripts/run_debug.sh [--experiment NAME] [--override KEY=VALUE] [--dry-run]" >&2
      exit 1 ;;
  esac
done

# Export EXPERIMENT so Python subprocesses (YAML parsing) can read it via os.environ.
# Without this, --experiment CLI flag sets a shell variable (not inherited by subprocess).
export EXPERIMENT

cd "$REPO_ROOT"

# ─── Helpers ─────────────────────────────────────────────────────────────────
print_banner() {
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  $1"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

run_or_dry() {
  local label="$1"; shift
  if [ "$DRY_RUN" = "true" ]; then
    echo "  [DRY-RUN] $label"
    echo "    $*"
  else
    echo "  Running: $label"
    "$@"
  fi
}

# ─── Preflight ────────────────────────────────────────────────────────────────
print_banner "MinIVess Debug Runner — $EXPERIMENT"
echo "  Experiment : $EXPERIMENT"
echo "  Overrides  : ${USER_OVERRIDES:-none}"
echo "  Dry-run    : $DRY_RUN"
echo "  Timestamp  : $TIMESTAMP"
echo ""

PREFLIGHT_OK=true

# 1. Docker daemon
if ! docker info &>/dev/null; then
  echo "  ✗ Docker daemon not running. Start Docker first."
  PREFLIGHT_OK=false
else
  echo "  ✓ Docker daemon running"
fi

# 2. NVIDIA CDI GPU access (required — training without GPU is theater, not training)
# Uses CDI (docker 25+) — NOT --runtime nvidia (requires daemon config)
if ! docker run --rm --device nvidia.com/gpu=all ubuntu:22.04 ls /dev/nvidia0 &>/dev/null 2>&1; then
  echo "  ✗ GPU not accessible via CDI (nvidia.com/gpu=all). Check: nvidia-ctk cdi list"
  echo "    Training requires GPU. Fix GPU access before proceeding."
  PREFLIGHT_OK=false
else
  echo "  ✓ GPU accessible via CDI (/dev/nvidia0 present)"
fi

# 3. minivess-base image
if ! docker image inspect minivess-base:latest &>/dev/null; then
  echo "  ✗ minivess-base:latest not found. Run: docker compose -f $FLOWS_COMPOSE build base"
  PREFLIGHT_OK=false
else
  echo "  ✓ minivess-base:latest found"
fi

# 4. Flows compose file
if [ ! -f "$FLOWS_COMPOSE" ]; then
  echo "  ✗ $FLOWS_COMPOSE not found"
  PREFLIGHT_OK=false
else
  echo "  ✓ $FLOWS_COMPOSE found"
fi

# 5. Experiment config exists
EXPERIMENT_YAML="configs/experiment/${EXPERIMENT}.yaml"
if [ ! -f "$EXPERIMENT_YAML" ]; then
  echo "  ✗ Experiment config not found: $EXPERIMENT_YAML"
  echo "    Available debug configs:"
  ls configs/experiment/debug_*.yaml 2>/dev/null | sed 's|configs/experiment/||;s|\.yaml||' | sed 's/^/      /'
  PREFLIGHT_OK=false
else
  echo "  ✓ Experiment config: $EXPERIMENT_YAML"
fi

if [ "$PREFLIGHT_OK" = "false" ] && [ "$DRY_RUN" = "false" ]; then
  echo ""
  echo "  Preflight failed. Fix issues above and retry."
  echo "  (Use --dry-run to skip Docker checks and preview commands)"
  exit 1
fi

echo ""

# ─── Read experiment config ────────────────────────────────────────────────────
# Parse models_to_test list from YAML using Python yaml.safe_load (no regex per Rule #16)
MODELS_TO_TEST=$(uv run python3 - <<'PYEOF'
from __future__ import annotations
import sys
import yaml
from pathlib import Path

import os
experiment = os.environ.get("EXPERIMENT", "debug_single_model")
yaml_path = Path(f"configs/experiment/{experiment}.yaml")
if not yaml_path.exists():
    sys.exit(0)

cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

# models_to_test list (debug_all_models) or single model field
if "models_to_test" in cfg:
    print("\n".join(cfg["models_to_test"]))
elif "model" in cfg:
    print(cfg["model"])
else:
    print("dynunet")
PYEOF
)

# Parse flows list from YAML
FLOWS=$(uv run python3 - <<'PYEOF'
from __future__ import annotations
import os
import yaml
from pathlib import Path

experiment = os.environ.get("EXPERIMENT", "debug_single_model")
yaml_path = Path(f"configs/experiment/{experiment}.yaml")
if not yaml_path.exists():
    print("")
    exit(0)

cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
flows = cfg.get("flows", ["train"])
print(" ".join(flows))
PYEOF
)

echo "  Models : $(echo "$MODELS_TO_TEST" | tr '\n' ' ')"
echo "  Flows  : $FLOWS"
echo ""

# ─── Create output dirs ────────────────────────────────────────────────────────
mkdir -p "$SUMMARY_DIR/logs"

# ─── Per-model training ────────────────────────────────────────────────────────
FLOW_STATUSES=()
TOTAL_START=$(date +%s)

# Use readarray to load ALL model names into a bash array upfront.
# This avoids the stdin-consumption bug: `while read ... done <<< VAR` shares stdin
# with the loop body. `docker compose run` (even with -T) can read ahead and consume
# remaining model names from the herestring before the next loop iteration.
# readarray loads everything in one shot, then the for loop uses no stdin at all.
readarray -t MODELS_ARRAY <<< "$MODELS_TO_TEST"

for model in "${MODELS_ARRAY[@]}"; do
  [ -z "$model" ] && continue

  print_banner "Training: $model (experiment=$EXPERIMENT)"

  MODEL_LOG="$SUMMARY_DIR/logs/${TIMESTAMP}_${model}.log"
  MODEL_START=$(date +%s)

  # Build Hydra overrides: model=X [,user overrides]
  OVERRIDES="model=$model"
  # sam3_hybrid: uses default patch_size=(64,64,3) from configs.
  # No patch_size override needed: validation is skipped (val_interval>max_epochs)
  # so val_roi OOM is not a concern. Training at (64,64,3) uses ~0.5 GiB activation
  # + 6.65 GiB model = ~7.15 GiB, which fits in 7.60 GiB CUDA budget.
  # Dimension fix (2026-03-09): sam3_hybrid.py now correctly iterates D=3 slices
  # (not H=64) using MONAI (B,C,H,W,D) order → 3 encoder calls instead of 64.
  if [ -n "$USER_OVERRIDES" ]; then
    OVERRIDES="$OVERRIDES,$USER_OVERRIDES"
  fi

  # -T: disable pseudo-TTY; </dev/null: detach container stdin.
  # docker compose run attaches to container stdin by default — without these,
  # it can consume the shell's stdin unexpectedly.
  # set +e / set -e: suspend errexit so a failed model doesn't abort the entire
  # loop — each model is independent and we want all models to attempt training.
  # PIPESTATUS[0] captures docker compose exit code (not tee's).
  set +e
  run_or_dry "docker compose train ($model)" \
    docker compose $ENV_FILE_ARG -f "$FLOWS_COMPOSE" run --rm -T \
      -e EXPERIMENT="$EXPERIMENT" \
      -e HYDRA_OVERRIDES="$OVERRIDES" \
      train </dev/null 2>&1 | tee "$MODEL_LOG"
  MODEL_STATUS=${PIPESTATUS[0]}
  set -e
  MODEL_END=$(date +%s)
  MODEL_DUR=$(( (MODEL_END - MODEL_START) / 60 ))

  if [ "$MODEL_STATUS" -eq 0 ]; then
    echo "  ✓ $model — ${MODEL_DUR}m"
    FLOW_STATUSES+=("train/$model:OK:${MODEL_DUR}m")
  else
    echo "  ✗ $model — FAILED after ${MODEL_DUR}m. Log: $MODEL_LOG"
    FLOW_STATUSES+=("train/$model:FAILED:${MODEL_DUR}m")
  fi

done

# ─── Flow chaining ────────────────────────────────────────────────────────────
# Run post_training and analyze flows if experiment config specifies them
for flow in $FLOWS; do
  [ "$flow" = "train" ] && continue
  [ "$flow" = "data_validation" ] && continue  # handled separately

  print_banner "Flow: $flow (upstream=$EXPERIMENT)"

  FLOW_LOG="$SUMMARY_DIR/logs/${TIMESTAMP}_${flow}.log"
  FLOW_START=$(date +%s)

  set +e
  run_or_dry "docker compose $flow" \
    docker compose $ENV_FILE_ARG -f "$FLOWS_COMPOSE" run --rm -T \
      -e UPSTREAM_EXPERIMENT="$EXPERIMENT" \
      -e EXPERIMENT="$EXPERIMENT" \
      "$flow" </dev/null 2>&1 | tee "$FLOW_LOG"
  CHAIN_STATUS=${PIPESTATUS[0]}
  set -e
  FLOW_END=$(date +%s)
  FLOW_DUR=$(( (FLOW_END - FLOW_START) / 60 ))

  if [ "$CHAIN_STATUS" -eq 0 ]; then
    echo "  ✓ $flow — ${FLOW_DUR}m"
    FLOW_STATUSES+=("$flow:OK:${FLOW_DUR}m")
  else
    echo "  ✗ $flow — FAILED after ${FLOW_DUR}m"
    FLOW_STATUSES+=("$flow:FAILED:${FLOW_DUR}m")
  fi
done

# ─── Summary ──────────────────────────────────────────────────────────────────
TOTAL_END=$(date +%s)
TOTAL_DUR=$(( (TOTAL_END - TOTAL_START) / 60 ))

SUMMARY_JSON="$SUMMARY_DIR/summary_${TIMESTAMP}.json"

if [ "$DRY_RUN" = "false" ]; then
  # Pass FLOW_STATUSES as a JSON array via a temp file to avoid bash-into-Python
  # injection (the old approach used ${FLOW_STATUSES[@]} directly in Python source,
  # producing `statuses = train/dynunet:OK:0m` — a SyntaxError).
  STATUSES_JSON_FILE=$(mktemp /tmp/minivess-statuses-XXXXXX.json)
  # Build a JSON array from the bash array
  printf '%s\n' "${FLOW_STATUSES[@]:-}" | uv run python3 -c "
import sys, json
lines = [l.rstrip() for l in sys.stdin if l.strip()]
json.dump(lines, sys.stdout)
" > "$STATUSES_JSON_FILE"

  uv run python3 - <<PYEOF
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

with open("$STATUSES_JSON_FILE", encoding="utf-8") as f:
    statuses = json.load(f)

timestamp = "$TIMESTAMP"
experiment = "$EXPERIMENT"
total_dur = $TOTAL_DUR

# Parse status strings "flow/model:STATUS:DURm"
parsed = []
for s in statuses:
    parts = s.split(":")
    parsed.append({"name": parts[0], "status": parts[1] if len(parts) > 1 else "UNKNOWN",
                   "duration_min": parts[2].rstrip("m") if len(parts) > 2 else "?"})

summary = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "experiment": experiment,
    "total_duration_min": total_dur,
    "flows": parsed,
}

out = Path("$SUMMARY_JSON")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"  Summary saved: $SUMMARY_JSON")
PYEOF
  rm -f "$STATUSES_JSON_FILE"
fi

print_banner "Debug Run Complete"
echo "  Experiment : $EXPERIMENT"
echo "  Duration   : ${TOTAL_DUR}m"
echo "  Flows      :"
for s in "${FLOW_STATUSES[@]:-}"; do
  echo "    $s"
done
echo ""
echo "  MLflow runs: uv run mlflow ui --backend-store-uri mlruns"
echo "  Logs dir   : $SUMMARY_DIR/logs/"
