#!/usr/bin/env bash
set -euo pipefail

# Run MinIVess training flow via Prefect deployment.
#
# Usage:
#   ./scripts/run_training.sh [loss] [model] [compute] [experiment]
#
# Examples:
#   ./scripts/run_training.sh cbdice_cldice dynunet auto minivess_training
#   ./scripts/run_training.sh dice_ce segresnet auto minivess_ablation
#
# Requires:
#   PREFECT_API_URL environment variable pointing to the Prefect server.

LOSS="${1:-cbdice_cldice}"
MODEL="${2:-dynunet}"
COMPUTE="${3:-auto}"
EXPERIMENT="${4:-minivess_training}"

echo "Submitting training flow: loss=${LOSS} model=${MODEL} compute=${COMPUTE} experiment=${EXPERIMENT}"

prefect deployment run 'training-flow/default' \
  --params "{\"loss_name\": \"${LOSS}\", \"model_family\": \"${MODEL}\", \"compute\": \"${COMPUTE}\", \"experiment_name\": \"${EXPERIMENT}\"}"
