#!/usr/bin/env bash
set -euo pipefail

# Run the full MinIVess pipeline trigger chain via Prefect deployments.
#
# Usage:
#   ./scripts/run_pipeline.sh [experiment]
#
# Triggers: data -> train -> analyze -> deploy -> dashboard
#
# Requires:
#   PREFECT_API_URL environment variable pointing to the Prefect server.

EXPERIMENT="${1:-minivess_training}"

echo "Triggering full pipeline: experiment=${EXPERIMENT}"

# Trigger data flow first
prefect deployment run 'data-engineering-flow/default' \
  --params "{\"experiment_name\": \"${EXPERIMENT}\"}"

echo "Pipeline trigger submitted. Monitor at: ${PREFECT_API_URL:-http://localhost:4200}"
