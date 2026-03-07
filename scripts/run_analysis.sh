#!/usr/bin/env bash
set -euo pipefail

# Run MinIVess analysis flow via Prefect deployment.
#
# Usage:
#   ./scripts/run_analysis.sh [experiment]
#
# Requires:
#   PREFECT_API_URL environment variable pointing to the Prefect server.

EXPERIMENT="${1:-minivess_training}"

echo "Submitting analysis flow: experiment=${EXPERIMENT}"

prefect deployment run 'analysis-flow/default' \
  --params "{\"experiment_name\": \"${EXPERIMENT}\"}"
