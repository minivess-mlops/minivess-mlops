#!/usr/bin/env bash
# Initialize MLflow basic-auth after mlflow-secure service starts.
#
# Prerequisites:
#   1. Copy auth config:  cp deployment/mlflow/auth.ini.example deployment/mlflow/auth.ini
#   2. Edit auth.ini:     set admin_password to a secure value
#   3. Start secure MLflow (MUTUALLY EXCLUSIVE with default mlflow service):
#      docker compose --env-file .env -f deployment/docker-compose.yml stop mlflow
#      docker compose --env-file .env -f deployment/docker-compose.yml --profile secure up -d mlflow-secure
#   4. Run this script:   bash scripts/init_mlflow_auth.sh
#
# MLflow 3.x basic-auth flag: --app-name basic-auth
# This flag enables the basic-auth plugin bundled with MLflow 3.x.
# MLflow auto-creates basic_auth.db in the server's working directory.
# No additional database URI flag is needed — SQLite is created automatically.

set -euo pipefail

MLFLOW_URL="${MLFLOW_TRACKING_URI:-http://localhost:5000}"

echo "[init_mlflow_auth] Waiting for MLflow server at ${MLFLOW_URL}..."
for i in $(seq 1 30); do
  if curl --silent --fail "${MLFLOW_URL}/health" > /dev/null 2>&1; then
    echo "[init_mlflow_auth] MLflow is up."
    break
  fi
  sleep 2
  if [[ $i -eq 30 ]]; then
    echo "[init_mlflow_auth] ERROR: MLflow did not respond after 60s."
    exit 1
  fi
done

echo ""
echo "[init_mlflow_auth] MLflow basic-auth is enabled via --app-name basic-auth."
echo "  The server auto-creates basic_auth.db on first start."
echo "  Admin credentials are set in deployment/mlflow/auth.ini."
echo ""
echo "  To create additional users, use the MLflow REST API:"
echo "    See: https://mlflow.org/docs/latest/auth/index.html"
echo "    POST ${MLFLOW_URL}/api/2.0/mlflow/users/create"
echo "    (authenticate with admin credentials from deployment/mlflow/auth.ini)"
echo ""
echo "  To set client credentials (add to .env):"
echo "    MLFLOW_TRACKING_USERNAME=admin"
echo "    MLFLOW_TRACKING_PASSWORD=your_admin_password"
echo ""
echo "[init_mlflow_auth] Done."
