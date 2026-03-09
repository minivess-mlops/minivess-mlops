#!/usr/bin/env bash
# MinIVess dev environment setup.
#
# Run once after cloning or when .env.enc changes:
#   bash scripts/setup_dev.sh
#
# Solo developer:  copy .env.example → .env, fill in your values.
# Team (SOPS+age): decrypt .env.enc → .env using your age key.
#   Install age:  sudo apt-get install age
#   Install sops: https://github.com/getsops/sops/releases
#   Decrypt:      sops --decrypt .env.enc > .env

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
ENC_FILE="$REPO_ROOT/.env.enc"

# ── Step 1: Ensure .env exists ───────────────────────────────────────────────
if [[ -f "$ENC_FILE" ]]; then
  echo "[setup_dev] Decrypting .env.enc → .env"
  echo "  Requires age key at: ~/.config/sops/age/keys.txt"
  sops --decrypt "$ENC_FILE" > "$ENV_FILE"
  echo "[setup_dev] .env decrypted successfully."
elif [[ -f "$ENV_FILE" ]]; then
  echo "[setup_dev] .env already present (solo dev mode — using as-is)."
else
  echo "[setup_dev] ERROR: neither .env nor .env.enc found."
  echo "  Solo developer setup:"
  echo "    cp .env.example .env"
  echo "    # Edit .env with your actual values"
  echo "  Team setup:"
  echo "    # Install sops + age, obtain .env.enc from your team"
  echo "    sops --decrypt .env.enc > .env"
  exit 1
fi

# ── Step 2: Validate required variables ─────────────────────────────────────
WARNINGS=0
required_vars=(
  "MODEL_CACHE_HOST_PATH"
  "MINIO_ROOT_PASSWORD"
  "POSTGRES_PASSWORD"
  "HF_TOKEN"
)

for var in "${required_vars[@]}"; do
  val=$(grep "^${var}=" "$ENV_FILE" 2>/dev/null | cut -d= -f2- || true)
  if [[ -z "$val" ]] || [[ "$val" == *"REPLACE"* ]] || [[ "$val" == *"hf_REPLACE"* ]]; then
    echo "[setup_dev] WARNING: $var is unset or still a placeholder in .env"
    WARNINGS=$((WARNINGS + 1))
  fi
done

if [[ $WARNINGS -gt 0 ]]; then
  echo ""
  echo "[setup_dev] $WARNINGS warning(s) — edit .env before running docker compose."
  echo "  Reference: .env.example for all required values."
fi

echo ""
echo "[setup_dev] Done. Run the infra stack:"
echo "  docker compose --env-file .env -f deployment/docker-compose.yml --profile dev up -d"
