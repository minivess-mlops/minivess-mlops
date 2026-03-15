#!/usr/bin/env bash
# hetzner-setup-script.sh — Programmatic Hetzner Cloud setup for MinIVess MLflow
#
# Creates: SSH key + firewall + VPS + Docker Compose MLflow stack
# Requires: hcloud CLI, HETZNER_API_TOKEN in .env
# Usage: bash scripts/hetzner-setup-script.sh [--teardown]
#
# Each phase asks for confirmation before executing.
# Total time: ~5 minutes from zero to running MLflow.

set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SSH_KEY_PATH="${HOME}/.ssh/hetzner_minivess"
SERVER_NAME="minivess-mlflow"
FW_NAME="minivess-fw"
SSH_KEY_NAME="minivess-key"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

confirm() {
    local msg="${1:-Continue?}"
    echo ""
    echo -e "${YELLOW}>>> ${msg} [y/N]${NC}"
    read -r response
    [[ "$response" =~ ^[Yy]$ ]]
}

# ─── Load .env ──────────────────────────────────────────────────────────────

load_env() {
    local env_file="$REPO_ROOT/.env"
    if [[ ! -f "$env_file" ]]; then
        err ".env not found at $env_file"
        err "Run: cp .env.example .env"
        exit 1
    fi
    while IFS='=' read -r key value; do
        [[ -z "$key" || "$key" =~ ^# ]] && continue
        value="${value%\"}"
        value="${value#\"}"
        case "$key" in
            HETZNER_API_TOKEN)   export HCLOUD_TOKEN="$value" ;;
            HETZNER_LOCATION)    export HETZNER_LOCATION="$value" ;;
            HETZNER_SERVER_TYPE) export HETZNER_SERVER_TYPE="$value" ;;
            CLOUDFLARE_API_TOKEN) export CLOUDFLARE_API_TOKEN="$value" ;;
            CLOUDFLARE_ZONE_ID)   export CLOUDFLARE_ZONE_ID="$value" ;;
        esac
    done < <(grep -v '^\s*#' "$env_file" | grep -v '^\s*$')

    HETZNER_LOCATION="${HETZNER_LOCATION:-fsn1}"
    HETZNER_SERVER_TYPE="${HETZNER_SERVER_TYPE:-cx22}"
}

# ─── Phase 0: Validate Prerequisites ───────────────────────────────────────

phase0_validate() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Phase 0: Validate Prerequisites"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    local errors=0

    # Check hcloud CLI
    if command -v hcloud &>/dev/null; then
        ok "hcloud CLI found: $(hcloud version 2>&1 | head -1)"
    else
        err "hcloud CLI not found."
        err "Install: wget https://github.com/hetznercloud/cli/releases/latest/download/hcloud-linux-amd64.tar.gz"
        err "         tar xzf hcloud-linux-amd64.tar.gz && sudo mv hcloud /usr/local/bin/"
        errors=$((errors + 1))
    fi

    # Load .env
    load_env

    # Check token
    if [[ -z "${HCLOUD_TOKEN:-}" ]]; then
        err "HETZNER_API_TOKEN is not set in .env"
        err "Get one: https://console.hetzner.cloud → Project → Security → API Tokens"
        errors=$((errors + 1))
    else
        ok "HETZNER_API_TOKEN set (${#HCLOUD_TOKEN} chars)"
    fi

    ok "Location: $HETZNER_LOCATION"
    ok "Server type: $HETZNER_SERVER_TYPE"

    # Test API access
    if [[ -n "${HCLOUD_TOKEN:-}" ]] && command -v hcloud &>/dev/null; then
        info "Testing Hetzner API access..."
        if hcloud server-type describe "$HETZNER_SERVER_TYPE" -o json &>/dev/null; then
            ok "API access verified"
        else
            err "API token invalid or expired. Check HETZNER_API_TOKEN in .env"
            errors=$((errors + 1))
        fi
    fi

    # Check / generate SSH key
    if [[ -f "$SSH_KEY_PATH" ]]; then
        ok "SSH key found: $SSH_KEY_PATH"
    else
        warn "SSH key not found at $SSH_KEY_PATH"
        if confirm "Generate SSH key pair?"; then
            ssh-keygen -t ed25519 -f "$SSH_KEY_PATH" -N "" -C "minivess-hetzner"
            ok "SSH key generated: $SSH_KEY_PATH"
        else
            errors=$((errors + 1))
        fi
    fi

    if [[ $errors -gt 0 ]]; then
        err "$errors prerequisite(s) failed. Fix them and re-run."
        exit 1
    fi

    ok "All prerequisites passed!"
}

# ─── Phase 1: Upload SSH Key ───────────────────────────────────────────────

phase1_ssh_key() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Phase 1: Upload SSH Key to Hetzner"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    # Check if already uploaded
    if hcloud ssh-key describe "$SSH_KEY_NAME" &>/dev/null 2>&1; then
        ok "SSH key '$SSH_KEY_NAME' already uploaded"
        return 0
    fi

    info "Uploading public key to Hetzner..."
    hcloud ssh-key create \
        --name "$SSH_KEY_NAME" \
        --public-key-from-file "${SSH_KEY_PATH}.pub"
    ok "SSH key uploaded"
}

# ─── Phase 2: Create Firewall ──────────────────────────────────────────────

phase2_firewall() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Phase 2: Create Firewall Rules"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    if hcloud firewall describe "$FW_NAME" &>/dev/null 2>&1; then
        ok "Firewall '$FW_NAME' already exists"
        return 0
    fi

    info "Creating firewall: SSH:22, HTTP:80, HTTPS:443, MLflow:5000"

    hcloud firewall create --name "$FW_NAME"

    # Add inbound rules
    hcloud firewall add-rule "$FW_NAME" --direction in --protocol tcp --port 22 \
        --source-ips 0.0.0.0/0 --source-ips ::/0 --description "SSH"
    hcloud firewall add-rule "$FW_NAME" --direction in --protocol tcp --port 80 \
        --source-ips 0.0.0.0/0 --source-ips ::/0 --description "HTTP"
    hcloud firewall add-rule "$FW_NAME" --direction in --protocol tcp --port 443 \
        --source-ips 0.0.0.0/0 --source-ips ::/0 --description "HTTPS"
    hcloud firewall add-rule "$FW_NAME" --direction in --protocol tcp --port 5000 \
        --source-ips 0.0.0.0/0 --source-ips ::/0 --description "MLflow"
    hcloud firewall add-rule "$FW_NAME" --direction in --protocol icmp \
        --source-ips 0.0.0.0/0 --source-ips ::/0 --description "Ping"

    ok "Firewall created with 5 inbound rules"
}

# ─── Phase 3: Create Server ────────────────────────────────────────────────

phase3_server() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Phase 3: Create Server ($HETZNER_SERVER_TYPE @ $HETZNER_LOCATION)"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    # Check if already exists
    if hcloud server describe "$SERVER_NAME" &>/dev/null 2>&1; then
        local ip
        ip=$(hcloud server ip "$SERVER_NAME")
        ok "Server '$SERVER_NAME' already exists at $ip"
        return 0
    fi

    # Get pricing info
    local price_monthly
    price_monthly=$(hcloud server-type describe "$HETZNER_SERVER_TYPE" -o json | \
        python3 -c "
import json,sys
d=json.load(sys.stdin)
for p in d.get('prices',[]):
    if p['location'] == '$HETZNER_LOCATION':
        print(f\"EUR {float(p['price_monthly']['gross']):.2f}/month\")
        break
else:
    print('price unknown')
")

    info "This creates:"
    info "  - $HETZNER_SERVER_TYPE server in $HETZNER_LOCATION"
    info "  - Ubuntu 24.04 + Docker CE pre-installed"
    info "  - Firewall: $FW_NAME"
    info "  - Cost: $price_monthly"

    if ! confirm "Create server?"; then
        info "Skipping."
        return 1
    fi

    info "Creating server (takes ~30 seconds)..."
    hcloud server create \
        --name "$SERVER_NAME" \
        --type "$HETZNER_SERVER_TYPE" \
        --image docker-ce \
        --location "$HETZNER_LOCATION" \
        --ssh-key "$SSH_KEY_NAME" \
        --firewall "$FW_NAME"

    local ip
    ip=$(hcloud server ip "$SERVER_NAME")
    ok "Server created: $ip"

    # Wait for SSH to be ready
    info "Waiting for SSH to become available..."
    local attempts=0
    while [[ $attempts -lt 30 ]]; do
        if ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=accept-new \
            -i "$SSH_KEY_PATH" "root@$ip" "echo ok" &>/dev/null; then
            ok "SSH is ready"
            return 0
        fi
        sleep 5
        attempts=$((attempts + 1))
    done
    warn "SSH not ready after 150s. Server may still be booting."
    info "Try manually: ssh -i $SSH_KEY_PATH root@$ip"
}

# ─── Phase 4: Deploy MLflow Stack ──────────────────────────────────────────

phase4_deploy() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Phase 4: Deploy MLflow + PostgreSQL + MinIO"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    local ip
    ip=$(hcloud server ip "$SERVER_NAME")

    local ssh_cmd="ssh -o StrictHostKeyChecking=accept-new -i $SSH_KEY_PATH root@$ip"

    # Generate passwords
    local pg_password
    pg_password=$(openssl rand -base64 24)
    local minio_password
    minio_password=$(openssl rand -base64 24)
    local mlflow_password
    mlflow_password=$(openssl rand -base64 24)

    info "Deploying Docker Compose stack to $ip..."

    if ! confirm "Deploy MLflow stack?"; then
        info "Skipping."
        return 1
    fi

    # Create compose file on server
    $ssh_cmd "bash -s" <<REMOTE_DEPLOY
set -euo pipefail

mkdir -p /opt/mlflow/nginx

# Docker Compose file
cat > /opt/mlflow/docker-compose.yml <<'COMPOSE_EOF'
services:
  postgres:
    image: postgres:16
    container_name: mlflow-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: minivess
      POSTGRES_PASSWORD: \${PG_PASSWORD}
      POSTGRES_DB: mlflow
    volumes:
      - pg_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U minivess -d mlflow"]
      interval: 10s
      timeout: 5s
      retries: 5

  minio:
    image: minio/minio
    container_name: mlflow-minio
    restart: unless-stopped
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: \${MINIO_PASSWORD}
    volumes:
      - minio_data:/data
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 10s
      timeout: 5s
      retries: 5

  minio-init:
    image: minio/mc
    depends_on:
      minio:
        condition: service_healthy
    entrypoint: >
      /bin/sh -c "
      mc alias set myminio http://minio:9000 minioadmin \${MINIO_PASSWORD};
      mc mb --ignore-existing myminio/mlflow-artifacts;
      echo 'Bucket ready';
      "
    environment:
      MINIO_PASSWORD: \${MINIO_PASSWORD}

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.20.3
    container_name: mlflow-server
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
      minio-init:
        condition: service_completed_successfully
    environment:
      AWS_ACCESS_KEY_ID: minioadmin
      AWS_SECRET_ACCESS_KEY: \${MINIO_PASSWORD}
      MLFLOW_S3_ENDPOINT_URL: "http://minio:9000"
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri postgresql://minivess:\${PG_PASSWORD}@postgres:5432/mlflow
      --artifacts-destination s3://mlflow-artifacts
      --serve-artifacts
    ports:
      - "5000:5000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 15s
      timeout: 5s
      retries: 5

  nginx:
    image: nginx:alpine
    container_name: mlflow-nginx
    restart: unless-stopped
    depends_on:
      mlflow:
        condition: service_healthy
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/default.conf:/etc/nginx/conf.d/default.conf:ro
      - ./nginx/.htpasswd:/etc/nginx/.htpasswd:ro
      - certbot_certs:/etc/letsencrypt:ro
      - certbot_www:/var/www/certbot:ro

volumes:
  pg_data:
  minio_data:
  certbot_certs:
  certbot_www:
COMPOSE_EOF

# .env for compose
cat > /opt/mlflow/.env <<ENV_EOF
PG_PASSWORD=${pg_password}
MINIO_PASSWORD=${minio_password}
ENV_EOF
chmod 600 /opt/mlflow/.env

# nginx config (HTTP with basic auth)
cat > /opt/mlflow/nginx/default.conf <<'NGINX_EOF'
server {
    listen 80;
    server_name _;

    location / {
        auth_basic "MLflow";
        auth_basic_user_file /etc/nginx/.htpasswd;
        proxy_pass http://mlflow:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /health {
        proxy_pass http://mlflow:5000/health;
    }

    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }
}
NGINX_EOF

# htpasswd
apt-get update -qq && apt-get install -y -qq apache2-utils 2>/dev/null
htpasswd -cb /opt/mlflow/nginx/.htpasswd minivess "${mlflow_password}"

# Start stack
cd /opt/mlflow
docker compose up -d

echo "[OK] Stack deployed"
REMOTE_DEPLOY

    ok "MLflow stack deployed"

    # Wait for health
    info "Waiting for MLflow health check (up to 90s)..."
    local attempts=0
    while [[ $attempts -lt 18 ]]; do
        if curl -sf "http://$ip:5000/health" &>/dev/null; then
            ok "MLflow is healthy!"
            break
        fi
        sleep 5
        attempts=$((attempts + 1))
    done

    if [[ $attempts -ge 18 ]]; then
        warn "Health check timed out. Check: ssh -i $SSH_KEY_PATH root@$ip docker compose -f /opt/mlflow/docker-compose.yml logs"
    fi

    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  MLflow Access Details"
    echo "═══════════════════════════════════════════════════════"
    echo ""
    echo "  URL:       http://$ip (nginx) or http://$ip:5000 (direct)"
    echo "  Username:  minivess"
    echo "  Password:  $mlflow_password"
    echo "  SSH:       ssh -i $SSH_KEY_PATH root@$ip"
    echo ""
    echo "  Add to .env:"
    echo "    MLFLOW_TRACKING_URI_REMOTE=http://$ip"
    echo "    MLFLOW_TRACKING_USERNAME_REMOTE=minivess"
    echo "    MLFLOW_TRACKING_PASSWORD_REMOTE=$mlflow_password"
    echo ""
}

# ─── Phase 5: DNS + TLS (Optional) ─────────────────────────────────────────

phase5_dns_tls() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Phase 5: Custom Domain + HTTPS (Optional)"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    local ip
    ip=$(hcloud server ip "$SERVER_NAME")

    if [[ -z "${CLOUDFLARE_API_TOKEN:-}" || -z "${CLOUDFLARE_ZONE_ID:-}" ]]; then
        info "Cloudflare not configured. MLflow accessible at http://$ip"
        info "To add HTTPS later: set CLOUDFLARE_API_TOKEN + CLOUDFLARE_ZONE_ID in .env"
        return 0
    fi

    # Get domain from Cloudflare
    local domain
    domain=$(curl -sf -X GET \
        "https://api.cloudflare.com/client/v4/zones/$CLOUDFLARE_ZONE_ID" \
        -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
        -H "Content-Type: application/json" | \
        python3 -c "import json,sys; print(json.load(sys.stdin).get('result',{}).get('name',''))")

    if [[ -z "$domain" ]]; then
        warn "Could not fetch domain from Cloudflare."
        return 0
    fi

    local fqdn="mlflow.$domain"
    info "Will create: $fqdn → $ip"

    if ! confirm "Set up DNS + HTTPS for $fqdn?"; then
        return 0
    fi

    # Create/update DNS record
    local existing
    existing=$(curl -sf -X GET \
        "https://api.cloudflare.com/client/v4/zones/$CLOUDFLARE_ZONE_ID/dns_records?type=A&name=$fqdn" \
        -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" | \
        python3 -c "import json,sys; r=json.load(sys.stdin).get('result',[]); print(r[0]['id'] if r else '')")

    if [[ -n "$existing" ]]; then
        curl -sf -X PUT \
            "https://api.cloudflare.com/client/v4/zones/$CLOUDFLARE_ZONE_ID/dns_records/$existing" \
            -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
            -H "Content-Type: application/json" \
            --data "{\"type\":\"A\",\"name\":\"mlflow\",\"content\":\"$ip\",\"ttl\":300,\"proxied\":false}" \
            >/dev/null
        ok "DNS record updated: $fqdn → $ip"
    else
        curl -sf -X POST \
            "https://api.cloudflare.com/client/v4/zones/$CLOUDFLARE_ZONE_ID/dns_records" \
            -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
            -H "Content-Type: application/json" \
            --data "{\"type\":\"A\",\"name\":\"mlflow\",\"content\":\"$ip\",\"ttl\":300,\"proxied\":false}" \
            >/dev/null
        ok "DNS record created: $fqdn → $ip"
    fi

    # Install certbot on server and get TLS cert
    local ssh_cmd="ssh -o StrictHostKeyChecking=accept-new -i $SSH_KEY_PATH root@$ip"

    info "Getting TLS certificate via certbot..."
    $ssh_cmd "bash -s" <<REMOTE_TLS
set -euo pipefail
apt-get update -qq && apt-get install -y -qq certbot

# Stop nginx temporarily for standalone cert
docker compose -f /opt/mlflow/docker-compose.yml stop nginx 2>/dev/null || true

certbot certonly --standalone -d ${fqdn} --non-interactive --agree-tos -m admin@${domain} || {
    echo "[WARN] certbot failed — DNS may not have propagated. Retry in 5 min."
    docker compose -f /opt/mlflow/docker-compose.yml start nginx
    exit 0
}

# Update nginx config for HTTPS
cat > /opt/mlflow/nginx/default.conf <<'NGINX_TLS_EOF'
server {
    listen 80;
    server_name ${fqdn};
    location /.well-known/acme-challenge/ { root /var/www/certbot; }
    location / { return 301 https://\$server_name\$request_uri; }
}
server {
    listen 443 ssl;
    server_name ${fqdn};
    ssl_certificate /etc/letsencrypt/live/${fqdn}/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/${fqdn}/privkey.pem;
    location / {
        auth_basic "MLflow";
        auth_basic_user_file /etc/nginx/.htpasswd;
        proxy_pass http://mlflow:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    location /health { proxy_pass http://mlflow:5000/health; }
}
NGINX_TLS_EOF

# Mount letsencrypt into nginx — update compose volumes
docker compose -f /opt/mlflow/docker-compose.yml up -d

# Auto-renewal cron
echo "0 3 * * * certbot renew --quiet --deploy-hook 'docker compose -f /opt/mlflow/docker-compose.yml restart nginx'" | crontab -

echo "[OK] TLS configured for ${fqdn}"
REMOTE_TLS

    ok "HTTPS configured: https://$fqdn"
    echo ""
    echo "  Update .env: MLFLOW_TRACKING_URI_REMOTE=https://$fqdn"
}

# ─── Phase 6: Verify ───────────────────────────────────────────────────────

phase6_verify() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Phase 6: Verification"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    local ip
    ip=$(hcloud server ip "$SERVER_NAME")
    local errors=0

    info "Check 1/3: SSH access..."
    if ssh -o ConnectTimeout=5 -i "$SSH_KEY_PATH" "root@$ip" "echo ok" &>/dev/null; then
        ok "SSH: accessible"
    else
        err "SSH: failed"
        errors=$((errors + 1))
    fi

    info "Check 2/3: MLflow health..."
    if curl -sf "http://$ip:5000/health" &>/dev/null; then
        ok "MLflow: healthy"
    elif curl -sf "http://$ip/health" &>/dev/null; then
        ok "MLflow: healthy (via nginx)"
    else
        err "MLflow: unreachable"
        errors=$((errors + 1))
    fi

    info "Check 3/3: Docker containers..."
    local containers
    containers=$(ssh -o ConnectTimeout=5 -i "$SSH_KEY_PATH" "root@$ip" \
        "docker ps --format '{{.Names}}: {{.Status}}'" 2>/dev/null)
    if [[ -n "$containers" ]]; then
        echo "$containers" | while read -r line; do ok "  $line"; done
    else
        err "Could not list containers"
        errors=$((errors + 1))
    fi

    echo ""
    if [[ $errors -eq 0 ]]; then
        echo -e "  ${GREEN}ALL CHECKS PASSED${NC}"
    else
        echo -e "  ${RED}$errors CHECK(S) FAILED${NC}"
    fi
}

# ─── Teardown ───────────────────────────────────────────────────────────────

teardown() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo -e "  ${RED}TEARDOWN: Destroy Hetzner Resources${NC}"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    load_env

    info "Resources to destroy:"
    info "  Server:   $SERVER_NAME"
    info "  Firewall: $FW_NAME"
    info "  SSH key:  $SSH_KEY_NAME"

    if ! confirm "DESTROY all Hetzner resources? This cannot be undone!"; then
        info "Cancelled."
        exit 0
    fi

    hcloud server delete "$SERVER_NAME" 2>/dev/null && ok "Server deleted" || warn "Server not found"
    hcloud firewall delete "$FW_NAME" 2>/dev/null && ok "Firewall deleted" || warn "Firewall not found"
    hcloud ssh-key delete "$SSH_KEY_NAME" 2>/dev/null && ok "SSH key deleted" || warn "SSH key not found"

    echo ""
    ok "Teardown complete."
}

# ─── Main ───────────────────────────────────────────────────────────────────

main() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  MinIVess MLOps — Hetzner Cloud Setup"
    echo "  MLflow + PostgreSQL + MinIO on Hetzner VPS"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "  Plan: docs/planning/hetzner-mlflow-plan.md"
    echo "  Tutorial: docs/planning/cloud-tutorial.md"
    echo ""

    if [[ "${1:-}" == "--teardown" ]]; then
        teardown
        exit 0
    fi

    phase0_validate
    phase1_ssh_key
    phase2_firewall
    phase3_server
    phase4_deploy
    phase5_dns_tls
    phase6_verify
}

main "$@"
