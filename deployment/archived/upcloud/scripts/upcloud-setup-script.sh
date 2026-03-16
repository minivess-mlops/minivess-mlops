#!/usr/bin/env bash
# upcloud-setup-script.sh — Programmatic UpCloud setup for MinIVess MLflow
#
# Creates: VPS + Docker + Docker Compose MLflow stack + firewall rules
# Requires: upctl CLI, UPCLOUD_TOKEN in .env
# Usage: bash scripts/upcloud-setup-script.sh [--teardown]
#
# Each phase asks for confirmation before executing.
# Total time: ~5 minutes from zero to running MLflow.

set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SSH_KEY_PATH="${HOME}/.ssh/upcloud_minivess"
SERVER_NAME="minivess-mlflow"

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
            UPCLOUD_TOKEN)        export UPCLOUD_TOKEN="$value" ;;
            UPCLOUD_ZONE)         export UPCLOUD_ZONE="$value" ;;
            UPCLOUD_PLAN)         export UPCLOUD_PLAN="$value" ;;
            CLOUDFLARE_API_TOKEN) export CLOUDFLARE_API_TOKEN="$value" ;;
            CLOUDFLARE_ZONE_ID)   export CLOUDFLARE_ZONE_ID="$value" ;;
        esac
    done < <(grep -v '^\s*#' "$env_file" | grep -v '^\s*$')

    UPCLOUD_ZONE="${UPCLOUD_ZONE:-fi-hel1}"
    UPCLOUD_PLAN="${UPCLOUD_PLAN:-DEV-2xCPU-4GB}"
}

# ─── Phase 0: Validate Prerequisites ───────────────────────────────────────

phase0_validate() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Phase 0: Validate Prerequisites"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    local errors=0

    # Check upctl CLI
    if command -v upctl &>/dev/null; then
        ok "upctl CLI found: $(upctl version 2>&1 | head -1)"
    else
        err "upctl CLI not found."
        err "Install:"
        err "  curl -Lo upcloud-cli.deb https://github.com/UpCloudLtd/upcloud-cli/releases/latest/download/upcloud-cli_3.30.0_amd64.deb"
        err "  sudo apt install ./upcloud-cli.deb"
        err "  rm upcloud-cli.deb"
        err ""
        err "Or see: https://github.com/UpCloudLtd/upcloud-cli"
        errors=$((errors + 1))
    fi

    # Load .env
    load_env

    # Check token
    if [[ -z "${UPCLOUD_TOKEN:-}" ]]; then
        err "UPCLOUD_TOKEN is not set in .env"
        err "Get one: https://hub.upcloud.com/account/api-tokens → Create"
        errors=$((errors + 1))
    else
        ok "UPCLOUD_TOKEN set (${#UPCLOUD_TOKEN} chars)"
    fi

    ok "Zone: $UPCLOUD_ZONE"
    ok "Plan: $UPCLOUD_PLAN"

    # Test API access
    if [[ -n "${UPCLOUD_TOKEN:-}" ]] && command -v upctl &>/dev/null; then
        info "Testing UpCloud API access..."
        if upctl account show &>/dev/null 2>&1; then
            ok "API access verified"
        else
            err "API token invalid or expired. Check UPCLOUD_TOKEN in .env"
            errors=$((errors + 1))
        fi
    fi

    # Check / generate SSH key
    if [[ -f "$SSH_KEY_PATH" ]]; then
        ok "SSH key found: $SSH_KEY_PATH"
    else
        warn "SSH key not found at $SSH_KEY_PATH"
        if confirm "Generate SSH key pair?"; then
            ssh-keygen -t ed25519 -f "$SSH_KEY_PATH" -N "" -C "minivess-upcloud"
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

# ─── Phase 1: Create Server ────────────────────────────────────────────────

phase1_server() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Phase 1: Create Server ($UPCLOUD_PLAN @ $UPCLOUD_ZONE)"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    # Check if already exists
    if upctl server show "$SERVER_NAME" &>/dev/null 2>&1; then
        local ip
        ip=$(upctl server show "$SERVER_NAME" -o json 2>/dev/null | \
            python3 -c "import json,sys; d=json.load(sys.stdin); ips=d.get('ip_addresses',d.get('networking',{}).get('interfaces',[])); print(next((i.get('address','') for i in (ips if isinstance(ips,list) else []) if i.get('access','')=='public' and i.get('family','')=='IPv4'),'unknown'))" 2>/dev/null || echo "unknown")
        ok "Server '$SERVER_NAME' already exists at $ip"
        return 0
    fi

    info "This creates:"
    info "  - $UPCLOUD_PLAN server in $UPCLOUD_ZONE"
    info "  - Ubuntu Server 24.04 LTS"
    info "  - SSH key: $SSH_KEY_PATH"
    echo ""

    # Show trial credit reminder
    info "Note: This uses your UpCloud trial credit (EUR 250 / 30 days)."
    info "Estimated cost: ~EUR 19/month for DEV-2xCPU-4GB."

    if ! confirm "Create server?"; then
        info "Skipping."
        return 1
    fi

    info "Creating server (takes ~45 seconds)..."
    upctl server create \
        --hostname "$SERVER_NAME" \
        --title "$SERVER_NAME" \
        --zone "$UPCLOUD_ZONE" \
        --plan "$UPCLOUD_PLAN" \
        --os "Ubuntu Server 24.04 LTS (Noble Numbat)" \
        --ssh-keys "$(cat "${SSH_KEY_PATH}.pub")" \
        --wait

    # Get the server IP
    local ip
    ip=$(upctl server show "$SERVER_NAME" -o json 2>/dev/null | \
        python3 -c "import json,sys; d=json.load(sys.stdin); ips=d.get('ip_addresses',d.get('networking',{}).get('interfaces',[])); print(next((i.get('address','') for i in (ips if isinstance(ips,list) else []) if i.get('access','')=='public' and i.get('family','')=='IPv4'),'unknown'))" 2>/dev/null || echo "unknown")
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

# ─── Helper: get server IP ──────────────────────────────────────────────────

get_server_ip() {
    upctl server show "$SERVER_NAME" -o json 2>/dev/null | \
        python3 -c "
import json, sys
d = json.load(sys.stdin)
ips = d.get('ip_addresses', d.get('networking', {}).get('interfaces', []))
if isinstance(ips, list):
    for i in ips:
        if i.get('access','') == 'public' and i.get('family','') == 'IPv4':
            print(i.get('address',''))
            break
" 2>/dev/null || echo ""
}

# ─── Phase 2: Install Docker ───────────────────────────────────────────────

phase2_docker() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Phase 2: Install Docker on Server"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    local ip
    ip=$(get_server_ip)
    if [[ -z "$ip" ]]; then
        err "Cannot determine server IP. Run phase 1 first."
        return 1
    fi

    local ssh_cmd="ssh -o StrictHostKeyChecking=accept-new -i $SSH_KEY_PATH root@$ip"

    # Check if Docker already installed
    if $ssh_cmd "command -v docker" &>/dev/null 2>&1; then
        ok "Docker already installed on $ip"
        return 0
    fi

    info "Installing Docker on $ip..."

    $ssh_cmd "bash -s" <<'DOCKER_INSTALL'
set -euo pipefail

# Install Docker via apt (Ubuntu 24.04)
apt-get update -qq
apt-get install -y -qq docker.io docker-compose-v2

# Enable Docker service
systemctl enable --now docker

# Verify
docker --version
docker compose version

echo "[OK] Docker installed"
DOCKER_INSTALL

    ok "Docker installed"
}

# ─── Phase 3: Deploy MLflow Stack ──────────────────────────────────────────

phase3_deploy() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Phase 3: Deploy MLflow + PostgreSQL + MinIO"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    local ip
    ip=$(get_server_ip)
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

# ─── Phase 4: Firewall Rules ───────────────────────────────────────────────

phase4_firewall() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Phase 4: Configure Firewall Rules"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    local ip
    ip=$(get_server_ip)
    local ssh_cmd="ssh -o StrictHostKeyChecking=accept-new -i $SSH_KEY_PATH root@$ip"

    info "Setting up UFW firewall on $ip..."

    $ssh_cmd "bash -s" <<'FIREWALL_SETUP'
set -euo pipefail

# Use UFW (Uncomplicated Firewall) — standard on Ubuntu
apt-get install -y -qq ufw 2>/dev/null

# Default: deny incoming, allow outgoing
ufw default deny incoming
ufw default allow outgoing

# Allow SSH (critical — must be first!)
ufw allow 22/tcp comment "SSH"

# Allow HTTP (for certbot challenge + nginx)
ufw allow 80/tcp comment "HTTP"

# Allow HTTPS
ufw allow 443/tcp comment "HTTPS"

# Allow MLflow direct (optional, for debugging)
ufw allow 5000/tcp comment "MLflow direct"

# Enable (non-interactive)
echo "y" | ufw enable

ufw status verbose

echo "[OK] Firewall configured"
FIREWALL_SETUP

    ok "Firewall rules applied (SSH:22, HTTP:80, HTTPS:443, MLflow:5000)"
}

# ─── Phase 5: DNS + TLS (Optional) ─────────────────────────────────────────

phase5_dns_tls() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Phase 5: Custom Domain + HTTPS (Optional)"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    local ip
    ip=$(get_server_ip)

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

# Restart stack with TLS
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
    ip=$(get_server_ip)
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
    echo -e "  ${RED}TEARDOWN: Destroy UpCloud Resources${NC}"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    load_env

    info "Resources to destroy:"
    info "  Server: $SERVER_NAME"

    if ! confirm "DESTROY UpCloud server? This cannot be undone!"; then
        info "Cancelled."
        exit 0
    fi

    # Stop server first (required before delete with storage)
    upctl server stop "$SERVER_NAME" --wait 2>/dev/null && ok "Server stopped" || warn "Server not running"

    # Delete server and its storage
    upctl server delete "$SERVER_NAME" --delete-storages 2>/dev/null && ok "Server + storage deleted" || warn "Server not found"

    echo ""
    ok "Teardown complete."
}

# ─── Main ───────────────────────────────────────────────────────────────────

main() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  MinIVess MLOps — UpCloud Setup"
    echo "  MLflow + PostgreSQL + MinIO on UpCloud VPS"
    echo "  Trial: EUR 250 / 30 days"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "  Plan: docs/planning/upcloud-mlflow-plan.md"
    echo "  Tutorial: docs/planning/cloud-tutorial.md"
    echo ""

    if [[ "${1:-}" == "--teardown" ]]; then
        teardown
        exit 0
    fi

    phase0_validate
    phase1_server
    phase2_docker
    phase3_deploy
    phase4_firewall
    phase5_dns_tls
    phase6_verify
}

main "$@"
