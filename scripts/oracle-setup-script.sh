#!/usr/bin/env bash
# oracle-setup-script.sh — Programmatic Oracle Cloud setup for MinIVess MLflow
#
# Creates: VCN + subnet + security list + ARM VM + block volume + Docker + MLflow stack
# Requires: oci-cli authenticated, .env with OCI_* variables
# Usage: bash scripts/oracle-setup-script.sh [--teardown] [--retry-vm]
#
# Each phase asks for confirmation before executing.
# Resource OCIDs are saved to ~/.oci/minivess-resources.json for teardown.

set -euo pipefail

# Suppress OCI CLI label warning globally
export SUPPRESS_LABEL_WARNING=True

# ─── Configuration ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESOURCE_FILE="${HOME}/.oci/minivess-resources.json"
SSH_KEY_PATH="${HOME}/.ssh/oci_vm_key"
VM_DISPLAY_NAME="minivess-mlflow"
VCN_CIDR="10.0.0.0/16"
SUBNET_CIDR="10.0.0.0/24"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ─── Helpers ────────────────────────────────────────────────────────────────

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

save_resource() {
    local key="$1" value="$2"
    if [[ ! -f "$RESOURCE_FILE" ]]; then
        echo '{}' > "$RESOURCE_FILE"
    fi
    # Use python3 for JSON manipulation (no jq dependency)
    python3 -c "
import json, sys
with open('$RESOURCE_FILE') as f:
    data = json.load(f)
data['$key'] = '$value'
with open('$RESOURCE_FILE', 'w') as f:
    json.dump(data, f, indent=2)
"
    info "Saved $key to $RESOURCE_FILE"
}

get_resource() {
    local key="$1"
    if [[ ! -f "$RESOURCE_FILE" ]]; then
        echo ""
        return
    fi
    python3 -c "
import json
with open('$RESOURCE_FILE') as f:
    data = json.load(f)
print(data.get('$key', ''))
"
}

# Extract a field from OCI CLI JSON output.
# OCI CLI --wait-for-state prints progress lines before JSON.
# We strip non-JSON lines before parsing.
extract_json() {
    # Read all stdin, find the first line starting with '{', take from there
    python3 -c "
import json, sys
raw = sys.stdin.read()
# Find first '{' — everything before is OCI progress output
idx = raw.find('{')
if idx < 0:
    print('{}', file=sys.stderr)
    sys.exit(1)
data = json.loads(raw[idx:])
json.dump(data, sys.stdout)
"
}

extract_id() {
    extract_json | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('data', {}).get('id', ''))
"
}

extract_field() {
    local field="$1"
    extract_json | python3 -c "
import json, sys
data = json.load(sys.stdin)
val = data.get('data', {})
for part in '$field'.split('.'):
    if isinstance(val, dict):
        val = val.get(part, '')
    else:
        val = ''
        break
print(val)
"
}

# ─── Load .env ──────────────────────────────────────────────────────────────

load_env() {
    local env_file="$REPO_ROOT/.env"
    if [[ ! -f "$env_file" ]]; then
        err ".env file not found at $env_file"
        err "Copy .env.example to .env and fill in your OCI values first."
        exit 1
    fi
    # Source .env (skip comments and empty lines)
    set -a
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ -z "$key" || "$key" =~ ^# ]] && continue
        # Remove surrounding quotes from value
        value="${value%\"}"
        value="${value#\"}"
        export "$key=$value"
    done < <(grep -v '^\s*#' "$env_file" | grep -v '^\s*$')
    set +a
}

# ─── Phase 0: Validate Prerequisites ───────────────────────────────────────

phase0_validate() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Phase 0: Validate Prerequisites"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    local errors=0

    # Check oci CLI
    if command -v oci &>/dev/null; then
        ok "oci CLI found: $(oci --version 2>&1 | head -1)"
    else
        err "oci CLI not found. Install: pip install oci-cli"
        errors=$((errors + 1))
    fi

    # Check python3
    if command -v python3 &>/dev/null; then
        ok "python3 found: $(python3 --version)"
    else
        err "python3 not found."
        errors=$((errors + 1))
    fi

    # Load and check .env variables
    load_env

    for var in OCI_TENANCY_OCID OCI_USER_OCID OCI_COMPARTMENT_OCID OCI_REGION OCI_FINGERPRINT; do
        if [[ -z "${!var:-}" ]]; then
            err "$var is not set in .env"
            errors=$((errors + 1))
        else
            ok "$var = ${!var:0:30}..."
        fi
    done

    # Verify OCI authentication works
    info "Testing OCI API authentication..."
    if oci iam region list --output table &>/dev/null; then
        ok "OCI API authentication successful"
    else
        err "OCI API authentication failed. Check your API key and fingerprint."
        err "Run: oci iam user get --user-id \$OCI_USER_OCID"
        errors=$((errors + 1))
    fi

    # Check SSH key
    if [[ -f "$SSH_KEY_PATH" ]]; then
        ok "SSH key found: $SSH_KEY_PATH"
    else
        warn "SSH key not found at $SSH_KEY_PATH"
        if confirm "Generate SSH key pair for VM access?"; then
            ssh-keygen -t ed25519 -f "$SSH_KEY_PATH" -N "" -C "minivess-oci-vm"
            ok "SSH key generated: $SSH_KEY_PATH"
        else
            err "SSH key required for VM access. Generate one or set SSH_KEY_PATH."
            errors=$((errors + 1))
        fi
    fi

    if [[ $errors -gt 0 ]]; then
        err "$errors prerequisite(s) failed. Fix them and re-run."
        exit 1
    fi

    ok "All prerequisites passed!"
}

# ─── Phase 1: Network (VCN + Subnet + Security List) ───────────────────────

phase1_network() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Phase 1: Create Network (VCN + Subnet + Firewall)"
    echo "═══════════════════════════════════════════════════════"
    echo ""
    info "This creates:"
    info "  - Virtual Cloud Network (VCN): 10.0.0.0/16"
    info "  - Internet Gateway (for public access)"
    info "  - Security List (SSH:22, HTTP:80, HTTPS:443, MLflow:5000)"
    info "  - Public Subnet: 10.0.0.0/24"
    info ""
    info "Cost: \$0 (included in Always Free)"

    # Check if already created
    local existing_vcn
    existing_vcn=$(get_resource "vcn_ocid")
    if [[ -n "$existing_vcn" ]]; then
        warn "VCN already exists: $existing_vcn"
        if ! confirm "Skip network creation and use existing?"; then
            warn "Delete existing resources first with --teardown, then re-run."
            exit 1
        fi
        return 0
    fi

    if ! confirm "Create network resources?"; then
        info "Skipping Phase 1."
        return 1
    fi

    # 1. Create VCN
    info "Creating VCN..."
    local vcn_json
    vcn_json=$(SUPPRESS_LABEL_WARNING=True oci network vcn create \
        --compartment-id "$OCI_COMPARTMENT_OCID" \
        --cidr-block "$VCN_CIDR" \
        --display-name "minivess-vcn" \
        --dns-label "minivess" \
        --output json 2>/dev/null)
    VCN_OCID=$(echo "$vcn_json" | extract_id)
    save_resource "vcn_ocid" "$VCN_OCID"
    ok "VCN created: $VCN_OCID"

    # Get default route table OCID
    local rt_ocid
    rt_ocid=$(echo "$vcn_json" | extract_field "default-route-table-id")
    save_resource "route_table_ocid" "$rt_ocid"

    # Get default security list OCID
    local default_sl_ocid
    default_sl_ocid=$(echo "$vcn_json" | extract_field "default-security-list-id")

    # 2. Create Internet Gateway
    info "Creating Internet Gateway..."
    local igw_json
    igw_json=$(SUPPRESS_LABEL_WARNING=True oci network internet-gateway create \
        --compartment-id "$OCI_COMPARTMENT_OCID" \
        --vcn-id "$VCN_OCID" \
        --display-name "minivess-igw" \
        --is-enabled true \
        --output json 2>/dev/null)
    IGW_OCID=$(echo "$igw_json" | extract_id)
    save_resource "igw_ocid" "$IGW_OCID"
    ok "Internet Gateway created: $IGW_OCID"

    # 3. Update Route Table (add 0.0.0.0/0 → IGW)
    info "Updating route table with internet route..."
    SUPPRESS_LABEL_WARNING=True oci network route-table update \
        --rt-id "$rt_ocid" \
        --route-rules "[{\"destination\":\"0.0.0.0/0\",\"destinationType\":\"CIDR_BLOCK\",\"networkEntityId\":\"$IGW_OCID\"}]" \
        --force \
        --output json >/dev/null 2>&1
    ok "Route table updated: 0.0.0.0/0 → IGW"

    # 4. Update default Security List (add ingress rules)
    info "Updating security list with firewall rules..."
    local ingress_rules
    ingress_rules='[
        {"source":"0.0.0.0/0","protocol":"6","tcpOptions":{"destinationPortRange":{"min":22,"max":22}},"description":"SSH","isStateless":false},
        {"source":"0.0.0.0/0","protocol":"6","tcpOptions":{"destinationPortRange":{"min":80,"max":80}},"description":"HTTP (certbot)","isStateless":false},
        {"source":"0.0.0.0/0","protocol":"6","tcpOptions":{"destinationPortRange":{"min":443,"max":443}},"description":"HTTPS (MLflow)","isStateless":false},
        {"source":"0.0.0.0/0","protocol":"6","tcpOptions":{"destinationPortRange":{"min":5000,"max":5000}},"description":"MLflow direct","isStateless":false},
        {"source":"0.0.0.0/0","protocol":"1","icmpOptions":{"type":3,"code":4},"description":"Path MTU Discovery","isStateless":false},
        {"source":"10.0.0.0/16","protocol":"1","icmpOptions":{"type":3},"description":"VCN ICMP","isStateless":false}
    ]'
    local egress_rules='[{"destination":"0.0.0.0/0","protocol":"all","description":"Allow all egress","isStateless":false}]'

    SUPPRESS_LABEL_WARNING=True oci network security-list update \
        --security-list-id "$default_sl_ocid" \
        --ingress-security-rules "$ingress_rules" \
        --egress-security-rules "$egress_rules" \
        --force \
        --output json >/dev/null 2>&1
    save_resource "security_list_ocid" "$default_sl_ocid"
    ok "Security list updated (SSH, HTTP, HTTPS, MLflow:5000)"

    # 5. Create Public Subnet
    info "Creating public subnet..."
    local subnet_json
    subnet_json=$(SUPPRESS_LABEL_WARNING=True oci network subnet create \
        --compartment-id "$OCI_COMPARTMENT_OCID" \
        --vcn-id "$VCN_OCID" \
        --cidr-block "$SUBNET_CIDR" \
        --display-name "minivess-public-subnet" \
        --dns-label "pub" \
        --route-table-id "$rt_ocid" \
        --security-list-ids "[\"$default_sl_ocid\"]" \
        --output json 2>/dev/null)
    SUBNET_OCID=$(echo "$subnet_json" | extract_id)
    save_resource "subnet_ocid" "$SUBNET_OCID"
    ok "Subnet created: $SUBNET_OCID"

    echo ""
    ok "Phase 1 complete! Network resources created."
}

# ─── Phase 2: Compute Instance ─────────────────────────────────────────────

phase2_compute() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Phase 2: Launch ARM VM (A1.Flex, 4 OCPU, 24 GB)"
    echo "═══════════════════════════════════════════════════════"
    echo ""
    info "This creates:"
    info "  - ARM Ampere Altra VM: 4 OCPU, 24 GB RAM"
    info "  - Ubuntu 24.04 LTS (aarch64)"
    info "  - 50 GB boot volume"
    info "  - Public IP (auto-assigned)"
    info ""
    info "Cost: \$0 (Always Free — 3000 OCPU-hours/month)"
    info ""
    warn "Frankfurt ARM capacity may be limited!"
    warn "If 'Out of host capacity', the script tries all 3 ADs."

    # Check if already created
    local existing_instance
    existing_instance=$(get_resource "instance_ocid")
    if [[ -n "$existing_instance" ]]; then
        warn "Instance already exists: $existing_instance"
        # Get public IP
        local public_ip
        public_ip=$(get_resource "public_ip")
        if [[ -n "$public_ip" ]]; then
            ok "Public IP: $public_ip"
            ok "SSH: ssh -i $SSH_KEY_PATH ubuntu@$public_ip"
        fi
        return 0
    fi

    local subnet_ocid
    subnet_ocid=$(get_resource "subnet_ocid")
    if [[ -z "$subnet_ocid" ]]; then
        err "No subnet found. Run Phase 1 first."
        return 1
    fi

    if ! confirm "Launch ARM VM? (This may take 2-5 minutes)"; then
        info "Skipping Phase 2."
        return 1
    fi

    # Find Ubuntu 24.04 ARM image
    info "Finding Ubuntu 24.04 ARM image..."
    local image_ocid
    image_ocid=$(oci compute image list \
        --compartment-id "$OCI_COMPARTMENT_OCID" \
        --operating-system "Canonical Ubuntu" \
        --operating-system-version "24.04" \
        --shape "VM.Standard.A1.Flex" \
        --sort-by TIMECREATED \
        --sort-order DESC \
        --limit 1 \
        --output json 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
items = data.get('data', [])
if items:
    print(items[0]['id'])
else:
    print('')
")

    if [[ -z "$image_ocid" ]]; then
        err "Could not find Ubuntu 24.04 ARM image."
        err "Try: oci compute image list --compartment-id $OCI_COMPARTMENT_OCID --operating-system 'Canonical Ubuntu' --shape VM.Standard.A1.Flex --output table"
        return 1
    fi
    save_resource "image_ocid" "$image_ocid"
    ok "Image found: $image_ocid"

    # Get availability domains
    info "Listing availability domains..."
    local ad_list
    ad_list=$(oci iam availability-domain list \
        --compartment-id "$OCI_TENANCY_OCID" \
        --output json 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
for ad in data.get('data', []):
    print(ad['name'])
")

    # Try each AD until one works
    local launched=false
    local instance_json=""
    while IFS= read -r ad_name; do
        [[ -z "$ad_name" ]] && continue
        info "Trying $ad_name ..."
        instance_json=$(SUPPRESS_LABEL_WARNING=True oci compute instance launch \
            --compartment-id "$OCI_COMPARTMENT_OCID" \
            --availability-domain "$ad_name" \
            --shape "VM.Standard.A1.Flex" \
            --shape-config '{"ocpus":4,"memoryInGBs":24}' \
            --display-name "$VM_DISPLAY_NAME" \
            --image-id "$image_ocid" \
            --subnet-id "$subnet_ocid" \
            --ssh-authorized-keys-file "${SSH_KEY_PATH}.pub" \
            --assign-public-ip true \
            --output json 2>/dev/null) && launched=true && break

        warn "Failed on $ad_name (likely capacity). Trying next AD..."
    done <<< "$ad_list"

    if [[ "$launched" != "true" ]]; then
        err "Could not launch VM in any availability domain."
        err ""
        err "This is likely the Frankfurt ARM capacity shortage."
        err "Options:"
        err "  1. Try again later (early morning CET has best availability)"
        err "  2. Start smaller: edit script to use 1 OCPU / 6 GB"
        err "  3. Use the capacity retry tool:"
        err "     https://github.com/hitrov/oci-arm-host-capacity"
        return 1
    fi

    local instance_ocid
    instance_ocid=$(echo "$instance_json" | extract_id)
    save_resource "instance_ocid" "$instance_ocid"
    save_resource "availability_domain" "$ad_name"
    ok "VM provisioning started: $instance_ocid"

    # Wait for instance to reach RUNNING state
    info "Waiting for VM to reach RUNNING state (this takes 2-5 minutes)..."
    local vm_state=""
    for i in $(seq 1 60); do
        vm_state=$(SUPPRESS_LABEL_WARNING=True oci compute instance get \
            --instance-id "$instance_ocid" --output json 2>/dev/null | \
            python3 -c "import json,sys; print(json.load(sys.stdin).get('data',{}).get('lifecycle-state',''))" 2>/dev/null)
        if [[ "$vm_state" == "RUNNING" ]]; then
            break
        fi
        printf "."
        sleep 10
    done
    echo ""

    if [[ "$vm_state" != "RUNNING" ]]; then
        warn "VM state is '$vm_state' after waiting. It may still be provisioning."
        warn "Check: oci compute instance get --instance-id $instance_ocid"
    else
        ok "VM is RUNNING"
    fi

    # Get public IP
    info "Retrieving public IP..."
    sleep 5
    local vnic_attachments
    vnic_attachments=$(oci compute vnic-attachment list \
        --compartment-id "$OCI_COMPARTMENT_OCID" \
        --instance-id "$instance_ocid" \
        --output json 2>/dev/null)

    local vnic_id
    vnic_id=$(echo "$vnic_attachments" | python3 -c "
import json, sys
data = json.load(sys.stdin)
items = data.get('data', [])
if items:
    print(items[0].get('vnic-id', ''))
")

    local public_ip=""
    if [[ -n "$vnic_id" ]]; then
        public_ip=$(oci network vnic get --vnic-id "$vnic_id" --output json 2>/dev/null | \
            python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('data', {}).get('public-ip', ''))
")
    fi

    if [[ -n "$public_ip" ]]; then
        save_resource "public_ip" "$public_ip"
        ok "Public IP: $public_ip"
        echo ""
        ok "SSH command: ssh -i $SSH_KEY_PATH ubuntu@$public_ip"
    else
        warn "Could not retrieve public IP yet. Check OCI Console."
    fi

    echo ""
    ok "Phase 2 complete! VM is running."
}

# ─── Phase 3: Block Volume ─────────────────────────────────────────────────

phase3_storage() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Phase 3: Attach 150 GB Block Volume"
    echo "═══════════════════════════════════════════════════════"
    echo ""
    info "This creates:"
    info "  - 150 GB block volume for Docker data"
    info "  - Attached to the VM as /dev/sdb"
    info "  - Formatted as ext4, mounted at /data"
    info ""
    info "Cost: \$0 (200 GB total free — 50 GB boot + 150 GB data)"

    local instance_ocid
    instance_ocid=$(get_resource "instance_ocid")
    if [[ -z "$instance_ocid" ]]; then
        err "No VM found. Run Phase 2 first."
        return 1
    fi

    local existing_vol
    existing_vol=$(get_resource "volume_ocid")
    if [[ -n "$existing_vol" ]]; then
        warn "Block volume already exists: $existing_vol"
        return 0
    fi

    if ! confirm "Create and attach 150 GB block volume?"; then
        info "Skipping Phase 3."
        return 1
    fi

    local ad_name
    ad_name=$(get_resource "availability_domain")

    # Create volume
    info "Creating 150 GB block volume (takes ~30s)..."
    local vol_json
    vol_json=$(SUPPRESS_LABEL_WARNING=True oci bv volume create \
        --compartment-id "$OCI_COMPARTMENT_OCID" \
        --availability-domain "$ad_name" \
        --display-name "minivess-data" \
        --size-in-gbs 150 \
        --output json 2>/dev/null)
    local volume_ocid
    volume_ocid=$(echo "$vol_json" | extract_id)
    save_resource "volume_ocid" "$volume_ocid"

    # Wait for volume to be AVAILABLE
    info "Waiting for volume..."
    for i in $(seq 1 30); do
        local vol_state
        vol_state=$(SUPPRESS_LABEL_WARNING=True oci bv volume get \
            --volume-id "$volume_ocid" --output json 2>/dev/null | \
            python3 -c "import json,sys; print(json.load(sys.stdin).get('data',{}).get('lifecycle-state',''))" 2>/dev/null)
        [[ "$vol_state" == "AVAILABLE" ]] && break
        printf "."
        sleep 5
    done
    echo ""
    ok "Volume created: $volume_ocid"

    # Attach volume
    info "Attaching volume to VM..."
    local instance_ocid
    instance_ocid=$(get_resource "instance_ocid")
    local attach_json
    attach_json=$(SUPPRESS_LABEL_WARNING=True oci compute volume-attachment attach \
        --instance-id "$instance_ocid" \
        --volume-id "$volume_ocid" \
        --type paravirtualized \
        --output json 2>/dev/null)
    local attach_ocid
    attach_ocid=$(echo "$attach_json" | extract_id)
    save_resource "volume_attachment_ocid" "$attach_ocid"

    # Wait for attachment
    info "Waiting for attachment..."
    for i in $(seq 1 30); do
        local att_state
        att_state=$(SUPPRESS_LABEL_WARNING=True oci compute volume-attachment get \
            --volume-attachment-id "$attach_ocid" --output json 2>/dev/null | \
            python3 -c "import json,sys; print(json.load(sys.stdin).get('data',{}).get('lifecycle-state',''))" 2>/dev/null)
        [[ "$att_state" == "ATTACHED" ]] && break
        printf "."
        sleep 5
    done
    echo ""
    ok "Volume attached: $attach_ocid"

    echo ""
    ok "Phase 3 complete! Volume attached."
    echo ""
    info "Next: SSH into the VM to format and mount the volume."
    info "The setup script in Phase 4 will do this automatically."
}

# ─── Phase 4: VM Setup (Docker + MLflow Stack) ─────────────────────────────

phase4_provision() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Phase 4: Install Docker + Deploy MLflow Stack"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    local public_ip
    public_ip=$(get_resource "public_ip")
    if [[ -z "$public_ip" ]]; then
        err "No public IP found. Run Phase 2 first."
        return 1
    fi

    info "This will SSH into $public_ip and:"
    info "  1. Format and mount the 150 GB data volume at /data"
    info "  2. Install Docker and docker-compose-plugin"
    info "  3. Deploy MLflow + PostgreSQL + MinIO + nginx"
    info "  4. Generate basic-auth credentials"
    info ""
    info "This takes ~3-5 minutes on first run."

    if ! confirm "SSH into VM and set up Docker + MLflow?"; then
        info "Skipping Phase 4."
        echo ""
        info "You can do it manually:"
        info "  ssh -i $SSH_KEY_PATH ubuntu@$public_ip"
        return 1
    fi

    # Generate a password for MLflow basic auth
    local mlflow_password
    mlflow_password=$(openssl rand -base64 24)
    save_resource "mlflow_password" "$mlflow_password"

    # Generate a password for PostgreSQL
    local pg_password
    pg_password=$(openssl rand -base64 24)
    save_resource "pg_password" "$pg_password"

    local ssh_cmd="ssh -o StrictHostKeyChecking=accept-new -i $SSH_KEY_PATH ubuntu@$public_ip"

    # Test SSH connectivity
    info "Testing SSH connection..."
    if ! $ssh_cmd "echo 'SSH OK'" 2>/dev/null; then
        err "Cannot SSH into $public_ip"
        err "The VM may still be booting. Wait 1-2 minutes and try:"
        err "  ssh -i $SSH_KEY_PATH ubuntu@$public_ip"
        return 1
    fi
    ok "SSH connection successful"

    # Step 1: Format and mount block volume
    info "Step 1/4: Formatting and mounting data volume..."
    $ssh_cmd "bash -s" <<'REMOTE_STORAGE'
set -euo pipefail

# Find the unformatted block device (not the boot disk)
DATA_DEV=""
for dev in /dev/sdb /dev/vdb /dev/xvdb; do
    if [[ -b "$dev" ]] && ! blkid "$dev" &>/dev/null; then
        DATA_DEV="$dev"
        break
    fi
done

# If no unformatted device, check if /data is already mounted
if [[ -z "$DATA_DEV" ]]; then
    if mountpoint -q /data 2>/dev/null; then
        echo "[OK] /data already mounted"
        exit 0
    fi
    # Check for any block device that might already be formatted
    for dev in /dev/sdb /dev/vdb /dev/xvdb; do
        if [[ -b "$dev" ]]; then
            DATA_DEV="$dev"
            break
        fi
    done
fi

if [[ -z "$DATA_DEV" ]]; then
    echo "[WARN] No block volume found. Skipping — using boot volume."
    sudo mkdir -p /data
    exit 0
fi

# Format if needed
if ! blkid "$DATA_DEV" &>/dev/null; then
    echo "Formatting $DATA_DEV as ext4..."
    sudo mkfs.ext4 -L minivess-data "$DATA_DEV"
fi

# Mount
sudo mkdir -p /data
if ! mountpoint -q /data; then
    sudo mount "$DATA_DEV" /data
fi

# Add to fstab if not already there
if ! grep -q '/data' /etc/fstab; then
    echo "$DATA_DEV /data ext4 defaults,_netdev 0 2" | sudo tee -a /etc/fstab
fi

echo "[OK] /data mounted ($(df -h /data | tail -1 | awk '{print $2}') total)"
REMOTE_STORAGE
    ok "Data volume ready"

    # Step 2: Install Docker
    info "Step 2/4: Installing Docker..."
    $ssh_cmd "bash -s" <<'REMOTE_DOCKER'
set -euo pipefail

if command -v docker &>/dev/null; then
    echo "[OK] Docker already installed: $(docker --version)"
else
    echo "Installing Docker..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker.io docker-compose-plugin
    sudo usermod -aG docker $USER
    echo "[OK] Docker installed: $(docker --version)"
fi

# Ensure docker is running
sudo systemctl enable docker
sudo systemctl start docker
REMOTE_DOCKER
    ok "Docker installed"

    # Step 3: Deploy MLflow stack
    info "Step 3/4: Deploying MLflow Docker Compose stack..."
    $ssh_cmd "bash -s" <<REMOTE_MLFLOW
set -euo pipefail

# Create directory structure
sudo mkdir -p /data/{postgres,minio,mlflow,nginx/ssl}
sudo chown -R \$USER:\$USER /data

# Create docker-compose.yml
cat > /data/docker-compose.yml <<'COMPOSE_EOF'
services:
  postgres:
    image: postgres:16
    container_name: mlflow-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: minivess
      POSTGRES_PASSWORD: ${PG_PASSWORD}
      POSTGRES_DB: mlflow
    volumes:
      - /data/postgres:/var/lib/postgresql/data
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
      MINIO_ROOT_PASSWORD: ${MINIO_PASSWORD}
    volumes:
      - /data/minio:/data
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
      echo 'Bucket mlflow-artifacts ready';
      "
    environment:
      MINIO_PASSWORD: ${MINIO_PASSWORD}

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
      MLFLOW_BACKEND_STORE_URI: "postgresql://minivess:\${PG_PASSWORD}@postgres:5432/mlflow"
      MLFLOW_ARTIFACTS_DESTINATION: "s3://mlflow-artifacts"
      AWS_ACCESS_KEY_ID: minioadmin
      AWS_SECRET_ACCESS_KEY: ${MINIO_PASSWORD}
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
      - /data/nginx/nginx.conf:/etc/nginx/conf.d/default.conf:ro
      - /data/nginx/.htpasswd:/etc/nginx/.htpasswd:ro
      - /data/nginx/ssl:/etc/nginx/ssl:ro
COMPOSE_EOF

# Create .env file for compose
cat > /data/.env <<ENV_EOF
PG_PASSWORD=${pg_password}
MINIO_PASSWORD=$(openssl rand -base64 24)
ENV_EOF

# Create nginx config (HTTP only initially — certbot adds HTTPS later)
cat > /data/nginx/nginx.conf <<'NGINX_EOF'
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

    # Health check endpoint (no auth)
    location /health {
        proxy_pass http://mlflow:5000/health;
    }
}
NGINX_EOF

# Create htpasswd (basic auth)
sudo apt-get install -y -qq apache2-utils 2>/dev/null || true
htpasswd -cb /data/nginx/.htpasswd minivess "${mlflow_password}"

echo "[OK] MLflow stack configured"
REMOTE_MLFLOW
    ok "MLflow stack configured"

    # Step 4: Start the stack
    info "Step 4/4: Starting Docker Compose stack..."
    # Need newgrp for docker group, or use sudo
    $ssh_cmd "sudo docker compose -f /data/docker-compose.yml --env-file /data/.env up -d"
    ok "Docker Compose stack started"

    # Wait for health
    info "Waiting for MLflow to be healthy (up to 60s)..."
    local attempts=0
    while [[ $attempts -lt 12 ]]; do
        if $ssh_cmd "curl -sf http://localhost:5000/health" &>/dev/null; then
            ok "MLflow is healthy!"
            break
        fi
        sleep 5
        attempts=$((attempts + 1))
    done

    if [[ $attempts -ge 12 ]]; then
        warn "MLflow health check timed out. Check logs:"
        warn "  ssh -i $SSH_KEY_PATH ubuntu@$public_ip"
        warn "  sudo docker compose -f /data/docker-compose.yml logs"
    fi

    echo ""
    ok "Phase 4 complete! MLflow stack deployed."
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Access Details"
    echo "═══════════════════════════════════════════════════════"
    echo ""
    info "MLflow UI:  http://$public_ip (nginx) or http://$public_ip:5000 (direct)"
    info "Username:   minivess"
    info "Password:   $mlflow_password"
    info "SSH:        ssh -i $SSH_KEY_PATH ubuntu@$public_ip"
    echo ""
    info "Add to your .env:"
    echo "  MLFLOW_TRACKING_URI_REMOTE=http://$public_ip"
    echo "  MLFLOW_TRACKING_USERNAME_REMOTE=minivess"
    echo "  MLFLOW_TRACKING_PASSWORD_REMOTE=$mlflow_password"
}

# ─── Phase 5: DNS + TLS (Optional) ─────────────────────────────────────────

phase5_dns_tls() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Phase 5: Custom Domain + HTTPS (Optional)"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    local public_ip
    public_ip=$(get_resource "public_ip")

    if [[ -z "${CLOUDFLARE_API_TOKEN:-}" || -z "${CLOUDFLARE_ZONE_ID:-}" ]]; then
        info "Cloudflare not configured (CLOUDFLARE_API_TOKEN / CLOUDFLARE_ZONE_ID empty)."
        info "Skipping DNS + TLS. MLflow is accessible at http://$public_ip"
        info ""
        info "To add a custom domain later:"
        info "  1. Set CLOUDFLARE_API_TOKEN and CLOUDFLARE_ZONE_ID in .env"
        info "  2. Re-run: bash scripts/oracle-setup-script.sh"
        return 0
    fi

    info "Cloudflare configured. This will:"
    info "  1. Create DNS A record: mlflow.minivess.fi → $public_ip"
    info "  2. Install certbot on the VM"
    info "  3. Get a free Let's Encrypt TLS certificate"
    info "  4. Update nginx for HTTPS"

    if ! confirm "Set up custom domain + HTTPS?"; then
        info "Skipping Phase 5."
        return 0
    fi

    # Get domain name from Cloudflare zone
    info "Fetching domain name from Cloudflare..."
    local zone_info
    zone_info=$(curl -sf -X GET \
        "https://api.cloudflare.com/client/v4/zones/$CLOUDFLARE_ZONE_ID" \
        -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
        -H "Content-Type: application/json")

    local domain
    domain=$(echo "$zone_info" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('result', {}).get('name', ''))
")

    if [[ -z "$domain" ]]; then
        err "Could not fetch domain from Cloudflare. Check CLOUDFLARE_ZONE_ID and token."
        return 1
    fi

    local fqdn="mlflow.$domain"
    info "Domain: $fqdn → $public_ip"

    # Create/update DNS A record
    info "Creating DNS A record..."
    # Check if record already exists
    local existing_record
    existing_record=$(curl -sf -X GET \
        "https://api.cloudflare.com/client/v4/zones/$CLOUDFLARE_ZONE_ID/dns_records?type=A&name=$fqdn" \
        -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
        -H "Content-Type: application/json")

    local record_id
    record_id=$(echo "$existing_record" | python3 -c "
import json, sys
data = json.load(sys.stdin)
records = data.get('result', [])
print(records[0]['id'] if records else '')
")

    if [[ -n "$record_id" ]]; then
        # Update existing
        curl -sf -X PUT \
            "https://api.cloudflare.com/client/v4/zones/$CLOUDFLARE_ZONE_ID/dns_records/$record_id" \
            -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
            -H "Content-Type: application/json" \
            --data "{\"type\":\"A\",\"name\":\"mlflow\",\"content\":\"$public_ip\",\"ttl\":300,\"proxied\":false}" \
            >/dev/null
        ok "DNS A record updated: $fqdn → $public_ip"
    else
        # Create new
        curl -sf -X POST \
            "https://api.cloudflare.com/client/v4/zones/$CLOUDFLARE_ZONE_ID/dns_records" \
            -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
            -H "Content-Type: application/json" \
            --data "{\"type\":\"A\",\"name\":\"mlflow\",\"content\":\"$public_ip\",\"ttl\":300,\"proxied\":false}" \
            >/dev/null
        ok "DNS A record created: $fqdn → $public_ip"
    fi

    save_resource "fqdn" "$fqdn"

    # Install certbot and get TLS cert
    info "Installing certbot and requesting TLS certificate..."
    local ssh_cmd="ssh -o StrictHostKeyChecking=accept-new -i $SSH_KEY_PATH ubuntu@$public_ip"

    $ssh_cmd "bash -s" <<REMOTE_TLS
set -euo pipefail
sudo apt-get update -qq
sudo apt-get install -y -qq certbot python3-certbot-nginx

# Update nginx config for the real domain
cat > /data/nginx/nginx.conf <<'NGINX_TLS_EOF'
server {
    listen 80;
    server_name ${fqdn};

    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    location / {
        return 301 https://\$server_name\$request_uri;
    }
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

    location /health {
        proxy_pass http://mlflow:5000/health;
    }
}
NGINX_TLS_EOF

# Get certificate (HTTP challenge — needs port 80 open)
# Stop nginx temporarily for standalone mode
sudo docker compose -f /data/docker-compose.yml stop nginx || true
sudo certbot certonly --standalone -d ${fqdn} --non-interactive --agree-tos -m admin@${domain} || {
    echo "[WARN] certbot failed — DNS may not have propagated yet."
    echo "       Wait 5 minutes, then run: sudo certbot certonly --standalone -d ${fqdn}"
    # Restart nginx on HTTP-only config
    sudo docker compose -f /data/docker-compose.yml start nginx
    exit 0
}

# Mount cert into nginx container
# Update compose to mount letsencrypt
sudo docker compose -f /data/docker-compose.yml up -d nginx

# Set up auto-renewal
echo "0 3 * * * certbot renew --quiet --deploy-hook 'docker compose -f /data/docker-compose.yml restart nginx'" | sudo crontab -

echo "[OK] TLS certificate installed for ${fqdn}"
REMOTE_TLS

    echo ""
    ok "Phase 5 complete! HTTPS configured."
    info "MLflow is now at: https://$fqdn"

    echo ""
    info "Update your .env:"
    echo "  MLFLOW_TRACKING_URI_REMOTE=https://$fqdn"
}

# ─── Phase 6: Verify Everything ────────────────────────────────────────────

phase6_verify() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Phase 6: Verification"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    local public_ip
    public_ip=$(get_resource "public_ip")
    local fqdn
    fqdn=$(get_resource "fqdn")
    local mlflow_password
    mlflow_password=$(get_resource "mlflow_password")
    local errors=0

    # Determine URL
    local mlflow_url
    if [[ -n "$fqdn" ]]; then
        mlflow_url="https://$fqdn"
    elif [[ -n "$public_ip" ]]; then
        mlflow_url="http://$public_ip"
    else
        err "No public IP or FQDN found."
        return 1
    fi

    # Check SSH
    info "Check 1/4: SSH access..."
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -i "$SSH_KEY_PATH" "ubuntu@$public_ip" "echo ok" &>/dev/null; then
        ok "SSH: accessible"
    else
        err "SSH: cannot connect"
        errors=$((errors + 1))
    fi

    # Check MLflow health
    info "Check 2/4: MLflow health endpoint..."
    if curl -sf "$mlflow_url/health" &>/dev/null; then
        ok "MLflow health: OK"
    elif curl -sf "http://$public_ip:5000/health" &>/dev/null; then
        ok "MLflow health: OK (direct port 5000)"
    else
        err "MLflow health: unreachable"
        errors=$((errors + 1))
    fi

    # Check MLflow API with auth
    info "Check 3/4: MLflow API (authenticated)..."
    if [[ -n "$mlflow_password" ]]; then
        local api_result
        api_result=$(curl -sf -u "minivess:$mlflow_password" \
            "$mlflow_url/api/2.0/mlflow/experiments/search" \
            -H "Content-Type: application/json" \
            -d '{"max_results": 1}' 2>/dev/null || echo "FAIL")
        if [[ "$api_result" != "FAIL" ]]; then
            ok "MLflow API: authenticated successfully"
        else
            warn "MLflow API: auth may not be working yet"
        fi
    fi

    # Check disk space
    info "Check 4/4: Disk space on VM..."
    local disk_info
    disk_info=$(ssh -o ConnectTimeout=5 -i "$SSH_KEY_PATH" "ubuntu@$public_ip" "df -h /data 2>/dev/null || df -h /" 2>/dev/null)
    if [[ -n "$disk_info" ]]; then
        ok "Disk: $(echo "$disk_info" | tail -1 | awk '{print $4 " available of " $2}')"
    fi

    echo ""
    if [[ $errors -eq 0 ]]; then
        echo "═══════════════════════════════════════════════════════"
        echo -e "  ${GREEN}ALL CHECKS PASSED${NC}"
        echo "═══════════════════════════════════════════════════════"
    else
        echo "═══════════════════════════════════════════════════════"
        echo -e "  ${RED}$errors CHECK(S) FAILED${NC}"
        echo "═══════════════════════════════════════════════════════"
    fi

    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Summary"
    echo "═══════════════════════════════════════════════════════"
    echo ""
    echo "  MLflow URL:    $mlflow_url"
    echo "  Username:      minivess"
    echo "  Password:      $mlflow_password"
    echo "  SSH:           ssh -i $SSH_KEY_PATH ubuntu@$public_ip"
    echo "  Resources:     $RESOURCE_FILE"
    echo ""
    echo "  Add to .env:"
    echo "    MLFLOW_TRACKING_URI_REMOTE=$mlflow_url"
    echo "    MLFLOW_TRACKING_USERNAME_REMOTE=minivess"
    echo "    MLFLOW_TRACKING_PASSWORD_REMOTE=$mlflow_password"
    echo ""
}

# ─── Teardown ───────────────────────────────────────────────────────────────

teardown() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo -e "  ${RED}TEARDOWN: Destroy All Oracle Cloud Resources${NC}"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    if [[ ! -f "$RESOURCE_FILE" ]]; then
        err "No resource file found at $RESOURCE_FILE"
        err "Nothing to tear down."
        exit 1
    fi

    info "Resources to destroy:"
    cat "$RESOURCE_FILE"
    echo ""

    if ! confirm "DESTROY all resources? This cannot be undone!"; then
        info "Teardown cancelled."
        exit 0
    fi

    load_env

    # Destroy in reverse order
    local volume_attachment_ocid instance_ocid volume_ocid subnet_ocid igw_ocid vcn_ocid

    # 1. Terminate instance (also detaches volumes)
    instance_ocid=$(get_resource "instance_ocid")
    if [[ -n "$instance_ocid" ]]; then
        info "Terminating instance..."
        oci compute instance terminate \
            --instance-id "$instance_ocid" \
            --preserve-boot-volume false \
            --force \
            --wait-for-state TERMINATED 2>/dev/null || warn "Instance may already be terminated"
        ok "Instance terminated"
    fi

    # 2. Delete block volume
    volume_ocid=$(get_resource "volume_ocid")
    if [[ -n "$volume_ocid" ]]; then
        info "Deleting block volume..."
        oci bv volume delete \
            --volume-id "$volume_ocid" \
            --force \
            --wait-for-state TERMINATED 2>/dev/null || warn "Volume may already be deleted"
        ok "Block volume deleted"
    fi

    # 3. Delete subnet
    subnet_ocid=$(get_resource "subnet_ocid")
    if [[ -n "$subnet_ocid" ]]; then
        info "Deleting subnet..."
        oci network subnet delete \
            --subnet-id "$subnet_ocid" \
            --force \
            --wait-for-state TERMINATED 2>/dev/null || warn "Subnet may already be deleted"
        ok "Subnet deleted"
    fi

    # 4. Delete internet gateway
    igw_ocid=$(get_resource "igw_ocid")
    if [[ -n "$igw_ocid" ]]; then
        # Must clear route table first
        local rt_ocid
        rt_ocid=$(get_resource "route_table_ocid")
        if [[ -n "$rt_ocid" ]]; then
            oci network route-table update --rt-id "$rt_ocid" --route-rules '[]' --force 2>/dev/null || true
        fi
        info "Deleting internet gateway..."
        oci network internet-gateway delete \
            --ig-id "$igw_ocid" \
            --force \
            --wait-for-state TERMINATED 2>/dev/null || warn "IGW may already be deleted"
        ok "Internet gateway deleted"
    fi

    # 5. Delete VCN
    vcn_ocid=$(get_resource "vcn_ocid")
    if [[ -n "$vcn_ocid" ]]; then
        info "Deleting VCN..."
        oci network vcn delete \
            --vcn-id "$vcn_ocid" \
            --force \
            --wait-for-state TERMINATED 2>/dev/null || warn "VCN may already be deleted"
        ok "VCN deleted"
    fi

    # Clean up resource file
    rm -f "$RESOURCE_FILE"
    ok "Resource file removed: $RESOURCE_FILE"

    echo ""
    ok "Teardown complete. All resources destroyed."
}

# ─── Retry VM Loop (for capacity-constrained regions) ───────────────────────

retry_vm_loop() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  VM Capacity Retry Loop"
    echo "═══════════════════════════════════════════════════════"
    echo ""
    info "Retrying ARM VM launch every 5 minutes until capacity is available."
    info "Frankfurt ARM is chronically oversubscribed — this may take hours."
    info "Press Ctrl+C to stop."
    echo ""

    load_env

    local subnet_ocid
    subnet_ocid=$(get_resource "subnet_ocid")
    if [[ -z "$subnet_ocid" ]]; then
        err "No subnet found. Run the full script first (Phase 1)."
        exit 1
    fi

    # Find image
    local image_ocid
    image_ocid=$(SUPPRESS_LABEL_WARNING=True oci compute image list \
        --compartment-id "$OCI_COMPARTMENT_OCID" \
        --operating-system "Canonical Ubuntu" \
        --operating-system-version "24.04" \
        --shape "VM.Standard.A1.Flex" \
        --sort-by TIMECREATED --sort-order DESC --limit 1 \
        --output json 2>/dev/null | python3 -c "
import json,sys
data=json.load(sys.stdin)
items=data.get('data',[])
print(items[0]['id'] if items else '')
" 2>/dev/null)

    if [[ -z "$image_ocid" ]]; then
        err "Could not find Ubuntu 24.04 ARM image."
        exit 1
    fi

    local ads=("dCPK:EU-FRANKFURT-1-AD-1" "dCPK:EU-FRANKFURT-1-AD-2" "dCPK:EU-FRANKFURT-1-AD-3")
    local attempt=0

    while true; do
        attempt=$((attempt + 1))
        echo ""
        info "Attempt #$attempt — $(date '+%Y-%m-%d %H:%M:%S')"

        for ad in "${ads[@]}"; do
            info "  Trying $ad ..."
            local result
            result=$(SUPPRESS_LABEL_WARNING=True oci compute instance launch \
                --compartment-id "$OCI_COMPARTMENT_OCID" \
                --availability-domain "$ad" \
                --shape "VM.Standard.A1.Flex" \
                --shape-config '{"ocpus":4,"memoryInGBs":24}' \
                --display-name "minivess-mlflow" \
                --image-id "$image_ocid" \
                --subnet-id "$subnet_ocid" \
                --ssh-authorized-keys-file "${SSH_KEY_PATH}.pub" \
                --assign-public-ip true \
                --output json 2>&1)

            local instance_id
            instance_id=$(echo "$result" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    print(d.get('data',{}).get('id',''))
except:
    print('')
" 2>/dev/null)

            if [[ -n "$instance_id" && "$instance_id" == ocid1.* ]]; then
                echo ""
                ok "VM LAUNCHED on $ad!"
                ok "Instance: $instance_id"
                save_resource "instance_ocid" "$instance_id"
                save_resource "availability_domain" "$ad"

                info "Waiting for RUNNING state..."
                for i in $(seq 1 60); do
                    local state
                    state=$(SUPPRESS_LABEL_WARNING=True oci compute instance get \
                        --instance-id "$instance_id" --output json 2>/dev/null | \
                        python3 -c "import json,sys; print(json.load(sys.stdin).get('data',{}).get('lifecycle-state',''))" 2>/dev/null)
                    [[ "$state" == "RUNNING" ]] && break
                    printf "."
                    sleep 10
                done
                echo ""
                ok "VM is RUNNING. Continue with: bash scripts/oracle-setup-script.sh"
                return 0
            fi
        done

        info "  All ADs exhausted. Waiting 5 minutes before retry..."
        sleep 300
    done
}

# ─── Main ───────────────────────────────────────────────────────────────────

main() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  MinIVess MLOps — Oracle Cloud Setup"
    echo "  MLflow + PostgreSQL + MinIO on Always Free ARM"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "  This script creates Oracle Cloud resources to host a remote"
    echo "  MLflow tracking server. Total cost: \$0/month (Always Free)."
    echo ""
    echo "  Plan: docs/planning/oracle-config-planning.md"
    echo "  Tutorial: docs/planning/cloud-tutorial.md"
    echo ""

    if [[ "${1:-}" == "--teardown" ]]; then
        teardown
        exit 0
    fi

    if [[ "${1:-}" == "--retry-vm" ]]; then
        retry_vm_loop
        exit 0
    fi

    phase0_validate
    phase1_network
    phase2_compute
    phase3_storage
    phase4_provision
    phase5_dns_tls
    phase6_verify
}

main "$@"
