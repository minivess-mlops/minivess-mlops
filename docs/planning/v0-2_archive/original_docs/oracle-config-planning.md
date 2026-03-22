---
title: "Oracle Cloud Configuration Plan — MLflow on Always Free ARM"
status: rejected
created: "2026-03-13"
depends_on:
  - cloud-tutorial.md
  - mlflow-deployment-storage-analysis.md
  - pulumi-iac-implementation-guide.md
---

# Oracle Cloud Configuration Plan

**Goal:** Deploy MLflow + PostgreSQL + MinIO on Oracle Cloud Always Free ARM VM,
accessible to SkyPilot training VMs and paper reviewers.

**Region:** Defined by `OCI_REGION` in `.env`. Default: `eu-milan-1`.
Avoid Frankfurt — chronically out of ARM capacity (confirmed 2026-03-13).
See `docs/planning/cloud-tutorial.md` for full region recommendations.

**Bootstrap flow (3 scripts, 1 browser action):**

```
1. Set OCI_USER_OCID + OCI_TENANCY_OCID + OCI_REGION in .env
2. bash scripts/oracle-bootstrap-local-key.sh     # generates key + paste script
3. Paste oci-cloud-shell-paste-this.sh into Cloud Shell  # ONE browser action
4. bash scripts/oracle-setup-script.sh             # everything else is automated
```

**Status: REJECTED (2026-03-13)**

Oracle Cloud Always Free was rejected after hands-on testing due to:
1. **Frankfurt ARM capacity exhausted** — all 3 ADs, even at 1 OCPU / 6 GB
2. **No region change** — home region is permanent, stuck with Frankfurt
3. **Card verification blocks second account** — cannot sign up on a better region
4. **Garbage DevEx** — API key bootstrap requires browser-based Cloud Shell,
   intermittent `IdcsConversionError` / `NotAuthenticated` from the OCI CLI,
   PKCS#1 vs PKCS#8 key format issues, and `--wait-for-state` corrupts JSON output

**Decision: Switch to Hetzner VPS** (EUR 4.50/month) for remote MLflow hosting.
See `docs/planning/hetzner-mlflow-plan.md` (to be created).

---

## Phase 1: Network Foundation (VCN + Subnet)

Oracle requires a Virtual Cloud Network before any compute instance.

| Resource | OCI Name | Purpose |
|----------|----------|---------|
| VCN | `minivess-vcn` | Isolated network (10.0.0.0/16) |
| Public Subnet | `minivess-public-subnet` | VM gets public IP (10.0.0.0/24) |
| Internet Gateway | `minivess-igw` | Outbound + inbound internet access |
| Route Table | `minivess-rt` | Routes 0.0.0.0/0 → IGW |
| Security List | `minivess-sl` | Firewall rules (ingress/egress) |

**Security List rules (ingress):**

| Port | Protocol | Source | Purpose |
|------|----------|--------|---------|
| 22 | TCP | 0.0.0.0/0 (or your IP) | SSH access |
| 80 | TCP | 0.0.0.0/0 | HTTP (certbot challenge) |
| 443 | TCP | 0.0.0.0/0 | HTTPS (MLflow via nginx) |
| 5000 | TCP | 0.0.0.0/0 | MLflow direct (before nginx) |

**OCI CLI commands:**
```bash
# Create VCN
oci network vcn create \
  --compartment-id "$OCI_COMPARTMENT_OCID" \
  --cidr-block "10.0.0.0/16" \
  --display-name "minivess-vcn" \
  --dns-label "minivess"

# Create Internet Gateway
oci network internet-gateway create \
  --compartment-id "$OCI_COMPARTMENT_OCID" \
  --vcn-id "$VCN_OCID" \
  --display-name "minivess-igw" \
  --is-enabled true

# Create Route Table (default 0.0.0.0/0 → IGW)
oci network route-table update \
  --rt-id "$RT_OCID" \
  --route-rules '[{"destination":"0.0.0.0/0","networkEntityId":"'$IGW_OCID'"}]'

# Create Security List
oci network security-list create \
  --compartment-id "$OCI_COMPARTMENT_OCID" \
  --vcn-id "$VCN_OCID" \
  --display-name "minivess-sl" \
  --ingress-security-rules '<see script for full JSON>'

# Create Public Subnet
oci network subnet create \
  --compartment-id "$OCI_COMPARTMENT_OCID" \
  --vcn-id "$VCN_OCID" \
  --cidr-block "10.0.0.0/24" \
  --display-name "minivess-public-subnet" \
  --dns-label "pub" \
  --route-table-id "$RT_OCID" \
  --security-list-ids '["'$SL_OCID'"]'
```

---

## Phase 2: SSH Key + Compute Instance

### SSH Key (separate from API signing key)

The API signing key authenticates CLI/API calls. The SSH key authenticates
interactive login to the VM. They are different key pairs.

```bash
ssh-keygen -t ed25519 -f ~/.ssh/oci_vm_key -N "" -C "minivess-oci-vm"
```

### ARM A1.Flex Instance

| Param | Value | Notes |
|-------|-------|-------|
| Shape | VM.Standard.A1.Flex | ARM Ampere Altra |
| OCPUs | 4 | Full free-tier allocation |
| RAM | 24 GB | Full free-tier allocation |
| Boot Volume | 50 GB | Minimum, leaves 150 GB for data |
| Image | Canonical-Ubuntu-24.04-aarch64 | Latest LTS for ARM |
| AD | Pick least-loaded of AD-1/2/3 | Frankfurt has 3 ADs |

**Known issue: Frankfurt capacity shortage**

ARM A1.Flex in `eu-frankfurt-1` is chronically over-subscribed. The `launch`
command may return `Out of host capacity`. Mitigations:

1. Try each AD sequentially (AD-1, AD-2, AD-3)
2. Start smaller (1 OCPU / 6 GB) → resize later
3. Use the retry script: [hitrov/oci-arm-host-capacity](https://github.com/hitrov/oci-arm-host-capacity)
4. Try off-peak hours (early morning CET)

```bash
oci compute instance launch \
  --compartment-id "$OCI_COMPARTMENT_OCID" \
  --availability-domain "$AD_NAME" \
  --shape "VM.Standard.A1.Flex" \
  --shape-config '{"ocpus":4,"memoryInGBs":24}' \
  --display-name "minivess-mlflow" \
  --image-id "$UBUNTU_IMAGE_OCID" \
  --subnet-id "$SUBNET_OCID" \
  --ssh-authorized-keys-file ~/.ssh/oci_vm_key.pub \
  --assign-public-ip true
```

### Block Volume (150 GB data)

```bash
oci bv volume create \
  --compartment-id "$OCI_COMPARTMENT_OCID" \
  --availability-domain "$AD_NAME" \
  --display-name "minivess-data" \
  --size-in-gbs 150

oci compute volume-attachment attach \
  --instance-id "$INSTANCE_OCID" \
  --volume-id "$VOLUME_OCID" \
  --type paravirtualized
```

On the VM:
```bash
# Find the attached volume
lsblk
# Format and mount
sudo mkfs.ext4 /dev/sdb
sudo mkdir -p /data
sudo mount /dev/sdb /data
echo '/dev/sdb /data ext4 defaults,_netdev 0 2' | sudo tee -a /etc/fstab
```

---

## Phase 3: VM Setup (Docker + MLflow Stack)

### Docker Installation (ARM64)
```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER
# Re-login for group change
```

### MLflow Docker Compose (Remote Server)

A minimal compose file for the remote server (subset of the full local stack):

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `postgres` | `postgres:16` | 5432 (internal) | MLflow backend store |
| `minio` | `minio/minio` | 9000/9001 (internal) | Artifact store (S3-compat) |
| `mlflow` | `ghcr.io/mlflow/mlflow:2.x` | 5000 | Tracking server |
| `nginx` | `nginx:alpine` | 80/443 | Reverse proxy + TLS + basic auth |

Key configuration:
```
MLflow backend:  postgresql://minivess:$PG_PASSWORD@postgres:5432/mlflow
MLflow artifacts: s3://mlflow-artifacts (via MinIO on same host)
MLflow auth:     nginx basic-auth in front of port 5000
```

### Directory Layout on VM
```
/data/
├── postgres/          # PostgreSQL data directory
├── minio/             # MinIO object storage
├── mlflow/            # MLflow artifacts (if not using MinIO)
├── nginx/
│   ├── nginx.conf     # Reverse proxy config
│   ├── .htpasswd      # Basic auth passwords
│   └── ssl/           # TLS certs (certbot)
└── docker-compose.yml # The compose file
```

---

## Phase 4: Networking + TLS (Optional)

### Without custom domain
- Access MLflow at `http://PUBLIC_IP:5000`
- Basic auth via nginx still recommended

### With custom domain (Cloudflare + Let's Encrypt)
1. Get VM public IP from OCI Console or `oci compute instance list`
2. Create DNS A record: `mlflow.minivess.fi` → `PUBLIC_IP` (via Cloudflare API)
3. Install certbot on VM
4. Get TLS cert: `sudo certbot --nginx -d mlflow.minivess.fi`
5. Certbot auto-renewal: `sudo certbot renew --dry-run`

---

## Phase 5: Verification

| Check | Command | Expected |
|-------|---------|----------|
| VM reachable | `ssh -i ~/.ssh/oci_vm_key ubuntu@PUBLIC_IP` | Login prompt |
| Docker running | `docker ps` (on VM) | 4 containers (postgres, minio, mlflow, nginx) |
| MLflow UI | `curl -u minivess:$MLFLOW_PW $MLFLOW_URL/api/2.0/mlflow/experiments/list` | JSON response |
| SkyPilot can reach | Set `MLFLOW_TRACKING_URI_REMOTE` and run test | Experiment created |
| Storage healthy | `df -h /data` | 150 GB available |

---

## Script Automation (`scripts/oracle-setup-script.sh`)

The script automates Phases 1-4 with human-in-the-loop confirmation at each phase.

**Architecture:**
```
oracle-setup-script.sh
├── Phase 0: Validate prerequisites (oci cli, .env vars, ssh key)
├── Phase 1: Create VCN + subnet + security list  [confirm]
├── Phase 2: Launch ARM VM (with AD failover)      [confirm]
├── Phase 3: Attach block volume + format           [confirm]
├── Phase 4: SSH into VM, install Docker + stack    [confirm]
├── Phase 5: DNS + TLS (if Cloudflare configured)  [confirm]
└── Phase 6: Verify everything + update .env        [auto]
```

**Each phase:**
1. Shows what will be created (resources + estimated cost = $0)
2. Asks `Continue? [y/N]`
3. Executes OCI CLI commands
4. Saves resource OCIDs to `.oci/minivess-resources.json` for teardown
5. Shows result summary

**Teardown:** `scripts/oracle-setup-script.sh --teardown` reads the saved OCIDs
and destroys everything in reverse order.

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Frankfurt ARM capacity exhausted | Cannot provision VM | Retry with smaller shape; try all 3 ADs; use capacity retry script |
| Oracle terminates Always Free | Service disruption | Stack is portable Docker Compose — redeploy to Hetzner in 1 hour |
| SSH key lost | Locked out of VM | Store key in password manager; OCI Console has serial console access |
| Data volume full (150 GB) | MLflow stops writing | Monitor with df; archive old experiments to OCI Object Storage (20 GB free) |
| TLS cert expires | HTTPS breaks | Certbot auto-renewal cron; Pulumi can manage this later |

---

## Future: Pulumi Automation

The manual script (`oracle-setup-script.sh`) is the first step. Once working,
the same infrastructure is encoded in Pulumi Python (task T1b in the XML plan):

```
deployment/pulumi/
├── __main__.py       # Pulumi program
├── Pulumi.yaml       # Project config
├── Pulumi.dev.yaml   # Dev stack config
├── oci_network.py    # VCN, subnet, security list
├── oci_compute.py    # VM instance, block volume
├── oci_provision.py  # Docker install via remote-exec
└── mlflow_stack.py   # Compose file generation + deployment
```

Benefits over the shell script:
- **Idempotent:** `pulumi up` converges to desired state (no duplicate resources)
- **State tracking:** Pulumi Cloud tracks all resources (no `.oci/minivess-resources.json`)
- **Drift detection:** `pulumi refresh` shows if someone changed resources manually
- **Teardown:** `pulumi destroy` (no reverse-order logic needed)
