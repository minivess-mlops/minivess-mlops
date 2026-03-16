---
title: "Pulumi + UpCloud Managed Services — MLflow Deployment Plan"
status: active
created: "2026-03-13"
replaces: upcloud-mlflow-plan.md (shell script approach → Pulumi IaC)
depends_on:
  - pulumi-iac-implementation-guide.md
  - cloud-tutorial.md
  - mlflow-deployment-storage-analysis.md
---

# Pulumi + UpCloud Managed Services Deployment

**Goal:** Deploy a remote MLflow tracking server using UpCloud's **managed services**
(Managed PostgreSQL + Managed Object Storage) provisioned entirely via **Pulumi Python**,
so that academic lab researchers with zero infrastructure experience can run
`pulumi up` and have a production MLflow instance in 5 minutes.

**Audience:** PhD researchers and academic lab groups with no DevOps/infra engineers.
The entire stack is defined in ~100 lines of Python. No shell scripts, no SSH, no
nginx, no manual server configuration.

---

## Why Managed Services (Not Self-Hosted)

The [previous approach](upcloud-mlflow-plan.md) used a shell script to create a VPS and
deploy PostgreSQL + MinIO + MLflow + nginx as 4 Docker containers. This works, but:

| Concern | Self-hosted (shell script) | Managed services (Pulumi) |
|---------|---------------------------|---------------------------|
| **PostgreSQL** | You manage backups, upgrades, HA | UpCloud manages everything |
| **Object Storage** | Self-hosted MinIO container | UpCloud S3-compatible, 99.99% durability |
| **Provisioning** | 6-phase bash script, SSH commands | `pulumi up` (one command) |
| **State tracking** | None (manual) | Pulumi state (drift detection, preview) |
| **Teardown** | Custom `--teardown` flag | `pulumi destroy` |
| **Reproducibility** | Best-effort | Deterministic (code = infrastructure) |
| **Security updates** | Manual `apt upgrade` | Managed by UpCloud |
| **PhD researcher can do it** | Needs SSH + Docker knowledge | Needs `pulumi up` only |
| **Cost (trial)** | EUR 0 | EUR 0 (managed services included in trial) |
| **Cost (post-trial)** | ~EUR 19/mo (VPS only) | ~EUR 32-39/mo (managed DB + S3 + small VPS) |

**The EUR 13-20/month premium buys zero-maintenance infrastructure.** For an academic
lab where nobody wants to be the "server person", this is the correct trade-off.

---

## Trial Eligibility (Verified 2026-03-13)

UpCloud's free trial (EUR 250 / 30 days) **explicitly includes managed services**:

> "Free access to test most of UpCloud's services, including Cloud Servers, Networking,
> VPN or NAT Gateway, and **Managed Services** such as Kubernetes, **Databases** and
> Load Balancers."

Source: [UpCloud Free Trial Documentation](https://upcloud.com/docs/getting-started/free-trial/)

This means during the 30-day trial, the entire managed stack (Managed PostgreSQL +
Managed Object Storage + VPS) costs EUR 0.

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              UpCloud (fi-hel1)              │
                    │                                             │
  Researcher's      │  ┌──────────────────────┐                   │
  laptop / SkyPilot │  │  VPS (DEV-1xCPU-2GB) │                   │
  training VM       │  │  ┌────────────────┐   │                   │
        ──────────────►│  │ MLflow server  │   │                   │
        HTTP :5000  │  │  │ --app-name     │   │                   │
                    │  │  │  basic-auth    │   │                   │
                    │  │  └───────┬────────┘   │                   │
                    │  └──────────┼────────────┘                   │
                    │             │                                │
                    │         ┌───┼────────────┐                   │
                    │         │   │            │                   │
                    │         ▼   ▼            │                   │
                    │  ┌────────────┐ ┌──────────┐                 │
                    │  │ Managed    │ │ Managed  │                 │
                    │  │ PostgreSQL │ │ Object   │                 │
                    │  │ (1x1xCPU-  │ │ Storage  │                 │
                    │  │  2GB-25GB) │ │ (250 GB) │                 │
                    │  │            │ │ S3-compat│                 │
                    │  │ Backups ✓  │ │ 99.99%   │                 │
                    │  │ HA-ready ✓ │ │ durable  │                 │
                    │  └────────────┘ └──────────┘                 │
                    │         Managed by UpCloud                   │
                    │         (no user maintenance)                │
                    └─────────────────────────────────────────────┘
```

**What you manage:** MLflow on a small VPS (1 container).
**What UpCloud manages:** PostgreSQL (backups, upgrades, HA) + S3 object storage.

**No nginx needed** — MLflow has built-in basic auth since v2.5 (`--app-name basic-auth`).
For TLS, use Cloudflare proxy (free tier) or accept HTTP during trial/internal use.

---

## Cost Breakdown

### During Trial (Day 1-30): EUR 0

| Service | Plan | Hourly | Monthly est. | Trial cost |
|---------|------|--------|--------------|------------|
| Managed PostgreSQL | `1x1xCPU-2GB-25GB` | ~EUR 0.048 | ~EUR 35 | EUR 0 (trial credit) |
| Managed Object Storage | 250 GB | — | EUR 5 | EUR 0 (trial credit) |
| Cloud Server (VPS) | `DEV-1xCPU-2GB` | ~EUR 0.012 | ~EUR 8.70 | EUR 0 (trial credit) |
| **Total** | | | **~EUR 49/mo** | **EUR 0** |

30-day projected spend: ~EUR 49, well within EUR 250 credit.

### Post-Trial Options

| Option | Monthly cost | Effort | Best for |
|--------|-------------|--------|----------|
| **A. Keep UpCloud managed** | ~EUR 39-49 | Zero maintenance | Labs with funding |
| **B. Downgrade VPS, keep managed DB** | ~EUR 38-44 | Zero maintenance | Budget-conscious |
| **C. Migrate to Hetzner (self-hosted)** | EUR 3.79 | Must manage DB + MinIO | Unfunded students |
| **D. Shut down, export data** | EUR 0 | Use local MLflow | Solo researcher |

**Recommendation for academic labs:** Option A or B. EUR 40/month is one conference
coffee session. The time saved by not debugging PostgreSQL backups or MinIO permissions
is worth 10x that in researcher hours.

---

## Provider Comparison (Updated with Managed Services)

| Provider | Managed PostgreSQL | Managed S3 | Trial | Total (managed) |
|----------|-------------------|------------|-------|-----------------|
| **UpCloud** | ~EUR 35/mo (1x1xCPU-2GB) | EUR 5/mo (250GB) | **EUR 250 / 30d** | ~EUR 49/mo |
| **Hetzner** | None (self-hosted only) | None | None | EUR 3.79/mo (VPS-only) |
| **DigitalOcean** | $15/mo (1vCPU/1GB) | $5/mo (250GB Spaces) | $200 / 60d | ~$28/mo |
| **Scaleway** | EUR 8.64/mo (DB-DEV-S) | EUR 0 (75GB free) | EUR 100 one-time | ~EUR 18/mo |

UpCloud is more expensive post-trial, but the trial is the most generous for immediate
use. Scaleway is the cheapest long-term option with managed services.

---

## Pulumi Stack Definition

The actual working code is in `deployment/pulumi/__main__.py` (~260 lines). Key resources:

| Resource | Pulumi Type | Purpose |
|----------|-------------|---------|
| `mlflow-postgres` | `ManagedDatabasePostgresql` | Backend store (auto-backups at 03:00 UTC) |
| `mlflow-s3` | `ManagedObjectStorage` | Artifact store (S3-compatible, europe-1) |
| `mlflow-artifacts` | `ManagedObjectStorageBucket` | S3 bucket for MLflow artifacts |
| `mlflow-s3-user` + key | `ManagedObjectStorageUser` + `AccessKey` | S3 credentials |
| `mlflow-server` | `Server` | VPS (DEV-1xCPU-2GB, Ubuntu 24.04) |
| `install-docker` | `command.remote.Command` | SSH provisioning: Docker install |
| `deploy-mlflow` | `command.remote.Command` | SSH provisioning: MLflow container |

### File: `deployment/pulumi/Pulumi.yaml`

```yaml
name: minivess-mlflow
runtime:
  name: python
  options:
    toolchain: uv
    virtualenv: .venv
config:
  minivess-mlflow:zone:
    default: fi-hel1
  minivess-mlflow:ssh_public_key:
    description: SSH public key content for VPS access
  minivess-mlflow:mlflow_admin_password:
    secret: true
    description: Password for MLflow basic auth admin user
```

---

## Bootstrap (Verified Working 2026-03-13)

```bash
# 1. Install Pulumi CLI (one-time)
curl -fsSL https://get.pulumi.com | sh

# 2. Generate SSH key for VPS access (one-time)
ssh-keygen -t ed25519 -f ~/.ssh/upcloud_minivess -N ""

# 3. Configure stack (local backend — no Pulumi Cloud account needed)
cd deployment/pulumi
PULUMI_CONFIG_PASSPHRASE="" pulumi stack init dev
pulumi config set upcloud:token "$UPCLOUD_TOKEN" --secret
pulumi config set ssh_public_key "$(cat ~/.ssh/upcloud_minivess.pub)"
pulumi config set ssh_private_key "$(cat ~/.ssh/upcloud_minivess)" --secret
pulumi config set mlflow_admin_password "$(openssl rand -base64 16)" --secret

# 4. Deploy everything (takes ~3-5 minutes first run)
PULUMI_CONFIG_PASSPHRASE="" pulumi up --yes
```

After `pulumi up` completes, everything is live:
- Managed PostgreSQL running (automatic daily backups at 03:00 UTC)
- Managed Object Storage with `mlflow-artifacts` bucket
- VPS running with Docker + MLflow container (built-in basic auth)
- MLflow accessible at `http://<server_ip>:5000`

**No post-provision steps needed** — Pulumi's `command.remote.Command` resources
handle Docker installation and MLflow deployment automatically via SSH.

The MLflow container uses a custom Dockerfile that adds `psycopg2-binary`, `boto3`,
and `flask-wtf` on top of the official MLflow image. The `basic_auth.ini` config:

```ini
[mlflow]
default_permission = READ
database_uri = sqlite:///basic_auth.db
admin_username = admin
admin_password = <generated-password>
```

**No nginx.** MLflow's built-in `--app-name basic-auth` (available since MLflow 2.5)
handles authentication directly. For TLS, use Cloudflare proxy (free tier) or the
nginx variant (see [#615](https://github.com/petteriTeikari/minivess-mlops/issues/615)).

Compare: the shell script approach needed **4 containers** (PostgreSQL + MinIO + MLflow +
nginx). The managed approach needs **1 container** (MLflow). PostgreSQL and S3
are fully managed — no container, no volume, no backup cron, no reverse proxy.

---

## Python Dependencies

Add to `pyproject.toml` in a new `infra` optional group:

```toml
[project.optional-dependencies]
infra = [
    "pulumi>=3.0",
    "pulumi-upcloud>=0.11",
    "pulumi-command>=1.0",       # for remote SSH provisioning
]
```

Install: `uv sync --extra infra` (or `uv sync --all-extras` which already covers it).

Pulumi CLI itself is a separate binary (`curl -fsSL https://get.pulumi.com | sh`),
not a Python package. The Python packages are the SDK bindings.

**Pulumi pricing:** Free for individuals (unlimited resources, 1 user).
See: [Pulumi Pricing](https://www.pulumi.com/pricing/)

---

## Backup Before Trial Ends

With managed services, backups are simpler:

```bash
# PostgreSQL: automatic daily backups by UpCloud (included)
# Manual export if migrating away:
pulumi stack output postgres_uri  # get connection string
pg_dump "$(pulumi stack output postgres_uri)" > mlflow_backup.sql

# Object Storage: already S3-compatible, use any S3 tool
aws s3 sync s3://mlflow-artifacts ./mlflow-artifacts-backup/ \
  --endpoint-url "$(pulumi stack output s3_endpoint)"
```

---

## Teardown

```bash
pulumi destroy    # removes ALL resources (VPS, managed DB, S3)
pulumi stack rm   # removes stack state
```

Unlike the shell script `--teardown`, Pulumi tracks exact state and removes exactly
what it created. No orphaned resources.

---

## Migration Path (Post-Trial)

| If you want to... | Do this |
|--------------------|---------|
| **Keep UpCloud** | Add payment method, keep running (~EUR 49/mo) |
| **Switch to Hetzner** | `pg_dump` + `s3 sync`, run Hetzner Pulumi stack |
| **Switch to Scaleway** | Same export, Scaleway has `pulumi-scaleway` provider |
| **Go local-only** | Export, `pulumi destroy`, use local `mlruns/` |

The Pulumi code is **provider-portable**. A Hetzner or Scaleway stack uses different
resource types but identical structure. The migration is writing a new `__main__.py`,
not re-learning infrastructure.

---

## Comparison with Shell Script Approach

| Aspect | `upcloud-setup-script.sh` | Pulumi `__main__.py` |
|--------|--------------------------|----------------------|
| Lines of code | 752 (bash) | ~120 (Python) |
| PostgreSQL | Self-hosted container | Managed (auto-backup, HA-ready) |
| Object Storage | Self-hosted MinIO | Managed S3-compatible |
| Containers on VPS | 4 (PG + MinIO + MLflow + nginx) | 1 (MLflow only) |
| VPS size needed | DEV-2xCPU-4GB (EUR 19/mo) | DEV-1xCPU-2GB (EUR 9/mo) |
| State tracking | None | Pulumi state (preview, diff, drift) |
| Idempotent | Manual checks | Built-in |
| Teardown | Custom bash function | `pulumi destroy` |
| PhD can use it | Needs SSH + Docker knowledge | Needs `pulumi up` |
| Testable | No | `pulumi preview` (dry run) |

---

## Deployment Notes (Verified 2026-03-13)

All questions from the planning phase have been answered through actual deployment:

1. **Managed Object Storage networking** — The VPS accesses S3 over the **public endpoint**
   (`https://pgopc.upcloudobjects.com`). No private networking needed. The `endpoints` list
   on `ManagedObjectStorage` contains entries with `type: "public"` — extract `domain_name`.

2. **Pulumi remote provisioning** — **`pulumi-command` remote exec** works perfectly.
   Two `command.remote.Command` resources: `install-docker` (apt-get docker.io) and
   `deploy-mlflow` (writes Dockerfile + docker-compose.yml + basic_auth.ini, runs
   `docker compose up -d --build`). Full automation — zero SSH needed.

3. **Managed PostgreSQL public access** — UpCloud managed DB defaults to **private only**.
   To connect from the VPS (which uses public networking), you MUST set:
   - `"public_access": True` in properties
   - `"ip_filters": ["0.0.0.0/0"]` (or restrict to VPS IP)
   - `"automatic_utility_network_ip_filter": False`
   The `service_uri` uses the **private hostname** — derive the public one from the
   `components` list (entry with `route: "public"`, `component: "pg"`).

4. **S3 credentials** — `ManagedObjectStorageUserAccessKey` provides `access_key_id` and
   `secret_access_key` as Pulumi Outputs. These are injected into the Docker Compose
   environment variables via `pulumi.Output.all(...).apply(...)`.

5. **URI scheme** — UpCloud returns `postgres://` but SQLAlchemy/MLflow requires
   `postgresql://`. Fix: `.replace("postgres://", "postgresql://", 1)`.

6. **MLflow basic auth requirements** (v2.20.3):
   - `flask-wtf` pip package is required (not in official MLflow image)
   - `MLFLOW_FLASK_SERVER_SECRET_KEY` env var is required for CSRF protection
   - `MLFLOW_AUTH_CONFIG_PATH` env var must point to `basic_auth.ini` location
   - `basic_auth.ini` must include `database_uri = sqlite:///basic_auth.db`
   - Default `authorization_function` works — do NOT set it explicitly (the function
     name changed between MLflow versions)

7. **Trial firewall limitation** — UpCloud trial accounts get 403 TRIAL_FIREWALL when
   creating `ServerFirewallRules`. Set `firewall=True` only after upgrading from trial.

8. **VPS network interface** — Use `iface["ip_address"]` and `iface.get("type") == "public"`
   to extract the IPv4 address. The SDK property name is `ip_address_family` (not `ipAddressFamily`).

---

## References

- [UpCloud Pulumi Provider (Official)](https://github.com/UpCloudLtd/pulumi-upcloud) — v0.11.1, Feb 2026
- [Pulumi Registry: UpCloud](https://www.pulumi.com/registry/packages/upcloud/) — Resource docs
- [UpCloud Free Trial](https://upcloud.com/docs/getting-started/free-trial/) — Includes managed services
- [UpCloud Managed PostgreSQL Docs](https://upcloud.com/docs/products/managed-postgresql/)
- [UpCloud Managed Object Storage Docs](https://upcloud.com/docs/products/managed-object-storage/)
- [Pulumi Pricing](https://www.pulumi.com/pricing/) — Free for individuals
- [MinIVess Pulumi IaC Guide](pulumi-iac-implementation-guide.md) — Full IaC comparison report
- [UpCloud Setup Script (Legacy)](../../scripts/upcloud-setup-script.sh) — Shell script approach (archived)
