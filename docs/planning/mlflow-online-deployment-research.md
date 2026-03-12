---
title: "MLflow Online Deployment Research: Oracle Cloud Always Free"
status: reference
created: "2026-03-12"
---

# MLflow Online Deployment Research: Oracle Cloud Always Free

**Date**: 2026-03-12
**Status**: DECIDED — Oracle Cloud Always Free (A1.Flex ARM)
**Related**: [SkyPilot + FinOps Report](skypilot-and-finops-complete-report.md) §10, [Pulumi IaC Guide](pulumi-iac-implementation-guide.md), [XML Plan](skypilot-compute-offloading-plan-for-vesselfm-sam3-and-synthetic-generation.xml) T1/T1b/T1c
**Decision context**: GitHub Issue [#564](https://github.com/petteriTeikari/minivess-mlops/issues/564) (SkyPilot compute offloading)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement: Why MLflow Must Be Online](#2-problem-statement-why-mlflow-must-be-online)
3. [Hosting Options Evaluated](#3-hosting-options-evaluated)
4. [Decision: Oracle Cloud Always Free](#4-decision-oracle-cloud-always-free)
5. [Region Selection: Frankfurt vs. Milan](#5-region-selection-frankfurt-vs-milan)
6. [Frankfurt Account: What You Get and How to Make It Work](#6-frankfurt-account-what-you-get-and-how-to-make-it-work)
7. [Storage Analysis Across All Deployment Targets](#7-storage-analysis-across-all-deployment-targets)
8. [Architecture: Current Docker Stack on ARM](#8-architecture-current-docker-stack-on-arm)
9. [Artifact Storage Strategy](#9-artifact-storage-strategy)
10. [Security Architecture](#10-security-architecture)
11. [Operational Concerns](#11-operational-concerns)
12. [Network Topology: SkyPilot ↔ Oracle MLflow](#12-network-topology-skypilot--oracle-mlflow)
13. [Implementation Roadmap](#13-implementation-roadmap)
14. [Paper Artifact Sharing](#14-paper-artifact-sharing)
15. [References](#15-references)

---

## 1. Executive Summary

MLflow experiment tracking is the inter-flow contract in MinIVess MLOps — all flows
communicate exclusively through MLflow artifacts and run metadata
([Zaharia et al. (2018). "Accelerating the Machine Learning Lifecycle with MLflow." *IEEE Data Eng. Bull.*](https://www.scinapse.io/papers/2952596791)).
The current deployment runs locally via Docker Compose (`minivess-mlflow` container on
`localhost:5000`), which is unreachable from SkyPilot cloud VMs. This is **Blocker #1**
for compute offloading.

After evaluating six hosting options across 11 criteria, **Oracle Cloud Always Free** was
selected: 4 ARM Ampere OCPUs, 24 GB RAM, 200 GB storage, 10 TB/month egress, at $0/month
forever. The existing Docker Compose stack (MLflow 3.10 + PostgreSQL 16 + MinIO) deploys
on ARM without image changes. The instance doubles as a public artifact archive for paper
submission — reviewers can browse experiments at `https://mlflow.minivess.fi`.

**Key architectural insight**: The MLflow *server stack* (PostgreSQL, MinIO, MLflow) uses
official multi-arch images with native arm64 support. The aarch64 concern applies only to
*training flow* images (which use `nvidia/cuda` amd64-only base), and training never runs
on this instance.

---

## 2. Problem Statement: Why MLflow Must Be Online

### 2.1 The Localhost Barrier

The MinIVess MLflow deployment runs as a Docker Compose service:

```yaml
# deployment/docker-compose.yml (current)
mlflow:
  image: minivess-mlflow:v3.10.0
  command: >
    mlflow server
      --backend-store-uri postgresql://minivess:...@postgres:5432/mlflow
      --default-artifact-root s3://mlflow-artifacts/
      --host 0.0.0.0 --port 5000
  environment:
    MLFLOW_S3_ENDPOINT_URL: http://minio:9000
```

SkyPilot VMs running on RunPod, AWS, or GCP cannot reach `localhost:5000` or
`minivess-mlflow:5000`. Without a network-accessible tracking URI, training runs on cloud
GPUs cannot log metrics, parameters, or artifacts to the central experiment store.

### 2.2 The Seven SkyPilot Integration Blockers

MLflow accessibility is one of seven blockers identified in the SkyPilot FinOps analysis.
The full list (from the [SkyPilot report](skypilot-and-finops-complete-report.md) §8):

1. **MLflow not accessible from cloud** ← this report
2. Compute dispatcher missing (SkyPilot YAML generation)
3. Docker gate blocks SkyPilot (STOP protocol adaptation)
4. YAML calls Prefect (won't exist on SkyPilot VM)
5. No config serialization (Hydra dict → YAML for SkyPilot)
6. No HPO barrier (Optuna storage needs network access)
7. No `compute_backend` config key

Solving blocker #1 also partially solves #6 — Optuna can use the same PostgreSQL instance
via `optuna.storages.RDBStorage(postgresql://...)` over the network.

### 2.3 Tracking URI Resolution Architecture

The Python client resolves the tracking URI via `resolve_tracking_uri()` in
`src/minivess/observability/tracking.py:55–104`:

| Priority | Source | Example |
|----------|--------|---------|
| 1 | Explicit parameter | `resolve_tracking_uri("http://mlflow.minivess.fi:5000")` |
| 2 | `MLFLOW_TRACKING_URI` env var | `http://localhost:5000` (local) or `https://mlflow.minivess.fi` (remote) |
| 3 | Dynaconf settings TOML | Fallback (not recommended — Rule #22) |
| 4 | Default `"mlruns"` | Local file backend (safety net) |

If `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` are set, basic auth
credentials are injected into the URI: `http://user:pass@hostname:port`.

For SkyPilot VMs, the `.env` file (or SkyPilot YAML `envs:` block) sets:
```bash
MLFLOW_TRACKING_URI=https://mlflow.minivess.fi
MLFLOW_TRACKING_USERNAME=minivess
MLFLOW_TRACKING_PASSWORD=<secret>
```

No code changes are needed — only an environment variable change.

---

## 3. Hosting Options Evaluated

Six options were analyzed across 11 criteria. Full analysis in the
[SkyPilot + FinOps Report](skypilot-and-finops-complete-report.md) §10.

### 3.1 Decision Matrix

| Criterion | A: DagsHub | B: CF Tunnel | C: Oracle Free | D: Hetzner | E: Post-Hoc Sync | F: Managed |
|-----------|:----------:|:------------:|:--------------:|:----------:|:-----------------:|:----------:|
| Monthly cost | $0–$39 | $0 | **$0** | €3.49+ | $0 | $50–200+ |
| Setup time (manual) | 5 min | 30 min | 2 hours | 1 hour | 0 min | 1 hour |
| Setup time (Pulumi) | N/A | N/A | **5 min** | 5 min | N/A | N/A |
| Artifact limit | 100 GB | **100 MB** | 200 GB | 20–160 GB | ∞ | ∞ |
| Uptime guarantee | 99.9% | Home ISP | 99.5%+ | 99.9% | N/A | 99.9%+ |
| TLS included | Yes | Yes | Manual (LE) | Manual (LE) | N/A | Yes |
| Auth included | Yes | Yes | Manual (nginx) | Manual | N/A | Yes |
| PostgreSQL backend | Yes | Self | Self | Self | N/A | Yes |
| Paper-shareable URL | Yes | No (home IP) | **Yes** | Yes | No | Yes |
| Full MLflow control | No | Yes | **Yes** | Yes | N/A | No |
| Pulumi-manageable | No | No | **Yes (OCI)** | Yes (Hetzner) | N/A | Partial |

### 3.2 Why Not the Others

**Option A (DagsHub)**: Fastest setup but no full MLflow control. Free tier limited to
1 user. Cannot self-host custom MLflow plugins or run custom queries against the backing
PostgreSQL. A migration script was scoped in [Issue #612](https://github.com/petteriTeikari/minivess-mlops/issues/612).

**Option B (Cloudflare Tunnel)**: Free, zero infrastructure. **Dealbreaker**: Cloudflare
imposes a 100 MB upload limit per request
([Cloudflare Docs — Upload Limits](https://developers.cloudflare.com/workers/platform/limits/)).
ONNX model artifacts routinely exceed 100 MB (SAM3 ViT-32L ≈ 2.5 GB). Also depends on
home machine uptime.

**Option D (Hetzner VPS)**: Excellent price-performance (CX22: 2 vCPU, 4 GB, €3.49/month).
Strong choice for teams willing to pay. Oracle Free wins on cost ($0 vs €42/year) and
raw resources (24 GB vs 4 GB RAM).

**Option E (Post-Hoc Sync)**: No infrastructure but breaks the real-time tracking contract.
SkyPilot VMs log to local `mlruns/`, then `rsync` post-training. Loses live monitoring,
early stopping visibility, and Prefect flow status updates during training.

**Option F (Managed MLflow)**: Databricks, Nebius, Azure ML. Best for enterprise teams.
Overkill for a single-researcher academic project at $50–200+/month.

### 3.3 Decision

**Oracle Cloud Always Free (Option C)** — selected 2026-03-12.

Rationale:
- $0/month forever (not a trial — permanently free tier)
- 24 GB RAM handles MLflow + PostgreSQL + MinIO with significant headroom
- 200 GB block storage accommodates years of experiment artifacts
- 10 TB/month egress enables paper artifact sharing (AWS free tier: 1 GB)
- Full MLflow control: custom plugins, direct PostgreSQL access, DuckDB analytics
- Pulumi-deployable: `pulumi up` in 5 minutes after one-time stack development
- Paper artifact: public `https://mlflow.minivess.fi` URL for reviewers

---

## 4. Decision: Oracle Cloud Always Free

### 4.1 Resource Allocation

Oracle Cloud Infrastructure (OCI) Always Free tier provides
([Oracle Always Free Resources](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)):

| Resource | Allocation | Our Usage |
|----------|-----------|-----------|
| **Compute** | VM.Standard.A1.Flex: 4 OCPU Arm Ampere, 24 GB RAM | MLflow + PostgreSQL + MinIO |
| **Boot + Block Storage** | 200 GB combined (up to 5 volumes) | 50 GB boot + 150 GB data |
| **Object Storage** | 20 GB, 50K API requests/month | Not primary — see §9 |
| **Outbound Data** | 10 TB/month | Paper reviewers, SkyPilot artifact pulls |
| **VCN** | 2 Virtual Cloud Networks | 1 for MLflow stack |
| **Load Balancer** | 1 Flexible (10 Mbps) | Optional — nginx on instance suffices |
| **Autonomous Database** | 2 instances, 20 GB each | Available but PostgreSQL in Docker preferred |
| **Monitoring** | 500M ingestion data points | Instance health metrics |

The 4 OCPU Arm Ampere A1 cores are based on the Ampere Altra processor (Neoverse N1
microarchitecture), delivering competitive single-threaded performance for server workloads
([Ampere Computing (2023). "Ampere Altra Family Technical Overview."](https://amperecomputing.com/products/processors/ampere-altra)).

### 4.2 Always Free vs. Trial Credits

| Aspect | Always Free | 30-Day Trial |
|--------|------------|-------------|
| Duration | **Permanent** | 30 days |
| Credits | N/A | $300 (US) |
| Resources | Fixed set (see table above) | Any OCI resource |
| After trial expires | Always Free continues | Paid resources terminated |
| Credit card | Required ($1 hold, refunded) | Same |
| Upgrade to PAYG | Always Free preserved | Recommended |

**Important**: After the 30-day trial, Always Free resources continue running at no cost.
Only resources provisioned beyond the free tier are terminated. Upgrading to Pay As You Go
(PAYG) preserves all Always Free resources and improves provisioning priority.

### 4.3 Account Setup Steps

1. Navigate to [https://cloud.oracle.com/](https://cloud.oracle.com/) → "Start for free"
2. Provide email, name, country. Verify email.
3. **Choose home region** (see §5–§6 — Frankfurt is viable with retry scripts)
4. Provide credit card (temporary $1 USD hold, refunded within days)
5. Set tenancy name (e.g., `minivess-research`)
6. Account activation: typically instant, occasionally up to 24 hours
7. Note down: Tenancy OCID, User OCID, Compartment OCID, Region identifier
8. Generate API signing key pair for Pulumi/CLI authentication:
   ```bash
   oci setup keys  # generates ~/.oci/oci_api_key.pem + oci_api_key_public.pem
   ```
9. Upload public key to OCI Console → User Settings → API Keys

### 4.4 Arm Architecture (aarch64) Implications

The A1.Flex shape uses Arm Ampere Altra processors (AArch64 ISA), not x86_64. This has
specific implications for Docker image compatibility:

| Component | Image | arm64 native? | Action needed |
|-----------|-------|:-------------:|---------------|
| PostgreSQL 16 | `postgres:16` | Yes | None |
| MinIO | `minio/minio:RELEASE.2025-02-08T19-54-51Z` | Yes | None |
| MinIO Client | `minio/mc:RELEASE.2025-02-08T19-54-51Z` | Yes | None |
| MLflow 3.10 | `ghcr.io/mlflow/mlflow:v3.10.0` | Yes | None |
| nginx | `nginx:alpine` | Yes | None |
| certbot | `certbot/certbot` | Yes | None |
| **minivess-base** | `nvidia/cuda:12.6.3-runtime-ubuntu24.04` | **No** | Not deployed here |

**Key insight**: All server-side images (MLflow, PostgreSQL, MinIO, nginx) already publish
multi-arch manifests with native arm64 variants. The `docker compose up` command on the
Oracle instance pulls arm64 images automatically via Docker's manifest-based resolution.

The `minivess-base:latest` training image (built on NVIDIA CUDA) is amd64-only, but
training never executes on the MLflow hosting instance — it runs on SkyPilot GPU VMs
(which are always amd64/x86_64). The XML plan task T1c (multi-arch Docker builds) is
therefore a **low-priority convenience**, not a deployment blocker.

---

## 5. Region Selection: Frankfurt vs. Milan

### 5.1 The Capacity Problem

Oracle Always Free A1.Flex instances are subject to chronic capacity constraints in popular
regions. The error `"Out of host capacity"` indicates physical servers are fully allocated
([Oracle Free Tier FAQ](https://www.oracle.com/cloud/free/faq/)).

Community reports confirm persistent shortages:

| Region | Capacity Status | Evidence |
|--------|----------------|----------|
| `eu-frankfurt-1` | **Severely constrained** | [Oracle Community (2024)](https://community.oracle.com/customerconnect/discussion/738457/ampere-out-of-capacity-eu-frankfurt-1) — weeks of retries |
| `us-ashburn-1` | **Severely constrained** | Most popular US region |
| `uk-london-1` | **Severely constrained** | Major EU/UK region |
| `eu-milan-1` | Good availability | Less popular, same EU jurisdiction |
| `eu-marseille-1` | Good availability | Newer French region |
| `eu-stockholm-1` | Good availability | Nordic region |
| `ap-osaka-1` | Good availability | Less popular APAC |
| `sa-saopaulo-1` | Good availability | Less popular LATAM |

### 5.2 Detailed Frankfurt vs. Milan Comparison

| Criterion | Frankfurt (`eu-frankfurt-1`) | Milan (`eu-milan-1`) |
|-----------|----------------------------|---------------------|
| A1.Flex capacity | **Poor** — chronic shortages, weeks of retries | **Good** — less popular region |
| E2.1.Micro capacity | **Good** — generally available | **Good** |
| Availability Domains | 3 (AD-1, AD-2, AD-3) | 1 |
| GDPR compliance | Yes (Germany) | Yes (Italy) |
| Latency to Finland | ~20 ms | ~35 ms |
| Latency to RunPod EU | ~5 ms | ~15 ms |
| Data center maturity | Oracle's flagship EU DC | Newer, less loaded |
| All required services | Yes | Yes |
| Peering quality | Excellent (DE-CIX Frankfurt) | Good |

### 5.3 Critical: Home Region Is Permanent

**You cannot change your home region after account creation.** Always Free resources
(A1.Flex, E2.1.Micro, Object Storage, Block Storage) are **locked to the home region**.
Subscribing to additional regions is possible, but non-home resources incur standard
charges — they are not free.

### 5.4 Second Account Policy

Oracle's policy is explicit: **"One Oracle Cloud Free Trial or Always Free account is
permitted per person"**
([Oracle Free Tier FAQ](https://www.oracle.com/cloud/free/faq/)). Enforcement is based on:

| Check | Enforcement |
|-------|-------------|
| Credit card | Must be unique per account (primary gate) |
| Phone number | Verified via SMS — must be unique |
| Name + address | Must match card billing info |
| Email | Must be unique (trivially bypassed) |

Creating a second account in a different region would require a different credit card in
your name. Oracle may suspend both accounts if detected. The safer path is to make
Frankfurt work (see §6).

---

## 6. Frankfurt Account: What You Get and How to Make It Work

### 6.1 Your Frankfurt Account Resources

Since you already registered with `eu-frankfurt-1`, your Always Free allocation is:

| Resource | Allocation | Status |
|----------|-----------|--------|
| **A1.Flex** (ARM) | 4 OCPU, 24 GB RAM | **Likely capacity-blocked** |
| **E2.1.Micro** (AMD x86_64) | 2 instances, 1/8 OCPU each, 1 GB each | **Usually available** |
| **Block Storage** | 200 GB total | Available once compute is provisioned |
| **Object Storage** | 20 GB, 50K API/month, S3-compatible | **Available immediately** |
| **Outbound Data** | 10 TB/month | Available |
| **VCN** | 2 networks | Available |
| **Autonomous Database** | 2 instances, 20 GB each | Available |
| **Load Balancer** | 1 Flexible (10 Mbps) | Available |

### 6.2 Strategy A: Get A1.Flex in Frankfurt (Recommended)

The A1.Flex is the best option (24 GB RAM, 4 OCPU). Getting it provisioned in Frankfurt
requires persistence:

**Step 1: Upgrade to Pay As You Go (PAYG)**

This is free and reportedly improves provisioning priority significantly. You still get
all Always Free resources at $0 — PAYG only charges if you exceed the free limits.

Console → Billing → Upgrade to Pay As You Go

**Step 2: Deploy automated retry script**

The most reliable approach is
[hitrov/oci-arm-host-capacity](https://github.com/hitrov/oci-arm-host-capacity) — a
PHP-based tool that polls the OCI API for capacity:

```bash
# Clone and configure
git clone https://github.com/hitrov/oci-arm-host-capacity.git
cd oci-arm-host-capacity
cp .env.example .env
# Edit .env with OCI credentials, shape, AD
```

Alternatively, run it as a **GitHub Actions workflow** (retries every 5–20 min, indefinitely,
at zero cost).

**Step 3: Start small, resize later**

Request **1 OCPU / 6 GB** first — smaller allocations succeed more often than 4/24.
Frankfurt has 3 availability domains; retry across all three. Once provisioned, resize
to 4 OCPU / 24 GB via Console → Instance → Edit (requires reboot).

**Step 4: Try off-peak hours**

Best availability: **01:00–05:00 CET** when European business usage drops. The retry
script handles this automatically if running 24/7.

**Expected timeline**: Days to weeks for Frankfurt. Some users report success within 24
hours after PAYG upgrade; others wait 1–2 weeks with retry scripts.

### 6.3 Strategy B: E2.1.Micro Fallback (Immediate, But Tight)

If A1.Flex is unavailable and you need MLflow running now, the 2x E2.1.Micro instances
are a functional (if constrained) fallback:

**E2.1.Micro specs** (per instance):

| Spec | Value |
|------|-------|
| CPU | 1/8 OCPU burstable (AMD EPYC) |
| RAM | 1 GB |
| Architecture | x86_64 (no arm64 concerns) |
| Network | 50 Mbps internet, 480 Mbps intra-region |
| Boot volume | 47 GB minimum (from 200 GB pool) |

**Can MLflow + PostgreSQL + MinIO run on E2.1.Micro?**

On a single 1 GB instance: **extremely tight but technically possible**.

| Service | Idle RAM | Notes |
|---------|----------|-------|
| PostgreSQL 16 | ~35 MB | With `shared_buffers=64MB` |
| MinIO | ~115 MB | Disable pre-allocation: `MINIO_CACHE=off` |
| MLflow 3.10 | ~400 MB | Single-worker `gunicorn` |
| OS + Docker | ~200 MB | Ubuntu minimal |
| **Total** | **~750 MB** | Leaves ~250 MB for bursts |

**Split across 2 instances** (better):

| Instance 1 | Instance 2 |
|------------|------------|
| PostgreSQL + MinIO | MLflow + nginx |
| ~350 MB used | ~600 MB used |
| Internal IP only | Public IP + TLS |

This works for a lightweight tracking server receiving metrics from 1–2 SkyPilot VMs.
**Not recommended for concurrent training runs** — the 50 Mbps bandwidth and 1 GB RAM
create bottlenecks during artifact upload. Add 2–4 GB swap as a safety net.

**Key advantage**: E2.1.Micro is **x86_64** — no arm64 image compatibility concerns.
All Docker images work without changes.

### 6.4 Strategy C: Wait for A1.Flex + Use Object Storage Now

While waiting for A1.Flex capacity, you can already use Oracle's S3-compatible Object
Storage (20 GB, available immediately) as a remote artifact store:

1. Create bucket `mlflow-artifacts` in OCI Console → Object Storage
2. Generate Customer Secret Key (OCI Console → User → Customer Secret Keys)
3. Configure MLflow locally to use OCI as artifact store:

```bash
# .env (local machine)
MLFLOW_S3_ENDPOINT_URL=https://<namespace>.compat.objectstorage.eu-frankfurt-1.oraclecloud.com
AWS_ACCESS_KEY_ID=<oci-customer-key-id>
AWS_SECRET_ACCESS_KEY=<oci-customer-secret>
```

This gives you 20 GB of cloud-accessible artifact storage immediately while keeping
MLflow metadata in local PostgreSQL. SkyPilot VMs can push artifacts to OCI Object Storage
directly. When A1.Flex becomes available, migrate to the full stack.

### 6.5 Recommendation: Frankfurt Is Viable

**Don't create a second account.** Frankfurt works — it just requires patience for A1.Flex:

1. **Immediately**: Upgrade to PAYG (free, improves priority)
2. **Immediately**: Set up retry script for A1.Flex (1 OCPU / 6 GB to start)
3. **Immediately**: Use Object Storage (20 GB) for artifact backup
4. **If urgent**: Deploy on 2x E2.1.Micro as interim solution
5. **Within 1–2 weeks**: A1.Flex should become available
6. **Once A1.Flex is up**: Migrate everything to the full 4 OCPU / 24 GB stack

Frankfurt's lower latency to Finland (~20 ms vs ~35 ms from Milan) and excellent peering
(DE-CIX) are genuine advantages once provisioned.

---

## 7. Storage Analysis Across All Deployment Targets

For the complete detailed storage analysis per provider, see the companion document:
[MLflow Deployment Targets: Detailed Storage Analysis](mlflow-deployment-storage-analysis.md).

### 7.1 Storage Comparison Summary

| Target | Block/Disk | Object Storage | Total Usable | Duration | Monthly Cost |
|--------|-----------|----------------|-------------|----------|-------------|
| **Oracle Always Free** | 200 GB (150 data) | 20 GB (S3-compat) | **170 GB** | **Forever** | **$0** |
| **Hetzner CX23** | 40 GB included | +EUR 4.99 for 1 TB | 40–1040 GB | Ongoing | EUR 4.50–9.49 |
| **Hetzner CX33** | 80 GB included | +EUR 4.99 for 1 TB | 80–1080 GB | Ongoing | EUR 8.00–12.99 |
| **DagsHub Free** | N/A (managed) | 20 GB (all-in) | 20 GB | Forever | $0 |
| **Cloudflare Tunnel** | Local disk (unlimited) | N/A | Unlimited | Forever | $0 |
| **AWS Free Tier** | 30 GB EBS | 5 GB S3 | 35 GB | **12 months** | $0 then ~$40 |
| **GCP Always Free** | 30 GB HDD | 5 GB Regional | 35 GB | Forever* | $0 |
| **Azure Free Tier** | 128 GB SSD | 5 GB Blob | 133 GB | **12 months** | $0 then ~$54 |

*GCP e2-micro is always free but has only 0.25 vCPU / 1 GB RAM — cannot run the stack.

### 7.2 Feasibility: Can It Run MLflow 3.10 + PostgreSQL 16 + MinIO?

The minimum viable stack requires ~1.7 GB idle RAM, ~3.5 GB under active artifact uploads:

| Target | RAM | vCPU | Feasible? | Notes |
|--------|-----|------|-----------|-------|
| **Oracle A1.Flex** | 24 GB | 4 | **Excellent** | 20+ GB headroom for Optuna, Grafana |
| **Oracle E2.1.Micro** (2x) | 2x 1 GB | 2x 1/8 | **Marginal** | Split services, add swap |
| **Hetzner CX23** | 4 GB | 2 | **Tight** | Solo researcher, light usage |
| **Hetzner CX33** | 8 GB | 4 | **Comfortable** | Best paid option |
| **DagsHub** | Managed | Managed | **Partial** | No PostgreSQL/MinIO access |
| **Cloudflare Tunnel** | Local | Local | **Yes** | 100 MB upload limit blocks SAM3 |
| **AWS t2.micro** | 1 GB | 1 | **No** | Insufficient RAM |
| **GCP e2-micro** | 1 GB | 0.25 | **No** | Insufficient RAM and CPU |
| **Azure B1s** | 1 GB | 1 | **No** | Insufficient RAM |

### 7.3 Oracle Storage Architecture Detail

The 200 GB block storage is a **combined pool** for boot volumes + data volumes:

| Allocation Strategy | Boot Volume | Data Volume | Free Headroom |
|---------------------|-------------|-------------|---------------|
| Max data (recommended) | 50 GB (minimum) | 150 GB attached | 0 GB |
| Balanced | 100 GB | 100 GB attached | 0 GB |
| All-in-one | 200 GB boot | None | 0 GB |

**Block storage** hosts Docker volumes: PostgreSQL data, MinIO artifacts, MLflow state.
**Object Storage** (20 GB, S3-compatible) serves as off-instance backup for disaster recovery.

OCI Object Storage S3 endpoint:
```
https://<namespace>.compat.objectstorage.eu-frankfurt-1.oraclecloud.com/<bucket>/<object>
```

Auth uses OCI Customer Secret Keys (compatible with `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`).
Tools: boto3, `aws s3`, `mc` (MinIO Client), rclone all work.

### 7.4 Cloudflare Tunnel Upload Limit Detail

The 100 MB per-request limit is per Cloudflare plan, **not bypassable**:

| Cloudflare Plan | Max Upload | Annual Cost |
|-----------------|-----------|-------------|
| Free | 100 MB | $0 |
| Pro | 100 MB | $240 |
| Business | 200 MB | $2,400 |
| Enterprise | 500 MB (negotiable) | Custom |

Affected MLflow artifacts:

| Artifact | Size | Blocked? |
|----------|------|----------|
| DynUNet checkpoint | 15–50 MB | No |
| SegResNet checkpoint | 30–80 MB | Borderline |
| SAM3 checkpoint | 600 MB–2.5 GB | **Yes** |
| ONNX export (SAM3) | 800 MB–2.5 GB | **Yes** |
| Config YAML | ~5 KB | No |
| Metrics/Figures | < 1 MB | No |

### 7.5 Five-Year Total Cost Comparison

| Target | Year 1 | Year 2+ (annual) | 5-Year Total |
|--------|--------|-------------------|-------------|
| **Oracle Always Free** | $0 | $0 | **$0** |
| **Hetzner CX23** | EUR 54 | EUR 54 | **EUR 270** |
| **Hetzner CX33** | EUR 96 | EUR 96 | **EUR 480** |
| **DagsHub Free** | $0 | $0 | **$0** (20 GB limit) |
| **Cloudflare Tunnel** | $0 | $0 | **$0** (100 MB limit) |
| **AWS** | ~$0 | ~$480 | **~$1,920** |
| **GCP** (unusable) | $0 | ~$486 | **~$1,944** |
| **Azure** | ~$0 | ~$648 | **~$2,592** |

---

## 8. Architecture: Current Docker Stack on ARM

### 8.1 Current Local Architecture

```
┌─────────────────── Docker Compose (localhost) ───────────────────┐
│                                                                   │
│  ┌──────────┐    ┌──────────────┐    ┌──────────┐               │
│  │PostgreSQL│◄───│  MLflow 3.10 │───►│  MinIO   │               │
│  │   :5432  │    │    :5000     │    │  :9000   │               │
│  └──────────┘    └──────────────┘    └──────────┘               │
│       │               ▲                   ▲                      │
│       │               │                   │                      │
│  ┌────┴────┐    ┌─────┴──────┐    ┌──────┴───────┐             │
│  │postgres_│    │mlflow_     │    │ minio_data   │             │
│  │  data   │    │artifacts   │    │              │             │
│  └─────────┘    └────────────┘    └──────────────┘             │
│                                                                   │
│  ┌──────────────────────────────────────────────┐               │
│  │  Flow containers (train, analyze, deploy...) │               │
│  │  → MLFLOW_TRACKING_URI=http://minivess-mlflow:5000          │
│  │  → MLFLOW_S3_ENDPOINT_URL=http://minio:9000                 │
│  └──────────────────────────────────────────────┘               │
│                                                                   │
│  Network: minivess-network (bridge)                              │
└──────────────────────────────────────────────────────────────────┘
```

### 8.2 Oracle Remote Architecture

```
┌──────── Oracle Cloud A1.Flex (Frankfurt) ────────────┐
│  4 OCPU ARM Ampere │ 24 GB RAM │ 200 GB block        │
│                                                        │
│  ┌──────────┐    ┌──────────────┐    ┌──────────┐    │
│  │PostgreSQL│◄───│  MLflow 3.10 │───►│  MinIO   │    │
│  │   :5432  │    │    :5000     │    │  :9000   │    │
│  └──────────┘    └──────────────┘    └──────────┘    │
│                        ▲                               │
│                        │ reverse proxy                 │
│                  ┌─────┴──────┐                        │
│                  │   nginx    │                        │
│                  │   :443     │ ← Let's Encrypt TLS   │
│                  │ + htpasswd │ ← Basic auth           │
│                  └────────────┘                        │
│                        ▲                               │
│  Network: Docker bridge (internal)                     │
└────────────────────────┼───────────────────────────────┘
                         │ HTTPS :443
            ┌────────────┼────────────┐
            │            │            │
    ┌───────┴──┐  ┌──────┴───┐  ┌────┴─────┐
    │ SkyPilot │  │ Local    │  │ Paper    │
    │ GPU VMs  │  │ dev      │  │ reviewers│
    │ (RunPod) │  │ machine  │  │ (public) │
    └──────────┘  └──────────┘  └──────────┘
```

### 8.3 What Changes vs. Local

| Aspect | Local | Oracle Remote |
|--------|-------|---------------|
| MLflow URI | `http://localhost:5000` | `https://mlflow.minivess.fi` |
| TLS | None | Let's Encrypt via nginx |
| Auth | None | Basic auth (htpasswd) via nginx |
| Artifact store | MinIO on localhost | MinIO on Oracle instance |
| S3 endpoint (server-side) | `http://minio:9000` | `http://minio:9000` (unchanged — internal) |
| PostgreSQL | `postgres:5432` (internal) | `postgres:5432` (internal — unchanged) |
| Docker images | Same | Same (native arm64) |
| `docker-compose.yml` | Used directly | Used with remote overrides (TLS, domain) |

The internal Docker networking is **identical**. MLflow still talks to `minio:9000` and
`postgres:5432` over the Docker bridge. The only additions are the nginx reverse proxy
(TLS termination + basic auth) and DNS configuration.

### 8.4 Current MLflow Image Details

The custom MLflow image (`deployment/docker/Dockerfile.mlflow`) adds two packages to the
official base:

```dockerfile
FROM ghcr.io/mlflow/mlflow:v3.10.0
RUN pip install --no-cache-dir \
    psycopg2-binary==2.9.10 \   # PostgreSQL driver
    boto3==1.34.162              # S3/MinIO client
```

Both `psycopg2-binary` and `boto3` publish arm64 wheels. The custom image builds on
ARM without changes.

### 8.5 Environment Variables for Remote Deployment

The `.env.example` already defines the required variables. For Oracle deployment, only
the *host-side* values change:

```bash
# .env (local dev)
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_DOCKER_HOST=minivess-mlflow
MINIO_DOCKER_HOST=minio

# .env (Oracle remote — on the instance itself)
MLFLOW_TRACKING_URI=http://minivess-mlflow:5000   # internal, unchanged
MLFLOW_DOCKER_HOST=minivess-mlflow                 # internal, unchanged
MINIO_DOCKER_HOST=minio                            # internal, unchanged

# .env (SkyPilot VMs / local dev connecting to remote)
MLFLOW_TRACKING_URI=https://mlflow.minivess.fi
MLFLOW_TRACKING_USERNAME=minivess
MLFLOW_TRACKING_PASSWORD=<secret>
```

---

## 9. Artifact Storage Strategy

### 9.1 Decision: MinIO on Instance (Option A)

Two artifact storage options were evaluated:

| | Option A: MinIO on Instance | Option B: OCI Object Storage |
|---|---|---|
| Storage limit | ~150 GB (block volume) | 20 GB (free tier) |
| API requests | Unlimited | 50K/month |
| Setup complexity | Zero changes to compose | Requires S3 endpoint + auth changes |
| Durability | Single disk (instance loss = data loss) | Oracle-managed (11 nines) |
| Cost | $0 (included in compute) | $0 (within free tier) |
| Latency | Sub-millisecond (same host) | ~5 ms (network) |

**Selected: Option A (MinIO on instance)**

Rationale:
- 150 GB >> 20 GB — free Object Storage is too small for years of experiments
- 50K API requests/month would be exhausted quickly by MLflow's per-metric write pattern
- Zero configuration changes to `docker-compose.yml`
- Data loss risk is mitigated by local `mlruns/` copies and periodic `mlflow` exports

### 9.2 OCI Object Storage as Backup (Optional)

OCI Object Storage provides S3-compatible API access
([OCI S3 Compatibility API](https://docs.oracle.com/en-us/iaas/Content/Object/Tasks/s3compatibleapi.htm))
using AWS Signature Version 4:

```
Endpoint: https://<namespace>.compat.objectstorage.<region>.oci.customer-oci.com
Auth:     OCI Customer Secret Keys (compatible with AWS_ACCESS_KEY_ID/SECRET_ACCESS_KEY)
Supported: GetObject, PutObject, Multipart Upload, ListObjects, etc.
```

This could serve as a **periodic backup target** for critical artifacts (champion models,
final paper figures) rather than the primary artifact store. A cron job running
`mc mirror` from the instance MinIO to OCI Object Storage provides off-instance durability
within the 20 GB free limit.

### 9.3 Artifact Size Estimates

| Artifact Type | Per-Run Size | Runs/Year (est.) | Annual Total |
|---------------|-------------|-------------------|-------------|
| Model checkpoints (.pth) | 15–50 MB (DynUNet) | ~200 | 3–10 GB |
| Model checkpoints (.pth) | 600–2500 MB (SAM3) | ~50 | 30–125 GB |
| Resolved configs (.yaml) | ~5 KB | ~250 | ~1.2 MB |
| Metrics (JSONL) | ~50 KB | ~250 | ~12 MB |
| Figures (PNG/SVG) | ~500 KB | ~250 | ~125 MB |
| ONNX exports | 50–800 MB | ~20 | 1–16 GB |
| **Total (DynUNet-heavy)** | | | **~15 GB/year** |
| **Total (SAM3-heavy)** | | | **~100 GB/year** |

With 150 GB available, the instance comfortably holds 1–10 years of experiments depending
on model mix. SAM3-heavy workflows may require periodic cleanup of non-champion runs.

### 9.4 MinIO Bucket Configuration

The existing `minio-init` service auto-creates required buckets on first startup:

```yaml
# deployment/docker-compose.yml — minio-init service
entrypoint: /bin/sh -c "
  mc alias set minio http://minio:9000 $$MINIO_ROOT_USER $$MINIO_ROOT_PASSWORD;
  mc mb --ignore-existing minio/mlflow-artifacts;
  mc mb --ignore-existing minio/model-registry;
"
```

No changes needed for Oracle deployment — the init service runs identically on ARM.

---

## 10. Security Architecture

### 10.1 Network Security

The Oracle instance exposes only two ports to the internet:

```
┌─── OCI Security List ───┐
│                          │
│  Ingress Rules:          │
│    TCP 443  (HTTPS)  ✓   │  ← MLflow UI + API (via nginx)
│    TCP 22   (SSH)    ✓   │  ← Administration (key-only)
│    *        (other)  ✗   │  ← All other ports blocked
│                          │
│  Egress Rules:           │
│    *        (all)    ✓   │  ← Docker pulls, apt, etc.
│                          │
└──────────────────────────┘
```

PostgreSQL (5432) and MinIO (9000, 9001) are **not exposed** — they communicate only over
the internal Docker bridge network. MLflow's port 5000 is proxied through nginx on 443.

### 10.2 TLS Configuration

Let's Encrypt certificates via certbot with nginx plugin:

```nginx
# /etc/nginx/sites-available/mlflow
server {
    listen 443 ssl http2;
    server_name mlflow.minivess.fi;

    ssl_certificate     /etc/letsencrypt/live/mlflow.minivess.fi/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mlflow.minivess.fi/privkey.pem;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;

        # Large artifact uploads (SAM3 checkpoints)
        client_max_body_size 3G;
        proxy_read_timeout 600s;
    }
}
```

Certificate auto-renewal via certbot cron (runs every 12 hours by default).

### 10.3 Authentication

**Phase 1 (MVP)**: nginx basic auth via htpasswd

```bash
htpasswd -c /etc/nginx/.htpasswd minivess
# MLflow Python client authenticates via:
# MLFLOW_TRACKING_USERNAME=minivess
# MLFLOW_TRACKING_PASSWORD=<secret>
```

**Phase 2 (optional)**: MLflow 3.x built-in auth (`mlflow-secure` profile in
docker-compose.yml) with per-user permissions. Currently available but not deployed.

### 10.4 SSH Hardening

- Key-only authentication (password auth disabled)
- fail2ban for brute-force protection
- Non-standard SSH port (optional, reduces log noise)

---

## 11. Operational Concerns

### 11.1 Idle Instance Reclamation

Oracle may reclaim Always Free instances after a 7-day evaluation period if **all three**
conditions are met simultaneously
([Oracle Free Tier FAQ](https://www.oracle.com/cloud/free/faq/)):

| Metric | Reclamation Threshold | MLflow Stack Typical |
|--------|----------------------|---------------------|
| CPU utilization (95th percentile) | < 20% | 5–15% baseline |
| Network utilization | < 20% | Variable |
| Memory utilization | < 20% | ~30–50% (PostgreSQL + MinIO + MLflow) |

**Risk assessment**: Memory utilization alone should keep the instance above the 20%
threshold — PostgreSQL and MinIO both maintain significant resident memory. However, as
a safety measure:

```bash
# /etc/cron.d/keep-alive — ping MLflow API every 4 hours
0 */4 * * * curl -sf https://mlflow.minivess.fi/health > /dev/null
```

Upgrading to PAYG ($0 additional cost) reportedly eliminates the reclamation policy
entirely.

### 11.2 Backup Strategy

| What | How | Frequency |
|------|-----|-----------|
| PostgreSQL dump | `pg_dump mlflow > backup.sql.gz` | Daily cron |
| MinIO artifacts | `mc mirror minio/mlflow-artifacts /backup/` | Weekly cron |
| Full instance | OCI Boot Volume Backup (5 free) | Weekly |
| Off-instance | `rsync` to local dev machine | Monthly or pre-paper |

The local dev machine also maintains `mlruns/` — in a disaster recovery scenario, the
Oracle instance can be re-provisioned via `pulumi up` and data restored from local copies
or OCI volume backups.

### 11.3 Monitoring

Oracle Cloud provides free monitoring (500M ingestion data points/month):
- CPU, memory, network, disk I/O metrics via OCI Monitoring
- Alerting via OCI Notifications (1M HTTPS + 1K emails/month free)

Application-level monitoring:
- MLflow `/health` endpoint for uptime checks
- PostgreSQL `pg_isready` for database health
- MinIO `mc admin info` for storage health

### 11.4 Maintenance

| Task | Frequency | Downtime |
|------|-----------|----------|
| `apt upgrade` | Monthly | ~2 min restart |
| Docker image updates | Quarterly | ~5 min `docker compose pull && up -d` |
| Let's Encrypt renewal | Auto (every 60 days) | Zero (graceful reload) |
| PostgreSQL vacuum | Auto (autovacuum) | Zero |
| Disk usage check | Monthly | Zero |

---

## 12. Network Topology: SkyPilot ↔ Oracle MLflow

### 12.1 Data Flow During Training

```
┌─────────────────┐         ┌──────────────────────┐
│  SkyPilot VM    │         │  Oracle Cloud (Milan) │
│  (RunPod A100)  │         │                        │
│                 │  HTTPS  │  ┌──────────────────┐  │
│  Training loop ─┼────────►│  │   nginx :443     │  │
│  mlflow.log_*() │         │  │   ↓ proxy_pass   │  │
│                 │         │  │   MLflow :5000    │  │
│  Artifact push ─┼────────►│  │   ↓ S3 write     │  │
│  boto3 PUT      │         │  │   MinIO :9000     │  │
│                 │         │  └──────────────────┘  │
└─────────────────┘         └──────────────────────┘
         │                            │
         │                            │
         ▼                            ▼
┌─────────────────┐         ┌──────────────────────┐
│  Local Dev      │  HTTPS  │  Paper Reviewers     │
│  machine        │────────►│  (browser, read-only) │
│  (monitoring)   │         │                        │
└─────────────────┘         └──────────────────────┘
```

### 12.2 Artifact Upload Path

When a SkyPilot VM calls `mlflow.log_artifact()`:

1. MLflow client sends metadata to MLflow server (`POST /api/2.0/mlflow/runs/log-artifact`)
2. MLflow server returns a pre-signed S3 URL pointing to MinIO
3. Client uploads artifact directly to MinIO via the pre-signed URL
4. **Problem**: MinIO is on `localhost:9000` inside the Oracle instance — not directly
   reachable from SkyPilot VMs

**Solution**: Configure MLflow to use **proxied artifact access** (MLflow 2.x+):

```bash
mlflow server \
  --serve-artifacts \          # Proxy artifacts through MLflow server
  --artifacts-destination s3://mlflow-artifacts/ \
  --default-artifact-root mlflow-artifacts:/  # Use proxy URI scheme
```

With `--serve-artifacts`, all artifact I/O flows through the MLflow server (port 5000,
proxied via nginx on 443). The client never needs direct MinIO access. This eliminates
the need to expose MinIO's port 9000 to the internet.

**Trade-off**: Artifact uploads are slightly slower (extra hop through MLflow), but
security is significantly improved (only port 443 exposed). For typical artifact sizes
(< 100 MB), the overhead is negligible over the 4 Gbps Oracle network bandwidth.

For SAM3 checkpoints (600 MB–2.5 GB), uploads may take 10–60 seconds at typical cloud
upload speeds — acceptable for end-of-training artifact logging.

### 12.3 SkyPilot YAML Configuration

```yaml
# deployment/skypilot/train_generic.yaml (excerpt)
envs:
  MLFLOW_TRACKING_URI: https://mlflow.minivess.fi
  MLFLOW_TRACKING_USERNAME: minivess
  EXPERIMENT: debug_single_model

file_mounts:
  /secrets/.env:
    source: .env.skypilot   # Contains MLFLOW_TRACKING_PASSWORD

setup: |
  pip install mlflow boto3
  # No MinIO endpoint needed — artifacts proxied through MLflow server

run: |
  source /secrets/.env
  python -m minivess.orchestration.flows.train_flow
```

---

## 13. Implementation Roadmap

Mapped to XML plan tasks (T1, T1b, T1c):

### Phase 0: Account Setup (Manual, 30 min)

| Step | Action | Blocker |
|------|--------|---------|
| 0.1 | Create OCI account (Milan region) | None |
| 0.2 | Generate API signing key pair | OCI account |
| 0.3 | Note Tenancy/User/Compartment OCIDs | OCI account |
| 0.4 | Configure Pulumi OCI provider auth | API keys |

### Phase 1: Pulumi Stack Development (T1b, 4h)

| Step | Action | Deliverable |
|------|--------|-------------|
| 1.1 | Create `deployment/pulumi/Pulumi.yaml` | Project definition |
| 1.2 | Write `__init__.py` — VCN, subnet, security list | Network infrastructure |
| 1.3 | Write `__init__.py` — A1.Flex instance + cloud-init | Compute + Docker install |
| 1.4 | Write `cloud-init.yaml` — Docker, nginx, certbot, fail2ban | Server provisioning |
| 1.5 | Write remote-exec — copy compose, `docker compose up` | Service deployment |
| 1.6 | Configure DNS (pulumi-cloudflare) | `mlflow.minivess.fi` |
| 1.7 | Export stack outputs: mlflow_url, postgres_conn | Pulumi outputs |
| 1.8 | `pulumi up --stack dev` → verify MLflow accessible | Live deployment |

### Phase 2: Integration (T2 + existing tasks, 2h)

| Step | Action | Deliverable |
|------|--------|-------------|
| 2.1 | Add OCI env vars to `.env.example` | Configuration contract |
| 2.2 | Add `MLFLOW_TRACKING_URI_REMOTE` to `.env.example` | SkyPilot URI |
| 2.3 | Update `resolve_tracking_uri()` if needed | Client compatibility |
| 2.4 | Test: local machine → Oracle MLflow logging | E2E validation |
| 2.5 | Test: mock SkyPilot env → Oracle MLflow logging | Compute offload readiness |

### Phase 3: Multi-Arch Builds (T1c, 2h, LOW PRIORITY)

Only needed if CPU flow images must run on Oracle ARM. Not required for MLflow hosting.

| Step | Action | Deliverable |
|------|--------|-------------|
| 3.1 | Add `docker buildx create --use` to Makefile | Build infrastructure |
| 3.2 | Add `make docker-build-multiarch` target | Multi-platform builds |
| 3.3 | Test arm64 build for `minivess-base-light` | Dashboard on ARM |

**Total effort**: ~6.5h (0.5h manual + 4h Pulumi + 2h integration). After initial setup,
subsequent deployments are `pulumi up` (5 min) or `pulumi destroy` + `pulumi up` for
rebuild.

---

## 14. Paper Artifact Sharing

### 14.1 Opportunity

A public MLflow instance at `https://mlflow.minivess.fi` serves as a **living supplement**
to the paper submission. Reviewers can:

- Browse all experiment runs (losses, metrics, hyperparameters)
- Compare models interactively (MLflow Compare Runs UI)
- Download artifacts (configs, checkpoints, figures)
- Verify reproducibility claims against logged system info

This aligns with growing expectations for computational reproducibility in ML research
([Pineau et al. (2021). "Improving Reproducibility in Machine Learning Research." *JMLR*, 22(164), 1–20.](https://jmlr.org/papers/v22/20-303.html);
[Gundersen & Kjensmo (2018). "State of the Art: Reproducibility in Artificial Intelligence." *AAAI*.](https://ojs.aaai.org/index.php/AAAI/article/view/11503)).

### 14.2 Access Control for Reviewers

**During review**: Read-only basic auth credentials shared in the paper's supplementary
materials or via anonymous review portal. The htpasswd file can include a dedicated
`reviewer` user with a separate password from the `minivess` admin account.

**Post-publication**: Consider making the instance fully public (no auth) as a permanent
research artifact. The 10 TB/month egress supports significant traffic.

### 14.3 What Reviewers See

| MLflow Feature | Reviewer Value |
|----------------|---------------|
| Experiment list | All training campaigns (loss variation, HPO, ablations) |
| Run comparison | Side-by-side metric plots (Dice, clDice, HD95) |
| Parameter table | Full hyperparameter configs per run |
| Artifact browser | Resolved YAML configs, training curves, model checksums |
| System tags | GPU model, CUDA version, PyTorch version, git hash |
| Model registry | Champion models with stage labels |

### 14.4 Egress Budget

At 10 TB/month free egress, even heavy reviewer traffic is well within limits:

| Scenario | Data per session | Sessions/month | Monthly egress |
|----------|-----------------|----------------|---------------|
| UI browsing | ~5 MB | 100 | 500 MB |
| Download 1 checkpoint | ~50 MB | 20 | 1 GB |
| Download all artifacts | ~5 GB | 5 | 25 GB |
| **Total** | | | **~26.5 GB** |

This is 0.26% of the 10 TB monthly allowance.

---

## 15. References

1. [Zaharia, M. et al. (2018). "Accelerating the Machine Learning Lifecycle with MLflow." *IEEE Data Eng. Bull.*, 41(4).](https://www.scinapse.io/papers/2952596791)

2. [Oracle Cloud Infrastructure (2026). "Always Free Resources."](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)

3. [Oracle Cloud Infrastructure (2026). "Free Tier FAQ."](https://www.oracle.com/cloud/free/faq/)

4. [Oracle Cloud Infrastructure (2026). "Amazon S3 Compatibility API."](https://docs.oracle.com/en-us/iaas/Content/Object/Tasks/s3compatibleapi.htm)

5. [Oracle Cloud Infrastructure (2026). "Compute Shapes."](https://docs.oracle.com/en-us/iaas/Content/Compute/References/computeshapes.htm)

6. [Ampere Computing (2023). "Ampere Altra Family Technical Overview."](https://amperecomputing.com/products/processors/ampere-altra)

7. [Pineau, J. et al. (2021). "Improving Reproducibility in Machine Learning Research." *JMLR*, 22(164), 1–20.](https://jmlr.org/papers/v22/20-303.html)

8. [Gundersen, O. E. & Kjensmo, S. (2018). "State of the Art: Reproducibility in Artificial Intelligence." *AAAI*.](https://ojs.aaai.org/index.php/AAAI/article/view/11503)

9. [Cloudflare (2026). "Workers Platform Limits — Upload Limits."](https://developers.cloudflare.com/workers/platform/limits/)

10. [Hitrov, A. (2024). "Resolving Oracle Cloud Out of Capacity Issue." *Medium*.](https://hitrov.medium.com/resolving-oracle-cloud-out-of-capacity-issue-and-getting-free-vps-with-4-arm-cores-24gb-of-a3d7e6a027a8)

11. [hitrov/oci-arm-host-capacity (2024). GitHub.](https://github.com/hitrov/oci-arm-host-capacity)

12. [Oracle Community (2024). "Ampere Out of Capacity eu-frankfurt-1."](https://community.oracle.com/customerconnect/discussion/738457/ampere-out-of-capacity-eu-frankfurt-1)

13. [MLflow Documentation (2026). "MLflow Tracking — Artifact Stores."](https://mlflow.org/docs/latest/tracking/artifacts-stores.html)

14. [Kreuzberger, D. et al. (2023). "Machine Learning Operations (MLOps): Overview, Definition, and Architecture." *IEEE Access*, 11, 31866–31879.](https://doi.org/10.1109/ACCESS.2023.3262138)

15. [Oracle Cloud Infrastructure (2026). "Signing Up for Oracle Cloud Free Tier."](https://docs.oracle.com/en-us/iaas/Content/GSG/Tasks/signingup_topic-Sign_Up_for_Free_Oracle_Cloud_Promotion.htm)

16. [Oracle Cloud Infrastructure (2026). "Managing Regions."](https://docs.oracle.com/en-us/iaas/Content/Identity/Tasks/managingregions.htm)

17. [Oracle Cloud Infrastructure (2026). "Changing the Shape of an Instance."](https://docs.oracle.com/en-us/iaas/Content/Compute/Tasks/resizinginstances.htm)

18. [Oracle Community (2024). "Host Capacity in Frankfurt."](https://community.oracle.com/customerconnect/discussion/593801/host-capacity-in-frankfurt)

19. [Hetzner (2026). "Cloud Servers."](https://www.hetzner.com/cloud/)

20. [Hetzner (2026). "Object Storage."](https://www.hetzner.com/storage/object-storage/)

21. [DagsHub (2026). "Pricing."](https://dagshub.com/pricing/)

22. [DagsHub (2026). "MLflow Integration Guide."](https://dagshub.com/docs/integration_guide/mlflow_tracking/)

23. [AWS (2026). "Free Tier FAQ."](https://aws.amazon.com/free/free-tier-faqs/)

24. [Google Cloud (2026). "Free Cloud Features."](https://cloud.google.com/free)

25. [Azure (2026). "Free Services."](https://azure.microsoft.com/en-us/pricing/free-services)

26. [MinIO (2025). "RAM Requirements Discussion." GitHub Discussions #19133.](https://github.com/minio/minio/discussions/19133)

27. [MLflow (2023). "Memory Usage Issue #5332." GitHub.](https://github.com/mlflow/mlflow/issues/5332)
