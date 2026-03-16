# Lambda Labs MLflow Artifact Failure: Root Cause & Multi-Cloud Architecture Decision

> **Status**: Active decision needed (2026-03-14)
> **Priority**: P0 — Pipeline non-functional without artifact uploads
> **Blocking**: Training completes but checkpoints can't reach MLflow → downstream flows broken

---

## User Prompt (Verbatim)

> Well we have essentially NONFUNCTIONAL pipeline if the .pth artifacts cannot move to our MLflow on UpCloud :S This is a critical failure as the whole point of the MLFlow artifact store is to train .pth files that we are using then later in our Prefect Flow. Create analysis, and optimize with reviewer agents and analyze if we really have to go towards "real big 3 clouds" (Azure, AWS, GCP) to have all the pieces there. Skypilot does not yet work with Hetzner then? If the issue is UpCloud only, I can try switching to Hetzner for the MLflow and S3 storage hosting. Obviously it would be in the end easiest have the MLflow and S3 hosted on same cloud with the Skypilot connection existing. Is this only now possible with the big clouds? I think MLflow, S3 and GPU instances are all available from Hetzner, but how about the Skypilot compatibility, or should we switch to dstack from SkyPilot. It is imperative that we use Pulumi for MLflow and S3, and Skypilot-like (can be dstack or something similar) orchestrator for multi-cloud job launcher, but the Docker image and large MLflow artifacts could be closer to the GPU instance right? What do you think? Do a deep exploration and with these constraints, do an open-ended multi-hypothesis decision matrix analysis and see if we need to change.

---

## Table of Contents

1. [Root Cause: Why the 910 MB Upload Fails](#1-root-cause)
2. [Quick Fix: Multipart Upload on UpCloud (Hypothesis A)](#2-quick-fix)
3. [Architecture Hypotheses](#3-hypotheses)
4. [Provider Capability Matrix](#4-capability-matrix)
5. [Orchestrator Comparison: SkyPilot vs dstack](#5-orchestrators)
6. [GCP Deep Dive (Preferred Big-3 Cloud)](#6-gcp)
7. [Hetzner Analysis](#7-hetzner)
8. [Spot Instance Checkpointing & Auto-Resume](#8-spot-recovery)
9. [Docker Registry Proximity](#9-registry)
10. [Decision Matrix](#10-decision-matrix)
11. [Recommendation](#11-recommendation)

---

## 1. Root Cause: Why the 910 MB Upload Fails {#1-root-cause}

### Current UpCloud MLflow Architecture

```
┌─────────────┐       HTTP PUT (910 MB)        ┌──────────────┐       S3 PUT        ┌──────────────┐
│ Lambda A100  │ ───────────────────────────► │  MLflow VPS   │ ──────────────────► │ UpCloud S3   │
│ (us-east-1)  │    via proxy (~100ms RTT)     │ 2vCPU, 4GB   │    (localhost)      │ (Helsinki)   │
│              │                               │  Helsinki     │                    │              │
└─────────────┘                               └──────────────┘                    └──────────────┘
```

### The Failure Chain

1. Our MLflow server runs with `--serve-artifacts --artifacts-destination s3://mlflow-artifacts`
2. `--serve-artifacts` means the **MLflow server proxies** all artifact uploads — the
   training client uploads to the MLflow server, which then forwards to S3
3. For files >500 MB, MLflow supports **multipart upload** — but it requires:
   ```
   MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=true
   ```
4. **This env var is NOT SET** on our UpCloud MLflow server
5. Without multipart, the 910 MB file is sent as a single HTTP PUT through the tiny VPS
6. The VPS (2 vCPU, 4 GB RAM) runs out of memory/timeout → HTTP 500 errors
7. MLflow client retries → "too many 500 error responses"

### Why Metrics Work But Artifacts Don't

- **Metrics/params**: Small JSON payloads (~KB) → stored in PostgreSQL (fast, reliable)
- **Artifacts**: Large binary files (~910 MB) → proxied through VPS to S3 (fails)

This is a **proxy bottleneck**, not a storage problem. The S3 bucket itself can handle
the data — the VPS can't proxy it.

---

## 2. Quick Fix: Multipart Upload on UpCloud (Hypothesis A) {#2-quick-fix}

### The One-Line Fix

Add to the MLflow Docker Compose environment on UpCloud:

```yaml
environment:
  MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD: "true"
  # Optional tuning:
  MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE: "104857600"  # 100 MB chunks (default)
  MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT: "1800"   # 30 min timeout
```

With multipart enabled, the MLflow server generates **presigned S3 URLs** and the
client uploads chunks **directly to S3**, bypassing the VPS proxy entirely.

### Why This Might Still Be Insufficient

Even with multipart upload:
- **Network path**: Lambda (US) → UpCloud S3 (Helsinki) = ~100ms RTT, ~5000 km
- **Upload speed**: Depends on UpCloud S3 ingest bandwidth (shared infrastructure)
- **910 MB at realistic cloud-to-cloud speeds**: 2-10 minutes per checkpoint
- **VPS RAM**: 4 GB may still struggle with orchestrating multipart chunks

### Verdict on Quick Fix

**Worth trying first** (5 minutes of work). But this is a bandaid — the fundamental
issue is that compute (US) and storage (Helsinki) are on different continents.

---

## 3. Architecture Hypotheses {#3-hypotheses}

### Hypothesis A: Fix UpCloud MLflow (Quick Fix)

**Change**: Add `MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=true` to UpCloud
**Cost**: $0 (already paying for UpCloud)
**Risk**: May still fail due to cross-Atlantic bandwidth and small VPS
**Effort**: 5 minutes (Pulumi config change)

### Hypothesis B: Move MLflow + S3 to Hetzner

**Change**: Migrate MLflow server and S3 to Hetzner (EU, cheaper)
**Cost**: €3.79/month (CX22 VPS) + €4.99/TB (Object Storage)
**Risk**: Hetzner has NO SkyPilot/dstack support for GPU compute
**Effort**: ~2 hours (new Pulumi stack)
**Problem**: Still cross-Atlantic to Lambda/GCP GPUs

### Hypothesis C: Move Everything to GCP

**Change**: MLflow on Cloud Run + GCS artifacts + Cloud SQL + SkyPilot GCP spot GPUs
**Cost**: ~$50-100/month base + GPU usage
**Risk**: Higher base cost but fastest artifact path
**Effort**: ~4-8 hours (new Pulumi stack + GCP setup)
**Advantage**: Same-region compute + storage → artifacts upload in seconds

### Hypothesis D: GCP for Compute + Storage, Keep UpCloud for MLflow UI

**Change**: Train on GCP spot, save artifacts to GCS, MLflow points to GCS
**Cost**: ~$20/month (GCS) + GPU usage
**Risk**: Medium — MLflow still on UpCloud but artifacts bypass it
**Effort**: ~3-4 hours

### Hypothesis E: Direct S3 Upload (Skip Proxy)

**Change**: Configure training to upload artifacts directly to S3, bypass MLflow proxy
**Cost**: $0 (infrastructure unchanged)
**Risk**: Requires training code to have S3 credentials + endpoint
**Effort**: ~2 hours (code changes)
**Problem**: Clients need S3 credentials, breaks the proxy abstraction

### Hypothesis F: Switch to dstack + Hetzner-native

**Change**: Replace SkyPilot with dstack, use Hetzner for everything
**Cost**: €3.79/month + Hetzner GPU pricing
**Risk**: dstack does NOT support Hetzner. Hetzner GPUs are dedicated servers only (no spot).
**Verdict**: **NOT VIABLE** — neither SkyPilot nor dstack supports Hetzner

---

## 4. Provider Capability Matrix {#4-capability-matrix}

### What We Need (All Must Be Present)

| Requirement | Description |
|-------------|-------------|
| **GPU Compute** | A100/H100 or equivalent, ideally spot for cost savings |
| **Docker** | `image_id: docker:` must work (VM-based, not container-based) |
| **S3/GCS Storage** | For MLflow artifacts (checkpoints ~1 GB each) |
| **PostgreSQL** | For MLflow backend + Optuna |
| **MLflow Server** | Tracking + artifact proxy |
| **Pulumi IaC** | Infrastructure as code for all infra |
| **SkyPilot/dstack** | Multi-cloud job orchestration |
| **Same-Region** | Compute and storage in same region for fast artifact upload |

### Provider Matrix

| Provider | GPU | Docker | S3-like | PostgreSQL | Pulumi | SkyPilot | dstack | Spot | Same-Region All-in-One |
|----------|-----|--------|---------|------------|--------|----------|--------|------|------------------------|
| **GCP** | A100, H100, T4, L4 | YES (VM) | GCS | Cloud SQL | YES | YES (best) | YES | YES (60-91% off) | **YES** |
| **AWS** | A100, H100, T4 | YES (VM) | S3 | RDS | YES | YES | YES | YES (60-90% off) | **YES** |
| **Azure** | A100, H100 | YES (VM) | Blob | Flexible Server | YES | YES | YES | YES (60-90% off) | **YES** |
| **Lambda** | A10-H100 | YES (VM) | **NO** | **NO** | **NO** | YES | YES | **NO** | **NO** |
| **Hetzner** | RTX 4000, RTX 6000 | YES | S3-compat | Managed DB? | YES | **NO** | **NO** | **NO** | **NO** |
| **UpCloud** | **NO GPU** | YES | S3-compat | Managed DB | YES (custom) | **NO** | **NO** | N/A | **NO** |
| **RunPod** | All consumer+DC | **NO** (container) | **NO** | **NO** | **NO** | YES | YES | YES | **NO** |
| **Vast.ai** | Marketplace | YES? | **NO** | **NO** | **NO** | YES | **NO** | YES | **NO** |
| **CoreWeave** | A100, H100 | YES (K8s) | S3 | **?** | YES | YES | **NO** | YES | Partial |
| **Nebius** | H100, H200 | YES (K8s) | S3-compat | Managed DB | **?** | YES | YES | **?** | Partial |

### Key Insight

**Only the Big 3 (GCP, AWS, Azure) offer ALL required components in the same cloud:**
GPU + Docker + Object Storage + PostgreSQL + Pulumi + SkyPilot + Spot instances.

Lambda Labs is great for compute but has **no storage or database services** — you
always need a second cloud for MLflow + artifacts.

Hetzner has storage + DB but **no SkyPilot/dstack support** for GPU orchestration.

---

## 5. Orchestrator Comparison: SkyPilot vs dstack {#5-orchestrators}

### Feature Comparison

| Feature | SkyPilot | dstack |
|---------|----------|--------|
| **Supported clouds** | 20+ (most) | 14 VM + 3 container |
| **Docker support** | `image_id: docker:` | Native Docker |
| **Managed spot jobs** | YES (auto-recovery) | YES |
| **Checkpointing** | MOUNT_CACHED (9.6x faster) | Volume mounting |
| **GCP support** | **BEST** (co-designed) | YES |
| **Lambda support** | YES | YES |
| **Hetzner support** | **NO** | **NO** |
| **Kubernetes** | YES | YES |
| **Multi-cloud failover** | YES (`any_of`) | YES |
| **Spot checkpointing** | GCS/S3 MOUNT_CACHED | Volume-based |
| **Maturity** | High (NSDI'23 paper) | Growing |
| **Pulumi integration** | None needed (orchestrator) | None needed |

### Verdict

**Stay with SkyPilot.** It's more mature, has better spot checkpointing (MOUNT_CACHED),
better GCP integration (co-designed at UC Berkeley), and supports more providers.
dstack offers no advantage for our use case and doesn't support Hetzner either.

---

## 6. GCP Deep Dive (Preferred Big-3 Cloud) {#6-gcp}

### GPU Spot Pricing (March 2026)

| GPU | VRAM | On-Demand $/hr | **Spot $/hr** | **Spot Discount** |
|-----|------|----------------|---------------|-------------------|
| T4 | 16 GB | $0.35 | **$0.14** | 60% off |
| L4 | 24 GB | $0.56 | **$0.22** | 60% off |
| A100 40GB | 40 GB | $2.93 | **$1.15** | 61% off |
| A100 80GB | 80 GB | $3.93 | **$1.57** | 60% off |
| H100 SXM | 80 GB | $9.80 | **$2.25** | 77% off |

**GCP L4 spot at $0.22/hr** is competitive with RunPod RTX 3090 spot ($0.22/hr)!
**GCP A100 40GB spot at $1.15/hr** is cheaper than Lambda A10 on-demand ($0.86/hr)
with 67% more VRAM (40 GB vs 24 GB)!

### GCP Architecture for MinIVess

```
┌──────────────────────────────────────────────────────────┐
│                    GCP (us-central1)                      │
│                                                          │
│  ┌──────────────┐   ┌───────────────┐   ┌──────────────┐ │
│  │ Cloud Run    │   │ Cloud SQL     │   │ GCS Bucket   │ │
│  │ (MLflow)     │──►│ (PostgreSQL)  │   │ (artifacts)  │ │
│  │ $0-15/month  │   │ $10-30/month  │   │ $2-5/month   │ │
│  └──────┬───────┘   └───────────────┘   └──────┬───────┘ │
│         │                                       │        │
│         │    ┌──────────────────────┐           │        │
│         └───►│   SkyPilot Spot VM   │◄──────────┘        │
│              │   A100 40GB ($1.15)  │  MOUNT_CACHED      │
│              │   Docker container   │  (same-region)     │
│              │   minivess-base      │                    │
│              └──────────────────────┘                    │
│                                                          │
│  ┌──────────────┐                                        │
│  │ Artifact     │  ◄── Docker images (gcr.io or GAR)     │
│  │ Registry     │      Same-region pull = fast            │
│  │ (GAR/GCR)    │                                        │
│  └──────────────┘                                        │
└──────────────────────────────────────────────────────────┘
```

### GCP Monthly Cost Estimate

| Service | Spec | Est. Cost/month |
|---------|------|-----------------|
| Cloud Run (MLflow) | 1 vCPU, 2 GB, always-on | $15-25 |
| Cloud SQL (PostgreSQL) | db-f1-micro, 10 GB | $10-15 |
| GCS (artifacts) | ~50 GB | $1-2 |
| GCS (DVC data) | ~5 GB | <$1 |
| GAR (Docker images) | ~20 GB | $1-2 |
| **Base infra total** | | **$28-45/month** |
| GPU training (spot) | A100 spot, ~10 hrs/month | $11.50/month |
| **Grand total** | | **$40-57/month** |

Compare: UpCloud trial is free for 30 days (then €19.42/month for VPS) + Lambda
at $1.48/hr is more expensive per GPU-hour than GCP spot.

### GCP Advantages for Our Use Case

1. **Same-region everything**: MLflow, artifacts, GPU, Docker registry — all in us-central1
2. **Spot instances with auto-recovery**: 60-77% discount, SkyPilot handles preemption
3. **MOUNT_CACHED checkpointing**: 9.6x faster checkpoint writes to GCS
4. **Artifact Registry (GAR)**: Docker images in same region → fast pulls
5. **Pulumi first-class support**: `@pulumi/gcp` is mature and well-documented
6. **SkyPilot co-designed with GCP**: Best integration, tested at UC Berkeley
7. **Free tier**: $300 credit for new accounts (90 days)
8. **Spot preemption simulation**: Can test auto-resume without waiting for real preemption

### GCP Setup with Pulumi

```python
# deployment/pulumi/gcp/__main__.py (sketch)
import pulumi_gcp as gcp

# Cloud SQL (PostgreSQL for MLflow + Optuna)
db = gcp.sql.DatabaseInstance("mlflow-db", ...)

# GCS Bucket (MLflow artifacts + DVC data + checkpoints)
artifacts_bucket = gcp.storage.Bucket("mlflow-artifacts", location="US-CENTRAL1")
dvc_bucket = gcp.storage.Bucket("minivess-dvc", location="US-CENTRAL1")
checkpoints_bucket = gcp.storage.Bucket("minivess-checkpoints", location="US-CENTRAL1")

# Artifact Registry (Docker images)
registry = gcp.artifactregistry.Repository("minivess", format="DOCKER", location="us-central1")

# Cloud Run (MLflow server)
mlflow_service = gcp.cloudrunv2.Service("mlflow",
    template=gcp.cloudrunv2.ServiceTemplateArgs(
        containers=[gcp.cloudrunv2.ServiceTemplateContainerArgs(
            image="ghcr.io/mlflow/mlflow:latest",
            envs=[
                {"name": "MLFLOW_BACKEND_STORE_URI", "value": db_url},
                {"name": "MLFLOW_ARTIFACTS_DESTINATION", "value": f"gs://{artifacts_bucket.name}"},
            ],
        )],
    ),
)
```

---

## 7. Hetzner Analysis {#7-hetzner}

### What Hetzner Offers

| Service | Available | Details |
|---------|-----------|---------|
| VPS | YES | CX22 from €3.79/month |
| GPU Servers | YES (dedicated only) | RTX 4000 SFF (20 GB), RTX 6000 Ada (48 GB) |
| Object Storage (S3) | YES | €4.99/TB/month (4.6x cheaper than AWS S3) |
| Managed Database | YES | PostgreSQL from €11.90/month |
| Pulumi support | YES | `@pulumi/hcloud` provider |

### What Hetzner Does NOT Offer

| Missing | Impact |
|---------|--------|
| **SkyPilot support** | Cannot use SkyPilot for GPU orchestration |
| **dstack support** | Cannot use dstack either |
| **Spot/preemptible instances** | No cost savings on GPU, no auto-recovery practice |
| **Cloud-managed GPU VMs** | Only dedicated servers (not ephemeral on-demand) |
| **Container registry** | Must use external registry (GHCR, Docker Hub) |

### Verdict on Hetzner

**Hetzner is great for MLflow hosting (cheap, EU, S3-compatible) but CANNOT replace
Lambda or GCP for GPU compute.** You'd still need a separate GPU cloud, creating the
same cross-cloud artifact transfer problem we have now.

The only scenario where Hetzner makes sense:
- MLflow + S3 on Hetzner EU (cheap, GDPR)
- GPU training on a supported SkyPilot cloud (GCP, Lambda, AWS)
- But artifacts still cross clouds → same latency issue

**Hetzner is NOT the answer to the architecture problem.**

---

## 8. Spot Instance Checkpointing & Auto-Resume {#8-spot-recovery}

### SkyPilot MOUNT_CACHED for GCP

```yaml
# SkyPilot YAML for GCP spot training with checkpointing
resources:
  image_id: docker:us-central1-docker.pkg.dev/PROJECT/minivess/base:latest
  accelerators: {A100: 1, L4: 1}
  cloud: gcp
  use_spot: true
  disk_tier: best  # SSD for fast local cache

file_mounts:
  /checkpoints:
    name: minivess-checkpoints  # GCS bucket
    mode: MOUNT_CACHED          # 9.6x faster writes

  /data:
    name: minivess-dvc-data
    mode: MOUNT                 # Read-only data

run: |
  # Load latest checkpoint if exists (spot recovery)
  python -m minivess.orchestration.flows.train_flow \
    --resume-from /checkpoints/latest.pth
```

### How MOUNT_CACHED Works

1. Checkpoint written to local SSD (fast, non-blocking)
2. Asynchronously uploaded to GCS bucket in background
3. If spot preempted: GCS has the last completed checkpoint
4. SkyPilot recovers → new VM → MOUNT_CACHED re-downloads checkpoint
5. Training resumes from last checkpoint

**Performance**: 9.6x faster than direct GCS writes. For a 910 MB checkpoint:
- Direct: ~45 seconds (blocking GPU)
- MOUNT_CACHED: ~5 seconds (GPU continues training)

### Simulating Spot Preemption

GCP provides a **preemption simulation** for testing:
```bash
# Simulate spot preemption on a running VM
gcloud compute instances simulate-maintenance-event INSTANCE_NAME --zone ZONE
```

This lets us test the full recovery pipeline without waiting for a real preemption.

---

## 9. Docker Registry Proximity {#9-registry}

### Current: GHCR (US-centric CDN)

- 20+ GB image pulled from GHCR (Fastly CDN)
- Lambda US-east-1: ~5-10 min pull
- GCP us-central1: ~5-10 min pull

### GCP Option: Artifact Registry (GAR)

- Docker image stored in same region (us-central1)
- Pull from same-region GCR: ~1-3 min for 20 GB
- **3-10x faster than cross-CDN GHCR pulls**

### Registry Recommendation

| Scenario | Registry | Pull Time (20 GB) |
|----------|----------|-------------------|
| GCP training | **GAR (us-central1)** | ~1-3 min |
| Lambda training | GHCR (US CDN) | ~5-10 min |
| AWS training | ECR (same-region) | ~1-3 min |
| Multi-cloud fallback | GHCR (public) | ~5-10 min |

---

## 10. Decision Matrix {#10-decision-matrix}

### Weighted Criteria

| Criterion | Weight | A: Fix UpCloud | B: Hetzner | C: All GCP | D: GCP compute + UpCloud MLflow | E: Direct S3 |
|-----------|--------|---------------|------------|------------|--------------------------------|--------------|
| **Artifacts work** | 30% | Maybe (3/5) | Maybe (3/5) | **YES (5/5)** | YES (4/5) | YES (4/5) |
| **Same-region speed** | 20% | NO (1/5) | NO (1/5) | **YES (5/5)** | Partial (3/5) | Partial (3/5) |
| **Spot instances** | 15% | NO (1/5) | NO (1/5) | **YES (5/5)** | YES (5/5) | NO (1/5) |
| **Cost** | 15% | FREE (5/5) | LOW (4/5) | MEDIUM (3/5) | MEDIUM (3/5) | FREE (5/5) |
| **Effort** | 10% | TINY (5/5) | MEDIUM (3/5) | HIGH (2/5) | MEDIUM (3/5) | MEDIUM (3/5) |
| **Long-term** | 10% | Poor (2/5) | Poor (2/5) | **BEST (5/5)** | Good (4/5) | OK (3/5) |
| **Weighted Score** | | **2.75** | **2.35** | **4.45** | **3.65** | **3.15** |

### Winner: Hypothesis C — All-GCP Architecture

But with a **phased approach**: try the quick fix first, then migrate to GCP.

---

## 11. Recommendation {#11-recommendation}

### Phase 1: Quick Fix (Today, 5 minutes)

Try fixing UpCloud MLflow artifact uploads:

```bash
# SSH to UpCloud server
ssh deploy@185.20.139.158

# Edit docker-compose to add multipart upload env var
# Add: MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD: "true"
# Add: MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT: "1800"
sudo docker compose restart mlflow
```

If this fixes 910 MB uploads, we have a working pipeline on Lambda + UpCloud.
This buys time for the proper GCP migration.

### Phase 2: GCP Migration (This Week)

Set up the all-GCP architecture:

1. **GCP account setup**: Enable Compute Engine, Cloud Run, Cloud SQL, GCS APIs
2. **Pulumi stack**: `deployment/pulumi/gcp/` with Cloud Run MLflow + Cloud SQL + GCS
3. **Docker registry**: Push `minivess-base` to GCP Artifact Registry (us-central1)
4. **SkyPilot config**: `cloud: gcp` with spot A100 + MOUNT_CACHED checkpointing
5. **DVC remote**: Switch from UpCloud S3 to GCS
6. **Test E2E**: sam3_vanilla smoke test on GCP spot A100

### Phase 3: Spot Recovery Testing (This Week)

1. Implement checkpoint loading at training startup
2. Add GCS MOUNT_CACHED to SkyPilot YAML
3. Run training on GCP spot
4. Simulate preemption: `gcloud compute instances simulate-maintenance-event`
5. Verify auto-resume from checkpoint

### Multi-Cloud Failover (Final Architecture)

```yaml
# deployment/skypilot/train_production.yaml
resources:
  image_id: docker:us-central1-docker.pkg.dev/minivess/repo/base:latest
  accelerators: {A100: 1, L4: 1, H100: 1}
  use_spot: true
  disk_tier: best

  any_of:
    - cloud: gcp          # Primary — same-region storage, spot, best SkyPilot support
      use_spot: true
    - cloud: lambda        # Fallback — on-demand only, cross-cloud artifacts
      use_spot: false
    - cloud: aws           # Emergency — expensive but available
      use_spot: true

file_mounts:
  /checkpoints:
    name: minivess-checkpoints
    mode: MOUNT_CACHED
```

---

## Sources

### MLflow
- [MLflow Artifact Store Architecture](https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/)
- [MLflow Large File Upload Bug #7564](https://github.com/mlflow/mlflow/issues/7564)
- [MLflow Multipart Upload Bug #11268](https://github.com/mlflow/mlflow/issues/11268)
- [MLflow Environment Variables](https://mlflow.org/docs/latest/python_api/mlflow.environment_variables.html)

### SkyPilot
- [SkyPilot Managed Jobs](https://docs.skypilot.co/en/latest/examples/managed-jobs.html)
- [SkyPilot High-Performance Checkpointing](https://blog.skypilot.co/high-performance-checkpointing/)
- [SkyPilot Training Guide](https://docs.skypilot.co/en/latest/reference/training-guide.html)
- [SkyPilot GCP Permissions](https://docs.skypilot.co/en/latest/cloud-setup/cloud-permissions/gcp.html)

### dstack
- [dstack Supported Backends](https://dstack.ai/docs/concepts/backends/)
- [dstack vs SkyPilot Discussion](https://news.ycombinator.com/item?id=42055036)

### GCP
- [GCP GPU Pricing 2026](https://gpucost.org/provider/gcp)
- [GCP Spot VMs Pricing](https://cloud.google.com/spot-vms/pricing)
- [GCP Cloud Run + Cloud SQL + Pulumi](https://www.pulumi.com/registry/packages/gcp/how-to-guides/gcp-py-cloudrun-cloudsql/)
- [MLflow on GCP Guide](https://dlabs.ai/blog/a-step-by-step-guide-to-setting-up-mlflow-on-the-google-cloud-platform/)

### Hetzner
- [Hetzner Object Storage](https://docs.hetzner.com/storage/object-storage/)
- [Hetzner GPU Servers](https://www.hetzner.com/dedicated-rootserver/gex44/)

### Provider Comparisons
- [State of Cloud GPUs 2025 (dstack)](https://dstack.ai/blog/state-of-cloud-gpu-2025/)
- [GPU Price Comparison 2026](https://getdeploying.com/gpus)
