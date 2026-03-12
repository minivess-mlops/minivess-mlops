# SkyPilot Remote GPU Training: Complete Architecture Report

**Date:** 2026-03-12
**Issue:** [#609](https://github.com/minivess-mlops/minivess-mlops/issues/609)
**Cross-ref:** [#366](https://github.com/minivess-mlops/minivess-mlops/issues/366) (RunPod via SkyPilot)
**Branch:** `feat/synthetic-vasculature-stack-generation`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Why SkyPilot (Not Pulumi, Not Raw APIs)](#2-why-skypilot)
3. [Current State: What Exists vs What's Missing](#3-current-state)
4. [Blocker Analysis](#4-blocker-analysis)
5. [SkyPilot Architecture for MinIVess](#5-skypilot-architecture)
6. [RunPod as Primary Backend](#6-runpod-as-primary-backend)
7. [Consumer GPU Economics: Buy vs Rent](#7-consumer-gpu-economics)
8. [FinOps Analysis: Datacenter GPUs](#8-finops-analysis)
9. [SkyPilot vs Pulumi Comparison](#9-skypilot-vs-pulumi)
10. [MLflow Accessibility Solutions](#10-mlflow-accessibility)
11. [Prefect + SkyPilot Integration](#11-prefect-skypilot-integration)
12. [HPO Sweep Architecture](#12-hpo-sweep-architecture)
13. [Flow Orchestration: Completion Barriers](#13-flow-orchestration)
14. [Spot Instance Checkpointing](#14-spot-checkpointing)
15. [Multi-Answer Questions](#15-multi-answer-questions)
16. [Minimum Viable Architecture](#16-minimum-viable-architecture)
17. [Implementation Roadmap](#17-implementation-roadmap)

---

## 1. Executive Summary

MinIVess MLOps has well-designed SkyPilot infrastructure that is **completely disconnected from the main pipeline**. The pieces exist (YAML tasks, Python launcher, Prefect bridge tasks, HPO engine) but the training flow never dispatches to SkyPilot. The `compute` parameter is accepted and ignored.

**The core problem is not GPU provisioning — it's service accessibility.** MLflow, Prefect, and PostgreSQL (Optuna) all run on `localhost` / Docker Compose internal network. SkyPilot-provisioned VMs cannot reach them. This is the #1 blocker.

**Recommended path:** Cloudflare Tunnel (zero-trust, free tier) to expose MLflow + Prefect + PostgreSQL to SkyPilot VMs without opening firewall ports or deploying to cloud. This works for solo-researcher scale; cloud-deployed services for team scale.

**Cost impact:** RunPod is 5-28x cheaper than AWS for single-GPU workloads (A100/H100). A 48-hour SAM3 fine-tuning run costs $96 on RunPod vs $2,642 on AWS. SkyPilot's multi-cloud failover (RunPod → Lambda → GCP Spot → AWS) automates cost optimization.

---

## 2. Why SkyPilot (Not Pulumi, Not Raw APIs) {#2-why-skypilot}

### What SkyPilot Does

[SkyPilot](https://docs.skypilot.co/en/latest/docs/index.html) is an open-source framework for running AI workloads on any cloud. It provides:

- **Multi-cloud failover:** Single YAML → automatically finds cheapest available GPU across 20+ clouds
- **Spot instance management:** Automatic preemption recovery with checkpointing
- **Managed job lifecycle:** `sky jobs launch` handles provisioning, execution, teardown
- **Cost optimization:** Region/zone arbitrage (2x savings) + spot (3-6x savings) = up to 90% reduction
- **Prefect integration:** [Official example](https://docs.skypilot.co/en/latest/examples/orchestrators/prefect.html) with Prefect task wrappers
- **Worker pools (v0.11+):** Pre-provisioned GPU pools for HPO sweeps

### What Pulumi Does (from the RunPods Reference Repo)

The [Mill Hill Garage RunPods repo](https://github.com/mill-hill-garage/runpods) uses Pulumi for:

- Pod lifecycle management (create, destroy, check status)
- Network volume provisioning (persistent storage across pod restarts)
- Container image building and pushing to GHCR
- Idle monitoring and auto-shutdown (cost control)
- Multi-pod stacks (testPod, visualTeam) with independent deploy cycles

Key patterns from the Pulumi approach:
- `create_or_find_storage()` — reuses existing NetworkVolumes by name
- Idle monitor daemon with CPU/GPU thresholds and SSH protection
- Multi-stage Docker builds (builder → runtime) for CUDA compilation
- Justfile orchestration: `just deploy-complete <pod>` = ship + deploy

### Why SkyPilot Wins for MinIVess

| Dimension | SkyPilot | Pulumi (RunPod) |
|-----------|----------|-----------------|
| **Multi-cloud** | 20+ clouds, automatic failover | RunPod only |
| **Spot management** | Automatic preemption + recovery | Manual (idle monitor only) |
| **Job lifecycle** | Fully managed (launch → teardown) | Manual via Justfile |
| **HPO sweeps** | Parallel managed jobs, worker pools | Must orchestrate manually |
| **Learning curve** | YAML task spec → `sky jobs launch` | Pulumi stacks + Python IaC |
| **MLflow integration** | Env var passthrough, works | Must build from scratch |
| **Prefect integration** | Official docs + example | No integration |
| **Cost optimization** | Region/zone + spot + autostop | Idle monitor only |
| **Vendor lock-in** | Zero (cloud-agnostic) | RunPod-specific provider |

**Verdict:** SkyPilot is the right abstraction for MinIVess. Pulumi is the right tool for *infrastructure management* (VPCs, databases, persistent services). They serve different layers:

```
Layer 4: SkyPilot  → Ephemeral GPU compute (training, HPO, inference)
Layer 3: Pulumi    → Persistent infrastructure (MLflow server, PostgreSQL, MinIO)
Layer 2: Docker    → Application packaging
Layer 1: Cloud API → Raw resources
```

For MinIVess today (solo researcher, local Docker Compose for infra), SkyPilot alone handles the GPU compute gap. Pulumi becomes relevant when deploying MLflow/Prefect to cloud for team access.

---

## 3. Current State: What Exists vs What's Missing {#3-current-state}

### Implemented (Tested, Not Integrated)

| Component | File | Status |
|-----------|------|--------|
| SkyPilot train YAML | `deployment/skypilot/train_generic.yaml` | Multi-cloud failover, spot, S3 mounts |
| SkyPilot HPO YAML | `deployment/skypilot/train_hpo_sweep.yaml` | HPO flow dispatch |
| Python SDK launcher | `src/minivess/compute/skypilot_launcher.py` | Dry-run fallback when not installed |
| Prefect bridge tasks | `src/minivess/compute/prefect_sky_tasks.py` | `launch_sky_training()`, `wait_sky_job()` |
| HPO engine | `src/minivess/optimization/hpo_engine.py` | Optuna + ASHA, PostgreSQL-only |
| HPO flow | `src/minivess/orchestration/flows/hpo_flow.py` | Trial iteration, local gpu-pool dispatch |
| Config YAML | `configs/hpo/dynunet_example.yaml` | Search space + allocation strategy |
| Environment vars | `.env.example` | `MLFLOW_SKYPILOT_HOST`, `OPTUNA_STORAGE_URL` |
| Unit tests | `tests/v2/unit/test_skypilot_*.py` | YAML validation, no-direct-script rules |

### Missing (Blockers for Remote Training)

| Gap | Impact | Effort |
|-----|--------|--------|
| **Compute dispatcher in training flow** | `compute` param accepted, never used | Medium |
| **MLflow accessible from cloud VMs** | Training can't log metrics/artifacts | High |
| **Prefect API accessible from cloud VMs** | Flows can't register/heartbeat | High |
| **PostgreSQL accessible from cloud VMs** | Optuna can't share study state | High |
| **HPO completion barrier** | Analysis Flow runs before all trials finish | Medium |
| **SkyPilot work pool in deployments.py** | No `sky-gpu-pool` for routing | Low |
| **`compute_backend` config** | No way to select local vs SkyPilot | Low |
| **Checkpoint-on-preemption logic** | Data lost on spot preemption | Medium |
| **SkyPilot in pyproject.toml extras** | Must manually install | Low |

---

## 4. Blocker Analysis {#4-blocker-analysis}

### B1: MLflow Tracking Server Accessibility (CRITICAL)

**Current state:** MLflow runs at `http://minivess-mlflow:5000` inside Docker Compose network. Only accessible from containers on `minivess-network`.

**Why it matters:** Every training fold logs metrics, params, artifacts, and system metrics to MLflow. Without MLflow access, SkyPilot-launched training produces orphaned results with no tracking.

**Options (detailed in [Section 10](#10-mlflow-accessibility)):**

| Option | Complexity | Cost | Security | Solo Researcher | Team |
|--------|-----------|------|----------|-----------------|------|
| A: Cloudflare Tunnel | Low | Free | Zero-trust | Best | OK |
| B: Cloud VM MLflow | Medium | $30-50/mo | VPC | OK | Best |
| C: Post-hoc sync | Low | Free | Excellent | Acceptable | No |
| D: Managed MLflow | Low | $100+/mo | Managed | Overkill | Good |

**Recommendation:** Option A (Cloudflare Tunnel) for immediate unblocking, migrate to Option B when team grows.

### B2: Prefect API Accessibility (HIGH)

**Current state:** Prefect Server at `http://minivess-prefect:4200`, same Docker Compose isolation.

**Impact:** Without Prefect access, SkyPilot jobs can't:
- Register flow runs
- Report heartbeats
- Update flow run status (COMPLETED/FAILED)
- Be visible in Prefect UI

**Mitigation options:**
1. **Same tunnel as MLflow** — Cloudflare Tunnel exposes Prefect alongside MLflow
2. **Standalone execution** — SkyPilot jobs run training without Prefect, use MLflow-only contract. Prefect orchestration happens locally (trigger chain polls MLflow for completion).
3. **Prefect Cloud** — Managed Prefect ($0 for 1 user, open-source tier)

**Recommendation:** Option 2 for MVP. SkyPilot jobs run training scripts directly (not Prefect deployments). The local Prefect flow wraps the SkyPilot launch/wait cycle. This means:

```
LOCAL: Prefect training_flow() → SkyPilotLauncher.launch() → wait_sky_job()
REMOTE (SkyPilot VM): python -m minivess.pipeline.train_fold --config ... → MLflow logging
LOCAL: Prefect detects SUCCEEDED → mark flow run COMPLETED → trigger Analysis Flow
```

### B3: HPO Completion Barrier (MEDIUM)

**Current state:** The `PipelineTriggerChain` runs flows sequentially:
```
data → train → analyze → deploy → dashboard
```

**Problem:** When training runs 128 HPO combinations, Analysis Flow should wait for ALL to complete before building ensembles. Currently:
- HPO flow iterates trials sequentially (one at a time)
- No fan-out/fan-in pattern for parallel SkyPilot trials

**Solution architecture (detailed in [Section 13](#13-flow-orchestration)):**

```
HPO Flow (local Prefect):
  ├── Launch trial 1 → SkyPilot job → MLflow run
  ├── Launch trial 2 → SkyPilot job → MLflow run
  ├── ...
  └── Launch trial N → SkyPilot job → MLflow run

  BARRIER: Wait for all N jobs to reach terminal state

Analysis Flow (local Prefect):
  └── Query MLflow for all runs in experiment → build ensembles → evaluate

Dashboard Flow (local Prefect):
  └── Read Analysis artifacts → generate figures → export Parquet
```

### B4: Optuna Storage Accessibility (MEDIUM)

**Current state:** PostgreSQL at `postgres:5432` inside Docker Compose, connection string in `.env.example`:
```
OPTUNA_STORAGE_URL=postgresql+psycopg2://minivess:minivess@postgres:5432/minivess
```

**Impact:** For parallel HPO trials on SkyPilot, all workers must connect to the same Optuna study. Without shared storage, each worker runs an independent study.

**Solutions:**
1. **Tunnel PostgreSQL** — Same Cloudflare Tunnel, add PostgreSQL route
2. **Cloud PostgreSQL** — AWS RDS / GCP Cloud SQL / Neon (free tier)
3. **Independent trials + post-hoc merge** — Each SkyPilot job gets env vars (LR, batch_size, etc.) from a pre-generated grid. No Optuna needed during training; results compared via MLflow after completion.

**Recommendation:** Option 3 for MVP. Pre-generate the trial grid locally (Optuna `suggest_*()` → env vars), launch SkyPilot jobs with those env vars, compare results in MLflow after completion. This is exactly how the SkyPilot docs recommend HPO:

```bash
for lr in 0.01 0.03 0.1; do
    sky jobs launch --async train.yaml --env LR=$lr --env MAX_STEPS=1000
done
```

No shared database needed. Optuna's TPE sampler can still be used locally to generate the trial configs before launching.

---

## 5. SkyPilot Architecture for MinIVess {#5-skypilot-architecture}

### Minimum Viable Architecture (Phase 0-1)

```
┌─────────────────────────────────────────────────────────────┐
│                     LOCAL MACHINE                            │
│                                                              │
│  ┌─────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Prefect     │  │ MLflow   │  │PostgreSQL│  │ MinIO    │ │
│  │ Server      │  │ Server   │  │ (Optuna) │  │ (S3)     │ │
│  │ :4200       │  │ :5000    │  │ :5432    │  │ :9000    │ │
│  └──────┬──────┘  └────┬─────┘  └──────────┘  └──────────┘ │
│         │              │                                     │
│  ┌──────┴──────────────┴──────────────────────────────────┐ │
│  │              Cloudflare Tunnel (cloudflared)            │ │
│  │  mlflow.minivess.example.com → localhost:5000          │ │
│  └────────────────────────┬───────────────────────────────┘ │
│                           │                                  │
│  ┌────────────────────────┴───────────────────────────────┐ │
│  │           Prefect Training Flow (local)                 │ │
│  │                                                         │ │
│  │  1. Compose experiment config (Hydra-zen)               │ │
│  │  2. if compute_backend == "skypilot":                   │ │
│  │       SkyPilotLauncher.launch_training_job()            │ │
│  │       wait_sky_job() → poll until SUCCEEDED/FAILED      │ │
│  │     else:                                               │ │
│  │       train_one_fold_task() → local Docker              │ │
│  │  3. Mark flow run COMPLETED                             │ │
│  │  4. Trigger Analysis Flow                               │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                    SkyPilot API (sky.jobs.launch)
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   ┌─────────┐        ┌─────────┐        ┌─────────┐
   │ RunPod  │        │ Lambda  │        │ GCP     │
   │ A100    │        │ A100    │        │ A100    │
   │ $1.74/h │        │ $0.78/h │        │ $1.20/h │
   └────┬────┘        └────┬────┘        └────┬────┘
        │                  │                   │
        └──────────────────┼───────────────────┘
                           │
                    MLFLOW_TRACKING_URI=
                    https://mlflow.minivess.example.com
                           │
                    Training script logs:
                    - Metrics (loss, DSC, clDice)
                    - Params (lr, batch_size, model)
                    - Artifacts (checkpoints, configs)
                    - System metrics (GPU util, memory)
```

### Full Architecture (Phase 2-4)

```
┌──────────────────────────────────────────────────────────────┐
│                      LOCAL / CLOUD VM                         │
│                                                               │
│  Prefect                 MLflow          PostgreSQL   MinIO   │
│  ┌──────────┐           ┌──────┐        ┌──────┐   ┌─────┐  │
│  │ data_flow│───────────│      │        │Optuna│   │ S3  │  │
│  │ train_fl │──┐        │Track │◄───────│Study │   │Arti │  │
│  │ hpo_flow │──┤        │ ing  │        │      │   │facts│  │
│  │ analyze  │──┤        │Server│        └──────┘   └─────┘  │
│  │ deploy   │──┤        └──────┘                             │
│  │ dashboard│──┤                                             │
│  └──────────┘  │                                             │
│                │  Compute Dispatcher                          │
│                │  ┌──────────────────────────────────┐        │
│                ├──│ if compute_backend == "local":   │        │
│                │  │   → Docker gpu-pool              │        │
│                │  │ elif compute_backend == "skypilot"│        │
│                │  │   → SkyPilotLauncher             │        │
│                │  │   → wait for completion           │        │
│                │  │   → log results to MLflow         │        │
│                │  └──────────────────────────────────┘        │
│                │                                              │
│                │  HPO Orchestrator                             │
│                │  ┌──────────────────────────────────┐        │
│                ├──│ 1. Optuna.suggest_*() → grid     │        │
│                │  │ 2. for trial in grid:            │        │
│                │  │      sky jobs launch --async     │        │
│                │  │ 3. BARRIER: wait all terminal    │        │
│                │  │ 4. Compare in MLflow             │        │
│                │  │ 5. Trigger Analysis Flow         │        │
│                │  └──────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
         ┌─────────┐  ┌─────────┐  ┌─────────┐
         │ RunPod  │  │ Lambda  │  │ GCP Spot│
         │ Trial 1 │  │ Trial 2 │  │ Trial 3 │
         │ A100    │  │ A100    │  │ A100    │
         └─────────┘  └─────────┘  └─────────┘
```

---

## 6. RunPod as Primary Backend {#6-runpod-as-primary-backend}

### Setup

1. Install RunPod CLI: `pip install "runpod>=1.6"`
2. Install SkyPilot with RunPod: `pip install "skypilot-nightly[runpod]"` (nightly required, not stable)
3. Configure: `runpod config` → paste API key from [RunPod Settings](https://www.runpod.io/console/user/settings)
4. Verify: `sky check` → RunPod should show as available

### GPU Availability

| GPU | VRAM | Community $/hr | Secure $/hr | Best For |
|-----|------|---------------|-------------|----------|
| H100 SXM | 80 GB | ~$1.99 | ~$2.39-2.79 | SAM3 fine-tuning, latent diffusion |
| A100 SXM | 80 GB | ~$1.74 | ~$1.94 | Standard training, HPO sweeps |
| A100 PCIe | 80 GB | ~$1.64 | ~$1.84 | Budget A100 option |
| RTX 4090 | 24 GB | ~$0.69 | ~$0.89 | VesselVAE, VQ-VAE, inference |
| A6000 | 48 GB | ~$0.76 | ~$0.96 | Good VRAM/price for medium models |
| RTX 3090 | 24 GB | ~$0.20 | N/A | Community only, cheapest |

### RunPod-Specific SkyPilot YAML

```yaml
name: minivess-train-runpod

resources:
  cloud: runpod
  accelerators: A100-80GB:1
  use_spot: true
  disk_size: 100
  image_id: docker:ghcr.io/minivess-mlops/minivess-base:latest

  # Failover: RunPod → Lambda → GCP Spot → AWS
  any_of:
    - cloud: runpod
      accelerators: A100-80GB:1
    - cloud: lambda
      accelerators: A100-80GB:1
    - cloud: gcp
      accelerators: A100-80GB:1
      use_spot: true
    - cloud: aws
      accelerators: A100:1
      use_spot: true

file_mounts:
  /checkpoints:
    name: minivess-checkpoints
    store: s3
    mode: MOUNT_CACHED  # 9x faster writes than MOUNT

envs:
  MLFLOW_TRACKING_URI: ${MLFLOW_TUNNEL_URL}
  EXPERIMENT_NAME: minivess_training
  LOSS_NAME: cbdice_cldice
  MODEL_FAMILY: dynunet
  MAX_EPOCHS: 100
  NUM_FOLDS: 3

secrets:
  HF_TOKEN: ${HF_TOKEN}

setup: |
  pip install uv
  cd ~/sky_workdir && uv sync --no-dev
  # Download MiniVess data if not cached
  python -m minivess.data.downloaders download_minivess --dest /data

run: |
  cd ~/sky_workdir
  python -m minivess.pipeline.train_fold \
    --loss ${LOSS_NAME} \
    --model ${MODEL_FAMILY} \
    --epochs ${MAX_EPOCHS} \
    --folds ${NUM_FOLDS} \
    --experiment ${EXPERIMENT_NAME} \
    --checkpoint-dir /checkpoints
```

### Known Gotchas

1. **Preemption (Community Cloud):** 5-second SIGTERM → SIGKILL. Checkpointing must save within 5 seconds or use cloud bucket mounts (persistent across preemptions).
2. **151+ outages in 6 months** ([StatusGator](https://statusgator.com/services/runpod)). Not enterprise-grade reliability — use multi-cloud failover.
3. **Catalog freshness:** [GitHub issue #3794](https://github.com/skypilot-org/skypilot/issues/3794) — RunPod GPU catalog in SkyPilot may lag behind actual offerings.
4. **Nightly required:** `skypilot-nightly[runpod]`, not the stable release.
5. **Zero egress fees** — major advantage over AWS ($0.09/GB) and GCP ($0.12/GB) for large checkpoint downloads.

---

## 7. Consumer GPU Economics: Buy vs Rent {#7-consumer-gpu-economics}

### Why Consumer GPUs Matter for MinIVess

The Big 3 (AWS, GCP, Azure) **do not offer consumer GPUs** (RTX 3090/4090/5090). They only offer datacenter cards (A100, H100, L4) bundled in large instances. RunPod and neoclouds fill this gap — and for MinIVess workloads, consumer GPUs are often the optimal choice:

- **DynUNet training**: 3.5 GB VRAM — fits RTX 3090 (24 GB) trivially
- **VesselVAE / VQ-VAE**: Fits 24 GB easily
- **SAM3 vanilla**: 2.9 GB — any card works
- **SAM3 hybrid**: 7.5 GB — fits 24 GB with room to spare
- **Only true A100+ need**: Multi-GPU DDP training, latent diffusion 3D, or batch size >>1 with large models

### GPU Specifications

| Spec | RTX 5090 | RTX 4090 | RTX 3090 | A100 80GB |
|------|----------|----------|----------|-----------|
| **VRAM** | 32 GB GDDR7 | 24 GB GDDR6X | 24 GB GDDR6X | 80 GB HBM2e |
| **Bandwidth** | 1,792 GB/s | 1,008 GB/s | 936 GB/s | 2,039 GB/s |
| **BF16 TFLOPS** | 209.5 | 82.6 | 35.6 | 312 |
| **TDP** | 575W | 450W | 350W | 300W (SXM) |
| **Architecture** | Blackwell | Ada Lovelace | Ampere | Ampere |

**Key benchmark:** RTX 5090 outperforms A100 in [92% of ML benchmarks](https://www.runpod.io/blog/rtx-5090-llm-benchmarks) (24/26 tests) including CV training, while costing far less per hour in the cloud.

### Cloud Rental Pricing: Consumer GPUs

| GPU | RunPod (On-Demand) | RunPod (Community) | Vast.ai | SaladCloud |
|-----|-------------------|-------------------|---------|------------|
| **RTX 5090** (32 GB) | **$0.69/hr** | — | ~$0.40/hr | ~$0.25/hr |
| **RTX 4090** (24 GB) | **$0.34/hr** | ~$0.20/hr | ~$0.17-0.32/hr | ~$0.16/hr |
| **RTX 3090** (24 GB) | **$0.22/hr** | ~$0.19/hr | ~$0.15-0.20/hr | — |

Compare to datacenter GPUs:
| GPU | RunPod | Lambda |
|-----|--------|--------|
| **A100 80GB** | $1.74/hr | $0.78-1.79/hr |
| **H100** | $1.99/hr | $1.38-2.99/hr |

**The consumer advantage is clear:** RTX 5090 at $0.69/hr delivers comparable or better ML performance than A100 at $1.74/hr — a **60% cost saving** for workloads that fit in 32 GB VRAM. RTX 4090 at $0.34/hr is **80% cheaper** than A100 for workloads fitting 24 GB.

**Lambda Labs does NOT offer consumer GPUs** — they focus on datacenter cards only. RunPod, Vast.ai, and SaladCloud are the providers for consumer GPU cloud.

### Buy vs Rent: Break-Even Analysis

#### Hardware Costs (March 2026)

| GPU | Purchase Price | System Total | 2yr Resale (est.) |
|-----|---------------|-------------|-------------------|
| **RTX 5090** (AIB, new) | $2,910-3,500 | $4,080-5,330 | $1,200-1,600 (50-55%) |
| **RTX 4090** (used) | $1,800-2,200 | $2,600-3,400 | $1,200-1,500 (60-70%) |
| **RTX 3090** (used) | $700-1,000 | $1,500-2,000 | $400-600 (50-60%) |

Note: RTX 4090 has **appreciated** since launch ($1,599 MSRP → $1,800+ used) due to production halt + DRAM shortage. RTX 5090 is 40-100% above MSRP due to supply constraints.

#### Electricity Costs

| GPU | TDP | System Draw | $/yr (US, $0.15/kWh, 8h/day) | $/yr (Finland, $0.18/kWh, 8h/day) | $/yr (Germany, $0.44/kWh, 8h/day) |
|-----|-----|------------|------------------------------|-----------------------------------|-----------------------------------|
| **RTX 5090** | 575W | ~700W | $307 | $368 | $899 |
| **RTX 4090** | 450W | ~575W | $252 | $303 | $740 |
| **RTX 3090** | 350W | ~475W | $208 | $250 | $610 |

#### Amortized Cost Per GPU-Hour (2-year ownership)

Formula: `(purchase_price + system_overhead + 2yr_electricity - resale_value) / total_hours`

| Scenario | Amortized $/GPU-hr | RunPod $/hr | Vast.ai $/hr |
|----------|-------------------|-------------|--------------|
| **RTX 5090 local (US, 8hr/day)** | **$0.58** | $0.69 | $0.40 |
| **RTX 5090 local (US, 24/7)** | **$0.19** | $0.69 | $0.40 |
| **RTX 5090 local (Finland, 8hr/day)** | **$0.61** | $0.69 | $0.40 |
| **RTX 5090 local (Germany, 8hr/day)** | **$0.79** | $0.69 | $0.40 |
| **RTX 4090 local (US, 8hr/day)** | **$0.40** | $0.34 | $0.20 |
| **RTX 4090 local (US, 24/7)** | **$0.13** | $0.34 | $0.20 |
| **RTX 4090 local (Finland, 8hr/day)** | **$0.42** | $0.34 | $0.20 |

#### Break-Even Hours

| GPU | vs RunPod | vs Vast.ai | At 8hr/day = months |
|-----|-----------|-----------|---------------------|
| **RTX 5090** | 4,948 hrs | 8,535 hrs | **21 mo** (RunPod) / **36 mo** (Vast.ai) |
| **RTX 4090 (used)** | 6,776 hrs | 11,520 hrs | **28 mo** (RunPod) / **48 mo** (Vast.ai) |
| **RTX 3090 (used)** | 4,545 hrs | 6,667 hrs | **19 mo** (RunPod) / **28 mo** (Vast.ai) |

### Decision Matrix: When to Buy vs Rent

| Usage Pattern | Recommendation | Why |
|--------------|---------------|-----|
| **< 4 hrs/day, intermittent** | **Rent (Vast.ai/RunPod)** | Cloud is 2-3x cheaper than ownership |
| **4-8 hrs/day, sustained >18 months** | **Buy local + rent burst** | Break-even reached; local for daily work, cloud for multi-GPU sweeps |
| **24/7 continuous training** | **Buy local** | Amortized drops to $0.13-0.19/hr, far below any cloud |
| **Multi-GPU HPO sweeps** | **Rent cloud** | Building multi-GPU workstation is $10K+, impractical |
| **EU/Germany (high electricity)** | **Rent cloud** | Local ownership exceeds even RunPod pricing |
| **EU/Finland or US** | **Buy viable at 6+ hrs/day** | Moderate electricity keeps ownership competitive |
| **Medical data (GDPR)** | **Buy local** | Avoids data transfer compliance complications |
| **Academic lab (grant-funded)** | **Hybrid** | See below |

### Academic Lab GPU Procurement Considerations

1. **Grant overhead:** University procurement adds 15-30% overhead to hardware purchases. A $3,200 GPU may cost $4,000+ through official channels. Cloud rentals are operational expenses — often easier to budget and faster to procure.

2. **Depreciation accounting:** Academic labs typically depreciate equipment over 3-5 years for accounting purposes. The GPU's market resale value follows a different curve:
   - Year 0: Purchase at $3,200 (RTX 5090 AIB)
   - Year 1: Market value ~$2,200 (depreciation: $1,000)
   - Year 2: Market value ~$1,600 (depreciation: $600)
   - Year 3: Market value ~$1,100 (depreciation: $500)
   - **Implication:** Most value loss happens in Year 1. If the lab plans to sell after 2 years, the effective rental cost is $(3,200 - 1,600) / 2yr = $800/yr.

3. **"Renting to yourself" calculation:** If a lab buys an RTX 5090 for $3,200 and a PhD student uses it 6 hrs/day for 2 years:
   - Total hours: 6 × 365 × 2 = 4,380 hours
   - Net cost: $3,200 - $1,600 (resale) + $614 (2yr electricity) = $2,214
   - **Effective internal rental rate: $0.51/hr** — cheaper than RunPod ($0.69) but more expensive than Vast.ai ($0.40)

4. **Startups should model it identically:** Hardware is a depreciating asset. The break-even calculation above IS the "internal rental rate" that should be compared against cloud pricing when making procurement decisions.

5. **[Lambda Labs offers 50% academic discount](https://lambda.ai/pricing)** — brings A100 40GB to ~$0.65/hr. Competitive with owning an RTX 5090 locally.

6. **Data sovereignty:** MiniVess data (EBRAINS, CC BY-NC-SA) and VesselNN data may have ethics board restrictions on cloud transfer. Local GPU avoids GDPR/IRB complications entirely.

### RTX 5090 vs A100: Which to Rent on RunPod?

For MinIVess workloads:

| Workload | VRAM Need | Best GPU | $/hr | Why |
|----------|-----------|----------|------|-----|
| DynUNet training (default) | 3.5 GB | RTX 4090 ($0.34) or 3090 ($0.22) | $0.22-0.34 | Massive overkill, cheapest wins |
| VesselVAE / VQ-VAE | 4-8 GB | RTX 4090 ($0.34) | $0.34 | 24 GB is plenty |
| SAM3 vanilla | 2.9 GB | RTX 3090 ($0.22) | $0.22 | Any card works |
| SAM3 hybrid | 7.5 GB | RTX 4090 ($0.34) | $0.34 | 24 GB has room |
| SAM3 full fine-tuning | 32+ GB | RTX 5090 ($0.69) or A100 ($1.74) | $0.69 | 32 GB of 5090 may suffice |
| Latent diffusion 3D | 40+ GB | A100 80GB ($1.74) | $1.74 | Only for heavy generative |
| 128-trial HPO sweep | 3.5-8 GB | 128x RTX 3090 ($0.22) | $0.22 each | Total: $0.22 × 128 × 4hr = $113 |

**Key insight:** For 80%+ of MinIVess workloads, consumer GPUs (RTX 3090/4090) at $0.22-0.34/hr are the optimal choice. A100/H100 should be reserved for the rare workloads that genuinely need >32 GB VRAM.

### SkyPilot Failover Chain (Updated with Consumer GPUs)

```yaml
resources:
  accelerators: RTX4090:1  # Start with cheapest sufficient GPU
  any_of:
    - cloud: runpod
      accelerators: RTX4090:1    # $0.34/hr — default for most workloads
    - cloud: runpod
      accelerators: RTX3090:1    # $0.22/hr — if 4090 unavailable
    - cloud: runpod
      accelerators: RTX5090:1    # $0.69/hr — if more VRAM needed
    - cloud: runpod
      accelerators: A100-80GB:1  # $1.74/hr — only for >32 GB VRAM
    - cloud: lambda
      accelerators: A100-80GB:1  # $0.78-1.79/hr — Lambda fallback
    - cloud: gcp
      accelerators: A100-80GB:1
      use_spot: true             # GCP spot last resort
```

---

## 8. FinOps Analysis: Datacenter GPUs {#8-finops-analysis}

### Cost Comparison by Workload

#### Scenario A: Single Model Training (1x A100 80GB, 24h)

| Provider | Tier | Cost | Notes |
|----------|------|------|-------|
| **Lambda Labs** | On-demand | **$18.72-42.96** | Best price IF available |
| **RunPod Community** | On-demand | **$41.76** | Reliable, single-GPU |
| **GCP Spot** | Spot | **$29-60** | Preemption risk |
| **RunPod Secure** | On-demand | **$46.56** | SLA available |
| **GCP** | On-demand | **~$96** | Expensive on-demand |
| **AWS p4de** | Spot | **$216-360** | 8 GPUs bundled (7 wasted) |
| **AWS p4de** | On-demand | **$681** | Catastrophically expensive |

#### Scenario B: HPO Sweep (4x A100 80GB, 8h parallel)

| Provider | Tier | Cost | Notes |
|----------|------|------|-------|
| **Lambda Labs** | On-demand | **$25-57** | 4 independent GPUs |
| **RunPod Community** | On-demand | **$56** | 4 pods, isolated |
| **GCP Spot** | Spot | **$38-80** | 4 instances, preemption risk |
| **RunPod Secure** | On-demand | **$62** | SLA |
| **GCP** | On-demand | **~$128** | |
| **AWS p4de** | On-demand | **$227** | Uses all 8 GPUs (4 idle) |

#### Scenario C: SAM3 Fine-Tuning (1x H100, 48h)

| Provider | Tier | Cost | Notes |
|----------|------|------|-------|
| **Lambda Labs** | On-demand | **$66-144** | Price range varies |
| **RunPod Community** | On-demand | **$96** | Best reliable price |
| **GCP Spot** | Spot | **$108** | |
| **RunPod Secure** | On-demand | **$115-134** | |
| **GCP** | On-demand | **$144** | |
| **AWS p5** | Spot | **$768-1,056** | 8 GPUs bundled |
| **AWS p5** | On-demand | **$2,642** | 27x more than RunPod |

### Key Insight: AWS is Not Cost-Competitive for Single-GPU

AWS bundles GPUs in large instances:
- A100 80GB: minimum `p4de.24xlarge` (8x A100s, $28.39/hr on-demand)
- H100: minimum `p5.48xlarge` (8x H100s, $55.04/hr on-demand)

For MinIVess workloads (single-GPU training, parallel HPO with isolated trials), RunPod and Lambda offer single-GPU pods at 5-28x lower cost.

### SkyPilot Failover Chain (Recommended)

```yaml
resources:
  accelerators: A100-80GB:1
  any_of:
    - cloud: runpod    # $1.74/hr — cheapest reliable
    - cloud: lambda    # $0.78-1.79/hr — cheapest IF available
    - cloud: gcp       # $1.20-2.50/hr spot
      use_spot: true
    - cloud: aws       # Last resort
      use_spot: true
```

### Cost Savings Summary

| Savings Source | Reduction | Mechanism |
|---------------|-----------|-----------|
| Neocloud vs AWS | 60-84% | Single-GPU instances |
| Spot vs on-demand | 50-70% | SkyPilot auto-recovery |
| Region arbitrage | ~2x | SkyPilot cheapest-zone selection |
| Autostop | Variable | Idle detection, no wasted hours |
| **Combined** | **70-90%** | All mechanisms stacked |

### Monthly Budget Estimate (Active Research Phase)

**Using consumer GPUs where possible (recommended):**

| Workload | GPU | Frequency | Cost/Run | Monthly |
|----------|-----|-----------|----------|---------|
| DynUNet training (3-fold, 100 epochs) | RTX 3090 ($0.22/hr) | 4x/month | $5.28 | $21 |
| HPO sweep (16 trials, 4 parallel) | RTX 4090 ($0.34/hr) | 2x/month | $22 | $44 |
| SAM3 fine-tuning | RTX 5090 ($0.69/hr) | 1x/month | $33 | $33 |
| VesselVAE/VQ-VAE training | RTX 4090 ($0.34/hr) | 4x/month | $3.40 | $14 |
| Latent diffusion 3D (heavy) | A100 80GB ($1.74/hr) | 1x/month | $42 | $42 |
| **Total (Consumer GPUs + 1 A100)** | | | | **$154** |
| **Total (A100 for everything)** | | | | **$556** |
| **Same on AWS** | | | | **$4,000+** |

**Consumer GPUs cut the monthly bill by 72%** compared to using A100 for everything.

---

## 9. SkyPilot vs Pulumi Comparison {#9-skypilot-vs-pulumi}

### When to Use Each

| Dimension | SkyPilot | Pulumi |
|-----------|----------|--------|
| **Purpose** | Ephemeral GPU compute | Persistent infrastructure |
| **Lifecycle** | Minutes to hours | Weeks to months |
| **State** | Managed by SkyPilot controller | Pulumi state backend |
| **Multi-cloud** | 20+ clouds, automatic | Per-provider plugin |
| **Configuration** | YAML task spec | Python/TypeScript IaC |
| **Cost model** | Pay per job | Pay per resource uptime |
| **Best for** | Training, HPO, inference | MLflow server, PostgreSQL, MinIO |
| **Spot support** | Native, auto-recovery | Manual (idle monitor) |
| **Scaling** | sky.jobs.launch (parallel) | Manual replicas |

### Complementary Architecture

```
Pulumi (persistent):                    SkyPilot (ephemeral):
┌─────────────────────┐                ┌─────────────────────┐
│ Cloud VM (always on) │                │ RunPod/Lambda GPUs  │
│ ├── MLflow Server   │                │ ├── Training jobs   │
│ ├── PostgreSQL      │◄───────────────│ ├── HPO trials      │
│ ├── Prefect Server  │  MLFLOW_URI    │ ├── SAM3 fine-tune  │
│ ├── MinIO           │                │ └── Synthetic gen   │
│ └── Grafana         │                └─────────────────────┘
└─────────────────────┘                     ↑ sky jobs launch
                                            │
                                    Local Prefect flow
```

### Recommendation

**Phase 0-1 (now):** SkyPilot only. Local Docker Compose for infra, Cloudflare Tunnel for accessibility. Zero Pulumi needed.

**Phase 2 (team grows):** Add Pulumi for cloud-deploying MLflow + PostgreSQL. SkyPilot for GPU compute.

**Phase 3 (production):** Pulumi for all infrastructure, SkyPilot for all GPU workloads, Prefect for orchestration.

---

## 10. MLflow Accessibility Solutions {#10-mlflow-accessibility}

### Option A: Cloudflare Tunnel (Recommended for Solo Researcher)

[Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) creates a secure outbound-only connection from your machine to Cloudflare's edge. No firewall ports opened, no public IP needed.

**Setup:**

```bash
# Install cloudflared
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
  -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared

# Login (one-time)
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create minivess

# Config file (~/.cloudflared/config.yml)
tunnel: <TUNNEL_ID>
credentials-file: ~/.cloudflared/<TUNNEL_ID>.json
ingress:
  - hostname: mlflow.minivess.example.com
    service: http://localhost:5000
  - hostname: prefect.minivess.example.com
    service: http://localhost:4200
  - service: http_status:404

# Run tunnel
cloudflared tunnel run minivess
```

**SkyPilot YAML:**
```yaml
envs:
  MLFLOW_TRACKING_URI: https://mlflow.minivess.example.com
```

**Pros:** Free tier, zero-trust security, no firewall changes, works behind NAT
**Cons:** Requires domain name, tunnel must run on dev machine during training, added latency (~50ms)

### Option B: Cloud VM MLflow (Recommended for Team)

Deploy MLflow + PostgreSQL to a cheap cloud VM ($10-30/month):

```bash
# Smallest VM with Docker
# AWS: t3.small ($0.02/hr ≈ $15/mo)
# GCP: e2-small ($0.017/hr ≈ $12/mo)
# Hetzner: CX22 ($4.15/mo)

# Docker Compose on cloud VM (reuse existing docker-compose.yml)
docker compose -f docker-compose.yml up -d mlflow postgres minio
```

**Pros:** Always accessible, team-ready, no tunnel dependency
**Cons:** Monthly cost, requires cloud account, security hardening needed

### Option C: Post-Hoc Sync (Simplest MVP)

Training logs to local filesystem on the SkyPilot VM. After completion, sync artifacts to local MLflow:

```bash
# In SkyPilot run section:
run: |
  # Train with local mlruns/ directory
  MLFLOW_TRACKING_URI=mlruns python -m minivess.pipeline.train_fold ...

  # Sync artifacts to cloud bucket after training
  aws s3 sync mlruns/ s3://minivess-mlruns/
```

Then locally:
```bash
# Download and import
aws s3 sync s3://minivess-mlruns/ /tmp/remote-mlruns/
python scripts/import_remote_runs.py /tmp/remote-mlruns/
```

**Pros:** Zero infrastructure changes, works immediately
**Cons:** No real-time monitoring, requires post-hoc import script, no system metrics during training

### Option D: Managed MLflow

- [Nebius Managed MLflow](https://nebius.com) — Integrated with SkyPilot
- [Databricks Managed MLflow](https://databricks.com/product/managed-mlflow) — Enterprise, $$
- [MLflow on Kubernetes](https://mlflow.org/docs/latest/deployment/deploy-model-to-kubernetes) — Self-managed

**Pros:** Zero maintenance, always accessible
**Cons:** Cost ($100+/month), vendor lock-in, may not support all MLflow features

---

## 11. Prefect + SkyPilot Integration {#11-prefect-skypilot-integration}

### Official Pattern

SkyPilot provides an [official Prefect integration example](https://docs.skypilot.co/en/latest/examples/orchestrators/prefect.html):

```python
@task(name='run_sky_task', retries=2, retry_delay_seconds=30)
def run_sky_task(task_config: dict, cluster_prefix: str = 'prefect') -> str:
    task = sky.Task.from_yaml_config(task_config)
    cluster_name = f"{cluster_prefix}-{task_name}-{uuid4().hex[:8]}"
    sky.launch(task, cluster_name=cluster_name, stream_logs=True)
    sky.down(cluster_name)
    return cluster_name
```

### MinIVess Integration Design

**Key insight:** The Prefect flow runs locally. Only the training computation runs on SkyPilot. The flow wraps the SkyPilot launch/poll/teardown cycle.

```python
@flow(name="training-flow")
def training_flow(
    config_dict: dict,
    compute_backend: str = "local_docker",  # "local_docker" | "skypilot"
) -> TrainingFlowResult:
    """Training flow with compute backend dispatch."""

    if compute_backend == "skypilot":
        # Launch on SkyPilot, poll until complete
        launcher = SkyPilotLauncher()
        result = launcher.launch_training_job(config_dict)
        job_status = wait_sky_job(result["job_id"])

        if job_status["status"] != "SUCCEEDED":
            raise RuntimeError(f"SkyPilot job failed: {job_status}")

        # Results already logged to MLflow by the remote training script
        return TrainingFlowResult(status="completed", source="skypilot")

    else:
        # Existing local Docker path
        for fold in range(config_dict["num_folds"]):
            train_one_fold_task(config_dict, fold=fold)
        return TrainingFlowResult(status="completed", source="local_docker")
```

### SkyPilot API Server (Optional, Recommended)

For shared state across Prefect tasks, use the SkyPilot API server:

```bash
export SKYPILOT_API_SERVER_ENDPOINT=http://localhost:46580
sky api start
```

All `sky.launch()`, `sky.jobs.launch()`, `sky.jobs.queue()` calls route through the server, enabling:
- Multiple Prefect tasks to see the same cluster/job state
- Dashboard at `http://localhost:46580`
- Persistent across process restarts

---

## 12. HPO Sweep Architecture {#12-hpo-sweep-architecture}

### Current Design (Local Only)

```python
# hpo_flow.py (simplified)
@flow(name="hpo-flow")
def hpo_flow(hpo_config: str, n_trials: int):
    engine = HPOEngine(study_name=..., storage=OPTUNA_STORAGE_URL)
    for i in range(n_trials):
        trial = engine.study.ask()
        # Dispatches to local Docker gpu-pool (one at a time)
        run_deployment("training-flow/default", params={...})
```

### SkyPilot HPO Design (Parallel Cloud Trials)

**Option 1: Pre-generated Grid (Recommended for MVP)**

```python
@flow(name="hpo-flow")
def hpo_flow(hpo_config: str, n_trials: int, compute_backend: str = "local_docker"):
    # Phase 1: Generate trial configs locally using Optuna
    engine = HPOEngine(study_name=..., storage=OPTUNA_STORAGE_URL)
    trial_configs = []
    for _ in range(n_trials):
        trial = engine.study.ask()
        trial_configs.append({
            "learning_rate": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [1, 2, 4]),
            "loss_name": trial.suggest_categorical("loss", ["dice_ce", "cbdice_cldice"]),
        })

    if compute_backend == "skypilot":
        # Phase 2: Launch all trials in parallel on SkyPilot
        launcher = SkyPilotLauncher()
        job_ids = []
        for i, config in enumerate(trial_configs):
            result = launcher.launch_training_job(
                config, task_yaml="deployment/skypilot/train_generic.yaml"
            )
            job_ids.append(result["job_id"])

        # Phase 3: BARRIER — wait for all trials to complete
        for job_id in job_ids:
            status = wait_sky_job(job_id, max_wait=timedelta(hours=24))
            if status["status"] != "SUCCEEDED":
                logger.warning("Trial %s failed: %s", job_id, status)

    else:
        # Local sequential execution (existing path)
        for config in trial_configs:
            run_deployment("training-flow/default", params=config)

    # Phase 4: Compare results in MLflow
    return HPOFlowResult(n_trials=n_trials, source=compute_backend)
```

**Option 2: SkyPilot Worker Pools (v0.11+ Beta)**

```yaml
# deployment/skypilot/hpo_pool.yaml
name: minivess-hpo-pool

pool:
  min_workers: 1
  max_workers: 8
  queue_length_threshold: 1

resources:
  cloud: runpod
  accelerators: A100-80GB:1

setup: |
  pip install uv && cd ~/sky_workdir && uv sync --no-dev
```

Worker pools pre-provision GPU instances and reuse them across trials. Each trial gets `$SKYPILOT_JOB_RANK` for identification.

**Option 3: Shared Optuna Study via Cloud PostgreSQL**

If PostgreSQL is accessible from SkyPilot VMs (tunnel or cloud deploy), each worker runs `study.optimize()` independently:

```python
# Each SkyPilot worker:
study = optuna.load_study(
    study_name="${STUDY_NAME}",
    storage="${OPTUNA_STORAGE_URL}"  # Must be routable
)
study.optimize(objective, n_trials=1)  # One trial per worker
```

This is the most sophisticated option — TPE sampler adapts in real-time as trials complete. But requires PostgreSQL accessibility.

---

## 13. Flow Orchestration: Completion Barriers {#13-flow-orchestration}

### The Problem

When training runs 128 HPO combinations, Analysis Flow should only run AFTER all 128 complete. The current trigger chain runs flows sequentially:

```python
# trigger.py (current)
chain = PipelineTriggerChain()
chain.register("data", data_flow)
chain.register("train", training_flow)   # Runs ONE training
chain.register("analyze", analysis_flow)  # Runs immediately after
chain.register("deploy", deploy_flow)
chain.register("dashboard", dashboard_flow)
```

### Solution: HPO Flow as the Barrier

The HPO flow already iterates trials. Making it the barrier point:

```
PipelineTriggerChain:
  data_flow → hpo_flow (contains training + barrier) → analysis_flow → deploy → dashboard
                │
                ├── trial 1 (SkyPilot or local)
                ├── trial 2
                ├── ...
                └── trial N
                │
                BARRIER: all trials terminal
                │
                ▼
           analysis_flow (reads ALL MLflow runs from experiment)
```

**Key design:** Analysis Flow doesn't need to know about HPO. It queries MLflow for all FINISHED runs in the experiment and builds ensembles from whatever it finds. The HPO flow is responsible for ensuring all trials are done before returning.

### Biostatistics Consideration

The user correctly identified: **biostatistics (Bayesian model comparison, posterior estimation) also needs all runs complete.** Running biostatistics after each individual MLflow run is wasteful — the posterior updates are meaningful only with the full ensemble.

**Solution:** Biostatistics runs as part of (or after) Analysis Flow, not Training Flow:

```
HPO Flow (all trials complete)
    → Analysis Flow (ensembles + biostatistics)
        → Deploy Flow (champion selection)
            → Dashboard Flow (figures + reports)
```

This is already the intended architecture — Analysis Flow already contains ensemble building and evaluation. Biostatistics is a natural extension of the analysis phase.

---

## 14. Spot Instance Checkpointing {#14-spot-checkpointing}

### SkyPilot Checkpointing Pattern

Data on the VM's local disk is **LOST** on preemption. Only data in cloud bucket mounts survives.

```yaml
file_mounts:
  /checkpoints:
    name: minivess-checkpoints
    store: s3
    mode: MOUNT_CACHED  # 9x faster writes than MOUNT
```

### Training Code Requirements

The training loop must:

1. **Save checkpoints periodically** to `/checkpoints/`:
   ```python
   # Every N epochs:
   torch.save({
       'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'scheduler_state_dict': scheduler.state_dict(),
       'best_metric': best_metric,
       'rng_state': torch.random.get_rng_state(),
   }, f"/checkpoints/{experiment_name}/fold{fold}/epoch_{epoch}.pt")
   ```

2. **Resume from latest checkpoint on startup:**
   ```python
   checkpoint_dir = Path("/checkpoints") / experiment_name / f"fold{fold}"
   checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"),
                        key=lambda p: int(p.stem.split('_')[-1]),
                        reverse=True)
   for ckpt_path in checkpoints:
       try:
           ckpt = torch.load(ckpt_path, weights_only=False)
           model.load_state_dict(ckpt['model_state_dict'])
           start_epoch = ckpt['epoch'] + 1
           break
       except Exception:
           continue
   ```

3. **Use `$SKYPILOT_TASK_ID`** for grouping runs across preemption recoveries:
   ```python
   mlflow.set_tag("skypilot_task_id", os.environ.get("SKYPILOT_TASK_ID", "local"))
   ```

### Recovery Strategy

```yaml
resources:
  job_recovery:
    strategy: FAILOVER  # Try same region, then failover to next
    max_restarts_on_errors: 3
    recover_on_exit_codes: [137]  # OOM kill → retry with adjusted batch size
```

---

## 15. Multi-Answer Questions {#15-multi-answer-questions}

These questions need resolution before or during implementation. Each has a recommended answer based on the analysis above.

### Q1: MLflow Accessibility Strategy

**Question:** How should MLflow be made accessible from SkyPilot VMs?

| Option | Complexity | Monthly Cost | Best For |
|--------|-----------|-------------|----------|
| **A: Cloudflare Tunnel** | Low | Free | Solo researcher (NOW) |
| B: Cloud VM | Medium | $15-30 | Team of 2-5 |
| C: Post-hoc sync | Lowest | Free | Offline-first workflow |
| D: Managed MLflow | Low | $100+ | Enterprise |

**Recommendation:** Start with **A** (Cloudflare Tunnel). Zero cost, zero firewall changes. Migrate to **B** when team grows or tunnel reliability becomes a bottleneck.

### Q2: Prefect on SkyPilot VMs

**Question:** Should SkyPilot-launched training scripts use Prefect?

| Option | Architecture | Pros | Cons |
|--------|-------------|------|------|
| **A: No Prefect on VM** | Local Prefect wraps SkyPilot launch/wait | Simpler, no Prefect accessibility needed | Less granular flow visibility |
| B: Prefect on VM | VM connects to Prefect API via tunnel | Full flow visibility, retries | Tunnel dependency, complexity |

**Recommendation:** **A** for MVP. The local Prefect flow owns the lifecycle. SkyPilot VMs are "dumb" training workers that log to MLflow only. Prefect visibility comes from the local flow wrapping the SkyPilot job.

### Q3: HPO Parallel Strategy

**Question:** How should parallel HPO trials execute on SkyPilot?

| Option | Mechanism | Optuna Integration | Complexity |
|--------|-----------|-------------------|-----------|
| **A: Pre-generated grid** | Optuna generates configs locally, SkyPilot launches N jobs | Static (no adaptive sampling) | Low |
| B: Worker pools | SkyPilot v0.11 pools, shared Optuna DB | Full (TPE adapts) | High |
| C: Independent Optuna | Each worker connects to shared PostgreSQL | Full | Medium (requires DB accessibility) |

**Recommendation:** **A** for MVP. Pre-generate trial configs using Optuna's TPE sampler, launch all as independent SkyPilot jobs with `--async`. For later phases, migrate to **C** when PostgreSQL is cloud-accessible.

### Q4: Checkpoint Storage Backend

**Question:** Where should SkyPilot training save checkpoints?

| Option | Speed | Cost | Persistence |
|--------|-------|------|-------------|
| **A: S3 via MOUNT_CACHED** | 9x faster than MOUNT | S3 pricing | Survives preemption |
| B: S3 via MOUNT | Slow writes | S3 pricing | Survives preemption |
| C: Local disk + post-hoc upload | Fast writes | Free during training | Lost on preemption |

**Recommendation:** **A** (MOUNT_CACHED). The existing `train_generic.yaml` uses MOUNT — should upgrade to MOUNT_CACHED for 9x faster checkpoint writes.

### Q5: Training Entry Point for SkyPilot

**Question:** What command should SkyPilot VMs run?

| Option | Command | Prefect Required | Docker Required |
|--------|---------|-----------------|-----------------|
| A: Prefect deployment | `prefect deployment run 'training-flow/default'` | Yes (tunnel) | No (SkyPilot VM) |
| **B: Direct Python module** | `python -m minivess.pipeline.train_fold` | No | No |
| C: Docker in SkyPilot | `docker run minivess-train` | Optional | Yes |

**Recommendation:** **B** for MVP. Direct Python module invocation avoids both Prefect and Docker dependencies on the SkyPilot VM. The module logs to MLflow directly. Note: this requires extracting the fold training logic into a standalone-importable module (not a Prefect task).

**Important caveat:** This does NOT violate the "no standalone scripts" rule (CLAUDE.md Rule #17). The SkyPilot VM is executing inside a managed compute environment with proper checkpointing, artifact storage, and lifecycle management — it's not a "dev shortcut." The Prefect flow still orchestrates the overall pipeline locally.

### Q6: When to Trigger Downstream Flows After HPO

**Question:** How does the pipeline know when ALL HPO trials are done?

| Option | Mechanism | Reliability |
|--------|-----------|-------------|
| **A: HPO flow polls all job statuses** | `wait_sky_job()` for each job_id | High (explicit barrier) |
| B: MLflow run count check | Query MLflow: `len(runs) == n_trials` | Medium (race conditions) |
| C: Prefect artifacts | Each trial writes a Prefect artifact on completion | High but complex |

**Recommendation:** **A**. The HPO flow maintains a list of SkyPilot job IDs and polls each until terminal. Only after ALL are terminal does it return, allowing the trigger chain to proceed to Analysis Flow.

---

## 16. Minimum Viable Architecture {#16-minimum-viable-architecture}

### What to Build First (Unblocks Everything)

1. **Cloudflare Tunnel for MLflow** — 1 hour setup, zero cost, unblocks all remote training
2. **`train_fold` standalone module** — Extract fold training logic from Prefect tasks into importable module
3. **Compute dispatcher in training flow** — `compute_backend` parameter: "local_docker" | "skypilot"
4. **Update `train_generic.yaml`** — MOUNT_CACHED, job_recovery, secrets, correct MLFLOW_TRACKING_URI
5. **HPO flow with SkyPilot dispatch** — Pre-generated grid, parallel `--async` launches, barrier

### What This Enables

After these 5 items:
- Any model can be trained on RunPod/Lambda/GCP via `compute_backend=skypilot`
- HPO sweeps run in parallel on cloud GPUs (N trials simultaneously)
- Results are tracked in local MLflow (via tunnel)
- Analysis Flow triggers after all HPO trials complete
- Dashboard Flow generates figures from the full experiment

### What to Defer

- Pulumi cloud infrastructure (not needed for solo researcher)
- SkyPilot worker pools (beta, complex)
- Shared Optuna PostgreSQL (pre-generated grid is sufficient)
- Cloud-deployed MLflow (tunnel works for now)
- Prefect on SkyPilot VMs (local Prefect wraps SkyPilot)

---

## 17. Implementation Roadmap {#17-implementation-roadmap}

### Phase 0: Service Accessibility (Week 1)

| Task | Description | Effort |
|------|-------------|--------|
| T0.1 | Install cloudflared, create tunnel for MLflow | 1h |
| T0.2 | Add `MLFLOW_TUNNEL_URL` to `.env.example` | 15min |
| T0.3 | Test MLflow logging from external network | 30min |
| T0.4 | Document tunnel setup in `deployment/CLAUDE.md` | 30min |

### Phase 1: Compute Dispatcher (Week 1-2)

| Task | Description | Effort |
|------|-------------|--------|
| T1.1 | Extract `train_fold` module from train_flow.py tasks | 4h |
| T1.2 | Add `compute_backend` parameter to training_flow() | 2h |
| T1.3 | Implement SkyPilot dispatch path (launch + wait) | 4h |
| T1.4 | Update `train_generic.yaml` (MOUNT_CACHED, recovery, secrets) | 2h |
| T1.5 | Add `skypilot` extras group to pyproject.toml | 30min |
| T1.6 | End-to-end test: DynUNet training on RunPod via SkyPilot | 2h |

### Phase 2: HPO on SkyPilot (Week 2-3)

| Task | Description | Effort |
|------|-------------|--------|
| T2.1 | HPO flow: pre-generate trial grid from Optuna | 3h |
| T2.2 | HPO flow: parallel SkyPilot `--async` launches | 3h |
| T2.3 | HPO flow: completion barrier (poll all job_ids) | 2h |
| T2.4 | HPO flow: results aggregation from MLflow | 2h |
| T2.5 | End-to-end test: 4-trial HPO sweep on RunPod | 2h |

### Phase 3: Trigger Chain + Analysis (Week 3-4)

| Task | Description | Effort |
|------|-------------|--------|
| T3.1 | Update PipelineTriggerChain for HPO → Analysis | 3h |
| T3.2 | Analysis Flow: handle multi-experiment ensemble building | 3h |
| T3.3 | Dashboard Flow: SkyPilot cost tracking section | 2h |
| T3.4 | End-to-end: HPO → Analysis → Deploy → Dashboard | 4h |

### Phase 4: Synthetic Generation on Cloud GPUs (Week 4+)

| Task | Description | Effort |
|------|-------------|--------|
| T4.1 | VascuSynth integration (CPU, runs anywhere) | 4h |
| T4.2 | VesselVAE training on RunPod RTX 4090 | 8h |
| T4.3 | VQ-VAE patch training on RunPod A100 | 8h |
| T4.4 | Synthetic → drift detection pipeline | 4h |
| T4.5 | SAM3 fine-tuning on RunPod H100 | 8h |

### Phase 5: Production Hardening (Deferred)

| Task | Description | When |
|------|-------------|------|
| T5.1 | Cloud-deploy MLflow + PostgreSQL (Pulumi) | When team grows |
| T5.2 | SkyPilot worker pools for HPO | When v0.11 stable |
| T5.3 | Shared Optuna study on cloud PostgreSQL | With T5.1 |
| T5.4 | SkyPilot API server for shared state | When multi-user |
| T5.5 | Cost alerting + budget caps | When monthly spend >$500 |

---

## Appendix A: Reference URLs

### SkyPilot Documentation
- [SkyPilot Docs Index](https://docs.skypilot.co/en/latest/docs/index.html)
- [YAML Spec Reference](https://docs.skypilot.co/en/latest/reference/yaml-spec.html)
- [Managed Jobs](https://docs.skypilot.co/en/latest/examples/managed-jobs.html)
- [Training Guide (Checkpointing)](https://docs.skypilot.co/en/latest/reference/training-guide.html)
- [Distributed Training](https://docs.skypilot.co/en/latest/running-jobs/distributed-jobs.html)
- [Many Parallel Jobs (HPO)](https://docs.skypilot.co/en/latest/running-jobs/many-jobs.html)
- [Worker Pools](https://docs.skypilot.co/en/latest/examples/pools.html)
- [Prefect Integration](https://docs.skypilot.co/en/latest/examples/orchestrators/prefect.html)

### RunPod
- [RunPod GPU Pricing](https://www.runpod.io/gpu-pricing)
- [RunPod SkyPilot Integration](https://docs.runpod.io/integrations/skypilot)
- [RunPod Pod Catalog (GitHub)](https://github.com/runpod/pod-catalog)
- [RunPod Status](https://uptime.runpod.io/)

### Production References
- [Nebius: Orchestrating LLM Fine-Tuning with SkyPilot + MLflow](https://nebius.com/blog/posts/orchestrating-llm-fine-tuning-k8s-skypilot-mlflow)
- [Alex Kim: ML Experiments with SkyPilot + MLflow](https://alex000kim.com/posts/2025-01-11-llm-fine-tune-skypilot-mlflow/)
- [Shopify: SkyPilot at Scale](https://shopify.engineering/skypilot)
- [SkyPilot Blog: GPU Neoclouds](https://blog.skypilot.co/ai-job-orchestration-pt1-gpu-neoclouds/)
- [SkyPilot Blog: Job Groups](https://blog.skypilot.co/job-groups/)

### Cost Comparison
- [ComputePrices: AWS vs RunPod](https://computeprices.com/compare/aws-vs-runpod)
- [Northflank: Cheapest Cloud GPU Providers 2026](https://northflank.com/blog/cheapest-cloud-gpu-providers)
- [AWS GPU Price Cuts (June 2025)](https://aws.amazon.com/blogs/aws/announcing-up-to-45-price-reduction-for-amazon-ec2-nvidia-gpu-accelerated-instances/)

---

## Appendix B: Existing Code Inventory

| File | Lines | Status | Gap |
|------|-------|--------|-----|
| `src/minivess/compute/skypilot_launcher.py` | 181 | Tested | Never called by training flow |
| `src/minivess/compute/prefect_sky_tasks.py` | 87 | Tested | Never imported by any flow |
| `deployment/skypilot/train_generic.yaml` | 58 | Valid | Uses MOUNT (not MOUNT_CACHED), no job_recovery |
| `deployment/skypilot/train_hpo_sweep.yaml` | 42 | Valid | Not integrated with HPO flow |
| `src/minivess/optimization/hpo_engine.py` | 100+ | Tested | Local only, no SkyPilot dispatch |
| `src/minivess/orchestration/flows/hpo_flow.py` | 100+ | Tested | Local gpu-pool only |
| `src/minivess/orchestration/flows/train_flow.py` | 775 | Tested | Ignores `compute` param |
| `src/minivess/orchestration/trigger.py` | 100+ | Tested | No HPO barrier pattern |
| `src/minivess/orchestration/deployments.py` | 50+ | Tested | No sky-gpu-pool |
| `.env.example` | 157 | Complete | MLFLOW_SKYPILOT_HOST defined but unused |
| `tests/v2/unit/test_skypilot_*.py` | 200+ | Passing | Good coverage of existing code |
