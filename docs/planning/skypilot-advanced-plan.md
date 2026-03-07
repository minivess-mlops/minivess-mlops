# SkyPilot Advanced Plan: Multi-Cloud Compute for All Prefect Flows

**Date:** 2026-03-07
**Branch:** `feat/flow-biostatistics` (planning only; implementation tracked per-issue)
**Closes/Informs:** #366 (VesselFM RunPod fine-tuning)
**Status:** Planning document — reviewed by multi-hypothesis decision matrix

---

## 1. Executive Summary

MinIVess MLOps uses Prefect 3.x for workflow orchestration and SkyPilot for multi-cloud
GPU compute provisioning. SkyPilot's multi-cloud failover (AWS → GCP → RunPod → Lambda)
provides the 3-6x cost savings described in CLAUDE.md.

The core value proposition is **transparent compute selection**: a researcher runs
`./scripts/train_all_best.sh` and the system finds the cheapest available GPU across all
configured cloud providers, handling provisioning, data mounting, training, and teardown.

This document:
1. Maps each of the six Prefect flows to its optimal compute target
2. Presents a multi-hypothesis decision matrix for how SkyPilot and Prefect integrate
3. Identifies gaps in the current implementation and prioritizes fixes
4. Plans the VesselFM fine-tuning pipeline on RunPod (closing issue #366)

---

## 2. Current State Assessment

### 2.1 What Exists

The codebase already has a solid SkyPilot foundation:

| File | Purpose | Status |
|------|---------|--------|
| `deployment/skypilot/train_generic.yaml` | Multi-cloud spot training (AWS→GCP→RunPod→Lambda) | Working, gaps below |
| `deployment/skypilot/train_hpo_sweep.yaml` | Parallel HPO sweep | Working, same gaps |
| `src/minivess/compute/skypilot_launcher.py` | Python SDK wrapper, dry-run support | Working |
| `src/minivess/compute/prefect_sky_tasks.py` | Prefect `@task` wrappers for SkyPilot | Working |

The existing `train_generic.yaml` already implements multi-cloud failover with RunPod:
```yaml
any_of:
  - cloud: aws
    accelerators: A100:1
  - cloud: gcp
    accelerators: A100:1
  - cloud: runpod
    accelerators: A100:1
  - cloud: lambda
    accelerators: A100:1
```

### 2.2 Confirmed Gaps

From research and codebase analysis, five concrete gaps need addressing:

| # | Gap | Impact | Fix |
|---|-----|--------|-----|
| G1 | `/checkpoints` uses `mode: MOUNT` (not `MOUNT_CACHED`) | 9.6x slower checkpoint writes | Switch to `MOUNT_CACHED` + `disk_tier: best` |
| G2 | `MLFLOW_TRACKING_URI: http://mlflow:5000` — unreachable from cloud | Tracking broken for remote runs | Must be public URI or MLflow S3 artifact backend |
| G3 | `wait_sky_job` polls with `time.sleep()` instead of `sky.stream_and_get()` | No log streaming to Prefect UI | Swap to blocking `sky.stream_and_get()` |
| G4 | No VesselFM-specific SkyPilot config (10GB VRAM, RunPod-first) | VesselFM fine-tuning unscripted | Add `finetune_vesselfm.yaml` |
| G5 | RunPod spot: 5-second SIGTERM warning (vs AWS 2-min) | Checkpoints lost on preemption | Set `CHECKPOINT_INTERVAL_EPOCHS=1` for RunPod |

---

## 3. Prefect Flow-to-Compute Mapping

Each flow has different compute requirements. The key insight: **only Flow 2 (Train) needs
GPU**. All other flows are CPU-bound and should stay on-prem.

| Flow | Name | GPU? | VRAM | Compute Target | Rationale |
|------|------|------|------|----------------|-----------|
| Flow 0 | Acquisition | No | — | On-prem Docker | Data import; no compute |
| Flow 1 | Data Engineering | No | — | On-prem Docker | Profiling, splits; CPU-only |
| **Flow 2** | **Training (DynUNet, Mamba)** | **Yes** | **8GB** | **SkyPilot → RunPod/Lambda/AWS** | Single A100; spot OK |
| **Flow 2** | **Training (VesselFM)** | **Yes** | **10GB** | **SkyPilot → RunPod A100 first** | Needs >8GB; RunPod cheapest |
| **Flow 2** | **Training (SAM3)** | **Yes** | **16GB+** | **SkyPilot → CoreWeave/AWS SXM** | 16GB min; high-bandwidth preferred |
| Flow 2.5 | Post-Training (SWA, CP) | No | — | On-prem Docker | Model averaging; CPU-bound |
| Flow 3 | Analysis | No | — | On-prem Docker | Ensemble eval; data is local |
| Flow 4 | Deployment | No | — | On-prem Docker | ONNX export; CPU-bound |
| Flow 5 | Dashboard | No | — | On-prem Docker | Figure generation; CPU |
| Flow 6 | QA | No | — | On-prem Docker | MLflow validation; trivial |
| HPO | Hyperparameter Search | Yes | 8GB+ | SkyPilot parallel sweep | N trials in parallel |

**Rule:** If a flow doesn't use GPU, it stays on-prem. SkyPilot manages only GPU workloads.

---

## 4. Multi-Hypothesis Decision Matrix: SkyPilot + Prefect Integration

There are four architectural hypotheses for how SkyPilot and Prefect connect. This matrix
evaluates them for the MinIVess use case (single researcher, on-prem base, cloud for GPU).

### H1: SkyPilot as Prefect @task (Current Pattern)

```
Local Machine:
  Prefect Worker (CPU)
    └── @flow training_flow()
          └── @task launch_sky_training()  ← SkyPilotLauncher.launch_training_job()
                    ↓ sky.jobs.launch()
              Cloud Instance (GPU):
                training_flow() [via Prefect deployment run]
```

**How it works**: The local Prefect worker calls `sky.jobs.launch()` inside a `@task`.
SkyPilot provisions a cloud GPU instance and runs the training there. The task blocks
until the job completes (via polling or `sky.stream_and_get()`).

| Dimension | Score | Notes |
|-----------|-------|-------|
| Prefect UI visibility | ★★☆ | SkyPilot job is a black box to Prefect during training |
| Multi-cloud failover | ★★★ | SkyPilot handles AWS → GCP → RunPod → Lambda |
| DevEx | ★★★ | Single `training_flow()` call handles everything |
| Ops overhead | ★★★ | No extra infrastructure |
| State management | ★★☆ | SkyPilot state is local; breaks on ephemeral Prefect workers |
| Log streaming | ★☆☆ | Manual polling unless using `sky.stream_and_get()` |

**Verdict**: Best for solo researcher, current implementation. Fix G3 (stream_and_get) to
improve log visibility.

---

### H2: SkyPilot API Server (Centralized State)

```
Shared Infrastructure:
  SkyPilot API Server (long-lived VM or K8s pod)
    ↑ SKYPILOT_API_SERVER_ENDPOINT env var
  Prefect Worker (ephemeral or local)
    └── @task launch_sky_training()
          └── sky.jobs.launch()  ← routed to API Server
```

**How it works**: A shared SkyPilot API Server holds cluster state. Multiple Prefect workers
(even ephemeral ones on different machines) share the same view of running jobs and clusters.

| Dimension | Score | Notes |
|-----------|-------|-------|
| Prefect UI visibility | ★★☆ | Same as H1 |
| Multi-cloud failover | ★★★ | Same as H1 |
| DevEx | ★★☆ | Requires API Server deployment |
| Ops overhead | ★☆☆ | Extra VM/pod for API Server |
| State management | ★★★ | Centralized, survives worker restarts |
| Log streaming | ★☆☆ | Same as H1 |

**Verdict**: Required for team environments. Deferred for solo researcher setup.
Implement when team grows or when running jobs from multiple machines.

---

### H3: SkyPilot Launches Prefect Flow on Remote (Reverse Pattern)

```
Local Machine:
  sky jobs launch train_generic.yaml
    ↓
Cloud Instance (GPU):
  prefect deployment run 'training-flow/default' ← run: block in YAML
    ↓ PREFECT_API_URL must be reachable from cloud
  Prefect runs training_flow() inside the GPU instance
```

**How it works**: `train_generic.yaml`'s `run:` block already calls
`prefect deployment run 'training-flow/default'`. SkyPilot acts as the compute provisioner;
Prefect orchestrates the training logic from *inside* the cloud instance.

| Dimension | Score | Notes |
|-----------|-------|-------|
| Prefect UI visibility | ★★★ | Full flow visibility — runs inside Prefect agent on GPU |
| Multi-cloud failover | ★★★ | SkyPilot handles |
| DevEx | ★★☆ | `PREFECT_API_URL` must be public (or VPN/tunnel) |
| Ops overhead | ★★☆ | Prefect Server must be reachable from cloud |
| State management | ★★★ | Prefect state stored on Prefect Server |
| Log streaming | ★★★ | Prefect streams logs from flow tasks |

**Verdict**: Best Prefect experience, but requires Prefect Server to be publicly accessible
(or use Prefect Cloud). Current local dev setup (`PREFECT_DISABLED=1`) won't work.
**Recommended for production (Prefect Cloud or public-IP self-hosted Prefect Server).**

---

### H4: Direct RunPod API (No SkyPilot)

```
Local Machine:
  runpod.create_pod() ← runpod-python SDK
    ↓ Pod IP
  SSH into pod → rsync data → run training
  Tail logs via SSH
  rsync artifacts back
  runpod.terminate_pod()
```

**How it works**: Bypass SkyPilot entirely; use RunPod's Python SDK or `runpodctl` CLI
directly to provision a pod, sync data, run training, and tear down.

| Dimension | Score | Notes |
|-----------|-------|-------|
| Prefect UI visibility | ★☆☆ | None — pure shell |
| Multi-cloud failover | ★☆☆ | RunPod only |
| DevEx | ★★★ | Simple; fastest pod start (~20s vs 2-5min for SkyPilot) |
| Ops overhead | ★★★ | Zero extra infrastructure |
| State management | ★☆☆ | All state in shell variables |
| Log streaming | ★★★ | SSH tail -f works perfectly |

**Verdict**: Best for one-off VesselFM fine-tuning on a specific RunPod GPU type, or when
SkyPilot's RunPod support is flaky. Implement as the fallback for issue #366.
**The `finetune_vesselfm_runpod.sh` script should try SkyPilot first, fall back to direct
RunPod API if SkyPilot unavailable.**

---

### Recommended Integration Strategy

| Use Case | Recommended Approach |
|----------|---------------------|
| Local dev (no cloud, 8GB GPU) | Direct `training_flow()` via `run_training_flow.py` |
| DynUNet/Mamba on cloud spot | H1: SkyPilot @task in `training_flow()` |
| VesselFM fine-tuning | H4 (direct RunPod) or H1 (SkyPilot YAML) — both in `finetune_vesselfm_runpod.sh` |
| SAM3 fine-tuning (16GB+) | H1: SkyPilot with CoreWeave/AWS SXM in `any_of:` |
| HPO sweep (N parallel trials) | H1: SkyPilot `launch_hpo_sweep()` |
| Team production (multi-user) | H2: SkyPilot API Server + H3 (Prefect Cloud) |

---

## 5. Cloud Provider Selection per Flow

Based on GPU requirements and cost analysis (2026-03-07 prices):

### Flow 2: Training (Standard — DynUNet, Mamba, ~8GB)

```yaml
# Priority: cheapest spot A100 PCIe
any_of:
  - cloud: runpod          # $0.60/hr A100 (Community Cloud)
    accelerators: A100:1
  - cloud: lambda          # $1.29/hr A100 (no spot, reliable)
    accelerators: A100:1
  - cloud: aws
    accelerators: A100:1   # ~$1.00-1.60/hr spot
  - cloud: gcp
    accelerators: A100:1   # ~$1.50/hr spot
```

### Flow 2: VesselFM Fine-Tuning (~10GB min, 16GB preferred)

```yaml
# Priority: cheapest large GPU; RunPod first
any_of:
  - cloud: runpod
    accelerators: {A100-80GB:1}  # $1.50/hr — sufficient headroom
  - cloud: runpod
    accelerators: {A100:1}       # $0.60/hr — tight on 10GB requirement
  - cloud: lambda
    accelerators: {A100:1}
  - cloud: aws
    accelerators: {A100:1}
```

### Flow 2: SAM3 Fine-Tuning (16GB+, 848M params)

```yaml
# Priority: high-bandwidth interconnect; SXM preferred
any_of:
  - cloud: coreweave
    accelerators: {A100-SXM:1}   # HPC-grade InfiniBand
  - cloud: aws
    accelerators: {A100:1}       # ~$2.00/hr spot
  - cloud: gcp
    accelerators: {A100:1}
  - cloud: runpod
    accelerators: {A100-80GB:1}  # fallback, 5s SIGTERM
```

### HPO Sweep (N parallel trials)

```yaml
# N separate managed jobs, each on cheapest available GPU
# SkyPilot 0.11 Pools feature ideal here
any_of:
  - cloud: runpod
    accelerators: A100:1
  - cloud: lambda
    accelerators: A100:1
  - cloud: aws
    accelerators: A100:1
```

---

## 6. Data Management for Remote Training

### Problem

The `MLFLOW_TRACKING_URI: http://mlflow:5000` in `train_generic.yaml` assumes Docker
Compose networking, which is unreachable from AWS/RunPod instances.

### Solutions (Ordered by Simplicity)

**Option A: MLflow on Public IP (Recommended for self-hosted)**
```bash
# In docker-compose.yml, expose MLflow on the host's public IP
ports:
  - "0.0.0.0:5000:5000"

# In train_generic.yaml:
envs:
  MLFLOW_TRACKING_URI: http://<your-public-ip>:5000
```

**Option B: MLflow S3 Artifact Store**
```bash
# MLflow stores to S3 directly; tracking server is optional
MLFLOW_TRACKING_URI=s3://minivess-mlruns/
# Remote instance needs AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
```

**Option C: Tunnel (for dev)**
```bash
# SSH tunnel from cloud instance to local MLflow
ssh -R 5000:localhost:5000 user@cloud-instance
```

### Dataset Mounting

Current `train_generic.yaml` uses `mode: COPY` for `/data` — correct (read-heavy,
pre-fetch at startup). The key fix is checkpoints:

```yaml
# BEFORE (slow):
/checkpoints:
  name: minivess-checkpoints
  store: s3
  mode: MOUNT

# AFTER (9.6x faster checkpoint writes):
/checkpoints:
  name: minivess-checkpoints
  store: s3
  mode: MOUNT_CACHED    # async write to S3 via rclone VFS cache
```

---

## 7. RunPod-Specific Caveats

RunPod is now a natively supported SkyPilot provider (via `skypilot[runpod]`). However,
several caveats apply that affect the VesselFM fine-tuning design:

| Caveat | Impact | Mitigation |
|--------|--------|------------|
| **5-second SIGTERM warning** (vs AWS 2-min) | Checkpoint at end of epoch may not complete | Set `CHECKPOINT_INTERVAL_EPOCHS=1`; use `MOUNT_CACHED` for async writes |
| **Spot maturity** | RunPod spot support in SkyPilot is newer; possible edge cases | Test dry-run first; have Lambda as immediate fallback |
| **Volume name ≤30 chars** | Long names like `minivess-checkpoints-vesselfm` will fail | Use `mnvss-ckpts-vfm` |
| **No label support** | Can't filter by tags on RunPod volumes | Document workaround |
| **~20s pod start** | Faster than hyperscalers (2-5 min SkyPilot provisioning) | Advantage when iteration speed matters |

---

## 8. Implementation Plan

### P0: Fix Current Gaps (Immediate)

- [ ] **G1**: Switch `/checkpoints` from `MOUNT` to `MOUNT_CACHED` in `train_generic.yaml`
      and `train_hpo_sweep.yaml`
- [ ] **G2**: Document `MLFLOW_TRACKING_URI` setup for remote runs in `train_generic.yaml`
      comments; add `MLFLOW_TRACKING_URI` as a required env var placeholder
- [ ] **G3**: Replace `wait_sky_job` polling loop with `sky.stream_and_get()` in
      `prefect_sky_tasks.py` — blocks inside the Prefect task and streams logs
- [ ] **G4**: Create `deployment/skypilot/finetune_vesselfm.yaml` (see section 9)
- [ ] **G5**: Add `CHECKPOINT_INTERVAL_EPOCHS=1` env var to RunPod-specific YAML

### P1: VesselFM Fine-Tuning on RunPod (Closes #366)

- [ ] `scripts/finetune_vesselfm_runpod.sh` — end-to-end launcher (see section 9)
- [ ] `deployment/skypilot/finetune_vesselfm.yaml` — SkyPilot config
- [ ] Credential setup documentation (RunPod API key via `runpodctl config`)
- [ ] Test dry-run mode locally (`--dry-run`)

### P2: MLflow Remote Tracking Fix (Required for Production)

- [ ] Update `train_generic.yaml` to use `${MLFLOW_TRACKING_URI}` (environment-injected)
- [ ] Add setup instructions in `deployment/skypilot/README.md` for setting the public URI
- [ ] Option A (public IP) or Option B (S3 backend) — document both

### P3: SkyPilot API Server Deployment (Team Use)

- [ ] `deployment/docker-compose.skypilot.yml` — SkyPilot API Server service
- [ ] `deployment/skypilot/api-server-config.yaml` — authentication and cloud credentials
- [ ] Update `skypilot_launcher.py` to support `SKYPILOT_API_SERVER_ENDPOINT` env var
- [ ] Document multi-user setup in `docs/ops/skypilot-api-server.md`

### P4: SAM3 Fine-Tuning Config

- [ ] `deployment/skypilot/finetune_sam3.yaml` — 16GB+ GPU, CoreWeave/AWS SXM priority
- [ ] `scripts/finetune_sam3_cloud.sh` — wrapper script

---

## 9. VesselFM Fine-Tuning on RunPod (Issue #366)

### Problem

VesselFM fine-tuning requires ~10GB VRAM. The development machine (RTX 2070 Super, 8GB)
is insufficient. We need a script that:
1. Provisions a RunPod A100 pod via SkyPilot
2. Syncs project + data
3. Runs VesselFM fine-tuning (via `training_flow()`)
4. Downloads MLflow artifacts and checkpoints
5. Tears down pod automatically

### Script Entry Point

```bash
./scripts/finetune_vesselfm_runpod.sh --gpu A100 --epochs 100
./scripts/finetune_vesselfm_runpod.sh --dry-run   # no credentials needed
```

### Architecture

The script uses **SkyPilot as the primary path** (multi-cloud failover, managed spot,
automatic teardown) with a direct RunPod CLI fallback.

```
finetune_vesselfm_runpod.sh
  ├── Check SkyPilot available?
  │     Yes → sky jobs launch finetune_vesselfm.yaml --down
  │     No  → runpodctl create pod ... (direct API fallback)
  └── Poll/stream logs until completion
  └── Download artifacts: mlruns/ + checkpoints/
  └── Pod torn down automatically (--down flag or trap EXIT)
```

### SkyPilot YAML

See `deployment/skypilot/finetune_vesselfm.yaml` (created alongside this document).

### Cost Estimate

| GPU | Provider | Price | 8h run |
|-----|----------|-------|--------|
| A100 80GB | RunPod Community | $1.50/hr | $12 |
| A100 40GB | RunPod Community | $0.60/hr | $5 |
| A100 | Lambda | $1.29/hr | $10 |
| A100 | AWS spot | ~$1.20/hr | $10 |

Expected fine-tuning time: 3-6 hours for 100 epochs on MiniVess (70 volumes, 3 folds).
Total cost: **$5-18** depending on GPU and provider.

---

## 10. GitHub Actions Integration

For P2 (team use), a manual-trigger GitHub Action that launches VesselFM fine-tuning:

```yaml
# .github/workflows/finetune-vesselfm.yml
name: Fine-tune VesselFM on RunPod

on:
  workflow_dispatch:
    inputs:
      gpu_type:
        description: GPU type (A100 / A100-80GB)
        default: A100
      max_epochs:
        description: Training epochs
        default: '100'
      dry_run:
        description: Dry run mode
        type: boolean
        default: false

jobs:
  finetune:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - name: Setup SkyPilot credentials
        run: |
          pip install "skypilot[runpod,lambda]"
          runpodctl config --apiKey "${{ secrets.RUNPOD_API_KEY }}"
          sky check
      - name: Launch fine-tuning
        run: |
          bash scripts/finetune_vesselfm_runpod.sh \
            --gpu "${{ inputs.gpu_type }}" \
            --epochs "${{ inputs.max_epochs }}" \
            ${{ inputs.dry_run == 'true' && '--dry-run' || '' }}
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
          RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}
```

**Note**: The GitHub Action runner provides ephemeral infrastructure; SkyPilot state must
be persisted elsewhere (S3 or API Server) for the job to survive runner restarts.

---

## 11. Pulumi IaC (P1 from Issue #366)

Pulumi provides IaC for RunPod pod definitions. The local-state-backend approach means
no Pulumi Cloud subscription is required:

```bash
# Setup (one-time)
pulumi login --local      # local state, no Cloud subscription
cd deployment/pulumi/runpod_gpu/
pulumi up                 # provision A100 pod

# Teardown
pulumi destroy
```

**Decision**: Pulumi is optional for MinIVess. SkyPilot handles provisioning more
flexibly (multi-cloud failover, managed jobs). Pulumi is valuable only if reproducible,
version-controlled pod configurations are needed across team members.

**Verdict for issue #366**: Implement SkyPilot approach first. Add Pulumi as P2 only if
team needs reproducible infrastructure definitions.

---

## 12. Key Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary GPU compute | SkyPilot managed jobs | Multi-cloud failover, automatic teardown |
| Integration pattern | H1 (SkyPilot @task) for now; H3 (remote Prefect) for production | DevEx balance |
| VesselFM target | RunPod A100 first (cheapest), Lambda/AWS fallback | Cost + VRAM requirements |
| SAM3 target | CoreWeave SXM first (HPC-grade), AWS A100 fallback | High-bandwidth NVLink |
| Data mounting | COPY for input, MOUNT_CACHED for checkpoints | Performance |
| Spot interrupt handling | `CHECKPOINT_INTERVAL_EPOCHS=1` + async S3 write | 5s SIGTERM on RunPod |
| MLflow remote | Public IP (simplest) or S3 backend (production) | Researcher context |
| GitHub Actions | Manual workflow_dispatch trigger | On-demand; not automated |
| Pulumi | Deferred to P2 | SkyPilot covers the use case |

---

## References

- SkyPilot docs: https://docs.skypilot.co/en/latest/
- RunPod SkyPilot integration: https://docs.runpod.io/integrations/skypilot
- SkyPilot 0.11 blog (Dec 2025): https://blog.skypilot.co/announcing-skypilot-0.11.0/
- SkyPilot MOUNT_CACHED benchmark: https://blog.skypilot.co/high-performance-checkpointing/
- SkyPilot Prefect integration: https://docs.skypilot.co/en/latest/examples/orchestrators/prefect.html
- VesselFM adapter: `src/minivess/adapters/vesselfm.py`
- Existing SkyPilot YAML: `deployment/skypilot/train_generic.yaml`
- Issue #366: https://github.com/minivess-mlops/minivess-mlops/issues/366
