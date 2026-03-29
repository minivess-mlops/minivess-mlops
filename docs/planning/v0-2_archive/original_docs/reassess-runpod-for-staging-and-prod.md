# Reassessing RunPod for Staging and Production — Multi-Hypothesis Decision Matrix

**Date**: 2026-03-28
**Status**: Decision recommendation
**Context**: 26+ hour GCP GPU drought across europe-west4 AND us-central1, zero successful completions in 11th pass
**Methodology**: Iterated LLM Council (5+ expert perspectives)
**Budget constraint**: Academic project, $50-100/experiment pass maximum
**Branch**: `fix/10th-pass-production-readiness`

---

## User Prompt (Verbatim)

> "Let's start planning in an open-ended multi-hypothesis decision matrix sense what are the pros and cons of this Dockerless execution at Runpod, what would be the cost, what are the tradeoffs? As in should the mlflow always be ephemeral? so that we sync back the files to my local filesystem-based mlruns? Or should we actually know switch to Dagshub as this is now an academic project and that is meant for academic projects? It should be free then? Examine different choices that we should be making here, and analyze also the costs if we have the MLflow/DVC from dagshub? SAM3 weights would be cached on the persistent volume on Runpod? What would you estimate the cost to be if we trained with 4090 which we previously identified as the most cost-efficient option over 5090 for example, and we would be totally happy with consumer-grade GPUs as we should also demonstrate the consumer GPU use if any of the researchers have their own GPUs with no access to cloud credits due to institutional policies or some other weirdness."

---

## Council Panel

| Expert | Domain | Key Contribution |
|--------|--------|------------------|
| **Academic Research Infrastructure Specialist** | University ML lab practices | What peer labs use, Nature Protocols norms, free-tier ecosystem |
| **RunPod Platform Expert** | RunPod execution model, SkyPilot integration | Dockerless execution tradeoffs, Network Volume, pricing |
| **MLOps Architect** | Experiment tracking, artifact management | Ephemeral vs persistent MLflow, DagsHub integration, DVC strategy |
| **Cost Engineer** | Total cost of ownership modeling | Debug pass, production pass, monthly infrastructure costs |
| **Reproducibility Expert** | Consumer GPU friendliness, paper compatibility | Nature Protocols framing, researcher independence, zero-cloud-credit path |

---

## 1. Situation Analysis: Why This Decision Is Needed Now

### 1.1 The GCP GPU Drought (Hard Evidence)

The 11th pass debug factorial experiment (2026-03-28/29) produced **zero successful GPU job completions** over a 26+ hour period:

| Phase | Region | Duration | Jobs Attempted | Successful | Status |
|-------|--------|----------|---------------|------------|--------|
| europe-west4 | Netherlands | 10 hours | 3 (IDs 159-161) | 0 | L4 capacity exhausted across all 3 zones |
| us-central1 | Iowa | 13+ hours | 4 (IDs 137-139, 171) | 0 | PENDING/FAILED_SETUP cycling |
| **Total** | | **26+ hours** | **7** | **0** | |

Infrastructure waste during drought: ~$5.22 (controller VMs + Cloud SQL + Cloud Run idle).

This is not an isolated incident. Passes 5-7 (europe-north1) experienced 12+ hours PENDING because the region had zero L4 GPUs. The GCP GPU drought pattern is now confirmed across 3 regions and 6 experiment passes.

### 1.2 Current GCP Infrastructure Costs (Monthly)

| Resource | Cost/month |
|----------|-----------|
| SkyPilot Controller VM (n4-standard-4, running ~10h/day average) | ~$40-100 |
| Cloud SQL (db-f1-micro, PostgreSQL 15) | ~$10-24 |
| Cloud Run (MLflow, 1 min instance) | ~$2-6 |
| GAR Storage (6.38 GB images) | ~$0.57 |
| GCS Storage (3 buckets, ~17 GB) | ~$0.45 |
| **Total always-on** | **~$53-131** |

This infrastructure runs whether or not GPUs are available. During a drought, the entire stack is idle waste.

### 1.3 The "Dockerless" Question Clarified

RunPod pods ARE Docker containers. The pod boots FROM a Docker image. This is called "container-native" execution. It is NOT Docker-in-Docker. The key distinction (from metalearning `2026-03-14-runpod-container-not-vm.md`):

| Provider Type | Examples | How `image_id: docker:` works |
|---------------|----------|-------------------------------|
| VM-based | GCP, AWS, Lambda Labs | VM boots, Docker daemon pulls image, runs container inside VM |
| Container-native | RunPod | Image IS the pod runtime, no nested Docker layer |

SkyPilot supports `image_id: docker:<your-image>` on RunPod. The image is used as the pod's runtime environment. SkyPilot injects its own setup into the container. Limitations: no custom entrypoints (must be `/bin/bash`), no Docker-in-Docker, private registry auth goes through RunPod API.

**Council consensus**: "Dockerless" is a misnomer. RunPod runs your Docker image -- it just runs it AS the pod rather than INSIDE the pod. For our use case (training code in a Docker image), this is functionally equivalent. The reproducibility guarantee is preserved: the same Docker image runs on both GCP (Docker-in-Docker) and RunPod (container-native).

---

## 2. Expert Perspectives

### 2.1 Academic Research Infrastructure Specialist

**What peer labs actually use:**

The academic ML ecosystem in 2025-2026 has converged on a few patterns:

1. **Weights & Biases (W&B)**: Free Academic plan -- all Pro features, unlimited tracked hours, 200 GB cloud storage, up to 100 seats. Requires `.edu` email. Most popular in ML/CV/NLP labs. ([W&B Academic](https://wandb.ai/site/research/))
2. **DagsHub**: Individual tier -- free for unlimited public repos, 20 GB storage, up to 2 collaborators in private repos, 100 tracked experiments in private repos. MLflow-compatible. ([DagsHub Pricing](https://dagshub.com/pricing/))
3. **Neptune.ai**: Free tier -- 1 project, 3 users, 100 monitoring hours/month.
4. **MLflow self-hosted**: Many labs run MLflow on a departmental server or localhost. Zero cost, full control, no vendor lock-in.
5. **TensorBoard + local files**: Still common in labs with minimal MLOps maturity.

**Nature Protocols compatibility**: The journal cares about reproducibility, not specific tooling. What matters: (a) code is open-source, (b) environment is reproducible (Docker), (c) instructions are complete, (d) a reviewer with a consumer GPU can run the pipeline. Using DagsHub/W&B is fine; using local MLflow is also fine. The paper should demonstrate the platform works, not require a specific tracking server.

**Verdict**: W&B Academic is the most generous free tier (200 GB, 100 seats, unlimited tracking). DagsHub is MLflow-native which matches our existing code. Both are valid. But for a Nature Protocols paper that emphasizes reproducibility and zero vendor lock-in, file-based MLflow with optional remote integration is the most honest architecture.

### 2.2 RunPod Platform Expert

**RunPod execution model for SkyPilot training:**

RunPod's SkyPilot integration works as follows:
1. SkyPilot creates a pod using the RunPod API with `image_id: docker:<your-image>`
2. The pod boots FROM your Docker image (container-native, not Docker-in-Docker)
3. SkyPilot SSHs into the container, sets up the environment, runs the training command
4. Pod storage is ephemeral UNLESS a Network Volume is mounted

**Network Volume architecture:**
- Persistent NVMe SSD storage that survives pod restarts
- Mounted at `/runpod-volume` or configured mount point (we use `/opt/vol/`)
- Pricing: $0.07/GB/month (under 1 TB), $0.05/GB/month (over 1 TB)
- Zero ingress/egress fees (critical advantage over GCP)
- Transfer speeds: 200-400 MB/s

**What goes on the Network Volume:**
- SAM3 pretrained weights (~9 GB): cached once, reused by all jobs
- VesselFM pretrained weights (~2 GB): cached once
- DVC data (MiniVess + DeepVess ~5 GB): uploaded once via `sky rsync up`
- MLflow mlruns/ directory: accumulates across training jobs
- Total: ~20-30 GB needed = **$1.40-2.10/month**

**RTX 4090 pricing (verified March 2026):**
- Community Cloud on-demand: **$0.34/hr**
- Community Cloud spot: ~$0.17-0.22/hr (when available)
- Secure Cloud on-demand: ~$0.61/hr
- RTX 5090 Community Cloud: **$0.58/hr** (32 GB VRAM)
- RTX 3090 Community Cloud: **$0.40/hr** (24 GB VRAM)

**RTX 4090 availability**: High. RunPod has a large pool of consumer GPUs from community providers. Unlike GCP where L4 capacity is shared with all Google Cloud customers, RunPod's consumer GPU pool is dedicated to ML workloads. Availability is rarely an issue for RTX 4090.

**Limitations:**
- No custom Docker entrypoints (must be `/bin/bash`)
- Private registry auth through RunPod API, not Docker daemon
- SkyPilot managed jobs (`sky jobs launch`) have limited spot recovery on RunPod
- No guaranteed SLA (community cloud hosts can go offline)

### 2.3 MLOps Architect

**Ephemeral vs Persistent MLflow -- the real tradeoffs:**

| Aspect | File-Based on Network Volume (Ephemeral sync) | DagsHub Hosted MLflow | Self-Hosted MLflow Server |
|--------|-----------------------------------------------|----------------------|--------------------------|
| Setup complexity | Zero (default MLflow behavior) | 30 min (point tracking URI to DagsHub) | 2-4 hours (server, DB, artifact store) |
| Cost | $0 (part of Network Volume) | $0 (free individual tier) | $10-50/month (Cloud SQL + Cloud Run or VPS) |
| Multi-user access | Local only (after rsync) | Web UI, shared access | Web UI, shared access |
| Vendor lock-in | Zero | Medium (DagsHub-specific features) | Zero (self-hosted MLflow) |
| Offline capability | Full (file-based) | None (requires internet) | Depends on hosting |
| Artifact storage | Local filesystem | DagsHub storage (20 GB free) | GCS/S3/local |
| Reproducibility | Perfect (mlruns/ is portable) | Good (MLflow-compatible) | Good |
| Nature Protocols | Best (zero external dependencies) | Good (free, documented) | Good |

**The ephemeral MLflow pattern (recommended):**

```
Train on RunPod → MLflow writes to /opt/vol/mlruns/ → sky rsync down → local mlruns/
                                                                           ↓
                                              Analysis Flow reads from local mlruns/
                                              Dashboard reads from local mlruns/
                                              Biostatistics reads from local mlruns/
```

This is already how the RunPod "env" path works in the current architecture. The laptop is the control plane and MLflow home. Remote compute is ephemeral. Results sync back.

**DagsHub integration (optional enhancement):**

DagsHub provides a hosted MLflow server for every repository. Integration is trivial:

```python
import mlflow
mlflow.set_tracking_uri("https://dagshub.com/<user>/<repo>.mlflow")
```

Free tier limits: 20 GB storage, 100 experiments in private repos, unlimited in public repos. For an open-source academic project, this means unlimited experiments with 20 GB of artifact storage. SAM3 checkpoints alone are ~900 MB per run, so 20 GB supports ~20 full runs before storage becomes an issue.

**DVC on DagsHub**: DagsHub provides DVC-compatible storage. You can push/pull data using standard DVC commands with DagsHub as the remote. Free tier shares the 20 GB storage limit.

**Architect's recommendation**: Use file-based MLflow as the primary path (zero dependencies, maximum reproducibility). Offer DagsHub as an optional "team collaboration" addon that researchers can enable if they want shared experiment tracking. This preserves the zero-vendor-lock-in philosophy while giving teams a free collaboration layer.

### 2.4 Cost Engineer

**Cost Model: Debug Factorial Pass (34 jobs, 2 epochs, half data)**

Training time estimates (from prior passes):
- DynUNet jobs: ~10 min training + 3 min setup = 13 min
- SAM3 Hybrid/TopoLoRA jobs: ~25 min training + 5 min setup (weights cached on Network Volume) = 30 min
- SAM3 Vanilla zero-shot: ~15 min + 5 min setup = 20 min
- VesselFM zero-shot: ~15 min + 5 min setup = 20 min
- MambaVesselNet: ~12 min + 3 min setup = 15 min

Job mix (34 jobs): 10 DynUNet, 16 SAM3, 4 zero-shot, 2 MambaVesselNet, 2 other

Weighted average time per job: ~22 min

| Scenario | GPU | Rate | Time/Job | Cost/Job | **34 Jobs** | Monthly Infra | **Total/Pass** |
|----------|-----|------|----------|----------|-------------|---------------|----------------|
| **H1: RunPod + File MLflow** | RTX 4090 | $0.34/hr | 22 min | $0.12 | **$4.21** | $1.75 (NV) | **$5.96** |
| **H2: RunPod + DagsHub** | RTX 4090 | $0.34/hr | 22 min | $0.12 | **$4.21** | $0 (free) | **$4.21** |
| **H3: RunPod + Self-Hosted MLflow** | RTX 4090 | $0.34/hr | 22 min | $0.12 | **$4.21** | ~$15 | **$19.21** |
| **H4: GCP + RunPod Hybrid** | L4 spot | $0.22/hr | 22 min | $0.08 | **$2.75** | $53-131 (GCP) | **$56-134** |
| **H5: Local + RunPod Compute** | RTX 4090 | $0.34/hr | 22 min | $0.12 | **$4.21** | $0 | **$4.21** |
| **H6: DagsHub All-In** | RTX 4090 | $0.34/hr | 22 min | $0.12 | **$4.21** | $0 | **$4.21** |

**RTX 4090 vs RTX 5090 vs A100 for debug pass:**

| GPU | RunPod Rate | Speedup | Time/Job | Cost/Job | 34 Jobs | Cost-Efficiency |
|-----|-------------|---------|----------|----------|---------|-----------------|
| **RTX 4090** | $0.34/hr | 1.0x | 22 min | $0.12 | **$4.21** | BEST |
| RTX 5090 | $0.58/hr | ~1.5x | 15 min | $0.15 | **$4.93** | 17% more expensive |
| RTX 3090 | $0.40/hr | ~0.85x | 26 min | $0.17 | **$5.87** | 39% more expensive |
| A100 (RunPod) | $1.33/hr | ~2.2x | 10 min | $0.22 | **$7.55** | 79% more expensive |

**RTX 4090 is the clear cost-efficiency winner** for this workload profile. The 5090 is faster but costs 70% more per hour for only 50% speedup. The A100 is overkill for single-GPU 3D segmentation.

**Cost Model: Production Factorial (50 epochs, full data, 3 folds)**

The production factorial has ~640 conditions (4 models x 4 losses x 2 calibration x multiple folds). Estimated:
- 640 jobs x ~120 min average training time = 1,280 GPU-hours
- At RTX 4090 $0.34/hr = **~$435 total GPU cost**
- Network Volume (30 GB, 2 months): ~$4.20
- **Total production run: ~$440**

Compare with GCP: 1,280 GPU-hours x $0.22/hr (L4 spot) = $282, but with $53-131/month infrastructure overhead and zero reliability guarantee. Two months of infrastructure alone costs $106-262.

### 2.5 Reproducibility Expert

**Consumer GPU friendliness for Nature Protocols:**

The paper's platform contribution is that ANY researcher can reproduce results. This means demonstrating:

1. **Local GPU path**: RTX 2070 Super (8 GB) -- fits DynUNet, not SAM3
2. **Consumer cloud GPU path**: RTX 4090 on RunPod ($0.34/hr) -- fits everything
3. **No institutional cloud account needed**: RunPod requires only a credit card, no institutional agreement, no GCP project, no AWS IAM
4. **Zero mandatory infrastructure**: No Cloud SQL, no Cloud Run, no Pulumi, no GCS buckets

This is a powerful narrative for Nature Protocols. The platform works at three scales:
- **$0**: Researcher's own GPU (RTX 3090+), local MLflow, local data
- **$5-10/pass**: RunPod RTX 4090, file-based MLflow, sync results home
- **$50-130/month**: Full GCP stack with managed MLflow (for labs that want it)

**The consumer GPU demonstration is essential.** Many academic researchers face:
- No cloud credits (institutional policy, budget constraints)
- No GPU cluster access (small departments, teaching-focused universities)
- No IT support (solo PhD students, small labs)

Showing that the full factorial experiment runs on a $0.34/hr RunPod RTX 4090 with zero infrastructure setup is a stronger reproducibility argument than showing it works on GCP with Pulumi + Cloud SQL + Cloud Run + GAR.

---

## 3. Hypothesis Evaluation

### H1: RunPod Primary + File-Based MLflow (Ephemeral Sync)

**Architecture:**
```
Laptop (control plane)
  ├── MLflow mlruns/ (canonical home)
  ├── DVC data/ (local, versioned)
  └── SkyPilot → RunPod RTX 4090
        ├── Docker image (pulled from GHCR or GAR)
        ├── Network Volume:
        │   ├── /opt/vol/data/ (DVC data, uploaded once)
        │   ├── /opt/vol/mlruns/ (training results)
        │   ├── /opt/vol/weights/ (SAM3 9 GB, VesselFM 2 GB)
        │   └── /opt/vol/checkpoints/ (model outputs)
        └── sky rsync down → laptop mlruns/
```

| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| Monthly cost (debug) | 5 | $5.96/pass |
| GPU availability | 5 | RTX 4090 widely available on RunPod |
| Setup complexity | 4 | 1-2 hours (Network Volume setup, data upload) |
| Reproducibility | 5 | Zero external dependencies, file-based |
| Consumer GPU friendly | 5 | $0.34/hr, credit card only |
| Data persistence | 4 | Network Volume persistent, but single provider |
| MLflow accessibility | 3 | Local only after rsync (no web sharing) |
| Vendor lock-in | 5 | Zero (file-based MLflow, standard Docker) |
| Nature Protocols | 5 | Best narrative: works with zero infrastructure |
| **Weighted Total** | **4.6** | |

**Pros:**
- Cheapest viable option ($5.96/pass including storage)
- Zero always-on infrastructure costs (no Cloud SQL, no Cloud Run)
- Maximum reproducibility (reviewer needs only Docker + RunPod account)
- Consumer GPU narrative for the paper
- Already partially implemented (RunPod "env" path exists)

**Cons:**
- No shared experiment tracking (local MLflow only)
- Must remember to `sky rsync down` after each job
- Network Volume is RunPod-specific (not portable)
- No MLflow web UI for team collaboration

### H2: RunPod Primary + DagsHub (Free MLflow + DVC)

**Architecture:**
```
DagsHub (free individual tier)
  ├── MLflow tracking server (hosted)
  ├── DVC remote (20 GB storage)
  └── Git mirror (optional)
       ↑
RunPod RTX 4090 → MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
       ↑
Laptop → DVC push → DagsHub storage
```

| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| Monthly cost (debug) | 5 | $4.21/pass (zero infra) |
| GPU availability | 5 | RTX 4090 on RunPod |
| Setup complexity | 4 | 1 hour (DagsHub repo + env vars) |
| Reproducibility | 4 | Depends on DagsHub availability |
| Consumer GPU friendly | 5 | $0.34/hr, credit card + DagsHub account |
| Data persistence | 4 | DagsHub storage (20 GB free) |
| MLflow accessibility | 5 | Web UI, sharable links |
| Vendor lock-in | 3 | DagsHub dependency (but MLflow-compatible) |
| Nature Protocols | 4 | Good, but adds external service dependency |
| **Weighted Total** | **4.4** | |

**Pros:**
- Zero infrastructure cost
- Shared MLflow web UI (collaborators can view experiments)
- DVC remote included (no GCS needed)
- MLflow-native (zero code changes to tracking calls)

**Cons:**
- 20 GB storage limit (SAM3 checkpoints ~900 MB each, limit at ~20 runs)
- 100 experiment limit in private repos (public repos unlimited)
- Adds external service dependency (DagsHub must be online)
- DagsHub is a smaller company -- longevity risk for a paper published in 2026-2027
- Latency: MLflow writes go over internet to DagsHub servers (vs local filesystem)
- Cannot work offline

### H3: RunPod Primary + Self-Hosted MLflow on RunPod

| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| Monthly cost (debug) | 3 | $19.21/pass (MLflow pod + storage) |
| GPU availability | 5 | RTX 4090 on RunPod |
| Setup complexity | 2 | 4-8 hours (MLflow server, PostgreSQL, networking) |
| Reproducibility | 3 | Reviewer must also set up MLflow server |
| Consumer GPU friendly | 3 | Additional complexity |
| Data persistence | 4 | Persistent if MLflow pod stays running |
| MLflow accessibility | 5 | Web UI, full MLflow server |
| Vendor lock-in | 4 | MLflow is open-source |
| Nature Protocols | 3 | Adds operational complexity to reproduction |
| **Weighted Total** | **3.5** | |

**Verdict**: Overengineered for a solo/small-team academic project. The self-hosted MLflow adds complexity without commensurate benefit over H1 or H2.

### H4: Hybrid GCP + RunPod (Current Architecture, Fixed)

| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| Monthly cost (debug) | 1 | $56-134/pass (GCP infra dominates) |
| GPU availability | 2 | GCP proved unreliable (26h drought) |
| Setup complexity | 1 | Already done but complex (Pulumi, IAM, etc.) |
| Reproducibility | 2 | Reviewer needs GCP project + Pulumi + Cloud SQL |
| Consumer GPU friendly | 1 | GCP requires institutional account |
| Data persistence | 5 | GCS + Cloud SQL (enterprise-grade) |
| MLflow accessibility | 5 | Cloud Run MLflow (full web UI) |
| Vendor lock-in | 2 | Heavy GCP dependency |
| Nature Protocols | 2 | High barrier for reviewers |
| **Weighted Total** | **2.3** | |

**Verdict**: The current architecture is the most expensive, least reliable, and hardest to reproduce. It was designed for a "production" scenario that never materialized because GPUs were never available. The $53-131/month always-on cost is wasted during drought periods.

### H5: Local-Only + RunPod Compute

| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| Monthly cost (debug) | 5 | $4.21/pass (GPU only) |
| GPU availability | 5 | RTX 4090 on RunPod |
| Setup complexity | 5 | Minimal (SkyPilot + RunPod API key) |
| Reproducibility | 5 | Zero external dependencies |
| Consumer GPU friendly | 5 | $0.34/hr |
| Data persistence | 5 | All on local filesystem |
| MLflow accessibility | 2 | Strictly local (no sharing) |
| Vendor lock-in | 5 | Zero |
| Nature Protocols | 5 | Maximum simplicity |
| **Weighted Total** | **4.5** | |

**Pros:** Simplest possible architecture. Everything lives on the researcher's machine.
**Cons:** Must upload data to RunPod for each job (no persistent Network Volume). Larger setup overhead per job.

### H6: DagsHub All-In (MLflow + DVC + Git)

| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| Monthly cost (debug) | 5 | $4.21/pass |
| GPU availability | 5 | RunPod RTX 4090 |
| Setup complexity | 3 | DagsHub repo + DVC remote + Git mirror |
| Reproducibility | 3 | Requires DagsHub account + service |
| Consumer GPU friendly | 4 | One more account to create |
| Data persistence | 3 | 20 GB limit (may need paid tier for production) |
| MLflow accessibility | 5 | Full web UI |
| Vendor lock-in | 2 | Deep DagsHub dependency |
| Nature Protocols | 3 | External service dependency |
| **Weighted Total** | **3.7** | |

**Verdict**: Going "all-in" on DagsHub creates unnecessary vendor lock-in. The Git mirror adds complexity without benefit (GitHub is already the repo home). Better to use DagsHub selectively (MLflow tracking only) than as the entire platform.

---

## 4. Decision Matrix Summary

| Criterion (Weight) | H1: RunPod+File | H2: RunPod+DagsHub | H3: RunPod+SelfMLflow | H4: GCP Hybrid | H5: Local+RunPod | H6: DagsHub All-In |
|---------------------|-----------------|--------------------|-----------------------|----------------|------------------|-------------------|
| Monthly cost (20%) | 5 | 5 | 3 | 1 | 5 | 5 |
| GPU availability (20%) | 5 | 5 | 5 | 2 | 5 | 5 |
| Setup complexity (10%) | 4 | 4 | 2 | 1 | 5 | 3 |
| Reproducibility (15%) | 5 | 4 | 3 | 2 | 5 | 3 |
| Consumer GPU (10%) | 5 | 5 | 3 | 1 | 5 | 4 |
| Data persistence (5%) | 4 | 4 | 4 | 5 | 5 | 3 |
| MLflow access (5%) | 3 | 5 | 5 | 5 | 2 | 5 |
| Vendor lock-in (10%) | 5 | 3 | 4 | 2 | 5 | 2 |
| Nature Protocols (5%) | 5 | 4 | 3 | 2 | 5 | 3 |
| **Weighted Total** | **4.65** | **4.40** | **3.45** | **1.85** | **4.65** | **3.90** |

---

## 5. Cost Comparison: Full Lifecycle

### 5.1 Three-Month Active Development (3 debug passes, 1 production pass)

| Cost Component | H1: RunPod+File | H2: RunPod+DagsHub | H4: GCP Hybrid |
|----------------|-----------------|--------------------|--------------------|
| Debug passes (3x) | $12.63 | $12.63 | $8.25 (if GPUs available) |
| Production pass (1x) | ~$435 | ~$435 | ~$282 (if GPUs available) |
| Network Volume (3 mo) | $5.25 | $0 | $0 |
| GCP infra (3 mo) | $0 | $0 | $159-393 |
| SkyPilot controller waste (drought days) | $0 | $0 | ~$40-100 |
| **Total 3-month** | **~$453** | **~$448** | **~$489-783** |

**Key insight from Cost Engineer**: The GCP path is MORE EXPENSIVE even when GPUs are available, because always-on infrastructure ($53-131/month) accumulates regardless of GPU usage. RunPod has zero always-on costs -- you only pay for what you compute.

### 5.2 Network Volume Sizing

| Asset | Size | Monthly Cost ($0.07/GB) |
|-------|------|------------------------|
| SAM3 weights (ViT-32L) | 9 GB | $0.63 |
| VesselFM weights | 2 GB | $0.14 |
| DVC data (MiniVess + DeepVess) | 5 GB | $0.35 |
| MLflow mlruns/ (per pass) | 2-5 GB | $0.14-0.35 |
| Docker layers (cached) | 0 GB | $0 (pulled at boot) |
| Headroom | 5 GB | $0.35 |
| **Total** | **~25 GB** | **$1.75/month** |

---

## 6. Addressing Specific User Questions

### 6.1 Should MLflow Always Be Ephemeral?

**Yes, for the primary path.** File-based MLflow on the Network Volume, synced back to local after each job, is the most robust and reproducible pattern. Reasons:

1. **Zero infrastructure cost**: No Cloud SQL, no Cloud Run, no managed service
2. **Works offline**: Researcher can analyze results on a plane
3. **Portable**: `mlruns/` directory works on any machine with MLflow installed
4. **No network dependency**: Training never fails because a tracking server is down
5. **Standard MLflow behavior**: This is how MLflow works by default

The sync-back step (`sky rsync down /opt/vol/mlruns/ mlruns/`) must be automated. The current `make dev-gpu-sync` command already does this.

### 6.2 Should We Switch to DagsHub?

**Optional enhancement, not primary path.** DagsHub's free individual tier (20 GB, unlimited public repos, MLflow hosting) is attractive but introduces constraints:

- 20 GB storage limit is tight for SAM3 artifacts (~900 MB per full run)
- Service dependency (DagsHub downtime = no experiment logging)
- Smaller company than Databricks (MLflow maintainer) -- longevity consideration

**Recommended approach**: Keep file-based MLflow as primary. Add DagsHub as an optional `MLFLOW_TRACKING_URI` override for researchers who want shared web access. Document both paths in the Nature Protocols paper.

### 6.3 SAM3 Weights on Persistent Volume?

**Yes, absolutely.** SAM3 weights (ViT-32L, ~9 GB) are the single largest setup cost on RunPod. Caching them on the Network Volume reduces per-job setup from ~18-20 min (HuggingFace download) to ~1-2 min (local NVMe read at 200-400 MB/s). This is the same strategy we planned for GCS (`pretrained_weight_caching` in `cloud.yaml`) but executed on RunPod storage instead.

The Network Volume persists across pod restarts. Upload once:
```bash
# Upload weights to Network Volume (one-time)
sky rsync up my-vol /path/to/sam3_weights/ /opt/vol/weights/sam3/
```

### 6.4 RTX 4090 vs RTX 5090 Cost Analysis

| Metric | RTX 4090 | RTX 5090 | Winner |
|--------|----------|----------|--------|
| Price/hr (RunPod) | $0.34 | $0.58 | 4090 (41% cheaper) |
| VRAM | 24 GB | 32 GB | 5090 (but 24 GB is sufficient) |
| BF16 TFLOPS | 82.6 | 209.5 | 5090 (2.5x faster peak) |
| Memory BW | 1,008 GB/s | 1,792 GB/s | 5090 (1.8x) |
| Estimated speedup | 1.0x | ~1.5x | 5090 |
| Cost per debug pass | $4.21 | $4.93 | 4090 (17% cheaper per pass) |
| Cost per prod pass | ~$435 | ~$493 | 4090 (13% cheaper per pass) |

**RTX 4090 wins on cost-efficiency.** The 5090's higher TFLOPS do not translate to proportional speedups on our workload (3D medical segmentation is memory-bandwidth-bound, and both GPUs have high bandwidth). The 4090 delivers 83% of the 5090's throughput at 59% of the cost.

**Consumer GPU demonstration value**: Showing the full factorial runs on an RTX 4090 (a GPU many researchers own) is more impactful for Nature Protocols than showing it on an RTX 5090 (newer, less common). The 4090 is the "everyman's GPU" in 2026.

---

## 7. Recommended Architecture

### Primary Recommendation: H1 (RunPod + File-Based MLflow) with H2 as Optional Enhancement

```
┌─────────────────────────────────────────────────────────┐
│  Researcher's Laptop (Control Plane)                     │
│  ├── MLflow mlruns/ (canonical, file-based)              │
│  ├── DVC data/ (local, versioned)                        │
│  ├── Prefect (local orchestration)                       │
│  └── SkyPilot CLI                                        │
│       │                                                   │
│       │ sky jobs launch                                   │
│       ▼                                                   │
│  ┌─────────────────────────────────────┐                 │
│  │  RunPod RTX 4090 ($0.34/hr)        │                 │
│  │  ├── Docker image (pulled at boot)  │                 │
│  │  └── Network Volume (/opt/vol/)     │                 │
│  │      ├── weights/ (SAM3 9 GB)       │                 │
│  │      ├── data/ (MiniVess 1 GB)      │                 │
│  │      ├── mlruns/ (training output)  │                 │
│  │      └── checkpoints/               │                 │
│  └─────────────────────────────────────┘                 │
│       │                                                   │
│       │ sky rsync down (post-job)                        │
│       ▼                                                   │
│  Laptop mlruns/ ← merged with remote results             │
│  ├── Analysis Flow (local)                               │
│  ├── Biostatistics Flow (local)                          │
│  └── Dashboard Flow (local)                              │
│                                                           │
│  [Optional] DagsHub MLflow (for team sharing)            │
│  MLFLOW_TRACKING_URI=https://dagshub.com/user/repo.mlflow│
└─────────────────────────────────────────────────────────┘
```

### Implementation Phases

**Phase 1 (Day 1, ~2 hours): RunPod as Primary Compute**
1. Create/verify RunPod Network Volume (~25 GB) in EU-NL datacenter
2. Upload SAM3 weights, VesselFM weights, DVC data to Network Volume
3. Update `deployment/skypilot/train_factorial.yaml` to support RunPod cloud target
4. Create `configs/cloud/runpod_primary.yaml` for factorial experiments
5. Test single DynUNet job end-to-end

**Phase 2 (Day 1, ~1 hour): Factorial on RunPod**
1. Run Phase 1 validation (3 jobs) on RunPod RTX 4090
2. Verify MLflow artifacts on Network Volume
3. Run `sky rsync down` and verify local mlruns/ merge
4. Launch full debug factorial (34 jobs)

**Phase 3 (Optional, Day 2): DagsHub Integration**
1. Create DagsHub repository (mirror from GitHub)
2. Configure `MLFLOW_TRACKING_URI` for DagsHub
3. Test DVC remote on DagsHub storage
4. Document dual-path (file-based + DagsHub) in README

**Phase 4 (Optional, Later): GCP Demotion**
1. Stop Cloud SQL instance (saves $10-24/month)
2. Stop SkyPilot controller (saves $40-100/month)
3. Keep GCS buckets as cold backup ($0.45/month)
4. Keep Cloud Run MLflow in sleep mode ($0/month when idle)
5. GCP becomes a documented "alternative path" in the paper, not the primary

### What Changes in CLAUDE.md

The two-provider architecture remains (RunPod + GCP), but roles shift:

| Provider | Current Role | New Role |
|----------|-------------|----------|
| RunPod | env (dev only) | **Primary compute** (debug + production factorial) |
| GCP | staging + prod | **Optional managed infrastructure** (for labs wanting Cloud SQL + MLflow server) |

The architectural principle is preserved: RunPod for compute, GCP for managed services (when desired). But the default path becomes RunPod-primary because it is cheaper, more reliable, and more reproducible.

---

## 8. Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| RunPod RTX 4090 shortage | Low | Medium | RTX 3090 ($0.40/hr) or RTX 5090 ($0.58/hr) as fallback |
| Network Volume data loss | Very Low | High | DVC-versioned locally, re-upload in 30 min |
| DagsHub discontinues free tier | Medium | Low | File-based MLflow is the primary path (zero dependency) |
| SkyPilot-RunPod integration breaks | Low | High | Pin SkyPilot version, test before each factorial pass |
| RunPod community host goes offline mid-job | Medium | Low | SkyPilot auto-retries, Network Volume persists |

---

## 9. Council Verdict

All five experts converge on the same recommendation:

**H1 (RunPod Primary + File-Based MLflow) is the clear winner**, with H2 (DagsHub) as an optional enhancement for team collaboration.

The reasoning:
1. **Cost**: $5.96/debug pass vs $56-134/pass on GCP (9-22x cheaper)
2. **Reliability**: RunPod RTX 4090 availability >> GCP L4 spot availability (empirically proven by 26h drought)
3. **Reproducibility**: Zero infrastructure dependencies, file-based MLflow, standard Docker image
4. **Consumer GPU narrative**: RTX 4090 at $0.34/hr directly addresses researchers with no cloud credits
5. **Nature Protocols alignment**: The simplest architecture that demonstrates the platform's flexibility

**The GCP full stack should be preserved as a documented alternative for well-funded labs**, not abandoned. But it should no longer be the default path. The default path is: laptop + RunPod RTX 4090 + file-based MLflow. Zero always-on costs. Pay only for compute.

---

## Sources

- [RunPod GPU Pricing](https://www.runpod.io/pricing)
- [RunPod RTX 4090 Cloud](https://www.runpod.io/gpu-models/rtx-4090)
- [RunPod SkyPilot Integration Blog](https://www.runpod.io/blog/runpod-skypilot-integration)
- [RunPod SkyPilot Documentation](https://docs.runpod.io/integrations/skypilot)
- [SkyPilot Docker Containers Documentation](https://docs.skypilot.co/en/latest/examples/docker-containers.html)
- [SkyPilot GitHub Issue #3096 - Docker on RunPod](https://github.com/skypilot-org/skypilot/issues/3096)
- [DagsHub Pricing](https://dagshub.com/pricing/)
- [DagsHub MLflow Integration](https://dagshub.com/docs/integration_guide/mlflow_tracking/)
- [DagsHub Storage Documentation](https://dagshub.com/docs/feature_guide/dagshub_storage/)
- [W&B Academic Research Plan](https://wandb.ai/site/research/)
- [MLflow Tracking Server Architecture](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/)
- [RunPod Network Volume Pricing](https://docs.runpod.io/pods/pricing)
- [GPU Cost Comparison - RunPod](https://gpucost.org/provider/runpod)
- [Cheapest Cloud GPU Providers 2026](https://northflank.com/blog/cheapest-cloud-gpu-providers)
