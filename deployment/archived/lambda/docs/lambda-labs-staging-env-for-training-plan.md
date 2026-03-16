# Lambda Labs Staging Environment for Training — Implementation Plan

> **Status**: Active (2026-03-14)
> **Priority**: P0 — Unblocks cloud GPU training
> **Issue**: #681
> **Replaces**: RunPod as primary SkyPilot cloud (RunPod → dev environment only)

---

## Context

RunPod pods ARE containers, not VMs. Docker-in-Docker is impossible on RunPod.
Lambda Labs provides actual Ubuntu VMs with Docker preinstalled — SkyPilot's
Docker abstraction works natively. See: `docs/runpod-vs-lambda-vs-rest-for-skypilot-with-docker.md`

## Lambda Labs GPU Inventory & Pricing

| GPU | VRAM | Price/hr | Quantities | Use Case |
|-----|------|----------|------------|----------|
| **A10** | 24 GB | **$0.86** | 1 | Smoke tests, all models fit |
| A100 40 GB | 40 GB | $1.48 | 1, 8 | Production training |
| **GH200** | 96 GB | **$1.99** | 1 | Future large models |
| H100 PCIe | 80 GB | $2.86 | 1 | Fast production training |
| H100 SXM5 | 80 GB | $3.78 | 1, 2, 4, 8 | Distributed training |
| B200 | 180 GB | $6.08 | 1, 2, 4, 8 | Future massive models |

**No spot instances on Lambda.** On-demand only. But A10 at $0.86/hr is cheaper
than RunPod RTX 4090 on-demand ($1.64/hr) with 8 GB more VRAM.

## Our Model VRAM Requirements vs Lambda GPUs

| Model | Training VRAM | Fits A10 (24 GB) | Fits A100 (40 GB) |
|-------|---------------|-------------------|---------------------|
| DynUNet | 3.5 GB | YES | YES |
| SAM3 Vanilla | 3.5 GB | YES | YES |
| SAM3 Hybrid | ~7.5 GB | YES | YES |
| VesselFM | ~10 GB | YES | YES |

**All models fit the cheapest Lambda GPU (A10, $0.86/hr).**

---

## Implementation Steps

### Step 1: Lambda Credentials (DONE)

- [x] Create Lambda API key at https://cloud.lambdalabs.com/api-keys
- [x] Write to `~/.lambda_cloud/lambda_keys`
- [x] Verify: `sky check lambda` → enabled

### Step 2: Create Lambda SkyPilot YAML

Create `deployment/skypilot/smoke_test_lambda.yaml`:
- `cloud: lambda` (not runpod)
- `image_id: docker:ghcr.io/petteriteikari/minivess-base:latest`
- Accelerator fallback: A10 → A100 → GH200 → H100
- No `use_spot` (Lambda doesn't support spot)
- Same setup/run as existing YAML (data pull + training)

### Step 3: Update Launch Script (DONE)

Updated `scripts/launch_smoke_test.py`:
- [x] `--cloud` argument (default: `lambda`)
- [x] Multi-region rotation for Lambda (17 regions, EU/Asia first → US last)
- [x] Real-time Lambda API availability check before launching
- [x] 3 retries per region, 10 rounds max (~30 min total)
- [x] Reorders regions by real-time availability each round
- [x] Falls back gracefully with clear instructions if globally sold out
- [x] Lambda is VM-based → `DockerLoginConfig` works natively
- [x] IPv4 forcing preserved

#### Multi-Region Rotation Strategy

```
Tier 1 (EU/ME, least popular):  europe-south-1 → europe-central-1 → me-west-1
Tier 2 (Asia-Pacific):          asia-south-1 → asia-northeast-2 → asia-northeast-1 → australia-east-1
Tier 3 (US peripheral):         us-midwest-1 → us-south-3 → us-south-2 → us-south-1
Tier 4 (US west):               us-west-3 → us-west-2 → us-west-1
Tier 5 (US east, most popular): us-east-3 → us-east-2 → us-east-1
```

Each region tries GPU types in price order: A10 ($0.86) → A100 ($1.48) → GH200 ($1.99) → H100 ($2.86)

### Step 4: Update `.env.example`

Add Lambda configuration section:
```
LAMBDA_API_KEY=
SKYPILOT_DEFAULT_CLOUD=lambda
```

Note: Fix typo `LAMBA_API_KEY` → `LAMBDA_API_KEY` in `.env`.

### Step 5: Smoke Test E2E

1. Launch `sam3_vanilla` on Lambda A10
2. Verify:
   - Docker image pulled successfully
   - DVC data pull from UpCloud S3
   - Training completes (2 epochs, 1 fold, 4 volumes)
   - MLflow metrics logged to UpCloud MLflow
3. If successful, test `sam3_hybrid` and `vesselfm`

### Step 6: Update Production YAML

Update `deployment/skypilot/train_production.yaml`:
- Change primary cloud to `lambda`
- Multi-cloud failover: Lambda → AWS → GCP
- Remove RunPod from production failover chain

### Step 7: Documentation Update

- Update `CLAUDE.md` SkyPilot section
- Update `.env.example` comments
- Create metalearning doc

---

## Architecture Diagram

```
┌──────────────┐     sky.launch()      ┌──────────────────┐
│  Local Dev   │ ──────────────────► │   Lambda VM        │
│  Machine     │                      │   (Ubuntu 22.04)   │
│              │     Docker pull      │  ┌──────────────┐  │
│  .env        │ ◄──── GHCR ────────►│  │  Docker       │  │
│  SkyPilot    │                      │  │  Container    │  │
│  Python API  │     MLflow metrics   │  │  minivess-    │  │
│              │ ◄────────────────── │  │  base:latest  │  │
└──────────────┘                      │  └──────────────┘  │
       │                              │  GPU: A10 (24 GB)  │
       │         DVC data pull        │                    │
       │    ┌──────────────────┐      │                    │
       └──► │  UpCloud S3      │ ◄────┘                    │
            │  (DVC remote)    │      └──────────────────────┘
            └──────────────────┘
                    │
            ┌──────────────────┐
            │  UpCloud MLflow  │
            │  (tracking)      │
            └──────────────────┘
```

## Key Differences from RunPod

| Aspect | RunPod (old) | Lambda (new) |
|--------|-------------|--------------|
| Architecture | Container-based (pod IS container) | VM-based (Docker runs inside VM) |
| Docker support | Runtime env only, no Docker daemon | Full Docker daemon, native container support |
| `image_id: docker:` | Used as pod base image | Docker pulls & runs container inside VM |
| Entrypoints | Must be `/bin/bash` | Any entrypoint supported |
| GPUs | Consumer (RTX 3090/4090) + Datacenter | Datacenter only (A10, A100, H100, B200) |
| Spot instances | Yes (but often sold out) | No (on-demand only) |
| Cheapest option | RTX 4090 $0.39/hr spot, $1.64 on-demand | A10 $0.86/hr (only on-demand) |
| Private registry | RunPod API injection | Standard `docker login` |

## Cost Estimate

| Job Type | GPU | Duration | Cost |
|----------|-----|----------|------|
| Smoke test (1 model) | A10 | ~15 min | ~$0.22 |
| Smoke test (3 models) | A10 | ~45 min | ~$0.65 |
| Production (100 ep, 3 folds) | A10 | ~6 hrs | ~$5.16 |
| Production (100 ep, 3 folds) | A100 | ~3 hrs | ~$4.44 |

## FinOps Analysis: The Docker Tax

See also: `docs/planning/skypilot-and-finops-complete-report.md` for full FinOps analysis.

### The Honest Cost Comparison

Lambda Labs is **more expensive per hour** than RunPod's consumer GPUs. This is the
"Docker tax" — the premium we pay for an architecture that actually supports our
Docker-mandatory workflow.

| Provider | GPU | $/hr | VRAM | Docker Support | Works for Us? |
|----------|-----|------|------|----------------|---------------|
| RunPod (spot) | RTX 3090 | **$0.22** | 24 GB | Runtime env only | **NO** (8 hrs debugging proved this) |
| RunPod (spot) | RTX 4090 | **$0.34** | 24 GB | Runtime env only | **NO** |
| RunPod (on-demand) | RTX 4090 | $1.64 | 24 GB | Runtime env only | **NO** |
| **Lambda** | **A10** | **$0.86** | **24 GB** | **Full Docker** | **YES** |
| Lambda | A100 40 GB | $1.48 | 40 GB | Full Docker | YES |
| Lambda | GH200 | $1.99 | 96 GB | Full Docker | YES |
| Lambda | H100 PCIe | $2.86 | 80 GB | Full Docker | YES |
| Vast.ai | A100 | ~$1.50 | 80 GB | Confirmed Docker | Likely YES |

### Absolute Training Costs (What Matters)

From our benchmarks (see FinOps report), $/hour is misleading. What matters is $/job:

#### DynUNet 3-fold x 100 epochs (CNN, ~12 hr on RTX 3090)

| GPU | Est. Wall Time | $/hr | **Total Cost** | Notes |
|-----|:-:|:-:|:-:|:--|
| RunPod RTX 3090 (spot) | ~12 hr | $0.22 | **$2.64** | **Doesn't work (no Docker)** |
| RunPod RTX 4090 (spot) | ~8.6 hr | $0.34 | **$2.92** | **Doesn't work (no Docker)** |
| **Lambda A10** | ~10 hr | **$0.86** | **$8.60** | **Works — cheapest Docker option** |
| Lambda A100 | ~6 hr | $1.48 | **$8.88** | Similar cost, 40% faster |

#### SAM3 Smoke Test (2 epochs, 1 fold, 4 volumes — ~15 min)

| GPU | Est. Wall Time | $/hr | **Total Cost** |
|-----|:-:|:-:|:-:|
| **Lambda A10** | ~15 min | **$0.86** | **$0.22** |
| Lambda A100 | ~10 min | $1.48 | **$0.25** |

### The Docker Tax is Worth It

The Docker tax for DynUNet training is ~$6/job ($8.60 on Lambda vs $2.64 hypothetical
RunPod). For SAM3 smoke tests it's negligible (~$0.22 vs ~$0.06 hypothetical).

**Why it's worth it:**
1. RunPod literally **does not work** for us — $0 saved, 8 hours wasted
2. Reproducibility guarantee (Docker) is the #2 repo priority
3. Lambda VMs are the well-tested SkyPilot path (same as AWS/GCP)
4. No "runtime environment" dependency injection issues
5. Custom entrypoints supported (future flexibility)

### Future Cost Optimization

1. **Vast.ai** — confirmed Docker `image_id` support, ~$1.50/hr for A100. Worth testing
   as cost-optimized fallback.
2. **Multi-cloud failover** — Lambda → Vast.ai → AWS spot → GCP preemptible
3. **Image size reduction** — 20+ GB → 15 GB target reduces pull time and cost
4. **Kubernetes path** — CoreWeave/Nebius for GPU scheduling at scale

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| GPU sold out | Medium | Multi-GPU fallback (A10→A100→GH200→H100) |
| GHCR pull slow from US Lambda | Low | US Lambda + Fastly US CDN = fast path |
| Docker auth failure | Low | DockerLoginConfig tested on VM providers |
| MLflow connectivity | Low | Already tested from RunPod (same UpCloud server) |
| Lambda API issues | Low | First-class SkyPilot provider since v0.3 |
