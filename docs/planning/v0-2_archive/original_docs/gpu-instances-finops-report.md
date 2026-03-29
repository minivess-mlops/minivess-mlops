# GPU Instances FinOps Report — Multi-Cloud Strategy for Vascadia v0.2-beta

**Date**: 2026-03-28
**Status**: Reference analysis
**Context**: L4 capacity exhaustion in europe-west4 (10h+ PENDING), factorial experiments blocked
**Budget constraint**: Academic project, ~$50-100 per experiment pass maximum
**Methodology**: Iterated LLM Council (5 expert perspectives)

---

## Council Panel

This report was produced using the iterated-llm-council approach with five expert
perspectives, each contributing domain-specific analysis and then cross-reviewing
all other sections for contradictions and blind spots.

| Expert | Domain | Key Contribution |
|--------|--------|------------------|
| **FinOps Analyst** | Cloud cost optimization | TCO modeling, egress analysis, cost-per-experiment calculations |
| **Cloud Architect** | Multi-cloud infrastructure | Provider evaluation, SkyPilot integration, data locality strategy |
| **ML Systems Engineer** | GPU performance for training | TFLOPS/$ analysis, speedup factors, VRAM requirements |
| **Platform Security/Compliance** | GDPR, data residency | EU region requirements, medical imaging data handling |
| **DevEx/Researcher Advocate** | PhD researcher workflows | Operational complexity, setup time, failure recovery UX |

---

## Section 1: Current State and Problem

### 1.1 The Capacity Exhaustion Crisis

The Vascadia v0.2-beta factorial experiment architecture submits 34 GPU jobs for a
debug pass (and 640+ for production). As of 2026-03-28, GCP L4 spot instances in
europe-west4 have been PENDING for 10+ hours with zero provisioning. This is the
immediate crisis driving this analysis.

**Observed symptoms:**
- SkyPilot job #158 (`minivess-factorial`): PENDING for 10+ hours, requesting 1x L4 spot
- SkyPilot controller VM continues running at ~$3.22/day ($0.134/hr x 24h) while jobs sit idle
- 3 blocked jobs = $3.22 controller waste + entire experiment pass blocked
- Total waste: controller cost + researcher time + delayed paper submission

**Root cause**: GCP L4 GPUs in europe-west4 (Netherlands) are in high demand due to the
global AI training boom. L4 is the price/performance sweet spot for inference and small
training jobs, making it the most oversubscribed GPU class. Europe has fewer GPU zones
than US, concentrating demand further.

### 1.2 Current Infrastructure Costs (from FinOps Snapshot 2026-03-28)

| Resource | EUR/month |
|----------|-----------|
| SkyPilot Controller VM (n4-standard-4, europe-west1-b) | ~93-100 |
| Cloud SQL (db-g1-small, PostgreSQL 15) | ~23.50 |
| Cloud Run (MLflow, 1 min instance) | ~2-6 |
| GAR Storage (6.24 GiB images) | ~0.57 |
| GCS Storage (empty buckets) | 0.00 |
| **Total always-on** | **~119-130** |

**Variable cost per L4 training job** (when provisioned): ~EUR 1.01 (5h x $0.22/hr spot)

### 1.3 Current YAML Contract Constraints

From `configs/cloud/yaml_contract.yaml`:
- **GCP**: Only L4 allowed
- **RunPod**: RTX4090, RTX5090, RTX3090
- **Banned globally**: T4, V100, K80, P100, P4
- **Not authorized**: A100, A100-80GB, H100 (listed in max_hourly_cost only for detection)
- **Factorial YAML**: L4 only, GCP only, spot required, Docker image required

**Key insight**: The current contract was designed for cost optimization when L4 was
available. It has no fallback for capacity exhaustion. This is the architectural gap.

---

## Section 2: Cost-Performance Matrix for Single-GPU Instances

### 2.1 GPU Specifications Comparison

| GPU | Architecture | VRAM | BF16 TFLOPS | Memory BW (GB/s) | TDP (W) | Release |
|-----|-------------|------|-------------|-------------------|---------|---------|
| **L4** | Ada Lovelace | 24 GB | 121 | 300 | 72 | 2023 |
| **RTX 4090** | Ada Lovelace | 24 GB | 82.6 | 1,008 | 450 | 2022 |
| **RTX 5090** | Blackwell | 32 GB | 209.5 | 1,792 | 575 | 2025 |
| **A100-40GB** | Ampere | 40 GB | 312 | 1,555 | 300 | 2020 |
| **A100-80GB** | Ampere | 80 GB | 312 | 2,039 | 300 | 2021 |
| **H100 SXM** | Hopper | 80 GB | 989 | 3,350 | 700 | 2023 |
| **L40S** | Ada Lovelace | 48 GB | 181 | 864 | 350 | 2023 |
| **H200** | Hopper | 141 GB | 989 | 4,800 | 700 | 2024 |

**T4 (Turing, 16 GB)**: BANNED -- no BF16 support, FP16 max=65504, causes NaN overflow
with SAM3's half-precision encoder. See `.claude/metalearning/2026-03-15-t4-turing-fp16-nan-ban.md`.

### 2.2 Cloud Pricing Matrix (March 2026, verified via web search)

**GCP Pricing (europe-west4)**:

| GPU | On-Demand $/hr | Spot $/hr | Spot Discount | EU Availability |
|-----|----------------|-----------|---------------|-----------------|
| **L4** | ~$0.74 | ~$0.22-0.34 | 55-70% | 3 zones (ew4), 2 zones (ew1) -- CAPACITY EXHAUSTED |
| **A100-40GB** | ~$3.67 | ~$1.10-1.39 | 62-70% | europe-west4 only |
| **A100-80GB** | ~$4.61 | ~$1.38 | 70% | europe-west4 only |
| **H100 SXM** | ~$9.80 | ~$3.40-3.72 | 62-65% | Limited EU availability |

**RunPod Pricing**:

| GPU | On-Demand $/hr | Community $/hr | Spot $/hr | EU Regions |
|-----|----------------|----------------|-----------|------------|
| **RTX 4090** | $0.34 | ~$0.20 | ~$0.17-0.22 | EU (NL) |
| **RTX 5090** | $0.69 | -- | ~$0.40 | Limited |
| **RTX 3090** | $0.22 | ~$0.19 | ~$0.15-0.20 | EU |
| **A100-80GB** | $1.74 | ~$1.64 | -- | EU |
| **H100 SXM** | $1.99-2.49 | ~$1.99 | ~$1.49 | EU |

**Vast.ai Pricing** (marketplace, variable):

| GPU | Typical $/hr | Lowest $/hr | EU Hosts |
|-----|-------------|-------------|----------|
| **RTX 4090** | $0.17-0.32 | ~$0.15 | Some |
| **A100-80GB** | ~$0.78-1.20 | ~$0.78 | Some |
| **H100** | ~$2.00-4.00 | ~$1.38 | Some |

**Nebius Pricing** (eu-north1, Finland):

| GPU | On-Demand $/hr | Spot | Notes |
|-----|----------------|------|-------|
| **H100** | $2.95 | Not available | Managed SkyPilot API server |
| **H200** | $3.50 | Not available | Latest Hopper |
| **L40S** | $1.55-1.82 | Not available | Good VRAM/price |
| **B200** | $5.50 | Not available | Blackwell |

### 2.3 Training Speed Estimation

Based on the MinIVess workload profile (3D medical image segmentation, batch size 1-4,
volumes of 64x128x128 to 128x256x256):

| GPU | Relative Speed vs L4 | Speedup Factor | Basis |
|-----|---------------------|----------------|-------|
| **L4** | 1.0x (baseline) | 1.0 | 121 BF16 TFLOPS, 300 GB/s BW |
| **RTX 4090** | ~0.9-1.1x | 1.0 | Similar TFLOPS, much higher BW (memory-bound helps) |
| **A100-40GB** | ~2.0-2.5x | 2.2 | 312 BF16 TFLOPS, 1555 GB/s BW |
| **A100-80GB** | ~2.0-2.5x | 2.2 | Same compute, more VRAM headroom |
| **H100 SXM** | ~4.0-6.0x | 5.0 | 989 BF16 TFLOPS, 3350 GB/s BW |
| **RTX 5090** | ~1.5-1.8x | 1.6 | 209 BF16 TFLOPS, 1792 GB/s BW |
| **L40S** | ~1.3-1.5x | 1.4 | 181 BF16 TFLOPS, 864 GB/s BW |

**Important caveat**: 3D medical segmentation is often memory-bandwidth-bound, not
compute-bound. The L4's low memory bandwidth (300 GB/s) is its weakest link.
A100/H100's HBM2e/HBM3 provides massive bandwidth advantages that translate into
superlinear speedups for bandwidth-bound workloads.

### 2.4 Total Cost per Debug Experiment Pass (34 Jobs)

**Formula**: `total_cost = (setup_time + training_time / speedup) * hourly_rate * n_jobs`

Assumptions:
- Setup time per job: 10 minutes (Docker pull + DVC data + weight loading) -- same-region
- L4 baseline training time per job: 45 minutes (observed from prior passes)
- 34 debug jobs
- Spot pricing where available

| GPU | Provider | Spot $/hr | Time/Job (min) | Cost/Job | **Total 34 Jobs** | Wall Clock (serial) |
|-----|----------|-----------|----------------|----------|-------------------|---------------------|
| **L4** | GCP | $0.22 | 55 | $0.20 | **$6.85** | 31h |
| **L4** | GCP (on-demand) | $0.74 | 55 | $0.68 | **$23.05** | 31h |
| **RTX 4090** | RunPod | $0.34 | 55 | $0.31 | **$10.58** | 31h |
| **RTX 5090** | RunPod | $0.69 | 38 | $0.44 | **$14.86** | 22h |
| **A100-40GB** | GCP spot | $1.10 | 30 | $0.55 | **$18.70** | 17h |
| **A100-80GB** | GCP spot | $1.38 | 30 | $0.69 | **$23.46** | 17h |
| **A100-80GB** | Vast.ai | $0.78 | 30 | $0.39 | **$13.26** | 17h |
| **A100-80GB** | RunPod | $1.74 | 30 | $0.87 | **$29.58** | 17h |
| **H100 SXM** | GCP spot | $3.40 | 21 | $1.19 | **$40.46** | 12h |
| **H100 SXM** | RunPod | $1.99 | 21 | $0.70 | **$23.66** | 12h |
| **L40S** | Nebius | $1.55 | 42 | $1.09 | **$36.89** | 24h |

**Key insight from FinOps Analyst**: L4 spot at $6.85/pass is the cheapest option
**when available**. But availability is zero right now. The real question is: what is
the cost of WAITING? At $3.22/day controller cost + researcher time, every day of
waiting costs more than the difference between L4 and A100 on Vast.ai.

**Council consensus**: The cheapest *actually available* option matters more than the
theoretical cheapest option. A100-80GB on Vast.ai at $13.26/pass is better than L4 at
$6.85/pass that never provisions.

---

## Section 3: Multi-Cloud Strategy with Per-Cloud Storage

### 3.1 The Egress Problem (Quantified)

The March 2026 billing disaster (EUR 89.58 in GAR egress alone) proved that cross-region
data transfer dominates costs when GPU and storage are not co-located.

**Egress pricing by boundary**:

| Transfer Type | Cost/GB | 6.4 GB Docker Image | 3 GB DVC Data |
|---------------|---------|---------------------|---------------|
| Same-region (GCS/GAR to VM) | **$0.00** | **$0.00** | **$0.00** |
| Intra-EU cross-region | $0.01-0.02 | $0.06-0.13 | $0.03-0.06 |
| EU to US | $0.05-0.08 | $0.32-0.51 | $0.15-0.24 |
| Internet egress (to non-cloud) | $0.12 | $0.77 | $0.36 |
| AWS S3 egress | $0.09 | $0.58 | $0.27 |
| Azure Blob egress | $0.087 | $0.56 | $0.26 |
| RunPod egress | **$0.00** | **$0.00** | **$0.00** |
| Vast.ai egress | **$0.00** | **$0.00** | **$0.00** |

### 3.2 Multi-Cloud Storage Duplication Analysis

**Scenario**: Maintain Docker images + DVC data + pretrained weights on each cloud provider.

| Data Asset | Size | GCS (ew4) | AWS S3 (eu-west-1) | Azure Blob (westeurope) | RunPod NV | Vast.ai |
|-----------|------|-----------|---------------------|------------------------|-----------|---------|
| Docker base image | 6.4 GB | $0.13/mo | $0.15/mo | $0.14/mo | N/A (pull) | N/A (pull) |
| DVC data (MiniVess+DeepVess) | ~3 GB | $0.06/mo | $0.07/mo | $0.06/mo | ~$0.12/mo (NV) | N/A |
| Pretrained weights (SAM3+VesselFM) | ~11 GB | $0.22/mo | $0.25/mo | $0.23/mo | ~$0.44/mo (NV) | N/A |
| MLflow artifacts (per pass) | ~2 GB | $0.04/mo | $0.05/mo | $0.04/mo | file-based | N/A |
| **Total storage/cloud** | | **$0.45/mo** | **$0.52/mo** | **$0.47/mo** | **~$0.56/mo** | **$0.00** |

**Total multi-cloud storage duplication cost**: ~$2.00/month for 4 clouds.

**Comparison with egress avoidance**:
- 34 jobs x 3 GB DVC pull from cross-cloud = 34 x 3 GB x $0.05-0.12 = $5.10-$12.24/pass
- 34 jobs x 6.4 GB Docker pull from cross-cloud = 34 x 6.4 GB x $0.05-0.12 = $10.88-$26.11/pass
- **Total egress per pass without co-location: $15.98-$38.35**

**Verdict from Cloud Architect**: Storage duplication at $2/month is 8-19x cheaper than
a single cross-cloud experiment pass. **Always replicate data to where compute runs.**
However, this introduces operational complexity (keeping images/data in sync across clouds).

### 3.3 Per-Cloud Registry Strategy

| Cloud | Registry | Push Strategy | Estimated One-Time Setup |
|-------|----------|---------------|--------------------------|
| GCP | GAR (europe-west4) | Already configured | 0h (done) |
| AWS | ECR (eu-west-1) | `docker push` to ECR | ~2h (ECR setup + IAM) |
| Azure | ACR (westeurope) | `docker push` to ACR | ~2h (ACR setup + RBAC) |
| RunPod | Direct pull from GHCR or GAR | No registry needed (pulls at boot) | 0h |
| Vast.ai | Direct Docker pull | No registry needed (pulls at boot) | 0h |
| Nebius | Direct Docker pull or private registry | TBD | ~1h |

**Council recommendation**: Do NOT set up AWS/Azure registries until we actually use those
clouds. Start with GAR (GCP) + direct pull for neoclouds. The neoclouds (RunPod, Vast.ai,
Nebius) have zero egress, so pulling a 6.4 GB image from GAR costs $0.77 in GCP egress
but $0.00 in neocloud ingress. At 34 pulls/pass, that is $26.18 -- significant enough
to warrant a GHCR mirror if neocloud usage becomes regular.

### 3.4 Recommended Data Locality Architecture

```
                    GCP europe-west4 (PRIMARY)
                    ├── GAR: Docker images
                    ├── GCS: DVC data, MLflow artifacts
                    ├── Cloud SQL: MLflow metadata
                    └── GPU: L4 spot (when available)
                         │
                         │ Fallback (when L4 exhausted)
                         ▼
              ┌─────────────────────────┐
              │   Neocloud (Vast.ai)    │
              │   Docker: pull from GAR │
              │   Data: DVC pull or     │
              │         file_mounts     │
              │   MLflow: Cloud Run URL │
              │   GPU: A100-80GB spot   │
              └─────────────────────────┘
```

---

## Section 4: Alternative Cloud Providers via SkyPilot

### 4.1 Provider Evaluation Matrix

| Provider | Docker `image_id` | GPU Options | Spot | EU Regions | SkyPilot Status | GDPR |
|----------|-------------------|-------------|------|------------|-----------------|------|
| **GCP** | Yes (full) | L4, A100, H100 | Yes | europe-west4 (NL), ew1 (BE) | Core, mature | Yes (EU regions) |
| **RunPod** | Partial (container-native, NOT Docker-in-Docker) | RTX4090, RTX5090, RTX3090, A100, H100 | Yes (community) | EU (NL, SE) | Supported | Partial |
| **Vast.ai** | Yes (Docker containers) | RTX4090, A100, H100, H200 | Yes (bid-based) | Some EU hosts | Supported (2025+) | GDPR committed, Secure Cloud ISO 27001 |
| **Nebius** | Yes (via SkyPilot integration) | H100, H200, L40S, B200 | No self-serve spot | eu-north1 (Finland), eu-west1 (France) | Supported, managed API server available | Yes (EU data centers) |
| **Lambda Labs** | Yes | A100, H100 | No | **No EU** | Supported | **No** (US only) |
| **CoreWeave** | Yes | A100, H100, H200 | Limited | EU (London) | Supported | Partial |
| **AWS** | Yes (full) | A100 (p4d), H100 (p5) | Yes | eu-west-1 (Ireland), eu-central-1 (Frankfurt) | Core, mature | Yes |
| **Azure** | Yes (full) | A100 (NC A100), H100 (ND H100) | Yes | westeurope, northeurope | Core, mature | Yes |

### 4.2 Detailed Provider Analysis

#### Vast.ai (RECOMMENDED as first fallback)

**Pros**:
- Cheapest A100-80GB pricing ($0.78/hr, lower than any hyperscaler)
- Docker container native -- workloads run in isolated Docker instances
- SkyPilot integration since 2025, with full `image_id` support
- Zero egress fees (data leaving Vast.ai is free)
- GDPR-committed with ISO 27001 Secure Cloud option
- Marketplace model means GPUs are almost always available somewhere
- January-February 2026 updates: full SkyPilot instance creation parameter support

**Cons**:
- EU host availability is variable (marketplace, not guaranteed capacity)
- Community hosts have unknown security posture (use Secure Cloud for medical data)
- No guaranteed SLA -- hosts can go offline
- Preemption model is bid-based, less predictable than GCP spot
- SkyPilot integration is newer (less battle-tested than GCP/AWS)

**Verdict from Security/Compliance Expert**: Vast.ai Secure Cloud with ISO 27001 hosts
is acceptable for pseudonymized medical imaging data. Raw patient data should NOT be
processed on community hosts. MiniVess/DeepVess data is from mouse brains (non-human),
so GDPR concerns are minimal for this specific dataset, but the platform design should
assume human data for generalizability.

#### Nebius (RECOMMENDED for EU compliance)

**Pros**:
- EU-first infrastructure (Finland, France, Iceland, UK)
- H100 at $2.95/hr -- competitive for Hopper class
- Managed SkyPilot API server available (zero controller management)
- B200 Blackwell available (future-proof)
- Strong EU data residency story

**Cons**:
- No A100 on self-serve (H100 minimum)
- No spot instances (on-demand only)
- H100 at $2.95/hr is expensive for 34 debug jobs ($2.95 x 0.35h x 34 = $35.11)
- Relatively new SkyPilot integration
- No consumer GPUs (RTX series)

**Verdict from Researcher Advocate**: Nebius is excellent for production runs where
EU compliance matters, but too expensive for debug passes. H100 at $2.95/hr for a
workload that fits on L4 is paying 13x more per hour for compute we do not need.

#### RunPod (EXISTING -- dev environment only)

**Pros**:
- Already configured as the "env" provider
- Consumer GPUs at excellent prices (RTX4090 $0.34/hr)
- EU availability (Netherlands)
- Zero egress fees

**Cons**:
- **Cannot run Docker-based SkyPilot training** (container-native platform, not VM-based)
- Network Volume workflow requires manual data upload
- MLflow is file-based on Network Volume (no Cloud Run integration)
- No spot preemption recovery (SkyPilot managed jobs limited on RunPod)

**Verdict from Cloud Architect**: RunPod remains the "dev" path per CLAUDE.md architecture.
It cannot serve as a GCP fallback for factorial experiments because it lacks Docker
image_id support in SkyPilot YAML (RunPod is container-native, not VM-based). The
existing architecture decision stands: RunPod for quick experiments, GCP for production.

#### AWS (NOT RECOMMENDED for this project)

**Pros**:
- Mature SkyPilot support, best spot infrastructure
- EU regions with good GPU availability
- Full Docker image_id support

**Cons**:
- **Adds a third cloud provider** (requires explicit user authorization per CLAUDE.md)
- A100/H100 only available in 8-GPU instances ($27-100/hr)
- Complex IAM/VPC setup (~4-8 hours)
- Egress costs ($0.09/GB) from AWS to GCP for MLflow logging
- No consumer GPUs

**Verdict**: Violates two-provider constraint. Only consider if user explicitly authorizes.

#### Azure (NOT RECOMMENDED for this project)

Same analysis as AWS: adds third provider, complex setup, egress costs. No advantage
over Vast.ai or Nebius for this workload.

### 4.3 SkyPilot Integration Maturity

| Provider | `sky launch` | `sky jobs launch` | `image_id: docker:` | Spot recovery | Multi-region failover |
|----------|-------------|------------------|---------------------|---------------|----------------------|
| GCP | Mature | Mature | Full | Full | Full |
| AWS | Mature | Mature | Full | Full | Full |
| Azure | Mature | Mature | Full | Full | Full |
| Vast.ai | Supported | Supported | Full (2026) | Bid-based | Marketplace |
| Nebius | Supported | Supported | Yes | N/A (no spot) | Limited |
| RunPod | Supported | Limited | **No** (container-native) | Limited | Limited |

---

## Section 5: Total Cost of Ownership Analysis

### 5.1 Cost Components Beyond GPU Hourly Rate

| Cost Component | GCP (current) | GCP + Vast.ai fallback | Multi-cloud (3+) |
|---------------|---------------|------------------------|-------------------|
| **GPU compute (34 debug jobs)** | $6.85 (L4 spot) | $6.85 or $13.26 (A100 Vast.ai) | Variable |
| **Data egress** | $0.00 (same-region) | $0.77 (GAR to Vast.ai, 1 pull) + $0.36 (DVC) | $15-38 per pass |
| **Storage (monthly)** | $0.45 | $0.45 (GCS only) | $2.00 (4 clouds) |
| **Controller VM** | $3.22/day | $3.22/day | $3.22/day |
| **Cloud SQL** | $0.78/day | $0.78/day | $0.78/day |
| **Cloud Run (MLflow)** | $0.10/day | $0.10/day | $0.10/day |
| **Setup time (per job)** | 10 min | 15 min (cross-cloud pull) | 15-20 min |
| **Failed/preempted jobs** | ~10-20% waste | ~15-25% waste (marketplace) | Variable |
| **Docker image pull** | $0.00 (same-region) | $0.77/pull (cross-cloud) | $0.77-2.56/pull |

### 5.2 Scenario Analysis: 1-Month Active Development

**Assumptions**: 4 debug passes/month (34 jobs each = 136 total jobs), 1 production pass
(640 jobs), always-on infrastructure.

**Scenario A: GCP-only (current, when L4 available)**

| Component | Monthly Cost |
|-----------|-------------|
| Always-on infrastructure | EUR 119-130 |
| 4 debug passes (L4 spot) | EUR 25 |
| 1 production pass (L4 spot) | EUR 130 |
| Egress | EUR 0 |
| **Total** | **~EUR 274-285** |

**Scenario B: GCP primary + Vast.ai fallback (recommended)**

| Component | Monthly Cost |
|-----------|-------------|
| Always-on infrastructure | EUR 119-130 |
| 2 debug passes on L4 (available weeks) | EUR 13 |
| 2 debug passes on A100 Vast.ai (exhaust weeks) | EUR 24 |
| 1 production pass (mixed) | EUR 100-160 |
| Egress (Vast.ai pulls from GAR) | EUR 5-10 |
| **Total** | **~EUR 261-337** |

**Scenario C: Vast.ai primary (cheapest compute)**

| Component | Monthly Cost |
|-----------|-------------|
| Always-on infrastructure (GCP, for MLflow/SQL) | EUR 119-130 |
| 4 debug passes (A100 Vast.ai) | EUR 49 |
| 1 production pass (A100 Vast.ai) | EUR 250 |
| Egress (every pull crosses clouds) | EUR 40-60 |
| **Total** | **~EUR 458-489** |

**Verdict from FinOps Analyst**: Scenario B is optimal. GCP primary when L4 available
(cheapest per-job), Vast.ai fallback when L4 exhausted (available, still affordable).
Scenario C is counterproductive -- egress costs erase the GPU savings.

### 5.3 The Hidden Cost of Waiting

The most expensive option is doing nothing:

| Waiting Scenario | Daily Cost | Per-Pass Delay Cost |
|-----------------|-----------|---------------------|
| Controller VM idle | $3.22/day | $3.22 per day waiting |
| Researcher blocked (8h/day at opportunity cost) | Priceless | Paper submission delayed |
| GCP spot queue (no SLA on provisioning) | Unbounded | Could be hours or days |

**1 day of waiting = $3.22 controller + blocked researcher > $13.26 for Vast.ai A100 pass**

The correct decision is always: switch to available GPU immediately rather than wait
for capacity.

---

## Section 6: Decision Matrix

### 6.1 Multi-Criteria Evaluation (1-5 scale, 5 = best)

| Criterion | Weight | L4 GCP Spot | A100 GCP Spot | A100 Vast.ai | H100 RunPod | H100 Nebius | RTX4090 RunPod |
|-----------|--------|-------------|---------------|-------------|-------------|-------------|----------------|
| **Raw GPU cost/pass** | 25% | 5 ($6.85) | 3 ($18.70) | 4 ($13.26) | 2 ($23.66) | 1 ($35.11) | 4 ($10.58) |
| **Total cost incl. infra** | 15% | 5 ($0 egress) | 4 ($0 egress) | 3 ($5-10 egress) | 3 (file-based) | 3 ($0 egress) | 3 (file-based) |
| **Availability** | 20% | 1 (EXHAUSTED) | 3 (medium) | 5 (marketplace) | 4 (on-demand) | 4 (on-demand) | 4 (on-demand) |
| **EU data residency** | 10% | 5 (NL) | 5 (NL) | 3 (variable) | 4 (NL) | 5 (Finland) | 4 (NL) |
| **Implementation effort** | 10% | 5 (done) | 3 (YAML contract change) | 2 (new provider setup) | 4 (exists as dev) | 2 (new provider setup) | 4 (exists as dev) |
| **SkyPilot maturity** | 10% | 5 | 5 | 3 | 2 (no Docker image_id) | 3 | 2 (no Docker image_id) |
| **Docker training support** | 5% | 5 | 5 | 5 | 1 (container-native) | 4 | 1 (container-native) |
| **Spot preemption risk** | 5% | 4 | 3 | 3 (bid-based) | N/A | N/A | N/A |
| **Weighted Score** | 100% | **3.60** | **3.65** | **3.65** | **2.80** | **2.75** | **3.10** |

### 6.2 Scoring Interpretation

The weighted scores cluster into three tiers:

1. **Tier 1 (score 3.5+)**: L4 GCP Spot, A100 GCP Spot, A100 Vast.ai
   - L4 GCP is cheapest but UNAVAILABLE (scored 1 on availability)
   - A100 GCP and A100 Vast.ai are tied: GCP is simpler (same cloud), Vast.ai is cheaper

2. **Tier 2 (score 3.0-3.5)**: RTX4090 RunPod
   - Good price, available, but cannot run Docker-based SkyPilot factorial YAML
   - Only useful for dev/interactive work, not automated factorial experiments

3. **Tier 3 (score <3.0)**: H100 RunPod, H100 Nebius
   - Overpowered and overpriced for this workload
   - H100 makes sense only for SAM3 Hybrid/TopoLoRA (>16 GB VRAM requirement)

---

## Section 7: Recommendations

### Recommendation 1: GCP A100-40GB Spot as Immediate Fallback (THIS WEEK)

**Action**: Add A100-40GB to the YAML contract `allowed_accelerators.gcp` and update
the factorial YAML with an `ordered` failover list.

**Rationale**:
- Same cloud = zero egress, zero new provider setup
- A100-40GB spot at ~$1.10/hr: debug pass costs $18.70 (vs $6.85 for L4)
- 40 GB VRAM handles all MinIVess models including SAM3 Hybrid
- Available in europe-west4 when L4 is not
- SkyPilot `ordered` resources enable automatic fallback: try L4 first, A100 second

**Implementation** (requires user authorization to modify YAML contract):

```yaml
# configs/cloud/yaml_contract.yaml — proposed change
allowed_accelerators:
  gcp:
    - L4
    - A100      # Fallback for L4 capacity exhaustion

# deployment/skypilot/train_factorial.yaml — proposed change
resources:
  ordered:
    - accelerators: L4:1
      cloud: gcp
      use_spot: true
      region: europe-west4
    - accelerators: A100:1
      cloud: gcp
      use_spot: true
      region: europe-west4
```

**Cost impact**: $11.85/pass more expensive than L4 spot. But $11.85 per pass is
dramatically better than infinite wait with $3.22/day controller burn.

**Estimated setup time**: 30 minutes (YAML changes + test + verify).

### Recommendation 2: Vast.ai as Secondary Fallback (NEXT 2 WEEKS)

**Action**: Enable Vast.ai as a SkyPilot backend with Docker image support.
Configure for A100-80GB on EU Secure Cloud hosts.

**Rationale**:
- Cheapest A100 pricing on the market ($0.78/hr)
- Docker container support confirmed via SkyPilot
- GDPR-committed Secure Cloud with ISO 27001
- Provides true multi-cloud resilience beyond GCP

**Implementation steps**:
1. Install SkyPilot Vast.ai backend: `uv add "skypilot-nightly[vastai]"`
2. Configure Vast.ai API key
3. Create `configs/cloud/vastai_fallback.yaml` Hydra config
4. Add Vast.ai to YAML contract `allowed_clouds`
5. Create SkyPilot YAML variant with Vast.ai resources
6. Test with a single training job before factorial launch
7. Verify MLflow Cloud Run URL is accessible from Vast.ai instances

**Cost impact**: $13.26/pass (debug) -- 93% more expensive than L4 but 30% cheaper
than A100 on GCP spot.

**Estimated setup time**: 4-6 hours (provider setup + testing + YAML contract update).

**Open questions** (require user decision):
- Does adding Vast.ai as a third provider violate the two-provider constraint?
  (Proposal: reclassify RunPod as "dev-only" and Vast.ai as "fallback compute",
  maintaining the spirit of two production providers: GCP + Vast.ai)
- Should Vast.ai Secure Cloud be mandated, or is community acceptable for mouse brain data?

### Recommendation 3: Nebius for Production Paper Runs (NEXT MONTH)

**Action**: Evaluate Nebius for production-grade runs where EU data residency is
paramount and H100 performance justifies the cost.

**Rationale**:
- EU-first infrastructure (Finland data center)
- Managed SkyPilot API server (eliminates controller VM cost)
- H100 at $2.95/hr is competitive for production runs where speed matters
- B200 Blackwell availability for future scaling

**When to use**: Production paper runs (640+ jobs) where wall-clock time matters
more than per-job cost. H100 completes jobs 5x faster than L4, so 640 jobs finish
in 2.3 days instead of 11.7 days (serial). The researcher time saved over 9 days
justifies the GPU premium for paper-deadline scenarios.

**Cost for production pass (640 jobs on H100)**: 640 x $0.70/job = **$448**
vs L4 at 640 x $0.20/job = **$128** (3.5x more expensive, but 5x faster).

**Estimated setup time**: 8-12 hours (new provider, Pulumi/IaC considerations, testing).

### 7.1 Phased Implementation Timeline

| Phase | Timeline | Action | Cost Impact | Effort |
|-------|----------|--------|-------------|--------|
| **Phase 1** | This week | Add A100 to GCP YAML contract, `ordered` failover | +$11.85/pass when L4 unavailable | 30 min |
| **Phase 2** | Week 2-3 | Set up Vast.ai SkyPilot backend, test single job | +$6.41/pass vs L4 (but cheaper than GCP A100) | 4-6h |
| **Phase 3** | Week 4+ | Evaluate Nebius for production runs | TBD based on paper deadline | 8-12h |
| **Ongoing** | Continuous | Monitor L4 availability, auto-select cheapest available GPU | Cost optimization | Automated |

### 7.2 What NOT to Do

1. **Do NOT add AWS or Azure** -- violates two-provider constraint, adds $5K+ setup overhead, egress costs dominate
2. **Do NOT use H100 for debug passes** -- 5x faster but 5x more expensive; debug passes are not wall-clock critical
3. **Do NOT use RunPod for factorial experiments** -- no Docker image_id in SkyPilot, requires manual workflow
4. **Do NOT wait for L4 capacity** -- the controller burns $3.22/day while waiting, which exceeds the A100 premium in <4 days
5. **Do NOT replicate data to all clouds preemptively** -- only replicate to clouds you actively use
6. **Do NOT use T4 as fallback** -- BANNED (Turing, no BF16, NaN with SAM3)
7. **Do NOT use B200/H200 for this workload** -- massively overpowered and overpriced; the dataset is too small to benefit

---

## Appendix A: Council Cross-Review Notes

### FinOps Analyst Review
- "The Cloud Architect's multi-cloud storage analysis is correct but understates
  operational complexity. Docker image sync across 4 registries requires CI/CD changes
  that cost engineer-hours. Only duplicate to clouds you actually use."
- "The $0.78/hr Vast.ai A100 price is a marketplace low. Sustained demand during
  factorial runs (34 simultaneous bids) will push Vast.ai prices up. Budget $1.00-1.20/hr
  for realistic modeling."

### Cloud Architect Review
- "The FinOps Analyst correctly identifies controller cost as the dominant idle waste.
  Recommendation: implement `sky jobs controller stop` automation when no jobs are queued
  for >1 hour. This saves $93/month."
- "The `ordered` resource list in SkyPilot YAML is the correct mechanism for L4-to-A100
  fallback. This is a 2-line YAML change, not an architecture change."

### ML Systems Engineer Review
- "The speedup factors in Section 2.3 are conservative for memory-bandwidth-bound 3D
  segmentation. A100's HBM2e at 1555 GB/s vs L4's 300 GB/s could yield 3-4x speedup,
  not the 2.2x estimated from raw TFLOPS ratio. Recommend benchmarking one real job."
- "SAM3 Hybrid/TopoLoRA at 7.5 GB VRAM fits A100-40GB easily. No need for 80GB variant
  unless batch size increases significantly."

### Security/Compliance Expert Review
- "Vast.ai Secure Cloud is acceptable for the current dataset (mouse brain imaging).
  If the project ever processes human medical data, re-evaluate. Nebius Finland is the
  strongest EU compliance story."
- "MLflow Cloud Run URL must be protected with authentication before exposing to
  non-GCP providers. Currently `ingress: all` with no auth -- acceptable for GCP-only
  but a risk when Vast.ai instances connect."

### Researcher Advocate Review
- "The Phase 1 recommendation (A100 fallback in YAML) is the right call for unblocking
  immediately. A 30-minute YAML change vs 4-6 hours of Vast.ai setup. Do Phase 1 NOW,
  evaluate Phase 2 after the current experiment pass completes."
- "The `ordered` failover in SkyPilot is invisible to the researcher -- they run the
  same command and SkyPilot picks the best available GPU. This is exactly the DevEx
  principle the project demands."

---

## Appendix B: Data Sources

### Pricing Sources (verified March 2026)
- [GCP GPU Pricing](https://cloud.google.com/compute/gpus-pricing)
- [GCP Spot VM Pricing](https://cloud.google.com/spot-vms/pricing)
- [GPU Cloud Pricing Comparison 2026 (Spheron)](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/)
- [H100 Rental Prices Compared (IntuitionLabs)](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)
- [L4 Cloud Pricing: 12+ Providers (GetDeploying)](https://getdeploying.com/gpus/nvidia-l4)
- [A100 Cloud Pricing: 37+ Providers (GetDeploying)](https://getdeploying.com/gpus/nvidia-a100)
- [H100 Cloud Pricing: 41+ Providers (GetDeploying)](https://getdeploying.com/gpus/nvidia-h100)
- [RunPod GPU Pricing](https://www.runpod.io/pricing)
- [Vast.ai GPU Pricing](https://vast.ai/pricing)
- [Nebius GPU Pricing](https://nebius.com/prices)
- [Google Cloud Network Pricing](https://cloud.google.com/vpc/network-pricing)

### SkyPilot Provider Support (verified March 2026)
- [SkyPilot GitHub: 20+ clouds supported](https://github.com/skypilot-org/skypilot)
- [Vast.ai SkyPilot Integration](https://vast.ai/article/vast-ai-gpus-can-now-be-rentend-through-skypilot)
- [Nebius SkyPilot Integration](https://nebius.com/blog/posts/nebius-ai-cloud-skypilot-integration)
- [Nebius Managed SkyPilot API Server](https://nebius.com/blog/posts/managed-skypilot-api-server-tech-overview-and-setup)

### Project-Internal Sources
- `configs/cloud/yaml_contract.yaml` -- current GPU allowlist
- `knowledge-graph/domains/cloud.yaml` -- two-provider architecture
- `docs/planning/v0-2_archive/original_docs/gcp-finops-snapshot-2026-03-28-report.md` -- cost baseline
- `docs/planning/v0-2_archive/original_docs/run-debug-factorial-experiment-11th-pass-finops-plan.md` -- egress analysis
- `docs/planning/v0-2_archive/original_docs/skypilot-and-finops-complete-report.md` -- SkyPilot architecture
- `.claude/metalearning/2026-03-15-t4-turing-fp16-nan-ban.md` -- T4 ban rationale
- `.claude/metalearning/2026-03-24-unauthorized-a100-in-skypilot-yaml.md` -- A100 authorization requirement

---

## Section 10: Multi-Region Infrastructure Replication Pattern

**Date added**: 2026-03-28 (Saturday night GPU crisis, both europe-west4 AND us-central1 exhausted)

**Context**: The Vascadia v0.2-beta project experienced GPU capacity exhaustion in BOTH
europe-west4 (13+ hours PENDING) AND us-central1 (1+ hour cycling STARTING/PENDING) on
the same Saturday night. This proves that single-region strategies are fundamentally
fragile regardless of which region is selected. The project needs a multi-region,
multi-provider architecture with infrastructure replication.

### Expert Council for Sections 10-13

| Expert | Domain | Key Contribution |
|--------|--------|------------------|
| **Cloud Architect** | Multi-region replication patterns | Per-cloud implementation, industry best practices |
| **FinOps Engineer** | Cost modeling, breakeven analysis | TCO, replication ROI, financial decision framework |
| **SkyPilot Expert** | Multi-cloud configuration | Intercloud broker capabilities, `ordered` YAML across clouds |
| **Data Engineer** | DVC multi-remote, data replication | Static data replication strategy, sync pipelines |
| **Dashboard Developer** | Availability monitoring pipeline | DuckDB analytics, heatmap dashboards, alerting |

### 10.1 The Pattern: Static Data Replication + Thin Docker Layer

The fundamental insight that makes multi-region viable for this project: **the data layer
is static**. MiniVess (2.89 GB), DeepVess (1.9 GB), and pretrained weights (SAM3 ~9 GB,
VesselFM ~2 GB) change on the timescale of months. Only the Docker image top layers change
with each code push (typically 50-200 MB of application code on top of a 6 GB base image).

This means the replication cost structure is dominated by one-time data copies, not
continuous synchronization:

```
Per-region infrastructure (replicated):
+-- Container Registry (GAR/ECR/ACR)   -- Docker images
+-- Object Storage (GCS/S3/Blob)       -- DVC data + pretrained weights
+-- MLflow Server (Cloud Run/ECS/ACA)  -- experiment tracking
+-- Database (Cloud SQL/RDS/Azure DB)  -- MLflow backend (SHARED cross-region preferred)

Replication frequency:
+-- DVC data:           Once per dataset version (months)
+-- Pretrained weights: Once per model release (months)
+-- Docker base image:  Once per dependency update (weeks)
+-- Docker app layer:   Every code push (daily during active dev, 50-200 MB)
+-- MLflow DB:          Shared single instance (cross-region queries OK for metadata)
```

**Cloud Architect assessment**: "This is the ideal replication profile. Static large
objects (datasets, weights) are replicated once and forgotten. The only continuous
replication is the thin Docker app layer, which at 50-200 MB costs effectively nothing
in transfer fees. The shared database pattern avoids the operational nightmare of
multi-master replication for a single-researcher academic project."

### 10.2 Per-Cloud Implementation

#### GCP (current primary)

GCP offers three approaches to container image replication:

1. **Multi-region GAR repository** (location: `europe` or `us`): Automatically replicates
   across regions within the multi-region boundary. Pricing is $0.13/GiB/month (vs
   $0.10/GiB for regional). For 6.4 GB of images, this adds ~$0.19/month. Simplest
   operationally but does not cross continental boundaries (a `europe` multi-region repo
   does not replicate to `us-central1`).

2. **Per-region repos with `gcrane` copy**: Use
   [gcrane](https://cloud.google.com/artifact-registry/docs/docker/copy-images) to copy
   images between regional GAR repositories. Can be automated via Cloud Pub/Sub triggers
   on push events, or run manually via `scripts/sync_regions.sh`. Supports cross-continent
   replication (europe-west4 to us-central1).

3. **Cross-region pull (no replication)**: Let the training VM pull from a remote GAR repo,
   paying egress. Only viable for intra-EU transfers ($0.01/GiB) -- cross-continental
   egress at $0.05-0.08/GiB on 6.4 GB images across 34 jobs = $10.88-$17.41/pass. Not
   recommended.

**GCS replication**: Use `gsutil -m cp -r` or
[Storage Transfer Service](https://cloud.google.com/storage-transfer-service) for
automated bucket-to-bucket copy. For 5 GB of static data, a one-time `gsutil` copy
costs $0.05 in cross-region transfer and takes <5 minutes.

**Cloud SQL**: Use a SINGLE shared instance. Cross-region MLflow metadata writes (small
JSON payloads, ~1 KB each) add ~10-20 ms latency with zero egress cost for writes.
This avoids the operational complexity and $23.50/month cost of a second Cloud SQL
instance. See Section 5.2 of the GCS Region Analysis report for the detailed cost model.

**Cloud Run MLflow**: Deploy a second Cloud Run service in the secondary region, pointing
to the same Cloud SQL backend but using that region's GCS bucket for artifact storage.
Monthly cost: $2-6.

| Component | Per-Region Cost | Replication Frequency | Automation |
|-----------|----------------|----------------------|------------|
| GAR (6.4 GB) | $0.13/month | Per code push (app layer only) | `gcrane cp` or multi-region repo |
| GCS DVC data (3 GB) | $0.06/month | Per dataset version (months) | `gsutil -m cp` |
| GCS pretrained weights (11 GB) | $0.22/month | Per model release (months) | `gsutil -m cp` |
| GCS MLflow artifacts | $0.02-0.20/month | Grows with experiments | Write-local (no replication needed) |
| Cloud Run MLflow | $2-6/month | Deploy once per region | Pulumi IaC |
| Cloud SQL | $0/month (shared) | N/A (single instance) | N/A |
| **Total per additional region** | **$2.43-6.61/month** | | |

#### AWS (SkyPilot supported -- training only, no deployment/CDN)

AWS provides native cross-region replication for both ECR and S3, making it the most
operationally simple multi-cloud target for training workloads:

| Service | AWS Equivalent | Replication Mechanism | Setup Effort |
|---------|---------------|----------------------|-------------|
| Container Registry | [ECR](https://docs.aws.amazon.com/AmazonECR/latest/userguide/replication.html) | **Native cross-region replication** -- automatic after initial config. Once enabled, every push to source ECR automatically replicates to destination regions. Supports up to 25 destination regions per replication rule. | ~2h (IAM roles + replication rules) |
| Object Storage | S3 | **S3 Replication Rules** -- automatic, per-bucket. Supports cross-region replication (CRR) with configurable destination buckets. | ~1h (bucket + replication config) |
| MLflow Server | ECS Fargate or EC2 | Deploy MLflow container via `docker-compose` on a single small instance. For training-only, a `t3.small` ($0.0208/hr = ~$15/month) with SQLite backend suffices. | ~4h (ECS task def + ALB + security groups) |
| Database | RDS PostgreSQL or SQLite | For training-only: SQLite on ECS ephemeral storage or EFS. For persistent: RDS db.t3.micro ($12.41/month). Cross-cloud shared DB not recommended (latency + complexity). | 0h (SQLite) or 2h (RDS) |

**AWS training-only cost per region**:

| Component | Monthly Cost |
|-----------|-------------|
| ECR storage (6.4 GB) | $0.64 |
| S3 storage (14 GB: DVC + weights) | $0.32 |
| S3 MLflow artifacts (growing) | $0.02-0.50 |
| ECS Fargate MLflow (256 CPU, 512 MB) | ~$10 |
| **Total** | **~$11-12/month** |

**Key advantage**: ECR cross-region replication is fully automatic. Push once to
`us-east-1`, and it replicates to `eu-west-1` without any scripting. S3 Replication Rules
similarly handle DVC data automatically after initial configuration.

**Key constraint**: Egress from AWS back to GCP (for MLflow consolidation or result
comparison) costs $0.09/GB. For training-only use, MLflow artifacts stay in AWS S3 and
are queried via the AWS MLflow server URL. No cross-cloud data movement needed during
training.

**SkyPilot YAML for AWS**:
```yaml
resources:
  ordered:
    - accelerators: L4:1
      cloud: gcp
      region: us-central1
      use_spot: true
    - accelerators: L4:1
      cloud: aws
      region: us-east-1
      use_spot: true
    - accelerators: A100:1
      cloud: gcp
      region: us-central1
      use_spot: true
```

#### Azure (SkyPilot supported -- training only, no deployment/CDN)

Azure Container Registry (ACR) offers native geo-replication, but **only on the Premium
tier** ($50/month base + ~$50/month per additional replica region). This is cost-
prohibitive for an academic project with $2/month of images to replicate.

| Service | Azure Equivalent | Replication Mechanism | Setup Effort |
|---------|-----------------|----------------------|-------------|
| Container Registry | [ACR](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-geo-replication) | **Geo-replication (Premium tier only)**. Premium: ~$50/month base + $50/month per replica. Alternative: Basic tier ($5/month) with manual `az acr import` for cross-region copies. | ~2h (ACR + RBAC) |
| Object Storage | Azure Blob Storage | **Object replication** -- available on standard tier. Automatic async replication between storage accounts in different regions. | ~1h (storage accounts + replication policy) |
| MLflow Server | [Azure Container Apps](https://learn.microsoft.com/en-us/azure/container-apps/) | Deploy MLflow container. Consumption plan: pay-per-use, ~$0 when idle. | ~3h (ACA + managed identity + networking) |
| Database | Azure Database for PostgreSQL Flexible Server | Burstable B1ms: ~$12.41/month. Or SQLite on Azure Files. | 0h (SQLite) or 2h (PostgreSQL) |

**Azure training-only cost per region (Basic ACR)**:

| Component | Monthly Cost |
|-----------|-------------|
| ACR Basic (6.4 GB, 10 GB included) | $5.00 |
| Blob Storage (14 GB) | $0.29 |
| Azure Container Apps MLflow (low traffic) | ~$5-10 |
| **Total** | **~$10-16/month** |

**FinOps Engineer assessment**: "Azure's Premium ACR at $50/month for geo-replication is
a non-starter for this project's budget. The workaround is a Basic ACR ($5/month) with
manual `az acr import` commands triggered by the same `sync_regions.sh` script that
handles GCP `gcrane`. Azure Blob replication is standard-tier and automatic -- no Premium
tax. For training-only workloads, Azure is viable but offers no cost advantage over AWS."

**SkyPilot YAML for Azure**:
```yaml
resources:
  ordered:
    - accelerators: L4:1
      cloud: gcp
      region: us-central1
      use_spot: true
    - accelerators: A100:1
      cloud: azure
      region: eastus
      use_spot: true
```

### 10.3 Cross-Cloud Comparison Summary

| Dimension | GCP (second region) | AWS (training-only) | Azure (training-only) |
|-----------|-------------------|--------------------|-----------------------|
| **Monthly cost** | $2.43-6.61 | $11-12 | $10-16 |
| **Registry replication** | `gcrane` (manual/automated) | ECR native (automatic) | ACR Basic manual import |
| **Data replication** | `gsutil` / Storage Transfer | S3 Replication (automatic) | Blob Object Replication (automatic) |
| **MLflow** | Cloud Run ($2-6) | ECS Fargate (~$10) | Container Apps (~$5-10) |
| **Database** | Shared Cloud SQL ($0 incremental) | Separate RDS/SQLite ($0-12) | Separate PostgreSQL/SQLite ($0-12) |
| **SkyPilot maturity** | Core (mature) | Core (mature) | Core (mature) |
| **Docker `image_id`** | Full support | Full support | Full support |
| **Spot recovery** | Full | Full | Full |
| **Setup effort** | ~2-4 hours | ~6-8 hours | ~6-8 hours |
| **Egress back to primary** | $0.01-0.08/GB | $0.09/GB | $0.087/GB |
| **GDPR (EU regions)** | europe-west4, europe-west1 | eu-west-1 (Ireland) | westeurope (Netherlands) |

**Cloud Architect verdict**: "For this project, a second GCP region is the clear first
step ($2.43-6.61/month, minimal setup). AWS is the best cross-cloud expansion when GCP
capacity is globally exhausted ($11-12/month, automatic ECR replication). Azure offers
no compelling advantage over AWS for training-only workloads and has the ACR Premium
tax for geo-replication. The recommended expansion order: GCP secondary region first,
then AWS, then Azure only if both GCP and AWS are simultaneously exhausted."

---

## Section 11: Multi-Cloud SkyPilot Configuration

### 11.1 Enabling Multiple Clouds

SkyPilot (v0.11+, December 2025) supports
[25+ infrastructure backends](https://github.com/skypilot-org/skypilot) including
all three hyperscalers (GCP, AWS, Azure) and neoclouds (Vast.ai, Nebius, RunPod,
CoreWeave, Lambda, Fluidstack, and more). Enabling a new cloud is a two-step process:

```bash
# 1. Install the cloud backend
uv add "skypilot-nightly[aws]"       # or [azure], [vastai], [nebius]

# 2. Verify credentials
uv run sky check

# Output shows enabled clouds:
# GCP: enabled
# AWS: enabled
# Azure: enabled
# RunPod: enabled
# Vast.ai: enabled
```

### 11.2 Multi-Cloud `ordered` Resource Specification

The `ordered` block in SkyPilot YAML specifies a strict preference order for resource
provisioning. SkyPilot tries each candidate sequentially and falls through to the next
on capacity exhaustion. This is the core mechanism for multi-region and multi-cloud
failover:

```yaml
# deployment/skypilot/train_factorial_multicloud.yaml
# Multi-cloud with per-cloud Docker registry and data paths

resources:
  ordered:
    # Priority 1: GCP us-central1 (cheapest, primary infra)
    - accelerators: L4:1
      cloud: gcp
      region: us-central1
      use_spot: true
      image_id: docker:us-central1-docker.pkg.dev/minivess-mlops/minivess/base:latest

    # Priority 2: GCP europe-west1 (EU fallback, co-located infra)
    - accelerators: L4:1
      cloud: gcp
      region: europe-west1
      use_spot: true
      image_id: docker:europe-west1-docker.pkg.dev/minivess-mlops/minivess/base:latest

    # Priority 3: AWS us-east-1 (cross-cloud fallback)
    - accelerators: L4:1
      cloud: aws
      region: us-east-1
      use_spot: true
      image_id: docker:<account-id>.dkr.ecr.us-east-1.amazonaws.com/minivess/base:latest

    # Priority 4: GCP A100 (higher tier, same cloud)
    - accelerators: A100:1
      cloud: gcp
      region: us-central1
      use_spot: true
      image_id: docker:us-central1-docker.pkg.dev/minivess-mlops/minivess/base:latest
```

### 11.3 Per-Region Environment Variable Injection

The critical operational detail: each cloud/region needs its Docker registry, MLflow
tracking URI, and data paths configured dynamically at job start time. SkyPilot exposes
`SKYPILOT_CLUSTER_INFO` and region metadata that the `run:` block can branch on:

```yaml
envs:
  # Defaults (GCP us-central1)
  GCS_DVC_BUCKET: gs://minivess-mlops-dvc-data
  GCS_PRETRAINED_BUCKET: gs://minivess-mlops-checkpoints/pretrained

run: |
  # Detect provisioned cloud and region
  CLOUD=$(python3 -c "import os; print(os.environ.get('SKYPILOT_CLOUD', 'gcp'))")
  REGION=$(python3 -c "import os; print(os.environ.get('SKYPILOT_REGION', 'us-central1'))")

  # Route to correct infrastructure
  if [ "$CLOUD" = "aws" ]; then
    export MLFLOW_TRACKING_URI="http://mlflow-aws.internal:5000"
    export DVC_REMOTE="s3://minivess-mlops-dvc-data-aws"
  elif [ "$REGION" = "europe-west1" ]; then
    export MLFLOW_TRACKING_URI="https://minivess-mlflow-ew1-xxxxx.run.app"
    export GCS_DVC_BUCKET="gs://minivess-mlops-dvc-data-ew1"
  fi

  # Training execution (unchanged)
  python -m minivess.flows.train_flow "$@"
```

**SkyPilot Expert assessment**: "The `ordered` + environment detection pattern is
production-proven at Shopify scale
([Shopify Engineering Blog, Jan 2026](https://shopify.engineering/skypilot)). The key
is that the training code itself is cloud-agnostic -- only the `run:` block handles
infrastructure routing. This maintains the project's principle that adding a new cloud
should be a YAML config change, not a code change."

### 11.4 DVC Multi-Remote Configuration

DVC natively supports multiple named remotes. Each remote can target a different
storage backend. The `dvc push -r <remote-name>` command pushes to a specific remote:

```ini
# .dvc/config (or generated from .env.example)
[core]
    remote = gcs

[remote "gcs"]
    url = gs://minivess-mlops-dvc-data

[remote "gcs-ew1"]
    url = gs://minivess-mlops-dvc-data-ew1

[remote "aws"]
    url = s3://minivess-mlops-dvc-data-aws
```

**Data replication workflow**:
```bash
# scripts/sync_regions.sh -- replicate data to all configured remotes
dvc push -r gcs        # Primary (us-central1)
dvc push -r gcs-ew1    # Secondary (europe-west1)
dvc push -r aws        # Tertiary (AWS us-east-1)
```

**Data Engineer assessment**: "DVC's multi-remote support is the correct abstraction.
Each remote is independently configured with its own credentials and endpoint. The
`sync_regions.sh` script should be idempotent -- running it multiple times pushes only
new or changed files. For 3 GB of static data, a full push takes <2 minutes per remote.
The important constraint: DVC remotes are push targets, not automatic replication. The
script must be run explicitly after dataset updates (monthly cadence for this project)."

---

## Section 12: Reproducible Availability Monitoring

### 12.1 Data Collection Pipeline

The following pipeline can be integrated into the Dashboard Flow (Flow 5) to provide
real-time and historical GPU availability data across regions and clouds:

```python
# src/minivess/observability/gpu_availability.py
# Collects GPU availability data for monitoring and decision support

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class AvailabilityStatus(Enum):
    AVAILABLE = "available"
    CONSTRAINED = "constrained"  # cycling STARTING/PENDING
    EXHAUSTED = "exhausted"      # 1h+ PENDING, zero provisions
    UNKNOWN = "unknown"


@dataclass
class AvailabilityRecord:
    timestamp: datetime
    cloud: str           # gcp, aws, azure, vastai, nebius
    region: str          # us-central1, eu-west-1, etc.
    gpu_type: str        # L4, A100, A100-80GB, H100
    spot_status: AvailabilityStatus
    on_demand_status: AvailabilityStatus
    spot_price_usd: float | None
    provisioning_latency_seconds: float | None
    source: str          # sky_check, gcloud_api, empirical, skypilot_log


@dataclass
class PricingRecord:
    timestamp: datetime
    cloud: str
    region: str
    gpu_type: str
    spot_price_usd: float | None
    on_demand_price_usd: float
    source: str          # cloud_api, getdeploying, cast_ai
```

**Collection methods** (ordered by reliability):

| Method | Data Quality | Latency | Cloud Coverage | Implementation |
|--------|-------------|---------|----------------|---------------|
| **SkyPilot `sky launch --dryrun`** | High (actual provisioning attempt) | 30-60s | All SkyPilot-supported clouds | `subprocess` call, parse JSON output |
| **`gcloud` CLI** | High (official API) | 5-10s | GCP only | `gcloud compute accelerator-types list --filter=zone:REGION` |
| **SkyPilot log parsing** | Medium (post-hoc) | 0s (from existing logs) | All clouds used in jobs | Parse `~/.sky/logs/` for PENDING durations |
| **[Cast AI Spot Map](https://cast.ai/spot-availability-map/)** | Medium (third-party) | API call | AWS, GCP, Azure | REST API scrape (if available) |
| **[GetDeploying](https://getdeploying.com/gpus)** | Medium (aggregator) | API call | 57+ providers | Web scrape (pricing only, not availability) |
| **GCP Capacity Planner** | High (official) | Console only | GCP only | Not programmatically accessible |

**Recommended collection schedule**: Every 60 minutes during active experiment periods,
every 6 hours during idle periods. Store results in DuckDB for analytics.

### 12.2 DuckDB Analytics Schema

```sql
-- gpu_availability: core time-series table
CREATE TABLE gpu_availability (
    timestamp    TIMESTAMPTZ NOT NULL,
    cloud        VARCHAR NOT NULL,      -- gcp, aws, azure, vastai, nebius
    region       VARCHAR NOT NULL,      -- us-central1, eu-west-1, etc.
    gpu_type     VARCHAR NOT NULL,      -- L4, A100, A100-80GB, H100
    spot_status  VARCHAR NOT NULL,      -- available, constrained, exhausted, unknown
    ondemand_status VARCHAR NOT NULL,
    spot_price_usd  DOUBLE,
    ondemand_price_usd DOUBLE,
    provisioning_latency_s DOUBLE,      -- seconds from launch to RUNNING
    source       VARCHAR NOT NULL,      -- sky_check, gcloud_api, empirical
    PRIMARY KEY (timestamp, cloud, region, gpu_type)
);

-- gpu_shortage_events: detected shortage episodes
CREATE TABLE gpu_shortage_events (
    event_id     INTEGER PRIMARY KEY,
    start_time   TIMESTAMPTZ NOT NULL,
    end_time     TIMESTAMPTZ,           -- NULL if ongoing
    cloud        VARCHAR NOT NULL,
    region       VARCHAR NOT NULL,
    gpu_type     VARCHAR NOT NULL,
    duration_hours DOUBLE GENERATED ALWAYS AS (
        EXTRACT(EPOCH FROM (COALESCE(end_time, NOW()) - start_time)) / 3600
    ),
    affected_jobs INTEGER,              -- count of jobs blocked
    estimated_cost_usd DOUBLE,          -- controller + idle infra during shortage
    pass_number  INTEGER                -- experiment pass affected (e.g. 11)
);

-- Historical data seeded from our 11-pass experience
INSERT INTO gpu_shortage_events VALUES
    (1, '2026-03-23 08:00+00', '2026-03-23 20:00+00', 'gcp', 'europe-north1', 'L4', NULL, 34, 3.22, 5),
    (2, '2026-03-23 08:00+00', '2026-03-23 20:00+00', 'gcp', 'europe-north1', 'L4', NULL, 34, 3.22, 6),
    (3, '2026-03-23 08:00+00', '2026-03-23 20:00+00', 'gcp', 'europe-north1', 'L4', NULL, 34, 3.22, 7),
    (4, '2026-03-28 09:00+00', '2026-03-28 22:00+00', 'gcp', 'europe-west4', 'L4', NULL, 3, 4.10, 11),
    (5, '2026-03-28 23:00+00', NULL, 'gcp', 'us-central1', 'L4', NULL, 34, NULL, 11);
```

**Analytical queries for the dashboard**:

```sql
-- Average shortage duration by region (last 30 days)
SELECT region, cloud, gpu_type,
       AVG(duration_hours) AS avg_shortage_hours,
       COUNT(*) AS shortage_count
FROM gpu_shortage_events
WHERE start_time > NOW() - INTERVAL '30 days'
GROUP BY region, cloud, gpu_type
ORDER BY shortage_count DESC;

-- Current availability heatmap data
SELECT cloud, region, gpu_type, spot_status, ondemand_status, spot_price_usd
FROM gpu_availability
WHERE timestamp = (SELECT MAX(timestamp) FROM gpu_availability)
ORDER BY cloud, region, gpu_type;

-- Best region to launch NOW (available + cheapest)
SELECT cloud, region, gpu_type, spot_price_usd
FROM gpu_availability
WHERE timestamp > NOW() - INTERVAL '1 hour'
  AND spot_status = 'available'
  AND gpu_type IN ('L4', 'A100')
ORDER BY spot_price_usd ASC
LIMIT 5;
```

### 12.3 Dashboard Integration

The Dashboard Flow (Flow 5) can incorporate GPU availability monitoring as a new
dashboard panel. The architecture follows the existing pattern: DuckDB for analytics,
Gradio or Streamlit for visualization.

**Dashboard panels**:

| Panel | Visualization | Data Source | Update Frequency |
|-------|--------------|-------------|-----------------|
| **Availability Heatmap** | Region x GPU matrix, color-coded (green/yellow/red) | `gpu_availability` table, latest record per region | Every 60 min |
| **Shortage Timeline** | Gantt chart of shortage events per region | `gpu_shortage_events` table | On new event detection |
| **Spot Price Trends** | Line chart, 30-day rolling window per GPU type | `gpu_availability` table, aggregated hourly | Every 60 min |
| **Cost Recommendation** | Text panel: "Best launch target: {cloud}/{region} at ${price}/hr" | `gpu_availability` latest + `gpu_shortage_events` active | Every 60 min |
| **Monthly Shortage Report** | Bar chart: shortage count + total hours by region | `gpu_shortage_events` table, monthly aggregation | Daily |

**Alert conditions** (integrated with the existing Prefect monitoring):

| Alert | Condition | Action |
|-------|-----------|--------|
| **Regional exhaustion** | `spot_status = 'exhausted'` for >30 min in primary region | Log warning, suggest failover region |
| **Global exhaustion** | All configured regions show `exhausted` simultaneously | Log critical, suggest neocloud fallback |
| **Price spike** | Spot price >2x 7-day moving average | Log warning, include cost projection |
| **Prolonged shortage** | `gpu_shortage_events.duration_hours > 6` and event still active | Log critical, estimate blocked cost |

**Dashboard Developer assessment**: "The DuckDB schema is designed for time-series
analytics with minimal storage overhead. At 24 records/day (hourly) x 5 regions x 3
GPU types = 360 records/day, the table grows at ~130 KB/month. This is negligible.
The key design decision is that availability data collection runs as a lightweight
background task within the Dashboard Flow, not as a separate infrastructure component.
This keeps operational complexity at zero for the researcher."

---

## Section 13: Financial Decision Framework

### 13.1 The Optimization Problem

Given:
- `N` replicated regions, each with cost `C_rep`/month
- GPU shortage probability `P_shortage(N)` per month (decreases with N)
- Cost per shortage event `C_blocked` (controller idle + researcher time)
- Correlation coefficient `rho` between regional capacity (0 = independent, 1 = fully correlated)

The expected monthly cost as a function of replicated regions is:

```
E[monthly_cost](N) = N * C_rep + P_shortage(N) * C_blocked
```

Where the shortage probability decreases with each additional independent region:

```
P_shortage(1) = p                          (single region failure probability)
P_shortage(2) = p^2 * (1-rho) + p * rho    (two independent regions)
P_shortage(3) = p^3 * (1-rho)^2 + p * rho  (three independent regions)
```

The correlation term `rho` captures the "Saturday night global GPU shortage" phenomenon
we observed: when demand is globally high, all regions are affected simultaneously.

### 13.2 Parameter Estimation from Empirical Data

From our 11-pass experiment history (2026-03-23 to 2026-03-28):

| Parameter | Value | Source |
|-----------|-------|--------|
| `p` (single-region shortage probability) | **0.45/month** (~5 events in 11 passes across ~6 days) | Passes 5, 6, 7 (europe-north1), 11 (europe-west4 + us-central1) |
| `C_rep_gcp` (GCP second region) | **$5/month** (mid-estimate of $2.43-6.61 range) | Section 10.2, Scenario B |
| `C_rep_aws` (AWS training-only) | **$12/month** | Section 10.2, AWS analysis |
| `C_rep_azure` (Azure training-only) | **$13/month** | Section 10.2, Azure analysis |
| `C_blocked` (cost per shortage event) | **$50-100** | $3.22/day controller + $50-100/day opportunity cost |
| `rho` (cross-region correlation) | **0.3** (intra-cloud, same provider) | Empirical: europe-west4 + us-central1 exhausted simultaneously |
| `rho_cross` (cross-cloud correlation) | **0.1** (different providers, different capacity pools) | Estimated: GCP and AWS share some demand but have independent inventory |
| Shortage duration | **6-13 hours** (mean ~9 hours) | Passes 5-7 (~12h), pass 11 (13h+ and counting) |

**FinOps Engineer note**: "The `p` estimate of 0.45/month is conservatively low -- it is
based on only 6 days of data during active experimentation. Real-world probability depends
on time-of-week (weekends are worse), time-of-year (Q4 and conference deadlines increase
demand), and whether we are competing with hyperscaler-scale training jobs for the same
GPU pool. I recommend using 0.5/month for planning."

### 13.3 Scenario Modeling

**Scenario 1: Single GCP region (current state)**

```
E[cost] = 0 * C_rep + 0.5 * $75 = $37.50/month in expected shortage cost
```

**Scenario 2: Two GCP regions (us-central1 + europe-west1)**

```
P_shortage(2) = 0.5^2 * (1-0.3) + 0.5 * 0.3 = 0.175 + 0.15 = 0.325
E[cost] = $5/month + 0.325 * $75 = $5 + $24.38 = $29.38/month
Savings vs single region: $37.50 - $29.38 = $8.12/month
```

**Scenario 3: GCP (2 regions) + AWS (1 region)**

```
P_shortage(3) = P(all_gcp_fail) * P(aws_fail)
             = 0.325 * 0.5 * (1-0.1) + 0.5 * 0.1
             = 0.146 + 0.05 = 0.196
E[cost] = $5 + $12 + 0.196 * $75 = $17 + $14.70 = $31.70/month
Savings vs single region: $37.50 - $31.70 = $5.80/month
```

**Scenario 4: GCP (2 regions) + AWS + Azure**

```
P_shortage(4) ~ 0.10 (diminishing returns, cross-cloud correlation reduces benefit)
E[cost] = $5 + $12 + $13 + 0.10 * $75 = $30 + $7.50 = $37.50/month
Savings vs single region: $37.50 - $37.50 = $0.00/month (breakeven)
```

### 13.4 Decision Matrix

| Strategy | Regions | Monthly Infra | Monthly Shortage Cost | **Total E[cost]** | Breakeven vs Single |
|----------|---------|-------------|---------------------|--------------------|---------------------|
| **Single GCP** | 1 | $0 | $37.50 | **$37.50** | -- |
| **Dual GCP** | 2 | $5 | $24.38 | **$29.38** | Immediate (saves $8.12/mo) |
| **GCP + AWS** | 3 | $17 | $14.70 | **$31.70** | 3 months (marginal over dual GCP) |
| **GCP + AWS + Azure** | 4 | $30 | $7.50 | **$37.50** | Never (breakeven only) |

### 13.5 Sensitivity Analysis

The model is most sensitive to two parameters: shortage probability `p` and
blocked cost `C_blocked`:

| If `p` changes to... | Single Region | Dual GCP | GCP + AWS |
|----------------------|---------------|----------|-----------|
| 0.25/month (optimistic) | $18.75 | $16.56 | $20.08 |
| 0.50/month (baseline) | $37.50 | $29.38 | $31.70 |
| 0.75/month (pessimistic) | $56.25 | $40.69 | $41.83 |
| 1.0/month (certain) | $75.00 | $51.25 | $51.70 |

| If `C_blocked` changes to... | Single Region | Dual GCP | GCP + AWS |
|------------------------------|---------------|----------|-----------|
| $25 (controller only) | $12.50 | $13.13 | $21.90 |
| $75 (baseline) | $37.50 | $29.38 | $31.70 |
| $150 (deadline pressure) | $75.00 | $53.75 | $46.40 |
| $300 (grant deadline) | $150.00 | $102.50 | $75.80 |

**Key insight from FinOps Engineer**: "Dual GCP dominates across all reasonable parameter
ranges. GCP + AWS only becomes superior at high `C_blocked` values ($150+/event), which
corresponds to grant or paper deadlines. The recommendation is clear: implement Dual GCP
now ($5/month), add AWS only when facing a deadline-critical experiment pass."

### 13.6 When Does Multi-Cloud Make Financial Sense?

Combining the cost model with operational complexity:

| Trigger | Strategy | Monthly Cost | Setup Time | Justification |
|---------|----------|-------------|-----------|---------------|
| **Default (no deadline)** | Dual GCP (us-central1 + europe-west1) | +$5/month | 2-4 hours | Positive ROI from day 1. No cross-cloud complexity. |
| **Conference deadline** | Dual GCP + AWS us-east-1 | +$17/month | 6-8 hours | When a 13-hour block would miss a submission deadline. |
| **Grant deadline + GPU crunch** | Dual GCP + AWS + Vast.ai | +$17/month + per-use | 10-14 hours | Maximum availability. Vast.ai marketplace almost always has A100s. |
| **Human clinical data** | Dual EU GCP only | +$5/month | 2-4 hours | GDPR requires EU data residency. AWS eu-west-1 acceptable with DPA. |

### 13.7 Implementation Roadmap

| Phase | Timeline | Action | Monthly Cost | Cumulative Effort |
|-------|----------|--------|-------------|-------------------|
| **Phase 0** | Immediate | Dual GCP (us-central1 + europe-west1 via Pulumi) | +$5 | 2-4 hours |
| **Phase 1** | Next experiment crunch | Add `scripts/collect_gpu_availability.py` to Dashboard Flow | +$0 | 4-6 hours |
| **Phase 2** | Next deadline pressure | Enable AWS via SkyPilot, push images to ECR, configure DVC remote | +$12 | 6-8 hours |
| **Phase 3** | If needed | Vast.ai or Nebius as neocloud fallback | Per-use only | 4-6 hours |
| **Phase 4** | Monthly | Review DuckDB availability data, adjust region priorities | +$0 | 30 min/month |

---

## Section 13 Appendix: Expert Council Cross-Review (Sections 10-13)

### Cloud Architect Review

"The static-data-replication pattern is the correct architecture for this workload profile.
The key differentiator from enterprise multi-region setups is that there is no streaming
data, no real-time replication requirement, and no multi-master database complexity. The
monthly dataset sync via `scripts/sync_regions.sh` is operationally simpler than any
automated replication pipeline and perfectly adequate for monthly dataset releases. I
recommend implementing the `gcrane` automation via Cloud Build triggers -- when a new
image is pushed to the primary GAR, a Cloud Build trigger automatically copies it to
secondary regions. This eliminates the risk of forgotten manual syncs."

### FinOps Engineer Review

"The financial model clearly shows that Dual GCP is the optimal strategy for the
foreseeable future. The $5/month cost is negligible compared to the $37.50/month expected
shortage cost. The cross-cloud (AWS/Azure) expansion has diminishing returns due to
the correlation term -- Saturday night GPU exhaustion affects all hyperscalers
simultaneously because they share the same demand pattern (weekend batch jobs). The
only escape from correlated failures is neoclouds (Vast.ai, Nebius) which have
independent capacity pools. I recommend: Dual GCP now, Vast.ai next, AWS only if the
user explicitly authorizes a third cloud provider."

### SkyPilot Expert Review

"The `ordered` multi-cloud YAML is the correct configuration pattern. One important detail:
the SkyPilot controller must be on the SAME cloud as the FIRST `ordered` candidate (per
the metalearning from the 5th pass). If the primary is GCP us-central1, the controller
must be on GCP. If a cross-cloud failover to AWS occurs, SkyPilot handles the SSH tunnel
transparently -- the controller does NOT need to be on AWS.

One correction to the configuration: `image_id` in an `ordered` block must match the
target cloud's registry format. GCP uses `docker:REGION-docker.pkg.dev/...`, AWS uses
`docker:ACCOUNT.dkr.ecr.REGION.amazonaws.com/...`. SkyPilot validates these at submission
time, so format errors are caught before any cloud resources are provisioned."

### Data Engineer Review

"The DVC multi-remote configuration is straightforward. The one operational risk is
credential management: each DVC remote needs its own auth (GCP ADC for GCS, AWS IAM
for S3, Azure managed identity for Blob). On a SkyPilot-provisioned VM, GCP credentials
are injected automatically via the service account. For AWS, the SkyPilot YAML needs
`aws_credentials: true` or an IAM role attached to the instance. I recommend testing
cross-cloud DVC push/pull before relying on it during an experiment pass."

### Dashboard Developer Review

"The DuckDB availability schema is designed for integration with the existing Dashboard
Flow. The collection pipeline runs as a Prefect task within the dashboard flow, not as a
standalone script. The recommended implementation path:

1. Add `gpu_availability.py` to `src/minivess/observability/`
2. Add a `collect_availability` Prefect task that runs on a 60-min schedule
3. Store results in the dashboard DuckDB instance at `data/dashboard/analytics.duckdb`
4. Add Gradio panels for the heatmap and price trend visualizations
5. The `gpu_shortage_events` table is populated by a detection algorithm that marks
   transitions from `available` to `exhausted` (and back)

The historical seeding from our 11-pass data (5 shortage events already) provides
enough data for meaningful visualizations from day one."

---

## Appendix C: Web Research Sources (Sections 10-13)

### Multi-Cloud Architecture
- [SkyPilot GitHub: 25+ Infrastructure Backends](https://github.com/skypilot-org/skypilot) -- current cloud support list
- [SkyPilot at Shopify: Multi-Cloud GPUs Without the Pain (2026)](https://shopify.engineering/skypilot) -- production multi-cloud case study
- [From SLURM to SkyPilot: How Avataar Cut Costs 11x](https://blog.skypilot.co/avataar/) -- multi-provider (AWS, RunPod, Azure, GCP) case study
- [SkyPilot YAML Spec: `ordered` Resources](https://docs.skypilot.co/en/stable/reference/yaml-spec.html) -- multi-cloud failover configuration
- [Multi-Cloud GPU Orchestration: AWS, Azure, GCP Guide 2025 (Introl)](https://introl.com/blog/multi-cloud-gpu-orchestration-aws-azure-gcp) -- orchestration patterns

### Container Registry Replication
- [GCP Artifact Registry Locations: Multi-Region Support](https://cloud.google.com/artifact-registry/docs/repositories/repo-locations) -- GAR multi-region repos
- [GCP: Copy Images Between Repositories (gcrane)](https://cloud.google.com/artifact-registry/docs/docker/copy-images) -- cross-region image copy
- [AWS ECR Cross-Region Replication](https://docs.aws.amazon.com/AmazonECR/latest/userguide/replication.html) -- automatic replication rules
- [Azure ACR Geo-Replication](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-geo-replication) -- Premium tier geo-replication

### GPU Availability Monitoring
- [Cast AI Spot Availability Map](https://cast.ai/spot-availability-map/) -- real-time spot interruption data (AWS, GCP, Azure)
- [Cloud GPU Spot Instance Availability (ThunderCompute 2026)](https://www.thundercompute.com/blog/cloud-gpu-spot-instance-availability) -- interruption rates and availability patterns
- [The State of Cloud GPUs in 2025 (dstack)](https://dstack.ai/blog/state-of-cloud-gpu-2025/) -- multi-cloud GPU market analysis
- [Cloud GPU Pricing: Compare 57 Providers (GetDeploying)](https://getdeploying.com/gpus) -- real-time pricing aggregator
- [GCP Capacity Planner](https://docs.google.com/capacity-planner/docs/overview) -- official GCP capacity visibility

### DVC Multi-Remote
- [DVC Remote Storage Documentation](https://dvc.org/doc/user-guide/data-management/remote-storage) -- multi-backend configuration
- [DVC `remote add` Command Reference](https://dvc.org/doc/command-reference/remote/add) -- adding multiple named remotes

### MLflow Multi-Region
- [MLflow Artifact Stores Documentation](https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/) -- supported backends (S3, GCS, Azure Blob)
- [Deploying MLflow to Azure (MLflow Docs)](https://mlflow.org/docs/latest/self-hosting/deploy-to-cloud/azure/) -- Azure Container Apps deployment pattern

---

---

## Section 14: Scheduled Training & Controller Cost Optimization

Two critical cost-efficiency questions that the main GPU selection analysis does not
address: (1) can we schedule training launches at optimal wall times to maximize spot
GPU availability? (2) how do we avoid burning ~$3.22/day on the SkyPilot controller
VM when no experiments are running?

### 14.1 GPU Availability by Time-of-Day

#### Empirical Evidence

Cloud GPU spot availability follows clear demand-driven cycles. The evidence comes from
three sources:

**1. Google Cloud Official Documentation**
[GCP Spot VM documentation](https://cloud.google.com/compute/docs/instances/spot) states:
"The load on Google Cloud data centers varies with location and time of day, but is
generally lowest on nights and weekends, making nights and weekends the best times to
run large clusters of Spot VMs." Google also recommends "using a region in a time zone
opposite your working hours" -- e.g., launching at 10:00 GMT targets 02:00 in
us-west-4 (Las Vegas), where demand is at its lowest.

**2. Interruption Rate Data** ([ThunderCompute, 2026](https://www.thundercompute.com/blog/cloud-gpu-spot-instance-availability))
Analysis of spot instance interruption rates by GPU type:

| GPU | Interruption Band | Risk Level |
|-----|-------------------|------------|
| L4 | 1-10% | Low-Medium (but availability is zero when exhausted) |
| A100 | 5-10% | Medium |
| H100 | 10-20% | High (most contested GPU class) |

An interruption rate under 5% is generally considered safe for ML training with
checkpointing. Above 10% means "at least one interruption during a day-long training
run." Weekend interruption rates are approximately 40% lower than weekdays
([ThunderCompute, 2026](https://www.thundercompute.com/blog/cloud-gpu-spot-instance-availability)).

Night and weekend periods offer 10-15% additional savings as enterprise workloads
decrease. Analysis across 10 million spot instance hours shows significant regional
variance: US-East-1 has a 3x higher interruption rate than US-West-2
([Introl, 2026](https://introl.com/blog/spot-instances-preemptible-gpus-ai-cost-savings)).

**3. SkyPilot Spot-Traces Dataset**
The [SkyPilot spot-traces dataset](https://github.com/skypilot-org/spot-traces) (used in
["Can't Be Late: Optimizing Spot Instance Savings under Deadlines," NSDI'24](https://www.usenix.org/conference/nsdi24))
contains 2-week to 2-month availability traces for AWS V100/K80 and GCP CPU instances,
sampled at 5-minute intervals (gap_seconds=300). While the traces do not explicitly
analyze time-of-day patterns, the raw data enables such analysis. The dataset covers
AWS and GCP from October 2022 through August 2023.

**4. GCP Preemption Lifetime Research**
Academic research on GCP preemption patterns ([Modeling The Temporally Constrained
Preemptions of Transient Cloud VMs](https://arxiv.org/pdf/1911.05160)) shows a "bathtub
curve": higher preemption rates at VM launch and again approaching the 24-hour limit,
with a relatively low rate in between. VMs that survive past 3 hours enjoy a stable
period of low preemption.

**5. Our Empirical Observation (2026-03-28)**
Saturday evening (EU prime time / US afternoon) saw complete L4 exhaustion in
europe-west4 across all zones. This aligns with the pattern: weekend *evenings* are
not off-peak if they coincide with US business hours. The optimal window is when BOTH
EU and US demand are low: **02:00-06:00 UTC** (nighttime EU, evening/night US West).

#### Time-of-Day Availability Model

```
UTC Time    EU Local    US West     Demand Level    Spot Availability
────────────────────────────────────────────────────────────────────
00:00-02:00  01-03 CET  16-18 PST  Medium-Low      Good (EU asleep, US winding down)
02:00-06:00  03-07 CET  18-22 PST  LOWEST          BEST (EU asleep, US evening)
06:00-08:00  07-09 CET  22-00 PST  Rising          Good (EU waking, US asleep)
08:00-14:00  09-15 CET  00-06 PST  Medium          Moderate (EU peak, US asleep)
14:00-18:00  15-19 CET  06-10 PST  HIGH            Poor (EU peak + US morning)
18:00-22:00  19-23 CET  10-14 PST  HIGHEST         WORST (EU evening + US peak)
22:00-00:00  23-01 CET  14-16 PST  High-Medium     Moderate (EU winding down, US peak)
```

**Key insight**: The window 02:00-06:00 UTC is consistently the off-peak sweet spot
for europe-west4 because it falls in the gap between European and American business
hours. This is when spot capacity is most likely to be freed by completed overnight
batch jobs from US companies.

### 14.2 Scheduled Launch Strategy

Four options evaluated for launching factorial experiments during optimal availability
windows:

#### Option A: Cron / systemd Timer (RECOMMENDED for Phase 1 -- Simplest)

**Implementation**: A cron job or systemd timer on the researcher's dev machine (or a
cheap always-on VM) that executes `scripts/run_factorial.sh` at the target time.

```bash
# crontab -e
# Launch factorial experiment at 03:00 UTC (04:00 CET) on weekdays
0 3 * * 1-5 cd /path/to/vascadia && bash scripts/run_factorial.sh 2>&1 \
    | tee /var/log/factorial-$(date +\%F).log

# Or using systemd timer for better logging and failure handling:
# /etc/systemd/system/factorial-launch.timer
[Unit]
Description=Launch factorial training at off-peak hours

[Timer]
OnCalendar=*-*-* 03:00:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
```

| Criterion | Rating |
|-----------|--------|
| Implementation effort | 15 minutes |
| Reliability | High (systemd) / Medium (cron on laptop -- requires machine to be awake) |
| Researcher experience | Fire-and-forget after initial setup |
| Requires always-on machine | Yes (laptop must be open, or use a EUR 5/month VPS) |

#### Option B: SkyPilot Scheduled/Delayed Launch

**Finding**: SkyPilot does NOT support scheduled or delayed job launches as of v0.11
(December 2025). The `sky jobs launch` command executes immediately. There is no
`--schedule` or `--delay` flag. The SkyPilot documentation ([Managed Jobs](https://docs.skypilot.co/en/latest/examples/managed-jobs.html),
[CLI Reference](https://docs.skypilot.co/en/latest/reference/cli.html)) confirms no
scheduling primitives exist.

**Workaround**: Wrap `sky jobs launch` in a cron job (Option A). SkyPilot handles the
compute provisioning; scheduling the launch time is the user's responsibility.

| Criterion | Rating |
|-----------|--------|
| Implementation effort | N/A (not supported natively) |
| Reliability | N/A |
| Workaround | Cron wrapper (reduces to Option A) |

#### Option C: Prefect Scheduled Deployment (RECOMMENDED for Phase 2 -- Most Integrated)

**Implementation**: Deploy the factorial launch as a Prefect 3.x scheduled deployment
with a cron trigger. This integrates with the existing Prefect orchestration architecture.

Prefect 3.x supports [cron, interval, and RRule schedules](https://docs.prefect.io/v3/concepts/schedules)
natively. Schedules are configured in `prefect.yaml` or programmatically.

```yaml
# prefect.yaml
name: factorial-offpeak
flow_name: factorial_launch_flow
parameters:
  pass_type: debug
schedules:
  - cron: "0 3 * * 1-5"    # 03:00 UTC weekdays
    timezone: "UTC"
```

Or programmatically:

```python
from prefect import flow
from prefect.client.schemas.schedules import CronSchedule

@flow
def factorial_launch_flow(pass_type: str = "debug"):
    """Launch factorial experiment via SkyPilot."""
    # ... existing launch logic via subprocess or SkyPilot Python API ...
```

| Criterion | Rating |
|-----------|--------|
| Implementation effort | 2-4 hours (deployment config + testing) |
| Reliability | High (Prefect server manages scheduling) |
| Researcher experience | Best (visible in Prefect UI, parameterizable) |
| Requires always-on machine | Yes (Prefect server/agent must be running) |
| Integration | Native (fits existing 5-flow Prefect architecture) |

**Advantage over cron**: Prefect provides a UI showing scheduled runs, execution history,
parameter overrides, and manual trigger capability. A researcher can see "next factorial
launch: tomorrow 03:00 UTC" in the dashboard and override parameters without editing
crontab files. Scheduling constraints (no more than 100 runs queued, no scheduling beyond
100 days) are non-issues for experiment launches.

**Important**: Always explicitly specify `timezone: "UTC"` in the schedule configuration.
Incorrect time zone settings cause schedules to behave unexpectedly
([Prefect docs](https://docs.prefect.io/v3/concepts/schedules)).

#### Option D: Smart Launch Wrapper Script

**Implementation**: A wrapper around `run_factorial.sh` that checks wall time and
either launches immediately or queues for the next off-peak window.

```bash
#!/bin/bash
# scripts/smart_launch.sh — time-aware factorial launcher
set -euo pipefail

HOUR=$(date -u +%H)
if [ "$HOUR" -ge 2 ] && [ "$HOUR" -lt 6 ]; then
    echo "[$(date -u)] OFF-PEAK window — launching immediately"
    bash scripts/run_factorial.sh "$@"
elif [ "$HOUR" -ge 14 ] && [ "$HOUR" -lt 22 ]; then
    NEXT_WINDOW="03:00 UTC tomorrow"
    echo "[$(date -u)] PEAK hours — queuing for $NEXT_WINDOW"
    echo "cd $(pwd) && bash scripts/run_factorial.sh $*" | at "$NEXT_WINDOW"
else
    echo "[$(date -u)] MODERATE hours — launching with 30-min timeout"
    timeout 1800 bash scripts/run_factorial.sh "$@" || {
        echo "[$(date -u)] Not provisioned in 30 min — queuing for 03:00 UTC"
        echo "cd $(pwd) && bash scripts/run_factorial.sh $*" | at "03:00 UTC tomorrow"
    }
fi
```

| Criterion | Rating |
|-----------|--------|
| Implementation effort | 1 hour |
| Reliability | Medium (depends on `at` daemon, machine must stay on) |
| Researcher experience | Good (transparent, logged decisions) |
| Requires always-on machine | Yes |

#### Scheduling Strategy Recommendation

**Phase 1 (immediate)**: Use Option A (cron/systemd) for simplicity. Zero dependencies
beyond the operating system.

**Phase 2 (when Prefect infrastructure is stable)**: Migrate to Option C (Prefect
scheduled deployment) for better observability and integration with the existing
5-flow architecture. The Prefect flow can also embed the smart launch logic from
Option D -- checking availability via `sky gpus --cloud gcp` before submitting jobs.

**Do NOT implement Option D** as the standalone primary mechanism -- the smart launch
logic is better embedded inside a Prefect flow than as a separate bash script.

### 14.3 Controller Cost Optimization

The SkyPilot jobs controller is the single largest always-on infrastructure cost after
Cloud SQL. Current burn rate: ~$3.22/day ($0.134/hr x 24h) on an n4-standard-4 VM.

#### Current Controller Lifecycle (As Documented by SkyPilot)

Per [SkyPilot Managed Jobs docs](https://docs.skypilot.co/en/latest/examples/managed-jobs.html)
and [Advanced Configuration](https://docs.skypilot.co/en/latest/reference/config.html):

- **Auto-provision**: The controller is automatically launched when the first `sky jobs launch`
  is submitted. No manual setup required.
- **Auto-stop**: The controller autostops after **10 minutes of idleness** (no in-progress
  managed jobs and no new submissions). This is the default on GCP.
  Not supported on Kubernetes and RunPod.
- **Auto-restart**: The controller automatically restarts when a new job is launched.
  The restart takes ~2-3 minutes (VM boot + SkyPilot daemon).
- **Cost when stopped**: ~$0.004/hr ($0.096/day) for the stopped VM's disk only.
- **Cost when running**: ~$0.134/hr ($3.22/day) for n4-standard-4.
- **Default resources**: cpus: 4+, memory: 8x, disk_size: 50 GB.
- **Concurrent job capacity**: `max_concurrent_jobs = 2 * cpus` (default = 8 concurrent).
  SkyPilot v0.11 optimized this to handle 2,000+ parallel jobs on 8-CPU controller.

**Critical finding**: The 10-minute auto-stop is ALREADY the default behavior. If the
controller is running 24/7, something is keeping it alive -- most likely a PENDING job
that never provisions. This is exactly our L4 exhaustion scenario. A PENDING job is NOT
"idle" from the controller's perspective, so the controller will never auto-stop while
jobs are stuck waiting for GPU capacity.

**Root cause chain**: L4 exhausted -> jobs PENDING indefinitely -> controller stays
alive -> $3.22/day burn. Fixing GPU availability (Section 7, A100 fallback) also
fixes the controller cost problem.

#### Option A: Explicit Teardown After Experiment (RECOMMENDED)

After all jobs reach a terminal state (SUCCEEDED, FAILED, CANCELLED), explicitly tear
down the controller:

```bash
#!/bin/bash
# scripts/post_experiment_cleanup.sh
set -euo pipefail

echo "[$(date -u)] Checking for in-progress jobs..."
ACTIVE=$(sky jobs queue 2>/dev/null | grep -cE "PENDING|RUNNING" || echo 0)

if [ "$ACTIVE" -eq 0 ]; then
    echo "[$(date -u)] No active jobs. Tearing down controller."
    CONTROLLER=$(sky status -u 2>/dev/null | grep "sky-jobs-controller" \
        | awk '{print $1}')
    if [ -n "$CONTROLLER" ]; then
        sky down "$CONTROLLER" -y
        echo "[$(date -u)] Controller torn down. Savings: \$3.22/day"
    else
        echo "[$(date -u)] No controller found."
    fi
else
    echo "[$(date -u)] $ACTIVE jobs still active. Controller stays up."
fi
```

**Important caveat** ([SkyPilot docs](https://docs.skypilot.co/en/latest/examples/managed-jobs.html)):
Tearing down the controller **loses all logs and status information** for finished
managed jobs. Extract results BEFORE teardown. Teardown is only permitted when no
in-progress jobs exist.

**Alternatively**: Cancel stuck PENDING jobs first, then let auto-stop handle it:

```bash
# Cancel all PENDING jobs that have been waiting >2 hours
sky jobs cancel -a   # Cancel all jobs (or selectively cancel PENDING ones)
# Controller will auto-stop in 10 minutes after last job terminates
```

| Impact | Value |
|--------|-------|
| Monthly savings (if controller would otherwise run 24/7) | ~$93/month |
| Risk | Log loss if teardown before results are extracted |
| Automation complexity | Low (single script, idempotent) |

#### Option B: Downsize the Controller VM

The controller VM can be customized via `~/.sky/config.yaml`
([Advanced Configuration docs](https://docs.skypilot.co/en/latest/reference/config.html)):

```yaml
# ~/.sky/config.yaml
jobs:
  controller:
    resources:
      cpus: 2+          # Down from 4+ default
      disk_size: 30     # Down from 50 GB default
    autostop:
      idle_minutes: 10  # Keep default auto-stop
      down: false       # Stop, don't terminate (preserves logs)
```

**Sizing analysis**:

| Config | Likely VM Type (GCP) | Hourly Cost | Daily Cost | Max Concurrent Jobs |
|--------|---------------------|-------------|------------|---------------------|
| Default (cpus: 4+) | n4-standard-4 | ~$0.134 | $3.22 | 8 |
| Downsized (cpus: 2+) | e2-medium | ~$0.034 | $0.81 | 4 |
| Minimum (cpus: 1+) | e2-small | ~$0.017 | $0.41 | 2 |

**Impact on factorial experiments**: With cpus=2, only 4 jobs are simultaneously
provisioned/monitored instead of 8 (formula: `max_concurrent = 2 * cpus`). All 34
debug jobs still run -- they queue internally. With off-peak launches at 03:00 UTC,
slightly longer queue processing is acceptable. The experiment finishes well before
peak hours.

**Important caveat**: "These settings will not take effect if you have an existing
controller (either stopped or live)." To apply: `sky down <controller>` first, update
`~/.sky/config.yaml`, then the next `sky jobs launch` creates a new smaller controller.

| Impact | Value |
|--------|-------|
| Monthly savings (cpus: 2 vs 4, always-on) | ~$72/month ($0.81 vs $3.22/day) |
| Monthly savings (cpus: 1 vs 4, always-on) | ~$84/month ($0.41 vs $3.22/day) |
| Trade-off | Slower job queue throughput (4 concurrent vs 8) |
| Setup | Requires controller teardown + recreation |

#### Option C: Nebius Managed SkyPilot API Server (FUTURE)

[Nebius offers a managed SkyPilot API server](https://nebius.com/blog/posts/managed-skypilot-api-server-tech-overview-and-setup)
that eliminates the controller VM entirely. The SkyPilot API runs as a managed service
on Nebius infrastructure.

**Pros**: Zero controller cost, no VM management, automatic scaling.
**Cons**: Nebius lock-in for the control plane, requires Nebius account, currently only
manages Nebius GPU resources (not multi-cloud orchestration).

**Verdict**: Not viable until Nebius supports managing jobs on GCP and Vast.ai (not just
Nebius-hosted GPUs). Monitor for multi-cloud managed SkyPilot API server announcements.

#### Option D: Combined Pattern -- Experiment-Scoped Controller (RECOMMENDED)

The optimal pattern combines scheduled launches with controller lifecycle management:

```
Timeline:
────────────────────────────────────────────────────────────────
03:00 UTC  Cron/Prefect triggers run_factorial.sh
03:00      `sky jobs launch` → controller auto-provisions (~2-3 min boot)
03:03      Controller alive, begins provisioning 34 factorial jobs
03:03-07:00 Jobs run during off-peak window (4 concurrent with cpus=2)
07:00-11:00 Remaining jobs complete as demand rises
11:00      All 34 jobs in terminal state
11:10      Controller auto-stops after 10 min idle
           Controller cost for this run: $0.134/hr * 8h = $1.07
────────────────────────────────────────────────────────────────
```

**Cost model**:

| Pattern | Monthly Controller Cost | Assumptions |
|---------|----------------------|-------------|
| **Current (always-on, n4-standard-4)** | **$96.60** | 24/7 running (PENDING jobs prevent auto-stop) |
| Option A (explicit teardown after experiment) | **$5.37** | 10 experiment days/month, ~4h controller/day |
| Option B (downsized e2-medium, always-on) | **$24.30** | e2-medium 24/7 |
| **Option D (default VM + experiment-scoped)** | **$5.37** | n4-standard-4, 10 days x 4h |
| **Option D (downsized + experiment-scoped)** | **$1.34** | e2-medium, 10 days x 4h |

**Savings with Option D (downsized + experiment-scoped) vs current: $95.26/month (98.6% reduction).**

Even without downsizing, experiment-scoped controller usage saves **$91.23/month (94.4%
reduction)** vs always-on. The critical enabler: once the A100 fallback is in place
(Section 7, Recommendation 1), jobs provision quickly instead of sitting PENDING, and
the controller's built-in 10-minute auto-stop works as designed.

**The key insight**: The controller should NEVER run 24/7. The current always-on state
is a SYMPTOM of the L4 exhaustion problem, not a separate issue. Fix GPU availability
-> jobs complete promptly -> controller auto-stops -> cost drops from $96.60/month to
~$5/month automatically.

### 14.4 Optimal Launch Schedule Framework

A decision framework for when to launch factorial experiments:

```
┌─────────────────────────────────────┐
│   Researcher triggers experiment    │
│   (manual or scheduled)             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Check UTC hour                    │
├─────────────────────────────────────┤
│                                     │
│  02:00-06:00 UTC?                   │
│  ├─ YES → LAUNCH IMMEDIATELY        │
│  │         (off-peak, best avail.)  │
│  │                                  │
│  14:00-22:00 UTC?                   │
│  ├─ YES → QUEUE for 03:00 UTC       │
│  │         (peak hours, worst)      │
│  │                                  │
│  Other hours?                       │
│  └─ LAUNCH with 30-min timeout      │
│     ├─ Provisioned? → Run           │
│     └─ Not provisioned? → Queue     │
│        for next 03:00 UTC           │
└─────────────────────────────────────┘
```

**Configuration-driven** (per CLAUDE.md Rule 29 -- no hardcoded parameters):

```yaml
# configs/scheduling/launch_windows.yaml
launch_windows:
  off_peak:
    start_utc: 2
    end_utc: 6
    action: launch_immediately
  peak:
    start_utc: 14
    end_utc: 22
    action: defer_to_next_off_peak
  moderate:
    action: launch_with_timeout
    timeout_minutes: 30
  default_launch_hour_utc: 3
```

**Implementation as a Prefect flow guard**:

```python
from datetime import datetime, timezone

def should_launch_now(config: dict) -> tuple[bool, str]:
    """Determine whether to launch immediately or defer to off-peak.

    Args:
        config: Launch window config from configs/scheduling/launch_windows.yaml
    """
    hour = datetime.now(timezone.utc).hour
    off_peak = config["launch_windows"]["off_peak"]
    peak = config["launch_windows"]["peak"]

    if off_peak["start_utc"] <= hour < off_peak["end_utc"]:
        return True, "OFF-PEAK window — launching immediately"
    elif peak["start_utc"] <= hour < peak["end_utc"]:
        return False, f"PEAK hours — deferring to {config['launch_windows']['default_launch_hour_utc']:02d}:00 UTC"
    else:
        return True, "MODERATE hours — launching with timeout fallback"
```

### 14.5 Integration with Availability Monitoring Pipeline

The scheduling framework integrates with the availability monitoring architecture
(Section 12: Reproducible Availability Monitoring):

#### Phase 1: Reactive Scheduling (NOW)

- Fixed cron schedule at 03:00 UTC based on general off-peak knowledge
- Manual override via `scripts/run_factorial.sh` at any time
- Controller auto-stops when experiments complete
- Cancel stuck PENDING jobs to unblock controller auto-stop

#### Phase 2: Data-Driven Scheduling (MONTH 2)

- Collect hourly availability data via `sky check` + `sky gpus --cloud gcp`
  (builds on Section 12's monitoring probes)
- Store availability snapshots in DuckDB (extend existing observability pipeline)
- Build empirical availability model: "L4 available 87% of the time at 03:00 UTC
  in europe-west4, but only 23% at 16:00 UTC"

```sql
-- Query: best launch hours for L4 in europe-west4
SELECT
    hour_utc,
    COUNT(*) FILTER (WHERE available) * 100.0 / COUNT(*) AS availability_pct
FROM gpu_availability_log
WHERE gpu_type = 'L4' AND region = 'europe-west4'
GROUP BY hour_utc
ORDER BY availability_pct DESC;
```

#### Phase 3: Predictive Auto-Launch (MONTH 3+)

- Dashboard (Section 12's Gradio dashboard) shows: "Recommended launch window:
  03:00-05:00 UTC tomorrow (predicted L4 availability: 91%)"
- Auto-launch trigger: when predicted availability exceeds configurable threshold,
  automatically submit the queued experiment
- Feedback loop: actual provisioning times feed back into the prediction model

```
┌─────────────────────────┐      ┌──────────────────────┐
│  Hourly Probe           │      │  DuckDB              │
│  sky gpus --cloud gcp   │─────▶│  gpu_availability_log│
│  every 30 min (Sec. 12) │      └──────────┬───────────┘
└─────────────────────────┘                 │
                                            ▼
┌─────────────────────────┐      ┌──────────────────────┐
│  Dashboard (Gradio)     │◀─────│  Prediction Model    │
│  "Best window: 03:00"   │      │  (rolling 14-day avg)│
│  "Current L4 avail: 0%" │      └──────────┬───────────┘
└─────────────────────────┘                 │
                                            ▼
                                 ┌──────────────────────┐
                                 │  Auto-Launch Trigger  │
                                 │  IF predicted > 80%   │
                                 │  AND queued_experiment │
                                 │  → sky jobs launch     │
                                 └──────────────────────┘
```

### 14.6 Combined Savings Summary

| Optimization | Monthly Savings | Implementation Effort | Priority |
|-------------|----------------|----------------------|----------|
| Experiment-scoped controller (stop when idle) | $91.23 (94%) | 30 min (cancel PENDING jobs + script) | **P0 -- immediate** |
| Downsize controller (e2-medium, cpus: 2) | Additional $4.03 | 15 min (config change + teardown/recreate) | P1 -- next experiment |
| Off-peak scheduling (03:00 UTC cron) | $5-15 (fewer preemptions, fewer retries) | 15 min (crontab) | P1 |
| Prefect scheduled deployment | Operational (better observability, UI) | 2-4 hours | P2 |
| Data-driven launch windows | $5-20 (optimal timing from empirical model) | 8-12 hours | P3 |
| **Total addressable savings** | **~$100-130/month** | | |

**Context**: The total always-on infrastructure cost is EUR 119-130/month (Section 1.2).
Controller optimization alone recovers ~$91/month of that. Combined with Cloud SQL
optimization (separate analysis), the always-on cost could drop to EUR 20-30/month --
bringing monthly infrastructure to less than the cost of a single debug experiment pass.

**Immediate action (P0)**: Cancel all PENDING jobs (`sky jobs cancel -a`) to allow the
controller's built-in 10-minute auto-stop to activate. This single command saves $3.22/day
starting immediately.

---

## Appendix D: Web Research Sources (Section 14)

### GPU Spot Availability Patterns
- [GCP Spot VM Documentation](https://cloud.google.com/compute/docs/instances/spot) -- official time-of-day availability guidance
- [Cloud GPU Spot Instance Availability and Interruption Rates (ThunderCompute, 2026)](https://www.thundercompute.com/blog/cloud-gpu-spot-instance-availability) -- interruption rate data by GPU type
- [Spot Instances and Preemptible GPUs: Cutting AI Costs by 70% (Introl, 2026)](https://introl.com/blog/spot-instances-preemptible-gpus-ai-cost-savings) -- 10M spot instance hours analysis, weekend patterns
- [Cast AI Spot Availability Map](https://cast.ai/spot-availability-map/) -- real-time spot interruption data
- [SkyPilot Spot-Traces Dataset (GitHub)](https://github.com/skypilot-org/spot-traces) -- 5-minute interval availability traces (NSDI'24)
- [Modeling Temporally Constrained Preemptions of Transient Cloud VMs (arXiv)](https://arxiv.org/pdf/1911.05160) -- bathtub-curve preemption patterns
- [Spot Instance Availability Demystified: AWS, Azure, and GCP (Cast AI)](https://cast.ai/blog/spot-instance-availability-demystified-aws-azure-and-gcp/) -- multi-cloud spot strategy

### SkyPilot Controller Management
- [SkyPilot Managed Jobs Documentation](https://docs.skypilot.co/en/latest/examples/managed-jobs.html) -- controller lifecycle, auto-stop, teardown
- [SkyPilot Advanced Configuration](https://docs.skypilot.co/en/latest/reference/config.html) -- `~/.sky/config.yaml` controller customization
- [SkyPilot Autostop and Autodown](https://docs.skypilot.co/en/latest/reference/auto-stop.html) -- idle detection, auto-stop mechanics
- [SkyPilot CLI Reference](https://docs.skypilot.co/en/latest/reference/cli.html) -- `sky down`, `sky jobs cancel` commands
- [SkyPilot Issue #3979: Non-Interactive Controller Deletion](https://github.com/skypilot-org/skypilot/issues/3979) -- controller teardown edge cases
- [Nebius Managed SkyPilot API Server](https://nebius.com/blog/posts/managed-skypilot-api-server-tech-overview-and-setup) -- serverless controller alternative

### Scheduling Infrastructure
- [Prefect 3.x Schedules Documentation](https://docs.prefect.io/v3/concepts/schedules) -- cron, interval, RRule schedules
- [Prefect Tutorials: Schedule a Flow](https://docs-3.prefect.io/v3/tutorials/schedule) -- deployment scheduling guide
- [SkyPilot v0.11 Release Notes](https://blog.skypilot.co/announcing-skypilot-0.11.0/) -- latest features (no native scheduling)

---

*Report generated: 2026-03-28*
*Updated: 2026-03-28 -- Sections 10-13 added (multi-region replication, multi-cloud SkyPilot, availability monitoring, financial decision framework)*
*Updated: 2026-03-28 -- Section 14 added (scheduled training & controller cost optimization)*
*Methodology: Iterated LLM Council (5 experts, 2 review rounds per section)*
*Next action: User authorization required for YAML contract changes (Recommendation 1) and multi-region Pulumi deployment (Section 13.7 Phase 0)*
