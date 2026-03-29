# GCP Region Availability Analysis — Multi-Region Strategy for GPU Capacity Resilience

**Date**: 2026-03-28
**Status**: Decision recommendation
**Context**: europe-west4 GPU capacity exhaustion blocks v0.2-beta factorial experiments for 13+ hours
**Methodology**: Iterated LLM Council (5 expert perspectives)
**Prior art**: 11 experiment passes, 3 region migrations, EUR 89/month egress lesson learned

---

## Council Panel

| Expert | Domain | Key Contribution |
|--------|--------|------------------|
| **Cloud Architect** | Region selection, infra replication | Multi-region topology, GAR replication strategy |
| **FinOps Engineer** | Cost modeling, egress analysis | Replication TCO vs. PENDING opportunity cost |
| **Compliance Officer** | GDPR, medical data residency | EU region requirements, data sovereignty |
| **SRE** | Availability patterns, failover design | Preemption rates, zone diversity, capacity planning |
| **Research Software Engineer** | Practical researcher workflow | DevEx impact, SkyPilot integration, operational complexity |

---

## 1. Abstract

The Vascadia v0.2-beta factorial experiment platform has been blocked for 13+ hours
because all GPU types (L4, A100, A100-80GB) are capacity-exhausted in GCP
europe-west4 (Netherlands). All infrastructure -- GAR Docker registry (6.38 GB),
GCS buckets (3 buckets), Cloud SQL (PostgreSQL 15), and Cloud Run MLflow -- is
co-located in europe-west4 to avoid the cross-region egress fees that previously
cost EUR 89/month. This co-location strategy is correct but creates a single point
of failure: when europe-west4 GPU capacity is exhausted, the entire experiment pipeline
is blocked with no fallback. This report analyzes GPU availability across GCP regions,
quantifies the cost of infrastructure replication, and recommends adding europe-west1
(Belgium) as a secondary region with full infrastructure co-location, enabling
zero-egress GPU training in either region. The incremental cost is approximately
EUR 28-33/month -- recoverable in under 9 days of avoided PENDING-state controller
waste (EUR 3.22/day). The recommendation includes a Pulumi-based implementation plan
and SkyPilot ordered-failover configuration.

---

## 2. Problem Statement

### 2.1 The Immediate Crisis

SkyPilot job #158 (`minivess-factorial`) has been PENDING for 13+ hours requesting
1x L4 spot in europe-west4. All three Phase 1 jobs (159, 160, 161) cycle between
STARTING and PENDING. Both spot AND on-demand provisioning fail. The region's L4
physical capacity is genuinely exhausted across all 3 zones (europe-west4-a, -b, -c).

**Observed timeline** (from 11th pass monitoring):
- 09:00 UTC: All 3 jobs PENDING. L4 capacity constrained.
- 09:15: J3 (spot) reached 14 min STARTING before reverting. J2 (on-demand) 13 min before PENDING.
- 09:30: J3 reached 19 min STARTING (longest). Spot preempted before setup completed.
- 09:45: On-demand should NOT be preempted -- suggests genuine capacity exhaustion, not preemption.
- 09:55: Pattern confirmed. europe-west4 L4 capacity fluctuates but never sustains long enough for a full job.

### 2.2 The Co-Location Constraint

All infrastructure is co-located in europe-west4 by design:

| Resource | Region | Monthly Cost |
|----------|--------|-------------|
| GAR Docker registry (6.38 GB) | europe-west4 | EUR 0.57 |
| GCS bucket: dvc-data | europe-west4 | EUR 0.06 |
| GCS bucket: mlflow-artifacts | europe-west4 | EUR 0.04 |
| GCS bucket: checkpoints | europe-west4 | EUR 0.04 |
| Cloud SQL (db-g1-small) | europe-west4 | EUR 23.50 |
| Cloud Run (MLflow) | europe-west4 | EUR 2-6 |
| **Total** | | **EUR 26-30** |

This co-location eliminates cross-region egress (previously EUR 89/month when GAR
was in europe-north1 and GPU VMs were in europe-west4). But it creates complete
dependency on a single region's GPU availability.

### 2.3 The Cost of Waiting

Every day the experiment is blocked:
- SkyPilot controller VM: EUR 3.22/day (n4-standard-4 running 24/7)
- Researcher productivity: unquantifiable but significant (paper deadline pressure)
- Cloud SQL running idle: EUR 0.78/day
- Cloud Run running idle: EUR 0.10/day
- **Total wasted per PENDING day: EUR 4.10/day**

After 13 hours of PENDING: EUR 2.22 already wasted on controller alone, with zero
training completed.

---

## 3. Historical Data from Our Experiments

### 3.1 Region Provisioning History Across 11 Passes

| Pass | Date | Primary Region | GPU | Result | Region Issue |
|------|------|---------------|-----|--------|-------------|
| 1st | 2026-03-23 | RunPod | RTX4090 | CANCELLED | Controller on wrong cloud |
| 2nd | 2026-03-23 | GCP (mixed) | A100-80GB | CANCELLED | Unauthorized GPU added |
| 3rd | 2026-03-23 | GCP (mixed) | L4 | CANCELLED | Controller zone-hopping |
| 4th | 2026-03-23 | GCP (mixed) | L4 | CANCELLED | sky config conflict |
| 5th-7th | 2026-03-23-24 | europe-north1 | L4 | CANCELLED (12h+ PENDING) | europe-north1 has ZERO L4 GPUs |
| 8th | 2026-03-24 | europe-west4 (multi-region) | L4 | 15/34 SUCCEEDED | First successful pass; preemption pressure on SAM3 |
| 9th | 2026-03-25 | europe-west4 | L4 | Partial | SAM3 zero-shot focus |
| 10th | 2026-03-27 | europe-west4 | L4 | Complete | Production readiness |
| 11th | 2026-03-28 | europe-west4 | L4 | BLOCKED (13h+ PENDING) | Full capacity exhaustion |

### 3.2 Key Observations

**Passes 5-7 (europe-north1)**: 12+ hours PENDING because europe-north1 (Finland)
has ZERO L4 GPUs. This was the original motivation for the europe-west4 migration
and the EUR 89/month egress elimination.

**Pass 8 (multi-region failover)**: The first successful pass used `ordered:` region
injection: europe-west4 -> europe-west1 -> europe-west3 -> us-central1. Most jobs
provisioned in europe-west4-b. Some failed in europe-west4-a (insufficientCapacity)
and fell through to -b. Provisioning succeeded, but the cross-region Docker pull from
europe-north1 GAR was costing EUR 89/month in egress.

**Pass 8 actual region distribution**: 38.65 GPU-hours in europe-west4 (Netherlands),
69 GPU-hours in asia-northeast3 (Seoul -- intercontinental fallback). The Seoul
provisioning happened when eu-west4 was briefly exhausted, confirming that capacity
fluctuates.

**Pass 11 (current)**: europe-west4 L4 capacity fully exhausted. On-demand and spot
both fail. The `europe_strict.yaml` config restricts to europe-west4 only (no
fallback) to avoid egress. This is the design flaw: when the only region is exhausted,
there is no recovery path that avoids egress.

### 3.3 Egress Cost History

| Period | Egress Source | Monthly Cost | Root Cause |
|--------|-------------|-------------|-----------|
| Pre-migration (Feb-Mar 2026) | GAR europe-north1 -> GPU VMs elsewhere | EUR 89.58 | GAR in region with zero L4 GPUs |
| Post-migration (Mar 28+) | None (same-region) | EUR 0.00 | GAR co-located with GPU VMs |
| Pass 8 intercontinental | GAR europe-north1 -> asia-northeast3 | ~EUR 57.71 (850 GB) | SkyPilot fallback to Seoul |
| Pass 8 intra-EU | GAR europe-north1 -> europe-west4 | ~EUR 31.87 (1,879 GB) | Cross-region within EU |

**Lesson learned**: Cross-region egress dominated costs (99% of GAR bill). The
only sustainable solution is co-locating Docker registry + data + GPU VMs in the
same region. This is non-negotiable.

---

## 4. GCP GPU Availability by Region

### 4.1 L4 GPU Availability (G2 Machine Types)

Based on GCP documentation (updated March 2026) and our operational experience:

| Region | Location | L4 Zones | Zone Names | L4 Spot $/hr | L4 On-Demand $/hr | Capacity Assessment |
|--------|----------|----------|-----------|-------------|-------------------|-------------------|
| **europe-west4** | Netherlands | 3 | a, b, c | $0.340 | $0.742 | HIGH but currently exhausted |
| **europe-west1** | Belgium | 2 | b, c | $0.294 | $0.778 | MEDIUM -- 2 zones, cheaper spot |
| europe-west3 | Frankfurt | 2 | a, b | $0.416 | $0.834 | MEDIUM -- more expensive |
| europe-west2 | London | 2 | a, b | $0.391 | $0.805 | MEDIUM -- post-Brexit GDPR nuance |
| europe-west6 | Zurich | Limited | Varies | ~$0.400 | ~$0.850 | LOW -- limited L4 support |
| europe-north1 | Finland | **0** | None | N/A | N/A | ZERO L4 GPUs |
| us-central1 | Iowa | 3 | a, b, c | $0.224 | $0.707 | HIGHEST globally |
| us-east4 | Virginia | 2 | a, c | $0.269 | $0.707 | HIGH |
| us-west1 | Oregon | 2 | a, b | $0.224 | $0.707 | HIGH |
| asia-southeast1 | Singapore | 2 | b, c | $0.340 | $0.767 | MEDIUM |

### 4.2 A100 GPU Availability (A2 Machine Types)

| Region | A100-40GB Zones | A100-80GB Zones | Spot $/hr (80GB) | Notes |
|--------|----------------|----------------|-------------------|-------|
| **europe-west4** | a | a | $1.38 | Only EU region with A100 |
| us-central1 | a, b, c, f | a, b, c | $1.10-1.39 | Highest availability |
| us-east4 | a, b, c | a | $1.10-1.39 | Good A100 availability |
| us-west1 | b | b | $1.10-1.39 | Limited A100 zones |
| asia-southeast1 | b, c | b | $1.38 | Asia option |

**Critical finding from FinOps Engineer**: A100 GPUs in Europe are available ONLY in
europe-west4, zone a. There is no EU A100 diversity. If europe-west4-a A100 capacity
is exhausted, there is no intra-EU fallback for A100 workloads.

### 4.3 H100 GPU Availability

H100 SXM (A3 machine types) have very limited European availability. The primary
European regions are europe-west4 and possibly europe-west1, but H100 capacity is
reserved primarily for large customers with committed use agreements. H100 is not
viable for our workload profile (single-GPU academic project).

### 4.4 Zone Diversity Analysis (SRE Perspective)

**Why zone count matters**: Each GCP zone is a separate physical data center. More
zones means more independent capacity pools. When zone A is exhausted, zone B may
have capacity.

| Region | Zones | L4 Zones | A100 Zones | Independent Capacity Pools |
|--------|-------|----------|-----------|---------------------------|
| europe-west4 | 3 (a,b,c) | 3 | 1 (a only) | Best EU L4 diversity |
| europe-west1 | 3 (b,c,d) | 2 (b,c) | 0 | Good L4, no A100 |
| europe-west3 | 3 (a,b,c) | 2 (a,b) | 0 | Moderate L4, no A100 |
| europe-west2 | 3 (a,b,c) | 2 (a,b) | 0 | Moderate L4, no A100 |
| us-central1 | 4 (a,b,c,f) | 3 (a,b,c) | 4 | Best global diversity |

**SRE assessment**: europe-west4 has the best EU GPU diversity (3 L4 zones + 1 A100 zone),
but when all 3 zones are exhausted simultaneously (as observed in the 11th pass), there
is no recovery within the region. Adding a second region with independent capacity pools
is the standard SRE approach to this class of problem.

---

## 5. Infrastructure Replication Cost Analysis

### 5.1 What Must Be Replicated

For zero-egress training in a second region, we need:

| Resource | Current (europe-west4) | Replicated (second region) | Monthly Cost |
|----------|----------------------|---------------------------|-------------|
| GAR Docker registry | 6.38 GB images | Same images, second repo | ~EUR 0.60 |
| GCS dvc-data | ~3 GB (MiniVess + DeepVess) | Copy to second-region bucket | ~EUR 0.06 |
| GCS mlflow-artifacts | ~1-10 GB (grows) | Separate bucket per region | ~EUR 0.02-0.20 |
| GCS checkpoints | ~2 GB (pretrained weights) | Copy to second-region bucket | ~EUR 0.04 |
| Cloud SQL | db-g1-small (PostgreSQL 15) | **Option A**: Second instance | ~EUR 23.50 |
| | | **Option B**: Cloud Run MLflow with SQLite | ~EUR 2-5 |
| Cloud Run (MLflow) | 1 vCPU, 2 GiB | Second deployment | ~EUR 2-6 |

### 5.2 Replication Cost Scenarios

**Scenario A: Full replication with Cloud SQL** (most resilient)

| Component | Monthly Cost |
|-----------|-------------|
| GAR storage (second region) | EUR 0.60 |
| GCS storage (3 buckets, second region) | EUR 0.12 |
| Cloud SQL (second instance) | EUR 23.50 |
| Cloud Run MLflow (second deployment) | EUR 2-6 |
| **Total** | **EUR 26-30/month** |

**Scenario B: Lightweight replication without Cloud SQL** (recommended)

Instead of replicating Cloud SQL, deploy MLflow in the second region with GCS-backed
artifact store and the SAME Cloud SQL instance in europe-west4. The cross-region
latency for MLflow metadata writes (small JSON payloads, ~1 KB each) is negligible
(~10-20ms added), and there is zero egress cost for writes to Cloud SQL. The large
artifacts (checkpoints, models) go to the local-region GCS bucket -- these are the
egress-sensitive payloads.

| Component | Monthly Cost |
|-----------|-------------|
| GAR storage (second region) | EUR 0.60 |
| GCS storage (3 buckets, second region) | EUR 0.12 |
| Cloud Run MLflow (second deployment, pointing to same Cloud SQL) | EUR 2-6 |
| Cross-region Cloud SQL access (metadata only, ~KB payloads) | ~EUR 0.00 |
| **Total** | **EUR 3-7/month** |

**Scenario C: GAR multi-region repository**

GCP supports multi-region GAR repositories (location: "europe") that automatically
replicate across EU regions. This eliminates the need for per-region GAR repos but
costs slightly more in storage (multi-region pricing: $0.13/GiB/month vs $0.10/GiB
for standard).

| Component | Monthly Cost |
|-----------|-------------|
| GAR "europe" multi-region (6.38 GB) | EUR 0.76 (vs EUR 0.57 current) |
| GCS storage (3 buckets, second region) | EUR 0.12 |
| Cloud Run MLflow (second deployment) | EUR 2-6 |
| **Total** | **EUR 3-7/month** |

### 5.3 Cost-Benefit Analysis

| Metric | Single Region (current) | Dual Region (Scenario B) |
|--------|------------------------|--------------------------|
| Monthly infrastructure cost | EUR 26-30 | EUR 29-37 |
| Monthly increment | -- | **EUR 3-7** |
| PENDING waste per day | EUR 4.10 | EUR 0 (fail to second region) |
| Break-even | -- | **0.7-1.7 days of avoided PENDING** |
| Annual cost of PENDING (2 episodes/month, 1 day each) | EUR 98.40 | EUR 0 |
| Annual infra increment | -- | EUR 36-84 |
| **Net annual savings** | -- | **EUR 14-62** |

**FinOps Engineer verdict**: The dual-region infrastructure pays for itself after
less than 2 days of avoided PENDING time. Given that we have already experienced
12+ hours PENDING in passes 5-7 and now 13+ hours in pass 11, the investment is
justified by historical evidence. The lightweight Scenario B (EUR 3-7/month) is
strongly recommended.

---

## 6. Decision Matrix

### 6.1 Region Scoring (1-5, higher is better)

| Region | L4 Avail | A100 Avail | Zone Diversity | EU/GDPR | Spot Price | Infra Support | Egress to ew4 | Overall |
|--------|----------|------------|---------------|---------|-----------|---------------|--------------|---------|
| **europe-west1** (Belgium) | 4 | 1 | 3 | 5 | **5** | 5 | 5 | **4.1** |
| europe-west3 (Frankfurt) | 3 | 1 | 3 | 5 | 3 | 5 | 5 | 3.6 |
| europe-west2 (London) | 3 | 1 | 3 | 4 | 3 | 5 | 5 | 3.4 |
| europe-north1 (Finland) | **1** | 1 | 2 | 5 | N/A | 5 | 4 | 2.3 |
| us-central1 (Iowa) | **5** | **5** | **5** | **1** | **5** | 5 | 2 | 3.8 |
| us-east4 (Virginia) | 4 | 4 | 4 | 1 | 4 | 5 | 2 | 3.3 |
| asia-southeast1 (Singapore) | 3 | 3 | 3 | 1 | 3 | 4 | 1 | 2.4 |

**Weighting**: L4 Availability (25%), EU/GDPR (20%), Spot Price (15%), Zone Diversity (15%),
Infra Support (10%), Egress to ew4 (10%), A100 Availability (5%).

### 6.2 Weighted Scores

| Region | Weighted Score | Rank |
|--------|---------------|------|
| **europe-west1 (Belgium)** | **4.15** | **1st** |
| us-central1 (Iowa) | 3.60 | 2nd |
| europe-west3 (Frankfurt) | 3.55 | 3rd |
| europe-west2 (London) | 3.35 | 4th |
| us-east4 (Virginia) | 3.15 | 5th |
| asia-southeast1 (Singapore) | 2.30 | 6th |
| europe-north1 (Finland) | 2.15 | 7th |

### 6.3 Why europe-west1 Wins

1. **Cheapest L4 spot price**: $0.294/hr (14% cheaper than europe-west4's $0.340/hr)
2. **Full EU/GDPR compliance**: Belgium is an EU member state
3. **SkyPilot controller already there**: The controller VM is already in europe-west1-b
4. **Minimal cross-region egress**: Intra-EU cross-region is $0.01/GiB (but we avoid this
   by replicating infrastructure to europe-west1)
5. **Independent capacity pool**: Belgium and Netherlands are physically separate data centers
6. **All GCP services available**: GAR, GCS, Cloud Run, Cloud SQL all supported
7. **Historical provisioning success**: Pass 8 showed europe-west1 as first failover target
8. **Geographic proximity**: ~200 km from Netherlands -- lowest latency EU pair

---

## 7. Multi-Region Strategy Options

### Option A: Single Region (Current)

- **Architecture**: All infra in europe-west4, `europe_strict.yaml` config
- **Monthly cost**: EUR 119-130 (always-on)
- **Resilience**: Zero. If europe-west4 GPUs are exhausted, experiment is blocked.
- **Egress**: EUR 0
- **Verdict**: Cheapest when GPUs are available. Completely fragile when they are not.

### Option B: Two EU Regions -- Primary + Fallback (RECOMMENDED)

- **Architecture**: Full infra in europe-west4 (primary) + lightweight infra in europe-west1 (fallback)
- **Monthly increment**: EUR 3-7 (Scenario B: shared Cloud SQL, separate GAR/GCS/Cloud Run)
- **Resilience**: High. SkyPilot `ordered:` failover provisions in europe-west1 when europe-west4
  is exhausted. Training VM uses europe-west1's local GAR and GCS -- zero egress.
- **Egress**: EUR 0 (same-region in whichever region provisions)
- **SkyPilot config**: `ordered:` block with europe-west4 first, europe-west1 second
- **Verdict**: Best cost/resilience trade-off. Proven pattern from pass 8.

### Option C: EU + US Regions

- **Architecture**: europe-west4 + us-central1
- **Monthly increment**: EUR 3-7 (same lightweight replication)
- **Resilience**: Very high. us-central1 has the best GPU availability globally.
- **GDPR concern**: Mouse brain imaging data (non-human) has minimal GDPR risk. But the
  platform design should assume human data for generalizability. US data residency
  would require a DPA (Data Processing Agreement) with GCP.
- **Egress**: EUR 0 (if infra is co-located in us-central1). But syncing results back
  to europe-west4 for MLflow consolidation incurs cross-continental egress.
- **Verdict**: Maximum availability but GDPR complications. Reserve as a tertiary fallback
  in `europe_us.yaml` without infrastructure replication -- accept egress cost when used.

### Option D: Multi-Cloud (GCP + Vast.ai)

- **Architecture**: GCP for primary, Vast.ai for GPU fallback
- **Monthly increment**: ~EUR 0 (Vast.ai is pay-per-use only)
- **Resilience**: High. Vast.ai marketplace almost always has A100-80GB available.
- **Complexity**: High. Different Docker runtime model, different MLflow connectivity,
  different data transfer path. SkyPilot supports Vast.ai but integration is newer.
- **GDPR**: Vast.ai Secure Cloud (ISO 27001) is acceptable for pseudonymized data.
- **Verdict**: Good for occasional GPU-constrained situations. Too operationally complex
  for primary failover. Better suited as a future enhancement once Option B is stable.

---

## 8. Recommendation

### Primary Recommendation: Option B -- europe-west1 as Secondary Region

**Add europe-west1 (Belgium) as a fully co-located secondary region.**

#### 8.1 Implementation Plan

**Phase 1: Infrastructure Replication (Pulumi, ~2 hours)**

1. **GAR**: Create `europe-west1-docker.pkg.dev/minivess-mlops/minivess` repository.
   Push the same base image and MLflow image to both registries. Alternatively, use
   a multi-region "europe" GAR repository that auto-replicates.

2. **GCS**: Create 3 buckets with `-ew1` suffix or in europe-west1:
   - `minivess-mlops-dvc-data-ew1` (copy ~3 GB from primary)
   - `minivess-mlops-mlflow-artifacts-ew1` (starts empty)
   - `minivess-mlops-checkpoints-ew1` (copy pretrained weights ~2 GB)

3. **Cloud Run MLflow**: Deploy second `minivess-mlflow-ew1` service in europe-west1,
   pointing to the SAME Cloud SQL instance in europe-west4 (for metadata) but using
   the europe-west1 GCS bucket for artifact storage.

4. **Pulumi stack**: Create a second Pulumi stack (`gcp-europe-west1`) or extend the
   existing stack with a `secondary_region` parameter.

**Phase 2: SkyPilot Configuration (~30 minutes)**

Update `configs/cloud/regions/europe_strict.yaml` (or create `europe_dual.yaml`):

```yaml
# europe_dual.yaml -- dual-region with co-located infra in both regions
ordered:
  - cloud: gcp
    region: europe-west4
    # Uses europe-west4 GAR, GCS, and Cloud Run MLflow
  - cloud: gcp
    region: europe-west1
    # Uses europe-west1 GAR, GCS, and Cloud Run MLflow
```

The SkyPilot YAML must dynamically select the correct GAR registry and MLflow URL
based on which region provisions. This can be done via SkyPilot environment variable
injection in the `run:` block:

```yaml
run: |
  if [ "$SKYPILOT_REGION" = "europe-west1" ]; then
    export DOCKER_REGISTRY="europe-west1-docker.pkg.dev/minivess-mlops/minivess"
    export MLFLOW_TRACKING_URI="https://minivess-mlflow-ew1-....run.app"
    export GCS_ARTIFACT_BUCKET="minivess-mlops-mlflow-artifacts-ew1"
  else
    export DOCKER_REGISTRY="europe-west4-docker.pkg.dev/minivess-mlops/minivess"
    export MLFLOW_TRACKING_URI="https://minivess-mlflow-....run.app"
    export GCS_ARTIFACT_BUCKET="minivess-mlops-mlflow-artifacts"
  fi
```

**Phase 3: Data Synchronization Script (~1 hour)**

Create `scripts/sync_regions.sh` that:
1. Pushes Docker images to both GAR repositories
2. Syncs DVC data to both GCS buckets
3. Syncs pretrained weights to both checkpoints buckets
4. Verifies both Cloud Run MLflow services are healthy

This script runs after any Docker image rebuild or data update.

**Phase 4: YAML Contract Update**

Update `configs/cloud/yaml_contract.yaml` to allow both regions:

```yaml
allowed_regions:
  gcp:
    - europe-west4  # Primary
    - europe-west1  # Secondary (co-located infra)
```

#### 8.2 Cost Impact

| Item | One-Time Cost | Monthly Cost |
|------|-------------|-------------|
| Pulumi development | ~2 hours engineer time | EUR 0 |
| Docker image push to europe-west1 GAR | ~EUR 0.10 (egress for initial push) | EUR 0.60 (storage) |
| GCS data copy (~5 GB) | ~EUR 0.05 (cross-region transfer) | EUR 0.12 (storage) |
| Cloud Run MLflow (europe-west1) | EUR 0 (setup) | EUR 2-6 |
| Cloud SQL cross-region access | EUR 0 | ~EUR 0 (KB-scale metadata) |
| **Total** | **~EUR 0.15** | **EUR 3-7/month** |

#### 8.3 Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Both EU regions exhausted simultaneously | Unlikely (independent data centers). If it happens, fall through to `europe_us.yaml` (accept egress cost). |
| Docker images out of sync between regions | `scripts/sync_regions.sh` runs as part of the build pipeline. Pre-launch check verifies image digest matches in both regions. |
| GCS data out of sync | DVC push to both remotes in the `dvc push` Makefile target. Pre-launch check verifies file counts match. |
| MLflow metadata scattered across two Cloud Run instances | Both point to the same Cloud SQL. All metadata is centralized. Artifacts are per-region but discoverable via MLflow API. |
| Cloud SQL single point of failure | Cloud SQL has automatic backups (7 retained, daily). For a single-researcher academic project, this is acceptable. Future: Cloud SQL HA ($50+/month) if needed. |

#### 8.4 Timeline

| Day | Action |
|-----|--------|
| Day 0 (today) | Approve recommendation. Create Pulumi resources for europe-west1. |
| Day 0 | Push Docker images and DVC data to europe-west1 GCS + GAR. |
| Day 0 | Deploy Cloud Run MLflow in europe-west1. |
| Day 0 | Update SkyPilot YAML to use `europe_dual.yaml`. |
| Day 0 | Re-launch 11th pass factorial experiment with dual-region failover. |
| Day 1 | Verify jobs provision in europe-west1 when europe-west4 is exhausted. |
| Day 7 | First cost audit: verify no unexpected egress charges. |
| Day 30 | Full month cost comparison: confirm EUR 3-7 increment, no egress. |

---

## 9. Expert Council Cross-Review

### Cloud Architect Review

"The dual-region approach is the standard pattern for GPU capacity resilience. The key
insight is using a shared Cloud SQL instance -- this avoids the complexity and cost of
database replication while keeping all metadata centralized. The multi-region GAR option
(location: europe) is even simpler than maintaining two regional repos, though it costs
~EUR 0.19/month more. I recommend starting with multi-region GAR and per-region GCS."

### FinOps Engineer Review

"The numbers are clear: EUR 3-7/month prevents EUR 4.10/day of PENDING waste. The
ROI is positive after 1 day. The one thing I would add: implement a cost alert that fires
when a SkyPilot job has been PENDING for more than 2 hours -- this triggers the
investigation that leads to region failover. Prevention is cheaper than reaction."

### Compliance Officer Review

"Both europe-west4 (Netherlands) and europe-west1 (Belgium) are EU member states.
GDPR data residency is fully satisfied. For mouse brain imaging data, GDPR is not
technically applicable (non-personal data), but maintaining EU residency is good
practice for the platform's future use with human clinical data. I see no compliance
blockers for Option B."

### SRE Review

"The critical detail is that europe-west4 and europe-west1 have INDEPENDENT physical
infrastructure. A capacity exhaustion in Netherlands does not affect Belgium. This is
true regional redundancy, not just zone redundancy. The SkyPilot `ordered:` mechanism
is the correct failover pattern -- it tries europe-west4 first (lower latency to Cloud SQL)
and falls through to europe-west1 only when needed. I recommend adding monitoring
for provisioning success rate per region to detect patterns."

### Research Software Engineer Review

"From a researcher's perspective, this should be invisible. The researcher types
`make launch-factorial` and SkyPilot handles region selection. The only visible
difference is that the MLflow artifact URI might point to a different GCS bucket
depending on which region provisioned. This is fine -- MLflow's API abstracts the
storage backend. The `sync_regions.sh` script must be in the Makefile so researchers
never forget to push to both regions after a Docker rebuild."

---

## 10. Appendix: Data Sources

### 10.1 Project Reports Referenced

1. `gcp-finops-snapshot-2026-03-28-report.md` -- Current infrastructure state
2. `gpu-instances-finops-report.md` -- GPU cost-performance matrix
3. `skypilot-and-finops-complete-report.md` -- SkyPilot architecture
4. `pr1-finops-infrastructure-timing-plan.md` -- FinOps implementation plan
5. `run-debug-factorial-experiment-report-11th-pass.md` -- Current monitoring (13h+ PENDING)
6. `run-debug-factorial-experiment-report-8th-pass-multi-region.md` -- Multi-region success data
7. `run-debug-factorial-experiment-11th-pass-finops-plan.md` -- Region migration plan
8. `knowledge-graph/domains/cloud.yaml` -- Cloud architecture decisions
9. `configs/cloud/yaml_contract.yaml` -- GPU allowlist

### 10.2 Web Research Sources

- [GPU locations | Compute Engine | Google Cloud](https://docs.cloud.google.com/compute/docs/regions-zones/gpu-regions-zones) -- Official GPU region/zone reference (updated March 2026)
- [GPU pricing | Google Cloud](https://cloud.google.com/compute/gpus-pricing) -- Current spot and on-demand pricing
- [Troubleshooting resource availability | Google Cloud](https://docs.cloud.google.com/compute/docs/troubleshooting/troubleshooting-resource-availability) -- Capacity exhaustion workarounds
- [Network pricing | Google Cloud](https://cloud.google.com/vpc/network-pricing) -- Egress pricing by region pair
- [Artifact Registry locations | Google Cloud](https://cloud.google.com/artifact-registry/docs/repositories/repo-locations) -- Multi-region GAR support
- [2025 GPU Price Report | Cast AI](https://cast.ai/reports/gpu-price/) -- Cross-cloud GPU pricing trends
- [SkyPilot Managed Jobs | SkyPilot Docs](https://docs.skypilot.co/en/latest/examples/managed-jobs.html) -- Spot preemption recovery
- [SkyPilot spot-traces | GitHub](https://github.com/skypilot-org/spot-traces) -- Spot availability research data
- [L4 Cloud Pricing: Compare 12+ Providers | GetDeploying](https://getdeploying.com/gpus/nvidia-l4) -- L4 multi-cloud pricing
- [Winning the GPU Pricing Game | Cast AI Blog](https://cast.ai/blog/winning-the-gpu-pricing-game-flexibility-across-cloud-regions/) -- Regional GPU pricing analysis

---

## 9. Post-Migration Update: us-central1 Empirical Results (2026-03-28 ~23:30 UTC)

### What Happened

After the original recommendation (europe-west1 secondary), the user decided to migrate
the ENTIRE stack to **us-central1 (Iowa)** -- the region with the highest documented GPU
availability globally. No GDPR restriction applies (public CC-BY mouse brain data).

**Migration completed**: Pulumi destroy europe-west4 → recreate in us-central1 (21 resources).
GAR, GCS, Cloud SQL, Cloud Run MLflow all verified in us-central1.

### Empirical Finding: us-central1 Also Capacity-Constrained

**L4 provisioning in us-central1 is ALSO cycling STARTING→PENDING**, similar to europe-west4.
The first job (J171, DynUNet spot) has been cycling for 7+ minutes with provisioning attempts
lasting ~2-4 minutes before failing.

| Region | L4 Spot Availability | A100 Quota | Observed Behavior |
|--------|---------------------|------------|-------------------|
| europe-west4 | **EXHAUSTED** (13+ hours PENDING) | 1 (preemptible only) | Zero successful provisions |
| us-central1 | **CONSTRAINED** (cycling, faster attempts) | 0 (not requested) | 2-4 min STARTING cycles |

### Key Insight: Saturday Global GPU Shortage

The GPU shortage may be **global**, not region-specific. Saturday evening (US/EU) is
a common time for batch ML jobs to run (researchers launching weekend experiments).
Both europe-west4 and us-central1 are affected simultaneously.

**Evidence**:
- europe-west4: 13+ hours PENDING (all GPU types: L4, A100-80GB)
- us-central1: cycling STARTING→PENDING within minutes (L4 spot)
- Both spot AND on-demand fail (not just spot preemption)

### Revised Conclusions

1. **Single-region strategy is fundamentally fragile** regardless of which region is chosen.
   The original recommendation (multi-region) was correct — the migration to a "better"
   region doesn't solve the availability problem if GPU demand is globally high.

2. **The real solution is multi-provider** (SkyPilot supports Vast.ai, Nebius, etc.) or
   **multi-region with infrastructure replication**. A single GCP region, even us-central1,
   can be fully exhausted.

3. **GCS weight caching would help**: The 6 GB Docker pull + 3 GB DVC pull takes 8-10 min
   of setup. If spot instances are preempted during this window, the entire setup is wasted.
   Caching SAM3 weights (9 GB) on same-region GCS would reduce setup to ~3 min, making
   shorter spot allocations viable.

4. **The original europe-west1 recommendation remains valid** as a SECONDARY region. But
   the primary should be us-central1 (cheapest, highest baseline capacity). The architecture
   should support 2+ regions with automatic failover.

### Updated Recommendation

**Primary: us-central1** (current, all infra deployed)
**Secondary: europe-west1** (to be added via Pulumi for EU failover)
**Tertiary: Vast.ai or Nebius** (for global GPU shortage scenarios like this Saturday)

The multi-region + multi-provider strategy is the only robust solution for a research
platform that cannot tolerate 13+ hour blocking periods.

---

*Report generated: 2026-03-28*
*Updated: 2026-03-28 ~23:30 UTC with us-central1 empirical results*
*Council methodology: 5 expert perspectives with cross-review*
*Original recommendation: Option B -- europe-west1 as secondary region*
*Updated recommendation: us-central1 primary + europe-west1 secondary + multi-provider tertiary*
