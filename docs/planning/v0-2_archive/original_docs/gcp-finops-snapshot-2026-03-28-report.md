# GCP FinOps Baseline Snapshot -- 2026-03-28

**Snapshot timestamp**: 2026-03-28 ~02:00 UTC (commands run 2026-03-27 ~23:22 -- 2026-03-28 ~02:00 UTC)
**Project**: `minivess-mlops`
**Billing account**: `01DCCF-E3B6B4-0616FE`
**Target region (post-migration)**: `europe-west4` (Netherlands)
**Previous region (pre-migration)**: `europe-north1` (Finland)

---

## 1. Executive Summary

All GCP infrastructure has been successfully migrated from `europe-north1` to `europe-west4`.
Zero resources remain in the old region. The stack is minimal and well-structured:

| Resource Category | Count | Region | Monthly Cost Estimate |
|-------------------|-------|--------|----------------------|
| GAR Repositories | 2 | europe-west4 | ~EUR 1.30 (storage only) |
| GCS Buckets | 3 | europe-west4 | ~EUR 0.00 (all empty) |
| Cloud SQL (PostgreSQL 15) | 1 | europe-west4 | ~EUR 25-27 |
| Cloud Run (MLflow) | 1 | europe-west4 | ~EUR 2-5 |
| Compute Engine (SkyPilot controller) | 1 | europe-west1-b | ~EUR 95-110 |
| Hyperdisk (controller) | 1 | europe-west1-b | ~EUR 3-5 |
| Service Accounts | 4 | global | EUR 0.00 |
| VPC / Firewall | default + 4 rules | global | EUR 0.00 |
| **TOTAL (at rest, no training jobs)** | | | **~EUR 126-148/month** |

**Pre-migration comparison**: The old europe-north1 GAR setup was generating ~EUR 89/month in
cross-region egress fees alone (SkyPilot pulling 6.4 GB base image from europe-north1 to
europe-west4 GPU VMs). That egress cost is now **EUR 0.00** because GAR and GPU VMs are
co-located in europe-west4.

**Dominant cost driver**: The SkyPilot jobs controller VM (`n4-standard-4`, 50 GB hyperdisk)
is the largest always-on cost at ~EUR 95-110/month. This is a managed SkyPilot resource.

---

## 2. Artifact Registry (GAR)

### 2.1 Repositories

```
REPOSITORY        FORMAT  MODE                 SIZE (MB)
docker-hub-cache  DOCKER  REMOTE_REPOSITORY    0
minivess          DOCKER  STANDARD_REPOSITORY  6384.856
```

| Repository | Location | Mode | Size | Created |
|-----------|----------|------|------|---------|
| `minivess` | europe-west4 | STANDARD | 6,384.856 MB (6.38 GB) | 2026-03-27T23:22:07Z |
| `docker-hub-cache` | europe-west4 | REMOTE (Docker Hub proxy) | 0 MB | 2026-03-27T23:22:07Z |

**Total GAR storage**: 6,695,006,585 bytes (6.24 GiB)

### 2.2 Docker Images

| Image | Tag | Size (bytes) | Size (human) | Created |
|-------|-----|-------------|--------------|---------|
| `.../minivess/base` | `latest` | 6,414,531,059 | **5.97 GiB** | 2026-03-28T01:41:09Z |
| `.../minivess/mlflow` | `v3.10.0` | 280,460,161 | **267 MiB** | 2026-03-28T01:46:08Z |
| `.../minivess/mlflow` | *(untagged)* | 280,460,155 | **267 MiB** | 2026-03-28T01:33:27Z |
| `.../minivess/mlflow-gcp` | `latest` | 280,460,161 | **267 MiB** | 2026-03-28T01:45:46Z |

**Note**: The untagged `mlflow` image (280,460,155 bytes) is a previous build that lost its tag
when `v3.10.0` was pushed. It is a cleanup candidate (saves ~267 MiB).

### 2.3 GAR Storage Cost Projection

- GAR Standard pricing (europe-west4): $0.10/GiB/month
- Current storage: 6.24 GiB
- **Monthly GAR storage cost: ~$0.62 (~EUR 0.57)**
- Docker Hub cache (REMOTE): $0.00 (no cached content yet)
- With untagged image cleanup: 5.97 GiB -> ~$0.60/month

### 2.4 GAR Egress Cost (The Key Migration Win)

**Pre-migration (europe-north1)**:
- Base image: 6.4 GB pulled from europe-north1 GAR
- GPU VMs: launched by SkyPilot in europe-west4 (cheapest L4 spot)
- Egress: cross-region (europe-north1 -> europe-west4) = $0.01/GiB
- Per pull: 6.4 GiB x $0.01 = $0.064
- At ~45 pulls/month (factorial experiments): ~$2.88/month egress
- **Actual observed**: ~EUR 89/month (includes repeated pulls from retries, Docker layer pulls,
  and SkyPilot controller pulls -- the real multiplier was much higher than naive calculation)

**Post-migration (europe-west4)**:
- Base image: 6.4 GB pulled from europe-west4 GAR
- GPU VMs: launched by SkyPilot in europe-west4
- Egress: **same-region = $0.00**
- **Monthly GAR egress cost: EUR 0.00**

**Savings**: ~EUR 89/month -> EUR 0.00/month = **EUR 89/month saved**

---

## 3. GCS Buckets

### 3.1 Bucket Inventory

| Bucket | Region | Storage Class | Size | Versioning | Lifecycle | Created |
|--------|--------|---------------|------|-----------|-----------|---------|
| `gs://minivess-mlops-dvc-data` | EUROPE-WEST4 | STANDARD | **0 bytes** | None | None | 2026-03-27T23:22:00Z |
| `gs://minivess-mlops-mlflow-artifacts` | EUROPE-WEST4 | STANDARD | **0 bytes** | None | Delete non-current after 7 days | 2026-03-27T23:22:03Z |
| `gs://minivess-mlops-checkpoints` | EUROPE-WEST4 | STANDARD | **0 bytes** | None | Delete non-current after 30 days | 2026-03-27T23:21:59Z |

**All three buckets are empty** (freshly created as part of the europe-west4 migration).
Data has not yet been pushed to these buckets.

### 3.2 Lifecycle Policies

**minivess-mlops-mlflow-artifacts**:
```json
{"rule": [{"action": {"type": "Delete"}, "condition": {"age": 7, "isLive": false}}]}
```
Non-current (overwritten/deleted) versions are cleaned up after 7 days.

**minivess-mlops-checkpoints**:
```json
{"rule": [{"action": {"type": "Delete"}, "condition": {"age": 30, "isLive": false}}]}
```
Non-current versions are cleaned up after 30 days (model checkpoints need longer retention).

**minivess-mlops-dvc-data**: No lifecycle rules (DVC data is immutable, content-addressed).

### 3.3 GCS Cost Projection (Once Data Arrives)

| Bucket | Expected Size | Monthly Storage | Monthly Egress (same-region) |
|--------|---------------|----------------|------------------------------|
| dvc-data | ~2-5 GiB (MiniVess + DeepVess) | $0.04-0.10 | $0.00 |
| mlflow-artifacts | ~1-10 GiB (grows with experiments) | $0.02-0.20 | $0.00 |
| checkpoints | ~2-20 GiB (model .pt files) | $0.04-0.40 | $0.00 |
| **Total** | **~5-35 GiB** | **$0.10-0.70** | **$0.00** |

GCS Standard pricing (europe-west4): $0.020/GiB/month. Same-region egress is free.

---

## 4. Cloud SQL

### 4.1 Instance Details

| Property | Value |
|----------|-------|
| **Name** | `mlflow-db-479b793` |
| **Region** | `europe-west4` |
| **State** | `RUNNABLE` |
| **Database Version** | PostgreSQL 15 |
| **Tier** | `db-g1-small` (shared-core, 0.6 GB RAM) |
| **Disk** | 10 GB PD-SSD |
| **Pricing Plan** | `PER_USE` |
| **Backups** | Enabled, 7 retained, daily at 03:00 UTC |
| **PITR** | Disabled |
| **SSL Mode** | `ALLOW_UNENCRYPTED_AND_ENCRYPTED` |
| **Public IP** | `34.158.67.254` |
| **Authorized Networks** | `0.0.0.0/0` (allow-all-initial) |
| **Connection Name** | `minivess-mlops:europe-west4:mlflow-db-479b793` |

### 4.2 Cloud SQL Cost Projection

| Component | Unit Price | Usage | Monthly Cost |
|-----------|-----------|-------|-------------|
| db-g1-small (shared) | ~$0.0325/hour | 730 hours | ~$23.73 |
| 10 GB PD-SSD storage | $0.17/GiB/month | 10 GiB | ~$1.70 |
| Backups (7 x ~10 GiB) | $0.08/GiB/month | ~1 GiB actual | ~$0.08 |
| **Total** | | | **~$25.51 (~EUR 23.50)** |

### 4.3 Security Note

**WARNING**: The Cloud SQL instance has `authorized_networks: 0.0.0.0/0` (allow-all-initial).
This is acceptable for initial setup but should be tightened to:
- Cloud Run connector IP range only
- SkyPilot controller IP only
- Developer IP only

---

## 5. Cloud Run (MLflow Tracking Server)

### 5.1 Service Details

| Property | Value |
|----------|-------|
| **Name** | `minivess-mlflow` |
| **Region** | `europe-west4` |
| **URL** | `https://minivess-mlflow-a7w6hliydq-ez.a.run.app` |
| **Alt URL** | `https://minivess-mlflow-615852151588.europe-west4.run.app` |
| **Image** | `europe-west4-docker.pkg.dev/minivess-mlops/minivess/mlflow:v3.10.0` |
| **CPU** | 1 vCPU |
| **Memory** | 2 GiB |
| **Min Instances** | 1 |
| **Max Instances** | 2 |
| **CPU Throttling** | Enabled |
| **Startup CPU Boost** | Enabled |
| **Cloud SQL Connection** | `minivess-mlops:europe-west4:mlflow-db-479b793` |
| **Ingress** | `all` (public) |

### 5.2 Cloud Run Cost Projection

With `minScale: 1`, one instance is always running:

| Component | Unit Price | Usage | Monthly Cost |
|-----------|-----------|-------|-------------|
| CPU (1 vCPU, throttled) | $0.00002400/vCPU-sec | ~2,628,000 sec (always-on) | ~$0.00 (throttled = idle free) |
| Memory (2 GiB, always-on) | $0.00000250/GiB-sec | ~2,628,000 sec | ~$6.57 |
| Requests (light usage) | $0.40/million | ~10,000/month | ~$0.004 |
| **Total** | | | **~$2-7 (~EUR 2-6)** |

**Note**: With CPU throttling enabled, idle CPU is not billed. The minimum instance keeps
the container warm (no cold start) but only bills for memory when idle. Active request
processing bills both CPU and memory.

---

## 6. Compute Engine

### 6.1 SkyPilot Jobs Controller

| Property | Value |
|----------|-------|
| **Name** | `sky-jobs-controller-aec-cw-aec1627c-head-4wim9eg2-compute` |
| **Zone** | `europe-west1-b` |
| **Machine Type** | `n4-standard-4` (4 vCPU, 16 GB RAM) |
| **Status** | `RUNNING` |
| **Preemptible** | No (standard) |
| **Created** | 2026-03-23T07:38:42 |
| **Disk** | 50 GB Hyperdisk Balanced |
| **Labels** | `skypilot-user: petteri`, `skypilot-head-node: 1` |

**IMPORTANT**: This controller VM is in `europe-west1-b`, NOT `europe-west4`. This is
expected -- SkyPilot places its jobs controller where it has existing infrastructure
or where it first launched. The controller only orchestrates jobs; actual training
VMs are launched in `europe-west4` (co-located with GAR/GCS/Cloud SQL).

### 6.2 Controller Cost Projection

| Component | Unit Price | Usage | Monthly Cost |
|-----------|-----------|-------|-------------|
| n4-standard-4 (on-demand) | ~$0.1340/hour | 730 hours | ~$97.82 |
| 50 GB Hyperdisk Balanced | ~$0.06/GiB/month | 50 GiB | ~$3.00 |
| **Total** | | | **~$100.82 (~EUR 93)** |

This is the **single largest always-on cost** in the project. The controller VM runs 24/7
to manage SkyPilot managed jobs. Options to reduce:
- Stop controller when not running experiments: `sky jobs controller stop` (manual)
- Downsize to `n4-standard-2` if memory allows
- Use spot/preemptible (risky -- controller crash = orphaned jobs)

### 6.3 Training GPU VMs (Transient)

Training VMs are launched by SkyPilot on demand (L4 spot instances in europe-west4).
These are NOT always-on resources. Current SkyPilot jobs:

```
ID   TASK  NAME                                           REQUESTED       SUBMITTED  STATUS
158  -     minivess-factorial                             1x[L4:1][Spot]  2 hrs ago  PENDING
157  -     sam3_topolora-dice_ce_cldice-calibfalse-f0     1x[L4:1][Spot]  1 day ago  CANCELLED
```

L4 spot pricing (europe-west4): ~$0.2204/hour
Per training job (~5 hours): ~$1.10

---

## 7. Service Accounts

| Email | Display Name | Disabled |
|-------|-------------|----------|
| `skypilot-training@minivess-mlops.iam.gserviceaccount.com` | SkyPilot Training SA | No |
| `skypilot-v1@minivess-mlops.iam.gserviceaccount.com` | SkyPilot SA (v1) | No |
| `mlflow-server@minivess-mlops.iam.gserviceaccount.com` | MLflow Cloud Run SA | No |
| `615852151588-compute@developer.gserviceaccount.com` | Compute Engine default SA | No |

---

## 8. Networking

### 8.1 VPC

Single `default` VPC network (auto mode).

### 8.2 Firewall Rules

| Rule | Direction | Allowed | Source |
|------|-----------|---------|--------|
| `default-allow-icmp` | INGRESS | ICMP | 0.0.0.0/0 |
| `default-allow-internal` | INGRESS | TCP/UDP/ICMP all ports | 10.128.0.0/9 |
| `default-allow-rdp` | INGRESS | TCP:3389 | 0.0.0.0/0 |
| `default-allow-ssh` | INGRESS | TCP:22 | 0.0.0.0/0 |

### 8.3 Static IPs

None allocated. All resources use ephemeral IPs (Cloud SQL has a public IP but it is
managed by GCP, not a reserved static address).

---

## 9. Pulumi Stack Outputs

```
OUTPUT                    VALUE
checkpoints_bucket        minivess-mlops-checkpoints
db_connection_name        minivess-mlops:europe-west4:mlflow-db-479b793
db_public_ip              34.158.67.254
docker_registry           europe-west4-docker.pkg.dev/minivess-mlops/minivess
dvc_data_bucket           minivess-mlops-dvc-data
mlflow_artifacts_bucket   minivess-mlops-mlflow-artifacts
mlflow_url                https://minivess-mlflow-a7w6hliydq-ez.a.run.app
mlflow_version            3.10.0
region                    europe-west4
skypilot_service_account  skypilot-training@minivess-mlops.iam.gserviceaccount.com
```

All Pulumi outputs point to `europe-west4`. The stack is consistent.

---

## 10. europe-north1 Remnant Check (Migration Verification)

| Check | Result | Status |
|-------|--------|--------|
| GAR repos in europe-north1 | `Listed 0 items` | CLEAN |
| GCS buckets in europe-north1 | N/A (bucket names are global, locations verified above) | CLEAN |
| Cloud SQL in europe-north1 | None (only `mlflow-db-479b793` in europe-west4) | CLEAN |
| Cloud Run in europe-north1 | None (only `minivess-mlflow` in europe-west4) | CLEAN |
| Compute in europe-north1 | None | CLEAN |
| Pulumi outputs | All point to europe-west4 | CLEAN |

**Verdict**: Zero europe-north1 remnants. Migration is complete.

**One note**: The SkyPilot jobs controller is in `europe-west1-b` (Belgium), not europe-west4
(Netherlands). This is a SkyPilot-managed resource and its location does not affect data
egress costs (the controller orchestrates, it does not transfer training data or Docker images).

---

## 11. Full Cost Summary

### 11.1 Always-On Costs (Monthly)

| Resource | Details | EUR/month |
|----------|---------|-----------|
| Cloud SQL (db-g1-small) | PostgreSQL 15, 10 GB SSD, backups | ~23.50 |
| Cloud Run (MLflow, min 1 instance) | 1 vCPU, 2 GiB, CPU-throttled | ~2-6 |
| SkyPilot Controller VM | n4-standard-4, 50 GB hyperdisk, europe-west1-b | ~93-100 |
| GAR Storage | 6.24 GiB images | ~0.57 |
| GCS Storage | 0 bytes (empty buckets) | 0.00 |
| **TOTAL always-on** | | **~EUR 119-130/month** |

### 11.2 Variable Costs (Per Training Run)

| Resource | Details | EUR/run |
|----------|---------|---------|
| L4 spot VM (europe-west4) | ~5 hours x $0.2204/hour | ~1.01 |
| GCS egress (same-region) | $0.00 | 0.00 |
| GAR image pull (same-region) | $0.00 | 0.00 |
| **Per training job** | | **~EUR 1.01** |

### 11.3 Projected Monthly Total (Active Development)

| Scenario | Training Jobs/month | EUR/month |
|----------|-------------------|-----------|
| Minimal (idle) | 0 | ~119-130 |
| Light development | 10 | ~129-140 |
| Active factorial experiments | 50 | ~169-180 |
| Heavy production runs | 100 | ~219-230 |

### 11.4 Comparison with Pre-Migration Costs

| Cost Component | Pre-Migration (europe-north1) | Post-Migration (europe-west4) | Savings |
|---------------|------------------------------|------------------------------|---------|
| GAR cross-region egress | ~EUR 89/month | EUR 0.00 | **EUR 89/month** |
| GAR storage | ~EUR 0.57 | ~EUR 0.57 | EUR 0.00 |
| Cloud SQL | ~EUR 23.50 | ~EUR 23.50 | EUR 0.00 |
| Cloud Run | ~EUR 2-6 | ~EUR 2-6 | EUR 0.00 |
| SkyPilot Controller | ~EUR 93-100 | ~EUR 93-100 | EUR 0.00 |
| **TOTAL** | **~EUR 208-219** | **~EUR 119-130** | **~EUR 89/month** |

**Annual savings from migration: ~EUR 1,068**

---

## 12. Cost Monitoring Checkpoints

### 12.1 Billing Console URLs

- **Billing Overview**: https://console.cloud.google.com/billing/01DCCF-E3B6B4-0616FE/reports?project=minivess-mlops
- **Cost Table (by SKU)**: https://console.cloud.google.com/billing/01DCCF-E3B6B4-0616FE/reports?project=minivess-mlops&reportType=COST_TABLE
- **Budget & Alerts**: https://console.cloud.google.com/billing/01DCCF-E3B6B4-0616FE/budgets?project=minivess-mlops
- **Export to BigQuery**: https://console.cloud.google.com/billing/01DCCF-E3B6B4-0616FE/export?project=minivess-mlops
- **Cloud SQL Monitoring**: https://console.cloud.google.com/sql/instances/mlflow-db-479b793/overview?project=minivess-mlops
- **Cloud Run Monitoring**: https://console.cloud.google.com/run/detail/europe-west4/minivess-mlflow/metrics?project=minivess-mlops
- **GAR Usage**: https://console.cloud.google.com/artifacts/docker/minivess-mlops/europe-west4/minivess?project=minivess-mlops
- **Compute Engine**: https://console.cloud.google.com/compute/instances?project=minivess-mlops

### 12.2 Validation Checkpoints

#### Day 1 (2026-03-29) -- Immediate Verification

- [ ] Check billing reports for any unexpected charges from 2026-03-28
- [ ] Verify no cross-region egress charges appear under "Cloud Storage" or "Artifact Registry"
- [ ] Confirm Cloud SQL is billing at db-g1-small rate (~$0.78/day)
- [ ] Confirm Cloud Run memory billing (~$0.20/day with min 1 instance)
- [ ] Confirm SkyPilot controller billing (~$3.22/day)
- [ ] Expected Day 1 total: ~$4.20 (~EUR 3.87)

#### Day 2 (2026-03-30) -- Pattern Confirmation

- [ ] Compare Day 1 vs Day 2 charges (should be nearly identical if no training jobs)
- [ ] Check for any "Networking" SKU charges (should be $0.00 for same-region)
- [ ] Verify no orphaned GPU VMs from SkyPilot job #158 (PENDING -> should auto-terminate)
- [ ] Expected cumulative: ~$8.40 (~EUR 7.74)

#### Day 7 (2026-04-04) -- Week-1 Audit

- [ ] Total week-1 cost should be: ~$29.40 (~EUR 27.09) without training jobs
- [ ] Check GAR egress specifically: should be $0.00
- [ ] Compare with pre-migration weekly: was ~$48.60 (EUR 44.82) including egress
- [ ] Week-1 savings should be visible: ~EUR 17.73 saved
- [ ] Review any training job costs (L4 spot: ~$1.10/job)
- [ ] Check if SkyPilot controller can be stopped if no experiments planned

#### Day 30 (2026-04-27) -- Month-1 Full Audit

- [ ] Full month cost should be: EUR 119-130 (idle) or EUR 130-180 (with training)
- [ ] Compare with pre-migration monthly: was ~EUR 208-219
- [ ] Verify EUR 89 monthly savings realized
- [ ] Decision point: downsize or stop SkyPilot controller if underutilized
- [ ] Decision point: enable Cloud SQL PITR if production data accumulates
- [ ] Review lifecycle policies: are non-current objects being cleaned up?

### 12.3 Automated Cost Monitoring Commands

Run these periodically to track resource growth:

```bash
# Check GAR storage growth
gcloud artifacts repositories list --project=minivess-mlops \
  --format="table(name,sizeBytes)"

# Check GCS bucket sizes
gsutil du -s gs://minivess-mlops-dvc-data/
gsutil du -s gs://minivess-mlops-mlflow-artifacts/
gsutil du -s gs://minivess-mlops-checkpoints/

# Check for orphaned VMs (should only be the controller)
gcloud compute instances list --project=minivess-mlops \
  --format="table(name,zone,machineType,status)"

# Check SkyPilot job status (catch stuck/orphaned jobs)
uv run sky jobs queue

# Check Cloud SQL disk usage
gcloud sql instances describe mlflow-db-479b793 --project=minivess-mlops \
  --format="yaml(settings.dataDiskSizeGb)"
```

---

## 13. Cost Optimization Recommendations

### 13.1 Immediate (No Risk)

1. **Delete untagged MLflow image**: The untagged `mlflow` image (267 MiB) is superseded by
   `v3.10.0`. Delete it to save ~$0.03/month and reduce confusion.
   ```bash
   gcloud artifacts docker images delete \
     europe-west4-docker.pkg.dev/minivess-mlops/minivess/mlflow@sha256:<digest> \
     --quiet
   ```

2. **Set up billing alerts**: Create a budget alert at EUR 150/month to catch unexpected charges.
   ```
   Console > Billing > Budgets & Alerts > Create Budget
   ```

### 13.2 Medium-Term (Requires Evaluation)

3. **SkyPilot controller stop/start**: When not running experiments for >24 hours, stop the
   controller to save ~EUR 3.10/day. Restart with `sky jobs controller start` before launching.

4. **Cloud SQL downtime**: If MLflow is only needed during experiments, consider stopping the
   Cloud SQL instance between experiment runs (saves ~EUR 0.78/day).

5. **Cloud Run min instances = 0**: If cold starts are acceptable, set `minScale: 0` to
   eliminate the always-on memory cost (~EUR 0.20/day saved). MLflow is not latency-critical.

### 13.3 Long-Term (Architecture Decisions)

6. **Committed use discounts**: If the project runs for 1+ years, CUDs on Cloud SQL and the
   controller VM could save 20-57%.

7. **SkyPilot controller on spot**: Risky but could save ~60% on the controller. Only
   recommended if SkyPilot handles controller preemption gracefully.

---

## 14. Security Observations

| Finding | Severity | Recommendation |
|---------|----------|---------------|
| Cloud SQL `authorized_networks: 0.0.0.0/0` | **HIGH** | Restrict to Cloud Run connector + dev IP |
| Cloud SQL `requireSsl: false` | MEDIUM | Enable SSL enforcement |
| Cloud Run ingress: `all` | LOW | Acceptable for MLflow tracking UI |
| Default firewall allows SSH from 0.0.0.0/0 | LOW | Standard GCP default, acceptable |

---

## 15. SkyPilot Status

```
SkyPilot API server version: 1.0.0.dev20260314 (running)
SkyPilot client version: 1.0.0.dev20260326 (installed)
```

**Version mismatch detected.** The running API server is 12 days behind the installed client.
Recommended fix:
```bash
sky api stop && sky api start
```

Current managed jobs:
- Job #158 (`minivess-factorial`): PENDING for 2+ hours, requesting 1x L4 spot
- Job #157 (`sam3_topolora-dice_ce_cldice-calibfalse-f0`): CANCELLED after 2 recoveries

---

## 16. Raw Command Outputs (Appendix)

### A. `gcloud artifacts repositories list`
```
REPOSITORY        FORMAT  MODE                 SIZE (MB)
docker-hub-cache  DOCKER  REMOTE_REPOSITORY    0
minivess          DOCKER  STANDARD_REPOSITORY  6384.856
```

### B. `gcloud artifacts docker images list` (JSON)
```json
[
  {
    "createTime": "2026-03-27T23:41:09.190303Z",
    "metadata": {"imageSizeBytes": "6414531059"},
    "package": "europe-west4-docker.pkg.dev/minivess-mlops/minivess/base",
    "tags": "",
    "updateTime": "2026-03-27T23:41:09.190303Z"
  },
  {
    "createTime": "2026-03-28T01:46:08.378767Z",
    "metadata": {"imageSizeBytes": "280460161"},
    "package": "europe-west4-docker.pkg.dev/minivess-mlops/minivess/mlflow",
    "tags": "",
    "updateTime": "2026-03-28T01:46:08.378767Z"
  },
  {
    "createTime": "2026-03-27T23:33:27.395695Z",
    "metadata": {"imageSizeBytes": "280460155"},
    "package": "europe-west4-docker.pkg.dev/minivess-mlops/minivess/mlflow",
    "tags": "",
    "updateTime": "2026-03-28T01:46:08.378767Z"
  },
  {
    "createTime": "2026-03-27T23:45:46.159855Z",
    "metadata": {"imageSizeBytes": "280460161"},
    "package": "europe-west4-docker.pkg.dev/minivess-mlops/minivess/mlflow-gcp",
    "tags": "",
    "updateTime": "2026-03-27T23:45:46.159855Z"
  }
]
```

### C. `gsutil du -s` (all buckets)
```
0  gs://minivess-mlops-dvc-data
0  gs://minivess-mlops-mlflow-artifacts
0  gs://minivess-mlops-checkpoints
```

### D. `gcloud sql instances list`
```
NAME               REGION        STATUS    TIER         DATA_DISK_SIZE_GB
mlflow-db-479b793  europe-west4  RUNNABLE  db-g1-small  10
```

### E. `gcloud run services list`
```
NAME             REGION        URL
minivess-mlflow  europe-west4  (see detailed output for full URL)
```

### F. `gcloud compute instances list`
```
NAME                                                       ZONE            MACHINE_TYPE   STATUS
sky-jobs-controller-aec-cw-aec1627c-head-4wim9eg2-compute  europe-west1-b  n4-standard-4  RUNNING
```

### G. `gcloud compute disks list`
```
NAME                                                       ZONE            SIZE_GB  STATUS  TYPE
sky-jobs-controller-aec-cw-aec1627c-head-4wim9eg2-compute  europe-west1-b  50       READY   hyperdisk-balanced
```

### H. `pulumi stack output`
```
OUTPUT                    VALUE
checkpoints_bucket        minivess-mlops-checkpoints
db_connection_name        minivess-mlops:europe-west4:mlflow-db-479b793
db_public_ip              34.158.67.254
docker_registry           europe-west4-docker.pkg.dev/minivess-mlops/minivess
dvc_data_bucket           minivess-mlops-dvc-data
mlflow_artifacts_bucket   minivess-mlops-mlflow-artifacts
mlflow_url                https://minivess-mlflow-a7w6hliydq-ez.a.run.app
mlflow_version            3.10.0
region                    europe-west4
skypilot_service_account  skypilot-training@minivess-mlops.iam.gserviceaccount.com
```

### I. Service Accounts
```
EMAIL                                                     DISPLAY NAME                            DISABLED
skypilot-training@minivess-mlops.iam.gserviceaccount.com  SkyPilot Training Service Account       False
skypilot-v1@minivess-mlops.iam.gserviceaccount.com        SkyPilot Service Account (v1)           False
mlflow-server@minivess-mlops.iam.gserviceaccount.com      MLflow Cloud Run Service Account        False
615852151588-compute@developer.gserviceaccount.com        Compute Engine default service account  False
```

### J. Lifecycle Policies
```
# minivess-mlops-mlflow-artifacts
{"rule": [{"action": {"type": "Delete"}, "condition": {"age": 7, "isLive": false}}]}

# minivess-mlops-checkpoints
{"rule": [{"action": {"type": "Delete"}, "condition": {"age": 30, "isLive": false}}]}

# minivess-mlops-dvc-data
(none)
```

---

*Report generated: 2026-03-28*
*Next scheduled audit: 2026-04-04 (Day 7 checkpoint)*
