---
title: "MLflow Deployment Targets: Detailed Storage Analysis"
status: reference
created: "2026-03-12"
---

# MLflow Deployment Targets: Detailed Storage Analysis

**Date**: 2026-03-12
**Status**: Reference document — supplements [mlflow-online-deployment-research.md](mlflow-online-deployment-research.md)
**Purpose**: Exhaustive storage analysis across all evaluated MLflow hosting targets

---

## Table of Contents

1. [Oracle Cloud Always Free](#1-oracle-cloud-always-free)
2. [Hetzner VPS (CX22/CX32/CX42)](#2-hetzner-vps)
3. [DagsHub Managed MLflow](#3-dagshub-managed-mlflow)
4. [Self-Hosted with Cloudflare Tunnel](#4-self-hosted-with-cloudflare-tunnel)
5. [AWS Free Tier](#5-aws-free-tier)
6. [GCP Free Tier](#6-gcp-free-tier)
7. [Azure Free Tier](#7-azure-free-tier)
8. [Comparison Summary](#8-comparison-summary)
9. [Can It Run MLflow + PostgreSQL + MinIO?](#9-feasibility-assessment)

---

## 1. Oracle Cloud Always Free

**Source**: [Oracle Always Free Resources](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm) |
[Oracle Free Tier FAQ](https://www.oracle.com/cloud/free/faq/)

### 1.1 Compute

| Resource | Spec |
|----------|------|
| Shape | VM.Standard.A1.Flex (Arm Ampere Altra) |
| OCPUs | 4 (3,000 OCPU-hours/month) |
| RAM | 24 GB (18,000 GB-hours/month) |
| Can split across VMs | Yes (e.g., 2x 2-OCPU/12-GB or 1x 4-OCPU/24-GB) |

Also available: 2x VM.Standard.E2.1.Micro (AMD, 1/8 OCPU, 1 GB each) — too small for
our stack but usable as jump hosts or monitoring.

### 1.2 Block Storage (200 GB Total)

The 200 GB is a **combined pool** for both boot volumes and data volumes. There is no
separate boot vs. data allocation — you split it however you choose.

| Allocation Strategy | Boot Volume | Data Volume(s) | Free Headroom |
|---------------------|-------------|-----------------|---------------|
| Single VM, max data | 50 GB (minimum) | 150 GB attached block | 0 GB |
| Single VM, balanced | 100 GB | 100 GB attached block | 0 GB |
| Two VMs | 50 GB + 50 GB boots | 100 GB shared block | 0 GB |
| Single VM, max boot | 200 GB (all-in-one) | None | 0 GB |

**Key constraints**:
- Minimum boot volume: **50 GB** (counts toward 200 GB limit)
- Maximum volumes: 5 total (boot + block combined)
- Volume backups: **5 free** (boot + block combined)
- **Must be in home region** for Always Free pricing
- Block volumes are attachable/detachable (can survive instance recreation)

**Recommended for MLflow**: 50 GB boot + 150 GB data volume mounted at `/data`.
Docker volumes for PostgreSQL, MinIO, and MLflow artifacts all live on the 150 GB volume.

### 1.3 Object Storage (20 GB)

| Detail | Spec |
|--------|------|
| Total storage | 20 GB (combined across all tiers) |
| API requests | 50,000/month |
| Tiers | Standard, Infrequent Access, Archive (shared 20 GB pool) |
| S3 compatibility | Yes — [Amazon S3 Compatibility API](https://docs.oracle.com/en-us/iaas/Content/Object/Tasks/s3compatibleapi.htm) |
| Durability | 11 nines (Oracle-managed) |

**S3 endpoint format**:
```
Path-style:    https://<namespace>.compat.objectstorage.<region>.oraclecloud.com/<bucket>/<object>
Virtual-host:  https://<bucket>.<namespace>.compat.objectstorage.<region>.oci.customer-oci.com/<object>
```

**Auth**: Uses OCI Customer Secret Keys (compatible with `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`).
Tools like `aws s3`, `mc` (MinIO Client), and `boto3` work with the S3 endpoint.

### 1.4 Can Both Be Used Simultaneously?

**Yes.** Block storage and Object Storage are independent services:
- **Block storage** (200 GB): Mounted as ext4/xfs filesystem to the VM. Used for boot, Docker volumes, PostgreSQL data, MinIO data.
- **Object Storage** (20 GB): Accessed via HTTP/S3 API. Used for off-instance backups, long-term archive of champion models.

Recommended pattern: MinIO on block storage as primary artifact store (150 GB available, unlimited API calls) with periodic `mc mirror` to OCI Object Storage for disaster recovery backup (within 20 GB free limit).

### 1.5 Autonomous Database

| Detail | Spec |
|--------|------|
| Instances | 2 Always Free |
| Storage per instance | 20 GB (non-scalable) |
| CPU | 1 OCPU (non-scalable) |
| Concurrent sessions | 20 |
| Database type | Oracle Autonomous (NOT PostgreSQL) |

**Relevance**: Could theoretically replace PostgreSQL as MLflow backend store (MLflow supports Oracle via SQLAlchemy). However, using Oracle Autonomous DB would lock us into Oracle-specific SQL dialect and complicate local dev parity. PostgreSQL in Docker is preferred for portability.

### 1.6 Outbound Data Transfer

| Detail | Spec |
|--------|------|
| Monthly egress | 10 TB/month |
| Ingress | Unlimited |
| Internal (within OCI) | Free |

### 1.7 Post-Trial Pricing (If Exceeding Free Tier)

| Resource | Overage Price |
|----------|---------------|
| Block storage beyond 200 GB | $0.0255/GB/month |
| Object storage beyond 20 GB | $0.0255/GB/month (Standard tier) |
| Outbound beyond 10 TB | $0.0085/GB |
| A1.Flex beyond free hours | $0.01/OCPU-hour |

### 1.8 Frankfurt Region Note

Frankfurt (`eu-frankfurt-1`) is Oracle's most popular EU region and suffers chronic
A1.Flex capacity shortages. Users report weeks of "Out of host capacity" errors.
Recommended: **Milan** (`eu-milan-1`) or **Marseille** (`eu-marseille-1`) for better
availability. Home region is permanent — cannot be changed after account creation.

**Source**: [Oracle Community — Ampere Out of Capacity eu-frankfurt-1](https://community.oracle.com/customerconnect/discussion/738457/ampere-out-of-capacity-eu-frankfurt-1) |
[hitrov/oci-arm-host-capacity](https://github.com/hitrov/oci-arm-host-capacity)

---

## 2. Hetzner VPS

**Source**: [Hetzner Cloud](https://www.hetzner.com/cloud/) |
[Hetzner Object Storage](https://www.hetzner.com/storage/object-storage/)

> **Note**: Hetzner renamed CX22/CX32/CX42 (Gen2) to CX23/CX33/CX43 (Gen3) in late
> 2025. CX22/CX32/CX42 are deprecated as of Feb 2026. Specs are nearly identical.
> Prices below reflect the new CX Gen3 plans after April 1, 2026 price adjustment.

### 2.1 Server Plans

| Plan | vCPU | RAM | Included Disk | Traffic (EU) | Price/month |
|------|------|-----|---------------|--------------|-------------|
| **CX22** (deprecated) | 2 shared | 4 GB | 40 GB NVMe | 20 TB | EUR 3.79 |
| **CX23** (current) | 2 shared | 4 GB | 40 GB NVMe | 20 TB | ~EUR 4.50 (post Apr 2026) |
| **CX32** (deprecated) | 4 shared | 8 GB | 80 GB NVMe | 20 TB | EUR 6.80 |
| **CX33** (current) | 4 shared | 8 GB | 80 GB NVMe | 20 TB | ~EUR 8.00 (post Apr 2026) |
| **CX42** (deprecated) | 8 shared | 16 GB | 160 GB NVMe | 20 TB | EUR 16.40 |
| **CX43** (current) | 8 shared | 16 GB | 160 GB NVMe | 20 TB | ~EUR 19.00 (post Apr 2026) |

**Billing**: Hourly with monthly cap. Delete server mid-month = only pay for hours used.
US locations are ~20% more expensive with only 1 TB traffic included.

**Source**: [Hetzner CX22 Benchmark](https://www.vpsbenchmarks.com/hosters/hetzner/plans/cx22) |
[Hetzner New CX Plans Announcement](https://www.hetzner.com/pressroom/new-cx-plans/) |
[Hetzner Price Adjustment 2026](https://docs.hetzner.com/general/infrastructure-and-availability/price-adjustment/)

### 2.2 Hetzner Volumes (Attachable Block Storage)

| Detail | Spec |
|--------|------|
| Size range | 10 GB to 10 TB (1 GB increments) |
| Pricing (pre-Apr 2026) | EUR 0.044/GB/month |
| Pricing (post-Apr 2026) | EUR 0.057/GB/month (~30% increase) |
| Type | SSD-based |
| Attachment | One server at a time |
| Persistence | Survives server deletion |

**Examples** (post-April 2026):
- 50 GB volume: EUR 2.85/month
- 100 GB volume: EUR 5.70/month
- 200 GB volume: EUR 11.40/month

**Source**: [Hetzner Cloud Volumes Overview](https://docs.hetzner.com/cloud/volumes/overview/)

### 2.3 Hetzner Object Storage (S3-Compatible)

| Detail | Spec |
|--------|------|
| Base price | EUR 4.99/month (USD 5.99) |
| Included storage | 1 TB |
| Included egress | 1 TB/month |
| Storage overage | EUR 0.0067/TB-hour |
| Egress overage | EUR 1.00/TB |
| Ingress | Free |
| S3 API calls | Free |
| Max buckets | 100 per account |
| Max per bucket | 100 TB / 50M objects |
| Min billable object | 64 KB |
| Regions | Falkenstein (FSN1), Helsinki (HEL1), Nuremberg (NBG1) |
| Billing start | First bucket creation (even if empty) |

**Source**: [Hetzner Object Storage](https://www.hetzner.com/storage/object-storage/)

### 2.4 Can CX22/CX23 (2 vCPU, 4 GB) Run MLflow + PostgreSQL + MinIO?

**Marginal but feasible** for light usage. Baseline memory footprint:

| Service | Idle RAM | Active RAM |
|---------|----------|------------|
| PostgreSQL 16 | ~35 MB | ~100-300 MB under load |
| MinIO (single node) | ~115 MB baseline, 1 GB pre-allocated | ~1-2 GB |
| MLflow 3.x server | ~400 MB | ~600-800 MB under load |
| OS + Docker overhead | ~300 MB | ~500 MB |
| **Total** | ~850 MB | **~2.4-3.6 GB** |

With 4 GB RAM, this leaves 400 MB to 1.15 GB headroom at idle, but heavy concurrent
artifact uploads could cause OOM. **Recommendation**: CX23 works for a solo researcher
with light usage. For production or concurrent SkyPilot VMs, CX33 (8 GB) is safer.

### 2.5 Total Cost Scenarios

| Scenario | Server | Extra Storage | Object Storage | Monthly Total |
|----------|--------|---------------|----------------|---------------|
| Minimal (CX23 only) | ~EUR 4.50 | None | None | **EUR 4.50** |
| CX23 + 100 GB volume | ~EUR 4.50 | EUR 5.70 | None | **EUR 10.20** |
| CX33 + 100 GB volume | ~EUR 8.00 | EUR 5.70 | None | **EUR 13.70** |
| CX33 + Object Storage | ~EUR 8.00 | None | EUR 4.99 | **EUR 12.99** |

Annual cost range: **EUR 54 to EUR 165**.

---

## 3. DagsHub Managed MLflow

**Source**: [DagsHub Pricing](https://dagshub.com/pricing/) |
[DagsHub MLflow Integration](https://dagshub.com/docs/integration_guide/mlflow_tracking/) |
[DagsHub MLflow 3.x Support](https://dagshub.com/blog/dagshub-now-supports-mlflow-3-x/)

### 3.1 Pricing Tiers

| Tier | Price | Storage | Key Limits |
|------|-------|---------|------------|
| **Individual** (Free) | $0/month | 20 GB DagsHub Storage | 1 user, unlimited public repos |
| **Team** | $99-$119/user/month | Up to 1 TB or 2M files | Private repos, team access |
| **Enterprise** | Custom | Petabyte-scale | SSO, audit logs, SLA |

### 3.2 Storage Details

- **Free tier**: 20 GB of DagsHub Storage total (covers DVC + MLflow artifacts + Git LFS)
- MLflow artifacts count against the 20 GB storage quota
- DagsHub generates a dedicated artifact location: `mlflow-artifacts:/<UUID>` per experiment
- No separate breakdown between MLflow tracking data and artifact data

### 3.3 Backend

- DagsHub hosts the MLflow tracking server on their infrastructure
- **Backend store**: Not publicly documented. DagsHub wraps MLflow API; users do not get
  direct database access.
- **Artifact store**: DagsHub native storage (S3-based internally) or user-provided S3 buckets
- Supports MLflow 3.x as of 2026 ([announcement](https://dagshub.com/blog/dagshub-now-supports-mlflow-3-x/))

### 3.4 Data Export / Migration

- **Export**: Yes, via `mlflow-export-import` tool. You can bulk-export experiments, runs,
  and models from DagsHub to any other MLflow server.
- **Programmatic access**: Standard MLflow API — `mlflow.search_runs()`, `mlflow.artifacts.download_artifacts()`
- **DVC integration**: Datasets tracked with DVC can be exported as HuggingFace Datasets
  or PyTorch DataLoaders
- **Limitation**: MLflow UI on DagsHub does not display artifacts stored in external S3 buckets

### 3.5 What You Don't Get

- No direct PostgreSQL access (cannot run DuckDB analytics against the backing store)
- No custom MLflow plugins
- No control over MLflow server version (DagsHub manages upgrades)
- No Optuna PostgreSQL co-hosting (would need separate database service)
- Limited to DagsHub's API rate limits (not publicly documented)

---

## 4. Self-Hosted with Cloudflare Tunnel

**Source**: [Cloudflare Tunnel Upload Limits](https://community.cloudflare.com/t/max-upload-size/630925) |
[Cloudflare Workers Limits](https://developers.cloudflare.com/workers/platform/limits/) |
[Bypassing Cloudflare Upload Limit](https://tpetrina.com/til/2025-01-02-cloudflare-upload-limit)

### 4.1 Storage

Storage is whatever the local machine has. For a typical dev workstation:
- 500 GB - 2 TB NVMe SSD
- Effectively unlimited for MLflow artifacts

**Cost**: $0 (existing hardware).

### 4.2 The 100 MB Upload Limit

**This is the critical constraint.** Cloudflare imposes a per-request body size limit
based on plan:

| Cloudflare Plan | Max Upload per Request |
|-----------------|----------------------|
| **Free** | **100 MB** |
| Pro ($20/month) | 100 MB |
| Business ($200/month) | 200 MB |
| Enterprise (custom) | 500 MB (can request increase) |

**Impact on MLflow**: This limit applies to **all HTTP requests** proxied through Cloudflare,
including MLflow artifact uploads via the REST API. Affected artifacts:

| Artifact Type | Typical Size | Blocked by 100 MB? |
|---------------|-------------|---------------------|
| DynUNet checkpoint (.pth) | 15-50 MB | No |
| SegResNet checkpoint (.pth) | 30-80 MB | Borderline |
| SAM3 checkpoint (.pth) | 600 MB - 2.5 GB | **Yes** |
| ONNX export (DynUNet) | 50-100 MB | Borderline |
| ONNX export (SAM3) | 800 MB - 2.5 GB | **Yes** |
| Resolved config YAML | ~5 KB | No |
| Metrics JSONL | ~50 KB | No |
| Figures (PNG/SVG) | ~500 KB | No |

### 4.3 Workarounds

1. **Bypass proxy for uploads**: Set DNS to "DNS only" (gray cloud) temporarily during
   large uploads, then re-enable proxy. Not automatable for SkyPilot VMs.
2. **Direct IP access**: Upload via server IP instead of domain. Requires exposing the
   server's IP and punching through NAT — defeats the purpose of Cloudflare Tunnel.
3. **Chunk uploads**: Not natively supported by MLflow's artifact upload API.
4. **Local DNS override**: Route local/VPN clients directly to the machine's LAN IP.
   Only works for same-network clients, not cloud VMs.

### 4.4 Bandwidth

- **No hard bandwidth cap** on Cloudflare free plan
- **No egress charges**
- Traffic is deprioritized vs. paid plans during congestion
- Tunnel creates a persistent outbound connection (no inbound port forwarding needed)

### 4.5 Uptime Dependency

The MLflow server is only accessible while:
- The local machine is running
- The Cloudflare Tunnel daemon (`cloudflared`) is running
- The home internet connection is up

Typical home ISP uptime: 99.0-99.5%. Power outages, ISP maintenance, machine reboots
all cause downtime. **Not suitable for SkyPilot VMs that may train overnight.**

---

## 5. AWS Free Tier

**Source**: [AWS Free Tier FAQ](https://aws.amazon.com/free/free-tier-faqs/) |
[AWS S3 Pricing](https://aws.amazon.com/s3/pricing/) |
[AWS EBS Pricing](https://aws.amazon.com/ebs/pricing/) |
[AWS Free Tier Usage Guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-free-tier-usage.html)

### 5.1 What AWS Free Tier Includes

| Resource | Free Tier Allocation | Duration |
|----------|---------------------|----------|
| **EC2 t2.micro** | 750 hrs/month (1 vCPU, 1 GB RAM) | **12 months only** |
| **EBS (gp2/gp3)** | 30 GB total | **12 months only** |
| **S3 Standard** | 5 GB storage | **12 months only** |
| S3 GET requests | 20,000/month | 12 months only |
| S3 PUT requests | 2,000/month | 12 months only |
| Data transfer out | 100 GB/month (across all services) | **Always free** (since Dec 2024) |

**Critical**: AWS Free Tier **expires after 12 months**. This is NOT a permanent free tier
like Oracle Cloud. After 12 months, all resources convert to pay-as-you-go.

### 5.2 Can t2.micro Run MLflow + PostgreSQL + MinIO?

**No.** The t2.micro has only **1 GB RAM**. Baseline memory for the stack:

| Service | Minimum RAM |
|---------|-------------|
| PostgreSQL | ~35 MB idle, 100+ MB active |
| MinIO | 1 GB pre-allocated minimum (single-node) |
| MLflow server | ~400 MB |
| OS + Docker | ~300 MB |
| **Total minimum** | **~1.7 GB** |

The 1 GB RAM of t2.micro is insufficient. You would need at minimum a t3.small (2 GB)
or t3.medium (4 GB), which are NOT free tier.

### 5.3 Storage Details

**EBS gp3** (30 GB free):
- Performance: 3,000 IOPS baseline, 125 MB/s throughput (free)
- After 12 months: $0.08/GB/month in us-east-1

**S3** (5 GB free):
- After 12 months: $0.023/GB/month (Standard, first 50 TB)
- PUT/GET: $0.005 per 1,000 PUT, $0.0004 per 1,000 GET
- Egress: $0.09/GB after 100 GB/month free

### 5.4 Post-Free-Tier Monthly Cost

| Resource | Size | Monthly Cost |
|----------|------|-------------|
| t3.medium (4 GB) | On-demand | ~$30/month |
| EBS gp3 | 50 GB | $4.00 |
| S3 Standard | 50 GB | $1.15 |
| Data transfer | 50 GB out | $4.50 (after 100 GB free) |
| **Total** | | **~$40/month** |

---

## 6. GCP Free Tier

**Source**: [GCP Free Cloud Features](https://cloud.google.com/free) |
[GCP Cloud Storage Pricing](https://cloud.google.com/storage/pricing) |
[GCP e2-micro Specs](https://gcloud-compute.com/e2-micro.html)

### 6.1 What GCP Always Free Includes

| Resource | Allocation | Duration |
|----------|-----------|----------|
| **e2-micro VM** | 0.25 vCPU shared (bursts to 2 vCPU), 1 GB RAM | **Always free** |
| **Persistent Disk** | 30 GB Standard (HDD, not SSD) | **Always free** |
| **Cloud Storage** | 5 GB Regional | **Always free** |
| Storage operations | 5,000 Class A + 50,000 Class B/month | Always free |
| Egress | 1 GB/month from North America | Always free |
| **$300 trial credit** | Any GCP resource | 90 days |

**Region restriction**: Always free resources limited to **us-west1, us-central1, or
us-east1**. No EU regions qualify for always-free compute.

### 6.2 Can e2-micro Run MLflow + PostgreSQL + MinIO?

**No.** The e2-micro has only **1 GB RAM** and 0.25 vCPU (shared). Same RAM constraint
as AWS t2.micro — the stack needs at minimum 1.7 GB. The 0.25 vCPU also makes PostgreSQL
query performance unacceptable.

### 6.3 Storage Details

**Persistent Disk** (30 GB free):
- Type: Standard (HDD) — NOT SSD
- Must be in us-west1/us-central1/us-east1
- After free limit: $0.04/GB/month (Standard), $0.17/GB/month (SSD)

**Cloud Storage** (5 GB free):
- Type: Regional Standard
- After free limit: ~$0.02/GB/month (us-central1)
- Egress: $0.12/GB after 1 GB/month free

### 6.4 Post-Free Monthly Cost (Comparable Setup)

| Resource | Size | Monthly Cost |
|----------|------|-------------|
| e2-medium (2 vCPU, 4 GB) | On-demand | ~$25/month |
| SSD Persistent Disk | 50 GB | $8.50 |
| Cloud Storage | 50 GB | $1.00 |
| Egress | 50 GB | $6.00 |
| **Total** | | **~$40.50/month** |

---

## 7. Azure Free Tier

**Source**: [Azure Free Services](https://azure.microsoft.com/en-us/pricing/free-services) |
[Azure B1s Specs](https://instances.vantage.sh/azure/vm/b1s) |
[Azure Blob Storage Pricing](https://azure.microsoft.com/en-us/pricing/details/storage/blobs/) |
[Azure Free Tier Analysis](https://costbench.com/software/cloud-infrastructure/azure/free-plan/)

### 7.1 What Azure Free Tier Includes

| Resource | Allocation | Duration |
|----------|-----------|----------|
| **B1s VM** | 750 hrs/month (1 vCPU, 1 GB RAM) | **12 months only** |
| **Managed Disks** | 2x 64 GB P6 SSD (128 GB total) | **12 months only** |
| **Blob Storage** | 5 GB LRS Hot tier | **12 months only** |
| Blob operations | 20,000 read + 10,000 write/month | 12 months only |
| **$200 trial credit** | Any Azure resource | 30 days |
| **Always free** services | 15+ services (Azure Functions, Cosmos DB free tier, etc.) | Permanent |

### 7.2 Can B1s Run MLflow + PostgreSQL + MinIO?

**No.** The B1s has only **1 GB RAM** and 1 vCPU (burstable, 10% baseline CPU).
Same constraint as AWS and GCP free-tier VMs.

The B1s is designed for low-traffic web servers, not multi-service Docker stacks.

### 7.3 Storage Details

**Managed Disks** (2x 64 GB P6 free for 12 months):
- 128 GB total usable (but only for 12 months)
- P6 SSD: 240 IOPS, 50 MB/s
- After 12 months: ~$9.60/disk/month ($19.20 for both)

**Blob Storage** (5 GB free for 12 months):
- Hot tier: $0.018/GB/month (after free tier)
- Cool tier: $0.01/GB/month
- Archive: $0.002/GB/month

### 7.4 Post-Free Monthly Cost (Comparable Setup)

| Resource | Size | Monthly Cost |
|----------|------|-------------|
| B2s (2 vCPU, 4 GB) | On-demand | ~$30/month |
| Managed Disk P10 | 128 GB | ~$19/month |
| Blob Storage (Hot) | 50 GB | $0.90 |
| Egress | 50 GB | $4.35 |
| **Total** | | **~$54/month** |

---

## 8. Comparison Summary

### 8.1 Storage Comparison Table

| Target | Block/Disk | Object Storage | Total Usable | Duration | Monthly Cost |
|--------|-----------|----------------|-------------|----------|-------------|
| **Oracle Always Free** | 200 GB (150 data) | 20 GB (S3-compat) | **170 GB** | **Forever** | **$0** |
| **Hetzner CX23** | 40 GB included | +EUR 4.99 for 1 TB | **40–1040 GB** | Ongoing | **EUR 4.50–9.49** |
| **Hetzner CX33** | 80 GB included | +EUR 4.99 for 1 TB | **80–1080 GB** | Ongoing | **EUR 8.00–12.99** |
| **DagsHub Free** | N/A (managed) | 20 GB (all-in) | **20 GB** | Forever | **$0** |
| **Cloudflare Tunnel** | Local disk (unlimited) | N/A | **Unlimited** | Forever | **$0** |
| **AWS Free Tier** | 30 GB EBS | 5 GB S3 | **35 GB** | **12 months** | $0 then ~$40 |
| **GCP Always Free** | 30 GB HDD | 5 GB Regional | **35 GB** | Forever | $0 |
| **Azure Free Tier** | 128 GB SSD | 5 GB Blob | **133 GB** | **12 months** | $0 then ~$54 |

### 8.2 Feasibility: Can It Run MLflow 3.10 + PostgreSQL 16 + MinIO?

| Target | RAM | vCPU | Feasible? | Notes |
|--------|-----|------|-----------|-------|
| **Oracle A1.Flex** | 24 GB | 4 | **Yes (excellent)** | Massive headroom; can add Optuna, Grafana |
| **Hetzner CX23** | 4 GB | 2 | **Yes (tight)** | Works for solo researcher; no headroom |
| **Hetzner CX33** | 8 GB | 4 | **Yes (comfortable)** | Recommended Hetzner plan |
| **DagsHub Free** | N/A | N/A | **Partially** | Managed MLflow only; no PostgreSQL/MinIO access |
| **Cloudflare Tunnel** | Local | Local | **Yes** (100 MB limit) | Blocks SAM3 uploads; uptime = home ISP |
| **AWS t2.micro** | 1 GB | 1 | **No** | Insufficient RAM |
| **GCP e2-micro** | 1 GB | 0.25 | **No** | Insufficient RAM and CPU |
| **Azure B1s** | 1 GB | 1 | **No** | Insufficient RAM |

### 8.3 Cost After Free Period Expires

| Target | Year 1 | Year 2+ | 5-Year Total |
|--------|--------|---------|-------------|
| **Oracle Always Free** | $0 | $0 | **$0** |
| **Hetzner CX23** | EUR 54 | EUR 54 | **EUR 270** |
| **Hetzner CX33** | EUR 96 | EUR 96 | **EUR 480** |
| **DagsHub Free** | $0 | $0 | **$0** (within 20 GB) |
| **Cloudflare Tunnel** | $0 | $0 | **$0** |
| **AWS** | ~$0 (free yr) | ~$480 | **~$1,920** |
| **GCP** | ~$0 | ~$486 | **~$1,944** |
| **Azure** | ~$0 (free yr) | ~$648 | **~$2,592** |

---

## 9. Feasibility Assessment

### 9.1 Recommended Stack Memory Budget

For MLflow 3.10 + PostgreSQL 16 + MinIO (single-node) + nginx reverse proxy:

| Component | Idle | Active (concurrent uploads) |
|-----------|------|-----------------------------|
| PostgreSQL 16 | 35 MB | 200-400 MB |
| MinIO (single-node) | 1 GB pre-alloc | 1-2 GB |
| MLflow 3.10 server | 400 MB | 600-800 MB |
| nginx | 5 MB | 20 MB |
| Docker daemon | 100 MB | 200 MB |
| OS (Ubuntu 22.04) | 200 MB | 300 MB |
| **Total** | **~1.7 GB** | **~2.5-3.7 GB** |

**Minimum viable RAM**: 4 GB (works with swapfile for peaks).
**Recommended RAM**: 8+ GB (handles concurrent SkyPilot VMs logging simultaneously).
**Oracle's 24 GB**: Leaves 20+ GB headroom for Optuna storage, Grafana, monitoring, etc.

### 9.2 Final Ranking

1. **Oracle Cloud Always Free** — Best overall: $0 forever, 24 GB RAM, 170 GB storage,
   10 TB egress. Only downside: ARM (aarch64) requires compatible images (all server
   images already support it) and A1.Flex capacity can be hard to provision in Frankfurt.

2. **Hetzner CX33** — Best paid option: EUR 8/month, 8 GB RAM, 80 GB disk, x86_64,
   instant provisioning, excellent EU locations. Add Volumes or Object Storage as needed.

3. **Cloudflare Tunnel** — Best for pure-local dev: $0, unlimited storage, but 100 MB
   upload limit blocks SAM3 artifacts and uptime depends on home ISP.

4. **DagsHub Free** — Best for zero-ops: $0, managed MLflow, but only 20 GB total and
   no database access for DuckDB analytics.

5. **Hetzner CX23** — Budget option: EUR 4.50/month, works for light use but 4 GB RAM
   is tight for concurrent access.

6. **AWS/GCP/Azure** — Not recommended: Free tiers have 1 GB RAM (insufficient), expire
   after 12 months (except GCP e2-micro which is permanent but unusable), and paid tiers
   are 5-10x more expensive than Hetzner.

---

## References

1. [Oracle Cloud Infrastructure (2026). "Always Free Resources."](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)
2. [Oracle Cloud Infrastructure (2026). "Free Tier FAQ."](https://www.oracle.com/cloud/free/faq/)
3. [Oracle Cloud Infrastructure (2026). "Amazon S3 Compatibility API."](https://docs.oracle.com/en-us/iaas/Content/Object/Tasks/s3compatibleapi.htm)
4. [Oracle Cloud Infrastructure (2026). "Object Storage Namespaces."](https://docs.oracle.com/en-us/iaas/Content/Object/Tasks/understandingnamespaces.htm)
5. [Hetzner (2026). "Cloud Servers."](https://www.hetzner.com/cloud/)
6. [Hetzner (2026). "Object Storage."](https://www.hetzner.com/storage/object-storage/)
7. [Hetzner (2026). "Cloud Volumes Overview."](https://docs.hetzner.com/cloud/volumes/overview/)
8. [Hetzner (2026). "New CX Plans Announcement."](https://www.hetzner.com/pressroom/new-cx-plans/)
9. [Hetzner (2026). "Price Adjustment Notice."](https://docs.hetzner.com/general/infrastructure-and-availability/price-adjustment/)
10. [DagsHub (2026). "Pricing."](https://dagshub.com/pricing/)
11. [DagsHub (2026). "MLflow Integration Guide."](https://dagshub.com/docs/integration_guide/mlflow_tracking/)
12. [DagsHub (2026). "DagsHub Now Supports MLflow 3.x."](https://dagshub.com/blog/dagshub-now-supports-mlflow-3-x/)
13. [Cloudflare Community (2026). "100MB Tunnel Limit."](https://community.cloudflare.com/t/100mb-tunnel-limit/901339)
14. [Cloudflare (2026). "Workers Platform Limits."](https://developers.cloudflare.com/workers/platform/limits/)
15. [Petrina, T. (2025). "Bypassing Cloudflare Upload Limit."](https://tpetrina.com/til/2025-01-02-cloudflare-upload-limit)
16. [AWS (2026). "Free Tier FAQ."](https://aws.amazon.com/free/free-tier-faqs/)
17. [AWS (2026). "S3 Pricing."](https://aws.amazon.com/s3/pricing/)
18. [AWS (2026). "EBS Pricing."](https://aws.amazon.com/ebs/pricing/)
19. [Google Cloud (2026). "Free Cloud Features."](https://cloud.google.com/free)
20. [Google Cloud (2026). "Cloud Storage Pricing."](https://cloud.google.com/storage/pricing)
21. [Azure (2026). "Free Services."](https://azure.microsoft.com/en-us/pricing/free-services)
22. [Azure (2026). "Blob Storage Pricing."](https://azure.microsoft.com/en-us/pricing/details/storage/blobs/)
23. [Oracle Community (2024). "Ampere Out of Capacity eu-frankfurt-1."](https://community.oracle.com/customerconnect/discussion/738457/ampere-out-of-capacity-eu-frankfurt-1)
24. [hitrov/oci-arm-host-capacity (2024). GitHub.](https://github.com/hitrov/oci-arm-host-capacity)
25. [MLflow (2026). "Artifact Stores."](https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/)
26. [MLflow (2026). "Filesystem Backend Deprecation Notice."](https://github.com/mlflow/mlflow/issues/18534)
27. [MinIO (2025). "RAM Requirements Discussion."](https://github.com/minio/minio/discussions/19133)
28. [MLflow (2023). "Memory Usage Issue #5332."](https://github.com/mlflow/mlflow/issues/5332)
