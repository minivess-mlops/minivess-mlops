---
title: "Hetzner Cloud MLflow Server — Setup Plan"
status: in-progress
created: "2026-03-13"
replaces: oracle-config-planning.md
depends_on:
  - cloud-tutorial.md
  - mlflow-deployment-storage-analysis.md
---

# Hetzner Cloud MLflow Server

**Goal:** Deploy MLflow + PostgreSQL + MinIO on a Hetzner VPS, accessible to
SkyPilot training VMs and paper reviewers.

**Why Hetzner (not Oracle):** Oracle Cloud Always Free was rejected (2026-03-13)
due to chronic ARM capacity shortage, no region change, and terrible DevEx.
Hetzner costs EUR 4-7/month but has: simple bearer token auth, `hcloud` CLI that
just works, pre-built Docker images, zero capacity issues, 5-minute setup.

**Cost:** CX22 = EUR 3.79/month (~$50/year), CX32 = EUR 6.80/month (~$82/year).
New accounts get $20 free credit (covers ~5 months of CX22).

---

## Bootstrap Flow (3 steps, 0 browser hacks)

```bash
# 1. Get API token from Hetzner Console (one-time, 30 seconds)
#    Console → Project → Security → API Tokens → Generate (Read+Write)
#    Paste into .env: HETZNER_API_TOKEN=...

# 2. Install hcloud CLI + create server:
bash scripts/hetzner-setup-script.sh

# 3. Done. MLflow is at http://SERVER_IP (or https://mlflow.yourdomain.com)
```

---

## Server Spec

| Param | Value | Notes |
|-------|-------|-------|
| Type | `cx22` (2 vCPU, 4 GB, 40 GB SSD) | Upgrade to `cx32` if needed |
| Image | `docker-ce` | Docker pre-installed, zero setup |
| Location | `fsn1` (Falkenstein, Germany) | Lowest latency to RunPod EU |
| SSH Key | Generated locally, uploaded via `hcloud` | No browser paste needed |
| Firewall | SSH:22, HTTP:80, HTTPS:443, MLflow:5000 | Created via `hcloud` |

---

## Stack on Server

Same Docker Compose stack as the Oracle plan, but simpler (no block volume dance):

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `postgres` | `postgres:16` | 5432 (internal) | MLflow backend store |
| `minio` | `minio/minio` | 9000/9001 (internal) | Artifact store (S3-compat) |
| `mlflow` | `ghcr.io/mlflow/mlflow:v2.20.3` | 5000 | Tracking server |
| `nginx` | `nginx:alpine` | 80/443 | Reverse proxy + TLS + basic auth |

40 GB SSD is enough for:
- PostgreSQL: ~1 GB for thousands of runs
- MinIO artifacts: ~20 GB for model checkpoints
- OS + Docker: ~10 GB
- Headroom: ~9 GB

---

## Script Phases (`scripts/hetzner-setup-script.sh`)

| Phase | What | Time |
|-------|------|------|
| 0 | Validate: `hcloud` CLI, `HETZNER_API_TOKEN`, SSH key | 5s |
| 1 | Upload SSH key to Hetzner | 2s |
| 2 | Create firewall (SSH, HTTP, HTTPS, MLflow) | 2s |
| 3 | Create server (Docker CE image) | 30s |
| 4 | SSH in, deploy MLflow compose stack | 2 min |
| 5 | (Optional) DNS + TLS via Cloudflare + certbot | 2 min |
| 6 | Verify: health check, API auth, update .env | 10s |

**Total: ~5 minutes** (vs 2+ hours for Oracle, which still didn't work).

---

## DevEx Comparison

| Aspect | Hetzner | Oracle (rejected) |
|--------|---------|-------------------|
| Auth setup | 1 env var (`HCLOUD_TOKEN`) | Config file + PKCS#8 key + fingerprint + Cloud Shell |
| CLI stability | Solid, single Go binary | Intermittent `NotAuthenticated`, `IdcsConversionError` |
| Server creation | `hcloud server create` → 30s → done | "Out of host capacity" across all ADs |
| Docker | Pre-built `docker-ce` image | Manual install after VM (if you can get one) |
| SSH key | `hcloud ssh-key create` from local | Requires browser Cloud Shell for first key |
| Firewall | `hcloud firewall create` + rules | Security list JSON, OCI CLI format quirks |
| Teardown | `hcloud server delete mlflow` | Reverse-order OCID cleanup, route table clearing |
| Cost | EUR 3.79/month (guaranteed) | $0 (if capacity exists, which it doesn't) |

---

## Teardown

```bash
bash scripts/hetzner-setup-script.sh --teardown
# or manually:
hcloud server delete minivess-mlflow
hcloud firewall delete minivess-fw
hcloud ssh-key delete minivess-key
```
