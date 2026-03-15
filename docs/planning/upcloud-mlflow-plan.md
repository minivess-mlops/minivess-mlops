---
title: "UpCloud MLflow Server — Setup Plan"
status: active
created: "2026-03-13"
replaces: hetzner-mlflow-plan.md (Hetzner archived as long-term fallback)
depends_on:
  - cloud-tutorial.md
  - mlflow-deployment-storage-analysis.md
---

# UpCloud MLflow Server

**Goal:** Deploy MLflow + PostgreSQL + MinIO on an UpCloud VPS, accessible to
SkyPilot training VMs and paper reviewers.

**Why UpCloud (not Hetzner right now):** UpCloud offers a **30-day free trial with
EUR 250 credit** — enough to run the MLflow stack for the entire trial period at zero
cost. Hetzner is archived as the **long-term fallback** at EUR 3.79/month.

**Why not the others?**
- **Oracle Cloud** — Rejected (2026-03-13): chronic ARM capacity shortage, garbage DevEx
- **Nebius** — GPU-focused, minimum VPS is 2vCPU/8GB at ~$40/month, overkill
- **Scaleway** — Decent (EUR 9.30/mo DEV1-S), but no trial credit advantage over UpCloud

**Cost:** EUR 0 during trial (30 days, EUR 250 credit). After trial, cheapest viable
plan is DEV-1xCPU-2GB at EUR 8.70/month, or switch to Hetzner CX22 at EUR 3.79/month.

---

## Provider Comparison

| Provider | Cheapest (2+ vCPU, 4 GB) | Trial/Free | CLI quality | Verdict |
|----------|--------------------------|------------|-------------|---------|
| **UpCloud** | EUR 19.42/mo (DEV-2xCPU-4GB) | **EUR 250 / 30 days** | `upctl` (good) | **Active — free trial** |
| **Hetzner** | EUR 3.79/mo (CX22) | None | `hcloud` (excellent) | **Archived fallback** |
| **Scaleway** | EUR 9.30/mo (DEV1-S + IPv4) | EUR 100 one-time | `scw` (good) | Alternative |
| **Nebius** | ~$40/mo (2vCPU/8GB min) | None | `nebius` CLI | Skip — GPU-focused |
| **Oracle** | $0/mo (if capacity exists) | Always Free | `oci` (terrible) | **Rejected** |

---

## Bootstrap Flow (3 steps, 0 browser hacks)

```bash
# 1. Get API token from UpCloud Hub (one-time, 30 seconds)
#    https://hub.upcloud.com/account/api-tokens → Create
#    Paste into .env: UPCLOUD_TOKEN=ucat_...

# 2. Install upctl CLI + create server:
bash scripts/upcloud-setup-script.sh

# 3. Done. MLflow is at http://SERVER_IP (or https://mlflow.yourdomain.com)
```

---

## Server Spec

| Param | Value | Notes |
|-------|-------|-------|
| Plan | `DEV-2xCPU-4GB` (2 vCPU, 4 GB, 60 GB SSD) | Comfortable for 4 containers |
| OS | Ubuntu Server 24.04 LTS | Docker installed via apt |
| Zone | `fi-hel1` (Helsinki, Finland) | Finnish company, low latency |
| SSH Key | Uploaded at server creation via `upctl` | No browser paste needed |
| Firewall | Per-server rules: SSH:22, HTTP:80, HTTPS:443, MLflow:5000 | Via `upctl` API |

**Alternative plan:** `DEV-1xCPU-2GB` (EUR 8.70/mo) — tight but workable for low-traffic.

---

## Stack on Server

Same Docker Compose stack across all providers (Hetzner, UpCloud):

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `postgres` | `postgres:16` | 5432 (internal) | MLflow backend store |
| `minio` | `minio/minio` | 9000/9001 (internal) | Artifact store (S3-compat) |
| `mlflow` | `ghcr.io/mlflow/mlflow:v2.20.3` | 5000 | Tracking server |
| `nginx` | `nginx:alpine` | 80/443 | Reverse proxy + TLS + basic auth |

60 GB SSD is enough for:
- PostgreSQL: ~1 GB for thousands of runs
- MinIO artifacts: ~30 GB for model checkpoints
- OS + Docker: ~15 GB
- Headroom: ~14 GB

**Optional:** UpCloud has Managed Object Storage (S3-compatible, EUR 5/250GB) which
could replace MinIO. For trial period, self-hosted MinIO is simpler and free.

---

## Script Phases (`scripts/upcloud-setup-script.sh`)

| Phase | What | Time |
|-------|------|------|
| 0 | Validate: `upctl` CLI, `UPCLOUD_TOKEN`, SSH key | 5s |
| 1 | Create server (Ubuntu 24.04) | 45s |
| 2 | Install Docker via SSH | 2 min |
| 3 | Deploy MLflow compose stack | 2 min |
| 4 | Set up firewall rules | 10s |
| 5 | (Optional) DNS + TLS via Cloudflare + certbot | 2 min |
| 6 | Verify: health check, API auth, print .env values | 10s |

**Total: ~5 minutes.**

---

## Trial Management

| Phase | Action |
|-------|--------|
| **Day 1** | Run setup script → MLflow running |
| **Day 1-30** | Run SkyPilot training jobs against remote MLflow |
| **Day 25** | Backup: `docker compose exec postgres pg_dump > backup.sql` |
| **Day 25** | Backup: `rsync -avz root@IP:/opt/mlflow/ ./mlflow-backup/` |
| **Day 30** | Trial ends. Either: deposit EUR 10 to keep, or migrate to Hetzner |

**Data retention after trial:** UpCloud preserves storage for ~60 days after credit
depletion, giving time to upgrade or extract data.

---

## Backup Before Trial Ends

```bash
# SSH into server
ssh -i ~/.ssh/upcloud_minivess root@SERVER_IP

# PostgreSQL dump (all MLflow metadata)
docker compose -f /opt/mlflow/docker-compose.yml exec -T postgres \
  pg_dump -U minivess mlflow > /tmp/mlflow_backup.sql

# MinIO artifacts (model checkpoints, etc.)
docker run --rm -v mlflow_minio_data:/data -v /tmp/backup:/backup \
  alpine tar czf /backup/minio_data.tar.gz /data

# Exit and download
exit
rsync -avz root@SERVER_IP:/tmp/mlflow_backup.sql ./
rsync -avz root@SERVER_IP:/tmp/backup/minio_data.tar.gz ./
```

---

## Migration to Hetzner (Post-Trial)

If you decide not to keep paying UpCloud after trial:

```bash
# 1. Backup from UpCloud (see above)
# 2. Set up Hetzner:
bash scripts/hetzner-setup-script.sh
# 3. Restore on Hetzner:
scp mlflow_backup.sql root@HETZNER_IP:/tmp/
ssh -i ~/.ssh/hetzner_minivess root@HETZNER_IP \
  'docker compose -f /opt/mlflow/docker-compose.yml exec -T postgres \
   psql -U minivess mlflow < /tmp/mlflow_backup.sql'
```

---

## Teardown

```bash
bash scripts/upcloud-setup-script.sh --teardown
# or manually:
upctl server stop minivess-mlflow --wait
upctl server delete minivess-mlflow
```

---

## DevEx Comparison

| Aspect | UpCloud | Hetzner (archived) | Oracle (rejected) |
|--------|---------|-------------------|-------------------|
| Auth setup | 1 env var (`UPCLOUD_TOKEN`) | 1 env var (`HCLOUD_TOKEN`) | Config file + PKCS#8 key |
| CLI | `upctl` (Go, good) | `hcloud` (Go, excellent) | `oci` (Python, flaky) |
| Server creation | `upctl server create` → 45s | `hcloud server create` → 30s | "Out of host capacity" |
| Docker | Manual apt install (~2 min) | Pre-built `docker-ce` image | Manual install |
| Firewall | Per-server rules via API | Separate firewall resource | Security list JSON |
| Trial | EUR 250 / 30 days | None | $0 (if capacity exists) |
| Long-term cost | EUR 19.42/mo (4 GB) | EUR 3.79/mo (4 GB) | $0 (theoretical) |
| Helsinki DC | fi-hel1, fi-hel2 | hel1 | N/A |
