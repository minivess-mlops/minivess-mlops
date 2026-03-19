# GCP Pulumi Stack — MinIVess MLOps

## GCP = Staging + Production Environment

GCP is the production-grade cloud for this project. Everything managed via Pulumi IaC.
RunPod is the separate "env" (dev) environment — it does NOT depend on GCP.

## Project Details

| Setting | Value | Source |
|---------|-------|--------|
| **Project ID** | `minivess-mlops` | `.env.example` line 395 |
| **Region** | `europe-north1` (Finland) | `.env.example` line 396 |
| **MLflow Version** | `3.10.0` (pinned) | `.env.example` line 379 |

## Resources Created by `pulumi up`

| Resource | Type | Purpose |
|----------|------|---------|
| `minivess-mlops-mlflow-artifacts` | GCS Bucket | MLflow run artifacts |
| `minivess-mlops-dvc-data` | GCS Bucket | **DVC training data (canonical cloud store)** |
| `minivess-mlops-checkpoints` | GCS Bucket | Model checkpoints (spot resume) |
| `mlflow` / `optuna` | Cloud SQL PostgreSQL | MLflow backend + Optuna HPO |
| `minivess` | Artifact Registry | Docker images (GAR, public read) |
| `skypilot-training` | Service Account | GPU training + GCS access |
| `mlflow-server` | Service Account | Cloud Run MLflow (if enabled) |
| MLflow Cloud Run | Cloud Run Service | **Disabled by default** (`enable_cloud_run: false`) |

## DVC Data on GCS

The canonical cloud data store is **GCS** (`gs://minivess-mlops-dvc-data`).

- **NOT AWS S3** — AWS is not part of the architecture
- `s3://minivessdataset` exists as a read-only public data origin, not a production backend
- DVC remote for GCP: `dvc remote add gcs gs://minivess-mlops-dvc-data`
- Config template: `configs/dvc/remotes.yaml` (gcs section)

## Usage

```bash
cd deployment/pulumi/gcp
pulumi stack init dev
pulumi config set minivess-gcp:mlflow_admin_password --secret "YOUR_PASSWORD"
pulumi config set minivess-gcp:db_password --secret "YOUR_DB_PASSWORD"
pulumi up
```

After provisioning, populate `.env` with outputs:
```bash
pulumi stack output mlflow_url       # → MLFLOW_TRACKING_URI (set in .env)
pulumi stack output dvc_data_bucket  # → GCS_DVC_BUCKET
```

## Prerequisites

1. `gcloud auth application-default login`
2. Enable APIs: compute, storage, iam, run, sqladmin, artifactregistry
3. GPU quota: `GPUS_ALL_REGIONS >= 1` in `europe-north1`
4. See: `docs/planning/gcp-setup-tutorial.md` for full walkthrough

## SkyPilot Integration

GCP SkyPilot config: `configs/cloud/gcp_spot.yaml`
- GPU: L4 preferred (BF16, 24 GB), A100 fallback
- T4 **BANNED** (Turing, no BF16 → FP16 overflow → NaN)
- Docker: `europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest`
- Spot: enabled (60-91% cheaper)

## Two-Provider Architecture

| Provider | Environment | Data | MLflow | Assumption |
|----------|------------|------|--------|------------|
| **RunPod** | env (dev) | Upload from local disk | File-based on Network Volume | Standalone, no GCP |
| **GCP** | staging + prod | GCS buckets | Cloud Run (optional) | Full managed stack |

**NEVER** add a third cloud provider without explicit user authorization.
