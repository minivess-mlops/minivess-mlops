---
name: skypilot-patterns
description: "SkyPilot YAML patterns for MinIVess — Docker image_id, Network Volume, spot recovery"
metadata:
  languages: "python"
  versions: "0.9.0"
  revision: 1
  updated-on: "2026-03-16"
  source: maintainer
  tags: "skypilot,cloud,gpu,docker,runpod,gcp"
---

# SkyPilot Patterns for MinIVess MLOps

SkyPilot = intercloud broker (Yang et al., NSDI'23). NOT IaC. NOT "a RunPod launcher."

## Mandatory: Docker image_id

ALL SkyPilot YAMLs MUST use Docker images. Bare VM setup scripts are BANNED.

```yaml
resources:
  image_id: docker:ghcr.io/petteriteikari/minivess-base:latest
  accelerators: {L4: 1, RTX4090: 1}
  cloud: runpod  # or gcp
  use_spot: true
  disk_size: 40
```

## Two Providers Only

| Provider | Environment | Config |
|----------|------------|--------|
| RunPod | env (dev) | `configs/cloud/runpod_dev.yaml` |
| GCP | staging + prod | `configs/cloud/gcp_spot.yaml` |

NEVER add AWS, Azure, or other providers without explicit user authorization.

## RunPod Pattern: Network Volume

```yaml
# Network Volume for data persistence across pods
volumes:
  /opt/vol: minivess-dev

envs:
  MLFLOW_TRACKING_URI: /opt/vol/mlruns  # File-based, no server
  DVC_REMOTE: remote_storage  # For initial data download only
```

## GCP Pattern: GCS + Cloud Run MLflow

```yaml
# GCP uses GCS buckets (created by Pulumi)
file_mounts:
  /app/checkpoints:
    source: gs://minivess-mlops-checkpoints
    mode: MOUNT

envs:
  MLFLOW_TRACKING_URI: ${MLFLOW_GCP_URI}  # Cloud Run endpoint
```

## Setup Block: DATA ONLY

```yaml
setup: |
  set -ex
  cd /app
  # BANNED: apt-get, uv sync, git clone, pip install
  # ALLOWED: dvc pull, cp configs, mkdir, nvidia-smi
  cp configs/splits/smoke_test_1fold_4vol.json configs/splits/splits.json
  mkdir -p /app/checkpoints /app/logs
  nvidia-smi
```

## T4 GPU BANNED

T4 is Turing architecture — no BF16 support. SAM3/VesselFM encoders overflow
during validation (FP16 max = 65504 → NaN). Use L4 (Ada Lovelace) or better.

## Key Files

- `deployment/skypilot/smoke_test_gpu.yaml` — RunPod smoke test
- `deployment/skypilot/smoke_test_gcp.yaml` — GCP smoke test
- `deployment/skypilot/smoke_test_mamba.yaml` — MambaVesselNet smoke test
- `knowledge-graph/domains/cloud.yaml` — Full cloud architecture
