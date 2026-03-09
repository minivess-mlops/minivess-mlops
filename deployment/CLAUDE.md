# Deployment — Docker Infrastructure

## Three-Layer Docker Hierarchy

```
Layer 1: nvidia/cuda:12.6.3-runtime-ubuntu24.04  (upstream, never modified)
Layer 2: minivess-base:latest                     (THIS — all shared deps)
Layer 3: Dockerfile.{flow}                        (thin — scripts, env, CMD only)
```

**Flow Dockerfiles NEVER run `apt-get` or `uv`** — all deps belong in Dockerfile.base.

## Building

```bash
# Base image (rebuild when pyproject.toml or Dockerfile.base changes):
docker build -t minivess-base:latest -f deployment/docker/Dockerfile.base .

# Match host UID for bind-mount dev:
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) \
  -t minivess-base:latest -f deployment/docker/Dockerfile.base .
```

## Docker Compose Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Infrastructure: PostgreSQL, MinIO, MLflow, Prefect, Grafana |
| `docker-compose.flows.yml` | Per-flow services: 12 flow containers |

## Volume Mount Rules (Non-Negotiable)

Every artifact that must survive the container MUST be volume-mounted:

| Mount | Used By | Mode |
|-------|---------|------|
| `raw_data:/app/data/raw` | acquisition, data(ro) | rw/ro |
| `data_cache:/app/data` | data, train(ro), analyze(ro), hpo(ro) | varies |
| `configs_splits:/app/configs/splits` | data, train(ro), analyze(ro), hpo(ro) | varies |
| `checkpoint_cache:/app/checkpoints` | train, post_training(ro), analyze(ro), hpo | varies |
| `mlruns_data:/app/mlruns` | most flows | varies |
| `logs_data:/app/logs` | acquisition, train, hpo | rw |

**/tmp and tempfile.mkdtemp() are FORBIDDEN for artifacts.**

## Network

All services use the `minivess-network` external network:
```bash
docker network create minivess-network
```

## GPU Reservation

Only `train` and `hpo` services reserve GPU devices using CDI (Docker 25+, no nvidia runtime config needed):
```yaml
# CDI GPU access — works without nvidia container runtime
devices:
  - "nvidia.com/gpu=all"
```

**BANNED** (requires daemon config that isn't set up):
```yaml
# BANNED — causes "GPU not accessible" even with nvidia-ctk installed
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

## Volume Initialization (One-Time Setup)

Named volumes start owned by `root`. Fix ownership before first run:
```bash
docker run --rm --user root \
  -v deployment_checkpoint_cache:/app/checkpoints \
  -v deployment_logs_data:/app/logs \
  -v deployment_mlruns_data:/app/mlruns \
  -v deployment_data_cache:/app/data \
  -v deployment_configs_splits:/app/configs/splits \
  ubuntu:22.04 \
  chown -R 1000:1000 /app/checkpoints /app/logs /app/mlruns /app/data /app/configs/splits
```

**Docker Compose project name**: The project name is `deployment` (derived from the
compose file's parent directory). Volumes are named `deployment_{volume_name}` NOT just
`{volume_name}`. Always use `deployment_` prefix when referencing volumes with `docker run`.

## Data Population (One-Time Setup)

The `data_cache` volume must be populated with MiniVess data before training:
```bash
docker run --rm --user root \
  -v deployment_data_cache:/app/data \
  -v /path/to/minivess-mlops/data:/src:ro \
  ubuntu:22.04 \
  sh -c "mkdir -p /app/data/raw && cp -r /src/raw/minivess /app/data/raw/ && chown -R 1000:1000 /app/data"
```

## MLflow 3.x Security Middleware

MLflow 3.x defaults to localhost-only binding. Required env var:
```yaml
environment:
  MLFLOW_SERVER_ALLOWED_HOSTS: "*"  # NOT MLFLOW_ALLOWED_HOSTS
```

**YAML entrypoint syntax**: Use list form to avoid newline-as-separator bugs:
```yaml
# CORRECT: list form passes args as a single string to /bin/sh
entrypoint:
  - /bin/sh
  - -c
  - >-
    mlflow server --host 0.0.0.0 --port 5000

# BANNED: ">" folded scalar preserves newlines → mlflow server runs with NO args
entrypoint: >
  sh -c "mlflow server
    --host 0.0.0.0"  # BUG: this newline is a shell command separator!
```

## Docker Compose V2 .env Loading — CRITICAL

Docker Compose V2 looks for `.env` in the **compose file's directory** (e.g., `deployment/`),
NOT the working directory. The repo root `.env` is INVISIBLE to Docker Compose V2 by default.

**ALL `docker compose` invocations MUST use `--env-file /path/to/repo/.env`:**
```bash
# CORRECT — .env loaded from repo root:
docker compose --env-file /path/to/repo/.env -f deployment/docker-compose.flows.yml run train

# BROKEN — .env at repo root NOT loaded (Docker Compose looks in deployment/):
docker compose -f deployment/docker-compose.flows.yml run train
```

`scripts/run_debug.sh` handles this automatically via `ENV_FILE_ARG`. Any new script or
Makefile target that invokes `docker compose` MUST include `--env-file`.

**NEVER** tell users to `export HF_TOKEN=...` — all secrets belong in `.env`.

## Required .env Variables (No Default — Must Be Set)

```
MODEL_CACHE_HOST_PATH=/your/local/model/cache
```

`MODEL_CACHE_HOST_PATH` is required. It has **no fallback** in docker-compose.flows.yml
(removed 2026-03-09, was `/home/petteri/download_cache` — machine-specific = broken on other machines).
Copy `.env.example` → `.env` and set this to your local model weight cache directory.
The cache persists across container restarts, preventing re-downloading SAM3 (~9 GB) etc.

## One-Time Stack Setup

After first `docker compose up`, run these initialization steps:

```bash
# 1. Initialize named volume ownership (fixes "permission denied" in containers)
make init-volumes

# MinIO buckets are created automatically by the minio-init service.
# If minio-init fails, create buckets manually:
docker compose exec minio mc mb --ignore-existing minio/mlflow-artifacts
```

## Makefile Targets

```bash
make init-volumes   # Fix Docker named volume ownership (run once after first up)
make scan           # Trivy vulnerability scan on all minivess-* images
```

## Running Flows

```bash
# Start infrastructure first:
docker compose -f deployment/docker-compose.yml --profile dev up -d

# Run a flow:
docker compose -f deployment/docker-compose.flows.yml run --rm \
  -e EXPERIMENT=debug_single_model train

# With Hydra overrides:
docker compose -f deployment/docker-compose.flows.yml run --rm \
  -e EXPERIMENT=debug_single_model \
  -e HYDRA_OVERRIDES="max_epochs=5,model=sam3_vanilla" \
  train
```
