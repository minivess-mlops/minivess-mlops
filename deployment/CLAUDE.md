# Deployment — Docker Infrastructure

## Three-Layer Docker Hierarchy

```
Layer 1: nvidia/cuda:12.6.3-runtime-ubuntu24.04  (upstream, never modified)
Layer 2: minivess-base:latest                     (THIS — all shared deps)
Layer 3: Dockerfile.{flow}                        (thin — scripts, env, CMD only)
```

**Flow Dockerfiles NEVER run `apt-get` or `uv`** — all deps belong in Dockerfile.base.

## Building

Dockerfile.base uses a **multi-stage build** (H3/H4 hardening):
- **builder** stage: `nvidia/cuda:12.6.3-devel-ubuntu24.04` — compiles packages
- **runner** stage: `nvidia/cuda:12.6.3-runtime-ubuntu24.04` — ships only the runtime

The builder stage installs all Python deps into `/app/.venv` using BuildKit cache
mounts. The runner stage copies only the `.venv`, `src/`, and `configs/` directories —
no compiler toolchain, no uv binary, no build artifacts in production.

```bash
# Base image (rebuild when pyproject.toml or Dockerfile.base changes):
DOCKER_BUILDKIT=1 docker build -t minivess-base:latest -f deployment/docker/Dockerfile.base .

# Match host UID for bind-mount dev:
DOCKER_BUILDKIT=1 docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) \
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
make init-volumes        # Fix Docker named volume ownership (run once after first up)
make scan                # Trivy vulnerability scan on all minivess-* images (CRITICAL+HIGH)
make sbom                # Generate CycloneDX SBOM for minivess-base
make seccomp-audit-train # Run train flow with seccomp audit profile (syscall discovery)
make install-trivy       # Install Trivy scanner to /usr/local/bin
```

## Running Flows

**`--shm-size` is REQUIRED for GPU flows (train, hpo, hpo-worker)**

`docker compose run` ignores `shm_size` from the compose file — it must be passed
explicitly. Without it, MONAI 3D DataLoader uses /dev/shm for IPC and triggers a
Bus error (SIGBUS) on large batch sizes.

```bash
# Start infrastructure first:
docker compose -f deployment/docker-compose.yml --profile dev up -d

# Run a GPU flow (note --shm-size 8g):
docker compose --env-file .env -f deployment/docker-compose.flows.yml run --rm \
  --shm-size 8g \
  -e EXPERIMENT=debug_single_model train

# CPU flows (no --shm-size needed):
docker compose --env-file .env -f deployment/docker-compose.flows.yml run --rm \
  -e EXPERIMENT=debug_single_model data

# With Hydra overrides:
docker compose --env-file .env -f deployment/docker-compose.flows.yml run --rm \
  --shm-size 8g \
  -e EXPERIMENT=debug_single_model \
  -e HYDRA_OVERRIDES="max_epochs=5,model=sam3_vanilla" \
  train
```

## CVE-2025-23266 — NVIDIA CTK Version Check

CVE-2025-23266 (CVSS 9.0): container-to-host escape in NVIDIA Container Toolkit
versions < 1.17.8. Before running GPU containers, verify the installed version:

```bash
nvidia-ctk --version
# Expected: v1.17.8 or later
```

If the version is below 1.17.8, upgrade before running any GPU workloads:
```bash
# Ubuntu:
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
# Verify:
nvidia-ctk --version
```

## Image Scanning (MLSecOps)

```bash
make scan          # CRITICAL+HIGH CVE scan on all built images (--ignore-unfixed)
make sbom          # CycloneDX SBOM for minivess-base (supply-chain transparency)
make install-trivy # Install Trivy scanner if not present
```

## Seccomp Profiles

Per-flow seccomp allowlist profiles reduce the attack surface of containers by
restricting available syscalls to only those the flow actually uses.

See `deployment/seccomp/README.md` for the full workflow:
1. Run flow with `deployment/seccomp/audit.json` (SCMP_ACT_LOG) to discover syscalls
2. Extract syscall list from audit log
3. Build per-flow allowlist profile (SCMP_ACT_ERRNO for unlisted)
4. Apply with `--security-opt seccomp=deployment/seccomp/{flow}.json`

```bash
make seccomp-audit-train  # Run train flow with audit profile
```

## Network Isolation Strategy

All MinIVess services communicate over `minivess-network` (custom user-defined bridge,
pre-created with `docker network create minivess-network`). Flow containers reach infra
services by container name:

```
[train/hpo/analyze/post_training]
    ──► minivess-mlflow:5000   (MLflow experiment tracking + artifact registry)
    ──► minio:9000             (MinIO S3-compatible artifact store)
    ──► minivess-prefect:4200  (Prefect orchestration API)
    ──► postgres:5432          (PostgreSQL — Optuna HPO storage)

[dashboard/qa/biostatistics]
    ──► minivess-mlflow:5000   (read-only run queries)
    ──► minio:9000             (artifact download)

[annotation]
    ──► minivess-mlflow:5000
    ──► minio:9000

All services share: minivess-network (external bridge, created once)
```

### Why `icc: false` Is BANNED

Setting `icc: false` in `/etc/docker/daemon.json` (CIS Docker Benchmark 2.1) is
**INCOMPATIBLE** with this project and must never be set.

**Reason**: `icc: false` disables inter-container communication on the default bridge.
While it technically only affects the default bridge, some Docker Compose versions apply
daemon-level `icc: false` to all networks including `minivess-network`. This breaks:
- MLflow tracking URI → 403 / connection refused from flow containers
- MinIO artifact uploads → silent boto3 connection failures
- Prefect heartbeats → flow runs stall with no error message

**Correct isolation**: explicit `networks:` declarations per service (already in place).
Each service only joins the networks it actually needs.

## Future Hardening (Upstream Blocked)

### Rootless Docker — Blocked (#549)

Rootless Docker (daemon without root socket) eliminates the host root socket attack
vector entirely. It is blocked by an upstream NVIDIA Container Toolkit bug on Ubuntu 24.04:
- **Blocker**: [NVIDIA CTK issue #434](https://github.com/NVIDIA/nvidia-container-toolkit/issues/434)
  — CDI spec generator does not produce correct rootless specs on Ubuntu 24.04.
- **Current mitigation**: `cap_drop: ALL` + `no-new-privileges: true` + non-root container
  user provides strong isolation without rootless mode.
- **When to revisit**: When NVIDIA CTK releases rootless CDI support for Ubuntu 24.04.
- **DO NOT attempt workarounds** — they break GPU CDI access and waste GPU hours.

## Falco Runtime Security Monitoring (Optional)

Falco detects anomalous container behaviour in real-time via eBPF kernel tracing.

Activate:
```bash
docker compose --env-file .env -f deployment/docker-compose.yml --profile security up falco
```

Custom ML-specific rules: `deployment/falco/minivess_rules.yaml`

Alerts on:
- **Model weight exfiltration** — process connecting to remote while accessing `/app/checkpoints`
- **Shell spawn in flow container** — unexpected bash/sh in a Python-only flow (deserialization attack)
- **Unexpected outbound connections** — flow container reaches IPs outside the trusted service set

**NOTE: Falco requires `privileged: true`** for eBPF kernel module access.
All other MinIVess services remain unprivileged (enforced by `test_falco.py`).
