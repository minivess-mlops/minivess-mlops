# MLflow Version Mismatch: 2.20.3 (server) vs 3.10.0 (client) — Reproducibility Failure (2026-03-14)

## The Failure

Spent 8+ hours debugging MLflow artifact upload failures (910 MB .pth checkpoints
returning HTTP 500). Added `MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=true` env var —
still failed. Root cause: **the MLflow server was running v2.20.3 while the client
(Docker training image) had v3.10.0**. A 1.x major version gap.

## Why This Is Unacceptable

This repo's #2 priority (CLAUDE.md TOP-2) is **reproducibility**. The whole point
of Docker, pinned deps, and `uv.lock` is to eliminate "works on my machine."

Yet we deployed an MLflow server with a hardcoded `FROM ghcr.io/mlflow/mlflow:v2.20.3`
in the UpCloud Dockerfile — completely disconnected from `pyproject.toml` where
MLflow is pinned to `>=3.0.0` (which resolved to 3.10.0 in the lock file).

## How This Happened

1. **UpCloud Pulumi stack** was deployed weeks ago with MLflow 2.20.3
2. **pyproject.toml** was updated to `mlflow>=3.0.0` during v2 modernization
3. **Docker training image** (`Dockerfile.base`) installs from `uv.lock` → MLflow 3.10.0
4. **Nobody verified** that the remote MLflow server matched the client version
5. **The multipart upload feature** requires BOTH client AND server on 3.x
6. **8 hours of debugging** before discovering the version mismatch

## The Root Cause Pattern

**The remote MLflow server is NOT managed by the same dependency system as the
training code.** It has its own Dockerfile with its own hardcoded version.

This is the "two Dockerfiles, two truths" anti-pattern:
- `deployment/docker/Dockerfile.base` → `uv sync` → `uv.lock` → MLflow 3.10.0
- `deployment/pulumi/__main__.py` → `FROM ghcr.io/mlflow/mlflow:v2.20.3` → MLflow 2.20.3

## The Fix

### Immediate (Applied)

Changed `deployment/pulumi/__main__.py` Dockerfile from:
```dockerfile
FROM ghcr.io/mlflow/mlflow:v2.20.3
```
to:
```dockerfile
FROM ghcr.io/mlflow/mlflow:v3.10.0
```

### Proper Fix (TODO)

The MLflow server version MUST be derived from the SAME source as the client:

1. **Option A**: Read MLflow version from `uv.lock` and inject into Pulumi template
   ```python
   mlflow_version = _extract_mlflow_version_from_lock()
   dockerfile = f"FROM ghcr.io/mlflow/mlflow:v{mlflow_version}"
   ```

2. **Option B**: Build the MLflow server image from the SAME Dockerfile.base
   (overkill — the server doesn't need PyTorch/MONAI, just MLflow + psycopg2 + boto3)

3. **Option C**: Pin the MLflow server version in `.env.example` as a single source:
   ```
   MLFLOW_SERVER_VERSION=3.10.0
   ```
   Both `deployment/pulumi/__main__.py` and `deployment/docker-compose.yml` reference it.

## Rules Derived

1. **NEVER hardcode library versions in Dockerfiles that are also in pyproject.toml.**
   The version must come from ONE source: `pyproject.toml` → `uv.lock`.

2. **All service versions (MLflow, PostgreSQL, MinIO, Prefect) must be pinned in
   `.env.example`** and referenced by both Pulumi and Docker Compose.

3. **Version compatibility tests**: Add a test that verifies the MLflow server version
   matches the client version. Run in pre-flight checks.

4. **The MLFLOW_SERVER_VERSION must be in .env.example** — single source of truth.

## Cost of This Failure

- 8+ hours of debugging (MLflow artifact upload failures)
- 2 Lambda A100 GPU-hours wasted (~$3)
- 1 UpCloud VPS rebuild
- Massive frustration for a repo that claims to solve reproducibility
- Trust erosion in the infrastructure

## Lesson

> "If the version isn't pinned and verified end-to-end, it's not reproducible.
> 'It's just the MLflow server' is not an excuse — the server IS part of the pipeline."
