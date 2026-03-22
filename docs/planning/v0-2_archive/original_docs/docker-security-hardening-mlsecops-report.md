---
title: "Docker Security Hardening and MLSecOps Audit"
status: reference
created: "2026-03-09"
---

# Docker Security Hardening & MLSecOps: Total Quality Audit

**Date:** 2026-03-09
**Branch:** feat/docker-hardening
**Scope:** Combined SecOps + efficiency + correctness audit for all Docker images, Compose files, volume handling, and infrastructure patterns in the MinIVess MLOps Prefect + Pydantic AI architecture.
**Audience:** Platform engineers, PhD researchers deploying production ML pipelines.
**Seeds:** GitHub issues #527–531, 20+ web sources listed in References.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview and Current Posture](#2-architecture-overview-and-current-posture)
3. [Threat Landscape for ML Containers](#3-threat-landscape-for-ml-containers)
4. [Multi-Hypothesis Analysis: Base Image Strategy](#4-multi-hypothesis-analysis-base-image-strategy)
5. [Dockerfile Quality Audit](#5-dockerfile-quality-audit)
6. [Docker Compose Quality Audit](#6-docker-compose-quality-audit)
7. [Volume Ownership and Data Population Patterns](#7-volume-ownership-and-data-population-patterns)
8. [Network, Endpoint, and Port Security](#8-network-endpoint-and-port-security)
9. [Secrets Management](#9-secrets-management)
10. [Core Hardening Techniques](#10-core-hardening-techniques)
11. [MLSecOps-Specific Considerations](#11-mlsecops-specific-considerations)
12. [Compliance Frameworks](#12-compliance-frameworks)
13. [Vulnerability Scanning Tooling](#13-vulnerability-scanning-tooling)
14. [Conclusion and Recommendations](#14-conclusion-and-recommendations)
15. [Implementation Checklist](#15-implementation-checklist)
16. [References](#16-references)

---

## 1. Executive Summary

This report is a total quality audit of the MinIVess MLOps Docker practices, covering security hardening (SecOps/MLSecOps), efficiency, correctness, and operational reliability. It synthesizes findings from five GitHub issues (#527–531 documenting production failures), direct Dockerfile and Compose file analysis, and 20+ external sources including governmental guidance (NIST SP 800-190, DoD DISA), industry frameworks (CIS Docker Benchmark v1.8.0, OWASP Docker Top 10), NVIDIA security bulletins (CVE-2025-23266 "NVIDIAScape"), and the OpenSSF MLSecOps Whitepaper (2025).

**The three-layer hierarchy is sound and should be preserved:**
```
Layer 1: nvidia/cuda:12.6.3-runtime-ubuntu24.04  (upstream, never modified)
Layer 2: minivess-base:latest                     (all deps, non-root user)
Layer 3: minivess-{flow}:latest                   (scripts + CMD only)
```

**Key findings:**

| Category | Status | Priority Findings |
|----------|--------|------------------|
| Critical CVE | ⚠️ Unverified | NVIDIA Container Toolkit must be ≥ 1.17.8 (CVE-2025-23266, CVSS 9.0) |
| Dockerfile efficiency | ❌ Issues | Build tools (git, curl, pip) in runtime image; no multi-stage build |
| Compose correctness | ⚠️ Partial | MLflow installs pip deps at every startup; image tags unpinned |
| Volume ownership | ❌ Manual | One-time manual chown step; no automated init container |
| Capabilities | ❌ Missing | No `cap_drop: ALL`; no `no-new-privileges` |
| Resource limits | ❌ Missing | No mem/CPU/PID limits on any service |
| Seccomp profiles | ❌ Missing | Default Docker seccomp only |
| Secrets | ⚠️ Partial | `.env.example` pattern is correct; credentials still default-weak |
| Supply chain | ❌ Missing | No Trivy scanning; no SBOM |

**Recommended approach:** H3 (systematic hardening of official NVIDIA CUDA images) combined with H4 (multi-stage builds) + targeted infrastructure fixes. Zero additional licensing cost; all improvements are additive to the existing architecture.

---

## 2. Architecture Overview and Current Posture

### 2.1 Three-Layer Docker Hierarchy

The existing architecture correctly separates concerns across three layers. This hierarchy is sound and must not be collapsed:

```
Layer 1: nvidia/cuda:12.6.3-runtime-ubuntu24.04
          ↓ FROM
Layer 2: minivess-base (Dockerfile.base)
          — all Python deps via uv
          — non-root user minivess:1000:1000
          — PYTHONPATH, PATH
          ↓ FROM
Layer 3: minivess-{flow} (Dockerfile.{flow} × 12)
          — scripts/ copied
          — DOCKER_CONTAINER=1 env var
          — CMD pointing to flow module
```

The base image is rebuilt only when `pyproject.toml`, `uv.lock`, or `Dockerfile.base` changes (Layer 2 rebuild). Flow images are rebuilt when scripts or flow-level configs change (Layer 3 rebuild). This layering ensures pip installs are cached and not re-run for every script change.

### 2.2 Issues Surfaced by #527–531

| Issue | Root Cause | Status | Lesson |
|-------|-----------|--------|--------|
| #527 | MinIO host resolution: service name `minio` unresolvable from flow network | Fixed | Use `container_name` for cross-compose DNS |
| #528 | MinIO bucket `mlflow-artifacts` not auto-created | Partially fixed | Needs `init-minio-buckets.sh` init container |
| #529 | `data_cache` volume empty on first run | Documented | Needs `make populate-data` / acquisition flow |
| #530 | GPU CDI: `--runtime nvidia` confused with CDI | Fixed | CDI (`nvidia.com/gpu=all`) is the only supported pattern |
| #531 | `docker compose run` consuming loop stdin | Fixed | Always use `-T < /dev/null` in scripts |

### 2.3 Security Strengths (Existing)

| Control | Evidence |
|---------|----------|
| Non-root user | `USER minivess` (UID=1000), configurable via build ARG |
| No credentials in Dockerfiles | `.env.example` single-source-of-truth (Rule #22) |
| Volume mounts for all artifacts | No `/tmp` for artifacts (Rule #18) |
| Correct CDI GPU access | `devices: ["nvidia.com/gpu=all"]` in flows.yml |
| Per-flow container isolation | 12 containers, MLflow-only inter-flow communication |
| Healthchecks on infra services | PostgreSQL, MinIO, MLflow, Prefect |
| YAML list form for entrypoints | Avoids folded scalar newline parsing bug in MLflow entrypoint |

### 2.4 Security and Efficiency Gaps (This Audit)

| Gap | Severity | Category |
|-----|----------|----------|
| NVIDIA Container Toolkit not verified ≥ 1.17.8 | **Critical** | Security/CVE |
| Build tools in runtime image (git, curl, pip) | High | Efficiency/Security |
| No capability dropping (`cap_drop: ALL`) | High | Security |
| No resource limits (mem, CPU, PID) | High | Reliability |
| MLflow pip installs at every container startup | High | Efficiency |
| Unpinned image tags (`minio/minio:latest`) | High | Reproducibility |
| No Trivy/CVE scanning | High | Supply chain |
| No `security_opt: no-new-privileges` | Medium | Security |
| No seccomp profiles | Medium | Security |
| Port bindings on `0.0.0.0` | Medium | Security |
| Minimal `.dockerignore` | Medium | Security/Efficiency |
| Volume ownership via manual root container | Medium | Operational |
| No OCI image labels | Low | Provenance |
| No HEALTHCHECK in base Dockerfile | Low | Reliability |
| World-executable venv (`chmod a+rX`) | Low | Security |

---

## 3. Threat Landscape for ML Containers

### 3.1 Shared Kernel Risk

Docker containers share the host Linux kernel. The canonical example of why this matters: Dirty Cow (CVE-2016-5195), a kernel privilege escalation that bypassed all container-level hardening because it operated below the container abstraction (Appsecco, 2024). This precedent establishes that host kernel patching is non-optional, regardless of container hardening. **37% of cloud environments running GPU workloads were vulnerable to container escape in 2025**, primarily from NVIDIA Container Toolkit vulnerabilities (vCluster Research, 2025).

### 3.2 ML-Specific Attack Vectors

| Threat Vector | Attack Surface | Specific Risk |
|---------------|---------------|---------------|
| Data poisoning | Training data pipeline | Corrupted training data → degraded model |
| Model theft | Model registry, BentoML serving | IP theft, competitive harm |
| Supply chain | Base images, pip/uv deps | One CVE in parent cascades to all 12 flow images |
| GPU driver escape | NVIDIA CTK hooks | Full host root (CVE-2025-23266, CVSS 9.0) |
| Credential exfiltration | HF token, MinIO keys in env | Registry access, data access |
| Agentic AI misuse | Pydantic AI tool execution | Unintended host actions via LLM-driven agent |
| Training DoS | Unconstrained container resources | OOM kills co-located MLflow/Prefect |
| Exposed Docker daemon | TCP 2375/2376, Docker socket | Full host root if reachable externally |

### 3.3 Docker Daemon Exposure

Publicly exposed Docker daemon sockets have been weaponized for DDoS botnet recruitment (Dark Reading, 2025). The Docker socket at `/var/run/docker.sock` grants root-equivalent access when mounted into a container — a frequent CI/CD misconfiguration. Prefect workers that need to spawn flow containers require Docker socket access; this must be locked down to the worker container only, never exposed to flow containers.

---

## 4. Multi-Hypothesis Analysis: Base Image Strategy

### H1: Docker Hardened Images (DHI) — Free Tier

**What it is:** In December 2025, Docker made its 1,000+ Docker Hardened Images free and open-source (Apache 2.0). DHI implements: up to 95% CVE reduction, distroless runtime (no shell/apt/apk), non-root by default, SLSA Build Level 3 provenance, and SBOM per image (Docker Press Release, 2025; InfoQ, 2025).

**Compliance coverage:** CIS Docker Benchmark v1.8.0 Section 4 (images/Dockerfile). Sections 1–3 (host, daemon, network) require separate operator action.

**NVIDIA/CUDA availability:** The DHI free catalog is **Debian and Alpine only** — no Ubuntu, no CUDA. As of March 2026, no CUDA-specific images exist in the free DHI catalog. NVIDIA CUDA images remain at `hub.docker.com/r/nvidia/cuda` (maintained by NVIDIA separately).

| Pros | Cons |
|------|------|
| Zero cost, Apache 2.0 license | No CUDA images in free tier |
| SLSA L3 provenance + SBOM built-in | Debian/Alpine only (no Ubuntu 24.04) |
| Continuous CVE patching by Docker | Enterprise features (FIPS, STIG) require paid tier |
| Up to 95% attack surface reduction | No shell makes ML debugging harder |
| Drop-in for non-GPU flows | Distroless requires sidecar debug containers |

**Verdict:** Excellent for **non-GPU flows** (data, qa, dashboard, biostatistics, annotation). Not viable for GPU flows (train, hpo, post\_training, deploy). Recommed a split base strategy: DHI for non-GPU, NVIDIA CUDA runtime for GPU.

---

### H2: Chainguard CUDA-Optimized Images

**What it is:** Chainguard's Early Access Program provides CUDA-optimized images: `pytorch` (CUDA 12), `nemo`, `vllm-openai`, `nvidia-container-toolkit`, `nvidia-gpu-driver` (Chainguard, 2025). Base: Wolfi "undistro" (minimal, musl-based).

**CVE comparison:**

| Image | CVEs | Packages |
|-------|------|---------|
| Official `pytorch` runtime | 145 CVEs (2 High) | 268 |
| Chainguard `pytorch` | **0 CVEs** | 33 (88% reduction) |

| Pros | Cons |
|------|------|
| Zero CVEs (commercial tier) | CUDA images not in free tier — commercial pricing |
| 88% package count reduction | Wolfi/musl — potential glibc incompatibility with CUDA libs |
| SLSA provenance + cosign + SBOM | Early Access only, not GA |
| Patch SLA measured in hours | Different debugging workflow from Ubuntu |
| Directly addresses ML dep jungle | Vendor relationship required |

**Verdict:** Technically superior for production at scale. Not the right first step for a research lab — commercial pricing unknown, musl/glibc risk with CUDA needs validation. **Aspirational target** for production deployment once platform has secured funding.

---

### H3: Custom Hardening of Official NVIDIA CUDA Images *(Recommended Baseline)*

**What it is:** Apply a systematic hardening checklist to `nvidia/cuda:12.6.3-runtime-ubuntu24.04` — the current base. Additive to existing architecture.

| Pros | Cons |
|------|------|
| Zero licensing cost | NVIDIA base CVEs non-zero; needs Trivy scanning |
| Ubuntu 24.04 — full glibc compat | No SLSA L3 out-of-box (add via GH Actions) |
| GPU acceleration preserved | Manual hardening requires CI enforcement discipline |
| Immediate actionability | No automated patch SLA |
| CIS Benchmark L1/L2 achievable | Ongoing NVIDIA security bulletin monitoring required |

**Verdict:** **Recommended baseline** — immediately actionable, zero cost, GPU-compatible, Ubuntu 24.04 compatible with all current tooling.

---

### H4: Multi-Stage Distroless Builds *(Required Complement to H3)*

**What it is:** Separate build-time environment from runtime using Docker multi-stage builds (SEI CMU, 2025; Northcode, 2025).

```dockerfile
# Stage 1: Builder (full — compilers, uv, git, curl, devel CUDA headers)
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv git curl && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project
COPY src/ src/
COPY configs/ configs/

# Stage 2: Runner (minimal — no build tools, runtime CUDA only)
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS runner
ARG UID=1000
ARG GID=1000
RUN apt-get update && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd -g $GID minivess && useradd -u $UID -g $GID -m minivess
COPY --from=builder --chown=minivess:minivess /app/.venv /app/.venv
COPY --from=builder --chown=minivess:minivess /app/src /app/src
COPY --from=builder --chown=minivess:minivess /app/configs /app/configs
USER minivess
WORKDIR /app
ENV PATH=/app/.venv/bin:$PATH
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
```

| Pros | Cons |
|------|------|
| Removes uv, git, curl from runtime | Slightly more complex Dockerfile |
| devel CUDA headers confined to builder | Must enumerate all COPY --from targets |
| Smaller runtime image | Debug containers need separate Dockerfile.debug |
| Composable with H3 hardening | Layer cache invalidation needs careful management |
| Compatible with all runtime hardening | |

**Verdict:** **High-value, low-cost improvement.** Combine with H3 as the primary approach. The existing `Dockerfile.base` is 80% of the way there — the primary change is moving uv, git, and curl to a builder stage.

---

## 5. Dockerfile Quality Audit

### 5.1 Dockerfile.base — Issues and Improvements

**Issue 1: Build tools in runtime image (Security High / Efficiency High)**

The current `Dockerfile.base` installs `git`, `curl`, and `python3-pip` in the runtime stage. These are build-time tools that:
- Increase the attack surface (attacker with RCE can `git clone`, `curl` exfiltrate data)
- Inflate image size unnecessarily
- Violate the principle of minimal attack surface (SEI CMU, 2025; Appsecco, 2024)

```dockerfile
# Current (problematic):
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

# Recommended: move git/curl to builder stage; remove pip (uv replaces it)
# In builder stage:
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv git curl && rm -rf /var/lib/apt/lists/*

# In runner stage:
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv && rm -rf /var/lib/apt/lists/*
# Note: python3-pip NOT needed — uv manages the venv directly
```

**Issue 2: `uv` binary in runtime image**

`uv` is a build tool — it resolves and installs packages. Including it in the runtime image allows arbitrary package installation from within the container. Move `COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv` to the builder stage only.

**Issue 3: Overly permissive venv permissions (Security Low)**

```dockerfile
# Current — world-readable AND world-executable:
RUN chmod -R a+rX /home/minivess/.local 2>/dev/null || true && \
    chmod -R a+rX /app/.venv

# Recommended — group-readable only (group=minivess):
# The minivess group is sufficient; world-readable is not needed since
# the container runs as minivess user. This avoids exposing venv contents
# to any other UID that might be injected at runtime.
RUN chmod -R g+rX /app/.venv
```

The rationale for `a+rX` was bind-mount flexibility with arbitrary `--user X:Y`. A better solution is to pass `--build-arg UID=$(id -u)` and use the matching UID, which the Dockerfile already supports.

**Issue 4: Missing OCI image labels**

OCI standard image labels enable registry traceability, SBOM association, and audit trails (NIST SP 800-190, 2017):

```dockerfile
# Add to Dockerfile.base (values injected at build time via --build-arg):
ARG GIT_COMMIT=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="minivess-base"
LABEL org.opencontainers.image.source="https://github.com/minivess-mlops/minivess-mlops"
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.licenses="CC-BY-NC-SA-4.0"
```

Build with:
```bash
docker build \
  --build-arg GIT_COMMIT=$(git rev-parse HEAD) \
  --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
  -t minivess-base:latest \
  -f deployment/docker/Dockerfile.base .
```

**Issue 5: `umask 0002` in `.bashrc` only (Low)**

`umask 0002` in `.bashrc` applies only to interactive bash sessions. The CMD process (`python -m minivess...`) does not source `.bashrc` and runs with the default umask (0022 = owner-writable, group/world-read-only). If group-writable output files are needed (for volume sharing), set umask in the entrypoint script or via `ENV`:

```dockerfile
# Option A: entrypoint.sh wrapper that sets umask before exec
ENTRYPOINT ["sh", "-c", "umask 0002 && exec \"$@\"", "--"]
CMD ["python", "-m", "minivess.orchestration.flows.train_flow"]

# Option B: Not currently supported natively by ENV — use entrypoint.sh
```

**Issue 6: Missing HEALTHCHECK**

No `HEALTHCHECK` instruction in `Dockerfile.base` (CIS Benchmark 4.6, scored). Per-flow Dockerfiles also lack healthchecks. Docker Compose services that `depends_on` with `condition: service_healthy` require healthchecks to gate startup correctly:

```dockerfile
# Add to each per-flow Dockerfile:
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
  CMD python -c "import minivess; print('healthy')" || exit 1
```

**Issue 7: `uv.lock*` glob (Low)**

```dockerfile
COPY --chown=minivess:minivess pyproject.toml uv.lock* ./
```

The `*` glob means the COPY succeeds even if `uv.lock` is absent, silently installing without a lockfile. This risks non-reproducible builds. Use the exact filename:

```dockerfile
COPY --chown=minivess:minivess pyproject.toml uv.lock ./
```

### 5.2 Per-Flow Dockerfiles — Current Pattern

All 12 per-flow Dockerfiles follow the same minimal pattern:
```dockerfile
FROM minivess-base:latest
LABEL flow="train"
COPY --chown=minivess:minivess scripts/ scripts/
ENV DOCKER_CONTAINER=1
CMD ["python", "-m", "minivess.orchestration.flows.train_flow"]
```

This is architecturally correct — flow Dockerfiles must not run `apt-get` or `uv`. However:

1. **COPY scripts/ invalidates cache frequently** — `scripts/` contains many files; any change to any script rebuilds all 12 flow images even if only one flow's script changed. Consider per-flow script directories or using the base image as CMD source.

2. **No HEALTHCHECK in flow Dockerfiles** — as noted above.

3. **`ENV DOCKER_CONTAINER=1`** — this is the right pattern for the STOP protocol's `_require_docker_context()` guard.

### 5.3 `.dockerignore` — Current State

The current `.dockerignore` is minimal/empty. A comprehensive file prevents sensitive files from entering the build context (Appsecco, 2024), reducing both security exposure and build time:

```dockerignore
# Security: never include secrets or credentials
.env
*.env
secrets/
.aws/

# Git history (not needed in image, ~50MB+ in large repos)
.git
.gitignore
.gitattributes

# Python artifacts (rebuild inside Docker, not on host)
__pycache__/
*.pyc
*.pyo
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
dist/
build/

# ML artifacts (large, not needed for build, gitignored separately)
mlruns/
outputs/duckdb/
data/minivess/
data/raw/
outputs/checkpoints/
*.onnx

# Node / frontend (if any)
node_modules/

# Development IDE files
.vscode/
.idea/
*.swp
*.swo

# Temporary files
*.tmp
*.log
*.bak

# Documentation (not needed at runtime)
docs/
wiki/
*.md

# CI/CD (not needed in image)
.github/

# Testing (not in production image)
tests/
```

---

## 6. Docker Compose Quality Audit

### 6.1 MLflow: pip installs at every container startup (Efficiency High)

The current `docker-compose.yml` MLflow entrypoint:
```yaml
entrypoint:
  - /bin/sh
  - -c
  - >-
    pip install --quiet psycopg2-binary boto3 &&
    mlflow server ...
```

`pip install psycopg2-binary boto3` runs every time MLflow starts. This:
- Adds 15–30 seconds to every infrastructure startup
- Requires internet access on startup (breaks air-gapped deployments)
- Is non-reproducible (latest PyPI version of each package)
- Violates immutable container principle (OWASP D09, 2024)

**Recommended fix:** Create a custom MLflow image:

```dockerfile
# deployment/docker/Dockerfile.mlflow
FROM ghcr.io/mlflow/mlflow:v3.10.0
RUN pip install --no-cache-dir psycopg2-binary==2.9.10 boto3==1.34.162
```

```yaml
# docker-compose.yml — use custom image, no pip at startup
mlflow:
  build:
    context: .
    dockerfile: docker/Dockerfile.mlflow
  image: minivess-mlflow:v3.10.0
  # no entrypoint override needed for pip install
  command: >
    mlflow server
    --backend-store-uri postgresql://...
    --default-artifact-root s3://...
    --host 0.0.0.0
    --port 5000
```

### 6.2 Unpinned Image Tags (Reproducibility High)

Several services use unpinned or insufficiently pinned tags:

| Service | Current Tag | Problem | Recommended |
|---------|------------|---------|-------------|
| minio | `minio/minio:latest` | Non-reproducible; breaking changes | `minio/minio:RELEASE.2025-02-28T09-55-16Z` |
| bentoml | `bentoml/bento-server:latest` | Same | Pin to specific release |
| Chainguard/DHI | (if adopted) | | Pin by digest `@sha256:...` |

MLflow (`ghcr.io/mlflow/mlflow:v3.10.0`) is version-pinned. `postgres:16` is major-version pinned but still receives floating patch updates — for strict reproducibility use `postgres:16.X` (minor-pinned) or a digest pin.

### 6.3 Missing `depends_on` with Health Conditions in Flows

Flow containers should not start before infrastructure is healthy. The current `docker-compose.flows.yml` has no `depends_on` — if a flow starts before MLflow or PostgreSQL is ready, it will fail with connection errors.

```yaml
# docker-compose.flows.yml — add to each flow service:
services:
  train:
    depends_on:
      mlflow:
        condition: service_healthy
      postgres:
        condition: service_healthy
```

This requires adding `HEALTHCHECK` to the MLflow and Prefect services in `docker-compose.yml`, or referencing them from an external compose file. Note: `depends_on` across compose files requires explicit `external: true` services in the flows compose.

### 6.4 MinIO Bucket Auto-Creation (Issue #528)

As documented in issue #528, MinIO does not auto-create S3 buckets. The `mlflow-artifacts` bucket must be created before first use. The current workaround is manual; the correct solution is an init container:

```yaml
# docker-compose.yml — add MinIO init service
minio-init:
  image: minio/mc:latest
  container_name: minivess-minio-init
  profiles: ["dev", "monitoring", "full"]
  depends_on:
    minio:
      condition: service_healthy
  entrypoint: >
    /bin/sh -c "
    mc alias set local http://minivess-minio:9000
      $${MINIO_ROOT_USER} $${MINIO_ROOT_PASSWORD} &&
    mc mb --ignore-existing local/mlflow-artifacts &&
    mc mb --ignore-existing local/model-registry &&
    echo 'MinIO buckets initialized'
    "
  environment:
    MINIO_ROOT_USER: ${MINIO_ROOT_USER:-minioadmin}
    MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD:-minioadmin_secret}
  networks:
    - minivess
  restart: "no"
```

### 6.5 Cross-Compose Network Resolution (Issue #527 — Lesson)

The fix for issue #527 (using `minivess-minio` container name instead of `minio` service name in `MLFLOW_S3_ENDPOINT_URL`) reveals a general principle:

**Rule:** When communicating across Docker Compose projects, **always use `container_name`** (resolvable on any shared network), never the service name (only resolves within the same compose project's default network).

This is already correctly documented in `deployment/CLAUDE.md`. A corresponding check should be added to `scripts/pr_readiness_check.sh`: scan `docker-compose.flows.yml` for bare service names (`http://minio:`, `http://postgres:`) that should be container names.

### 6.6 `docker compose run` and stdin (Issue #531 — Lesson)

All `docker compose run` invocations in scripts must use:
```bash
docker compose ... run --rm -T ... < /dev/null
```
- `-T` disables TTY allocation (prevents stdin consumption from the calling script's loop)
- `< /dev/null` explicitly detaches container stdin (defense-in-depth for nested loops)

This is already documented in `deployment/CLAUDE.md` and fixed in `scripts/run_debug.sh`.

---

## 7. Volume Ownership and Data Population Patterns

### 7.1 Current Pattern: Manual Root Container

The current volume initialization requires:
```bash
docker run --rm --user root \
  -v deployment_checkpoint_cache:/app/checkpoints \
  -v deployment_logs_data:/app/logs \
  ... \
  ubuntu:22.04 \
  chown -R 1000:1000 /app/checkpoints /app/logs /app/mlruns ...
```

**Problems:**
- Manual step (forgotten on new machines)
- Root execution on host Docker daemon
- No idempotency check — runs every time unnecessarily
- Volume name prefix `deployment_` depends on compose project name (fragile)

**Alternative A: Compose init service (Recommended)**

```yaml
# docker-compose.yml
volume-init:
  image: ubuntu:22.04
  container_name: minivess-volume-init
  user: root
  command: >
    chown -R 1000:1000
    /app/checkpoints /app/logs /app/mlruns /app/data /app/configs/splits
  volumes:
    - checkpoint_cache:/app/checkpoints
    - logs_data:/app/logs
    - mlruns_data:/app/mlruns
    - data_cache:/app/data
    - configs_splits:/app/configs/splits
  restart: "no"  # run once only
```

Run with: `docker compose run --rm volume-init` — idempotent, documented, no manual steps.

**Alternative B: Entrypoint-based ownership fix**

If flow containers occasionally start before volume-init runs, use an entrypoint.sh that detects and fixes ownership:

```bash
#!/bin/sh
# entrypoint.sh — fix volume ownership if needed
if [ "$(stat -c %u /app/checkpoints)" != "$(id -u)" ]; then
  echo "WARNING: /app/checkpoints not owned by $(id -u), volume-init may not have run"
fi
exec "$@"
```

**Alternative C: Match host UID at build time**

Build the base image with `--build-arg UID=$(id -u) --build-arg GID=$(id -g)` — then the container user already matches the volume owner. This is the cleanest solution for single-developer setups and is already documented in the existing `Dockerfile.base`.

### 7.2 Data Population (Issue #529)

The `data_cache` volume must be populated before training. The long-term fix is the Acquisition flow (Flow 0) which downloads from DVC/EBRAINS automatically. Until then:

```makefile
# Makefile target:
populate-data:
	@echo "Populating data_cache volume from local data/..."
	docker run --rm --user root \
	  -v deployment_data_cache:/app/data \
	  -v $(PWD)/data:/src:ro \
	  ubuntu:22.04 \
	  sh -c "mkdir -p /app/data/raw && \
	         cp -r /src/raw/minivess /app/data/raw/ && \
	         chown -R 1000:1000 /app/data && \
	         echo 'Data populated'"
```

### 7.3 MODEL_CACHE_HOST_PATH Hardcoded Home Directory

```yaml
# Current:
- ${MODEL_CACHE_HOST_PATH:-/home/petteri/download_cache}:/model_cache
```

The fallback `/home/petteri/download_cache` hardcodes the developer's home directory. On CI or other machines this breaks silently (empty `/model_cache`). Set the default in `.env.example` to a project-relative path or explicitly document that `MODEL_CACHE_HOST_PATH` must always be set:

```yaml
# Recommended: fail loudly if not set (no fallback)
- ${MODEL_CACHE_HOST_PATH}:/model_cache
```

With `.env.example` documenting: `MODEL_CACHE_HOST_PATH=/path/to/your/download_cache`.

---

## 8. Network, Endpoint, and Port Security

### 8.1 Port Binding Scope

All services in `docker-compose.yml` bind to `0.0.0.0` (all interfaces):
```yaml
ports:
  - "${POSTGRES_PORT:-5432}:5432"    # accessible from LAN
  - "${MINIO_API_PORT:-9000}:9000"   # accessible from LAN
  - "${MLFLOW_PORT:-5000}:5000"      # accessible from LAN
  - "${PREFECT_PORT:-4200}:4200"     # accessible from LAN
```

For a workstation in a trusted intranet, this is acceptable but not ideal. For any cloud deployment or shared server, these ports must be bound to `127.0.0.1` only:

```yaml
# Development (localhost-only):
ports:
  - "127.0.0.1:${POSTGRES_PORT:-5432}:5432"
  - "127.0.0.1:${MINIO_API_PORT:-9000}:9000"
  - "127.0.0.1:${MLFLOW_PORT:-5000}:5000"

# Production: remove port bindings entirely; use reverse proxy (nginx/Traefik)
# with TLS termination. Services communicate only via minivess-network.
```

### 8.2 MLflow Allowed Hosts

```yaml
MLFLOW_SERVER_ALLOWED_HOSTS: "*"
```

This allows any HTTP Host header, making MLflow accessible from outside the Docker network if the port is bound to `0.0.0.0`. For intranet deployment, restrict to the Docker network CIDR or specific hostnames:

```yaml
# Development (bind to localhost):
MLFLOW_SERVER_ALLOWED_HOSTS: "127.0.0.1,minivess-mlflow,localhost"

# Docker-internal only (no port binding, accessed via minivess-network):
MLFLOW_SERVER_ALLOWED_HOSTS: "minivess-mlflow"
```

### 8.3 Inter-Service Communication: TLS vs. Plain HTTP

All internal service communication uses `http://`. For intranet/localhost-only deployments this is acceptable (traffic never leaves the host network interface). For production:

1. Add Traefik or nginx as a reverse proxy with Let's Encrypt TLS termination
2. Services communicate via HTTPS through the reverse proxy
3. No direct port exposure on `0.0.0.0`

This is a medium-term production concern. For the current research lab setup (single machine, localhost-only), plaintext internal HTTP is acceptable if ports are bound to `127.0.0.1`.

### 8.4 Docker Socket in Prefect Worker

The Prefect Docker worker needs access to `/var/run/docker.sock` to spawn flow containers. This must be treated with extreme care:

```yaml
# docker-compose.yml Prefect worker:
prefect-worker-docker:
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock  # CRITICAL: limit access
  # Never mount Docker socket in flow containers — only in the worker
```

**Mitigation:** Consider rootless Docker (daemon runs as non-root user) which makes socket exposure significantly less dangerous. Rootless Docker is available on Ubuntu 22.04+ and does not require `sudo` for daemon operations (Appsecco, 2024).

---

## 9. Secrets Management

### 9.1 Current Pattern Assessment

The `.env.example` + `--env-file .env` pattern is the correct approach for Docker Compose-based secrets management (Rule #22). This pattern correctly:
- Never bakes credentials into images
- Provides a documented template (`env.example`)
- Uses `--env-file` to inject at runtime (not `ENV` in Dockerfile)

**Gaps:**

| Secret | Current Risk | Mitigation |
|--------|-------------|------------|
| `MINIO_ROOT_PASSWORD=minioadmin_secret` | Default, guessable | Change before any network exposure |
| `LANGFUSE_SECRET=changeme-langfuse-secret` | Default, guessable | Rotate: `openssl rand -hex 32` |
| `LANGFUSE_SALT=changeme-langfuse-salt` | Default, guessable | Rotate: `openssl rand -hex 32` |
| `GRAFANA_PASSWORD=admin` | Default | Change before any network exposure |
| `HF_TOKEN` | Visible in `docker inspect` | Use Docker secrets when Swarm available |
| `POSTGRES_PASSWORD=minivess_secret` | Default, per-service | Rotate; use per-service credentials |

### 9.2 `docker inspect` Exposure

Any `environment:` variable in Docker Compose is visible via `docker inspect <container>` to any user with Docker daemon access. This includes `HF_TOKEN`, `MINIO_ROOT_PASSWORD`, and all database credentials. This is an inherent limitation of Docker Compose environment injection.

**Mitigation path (progressive):**
1. **Now:** Rotate all default credentials; restrict Docker daemon access to `docker` group only
2. **Short-term:** Use Docker Swarm secrets for `HF_TOKEN` (eliminates it from `docker inspect`)
3. **Medium-term:** HashiCorp Vault (Community Edition, free) with Vault Agent sidecar for all credentials

### 9.3 Principle of Least Privilege for MinIO

All flow containers currently share the root MinIO credentials (`MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`). Production best practice: create per-flow MinIO users with bucket-scoped policies:

```python
# init-minio.py — run once after minio-init container
from minio import Minio
client = Minio("minivess-minio:9000", ...)
# Create policy: train flow can only PUT to mlflow-artifacts/train-*
train_policy = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["s3:PutObject"],"Resource":["arn:aws:s3:::mlflow-artifacts/train-*"]}]}'
client.set_bucket_policy("mlflow-artifacts", train_policy)
```

---

## 10. Core Hardening Techniques

### 10.1 Capability Dropping

Linux capabilities divide root privilege into granular units. Default Docker containers start with ~14 capabilities, most of which are not needed for ML training (CIS Security, 2024). OWASP Docker Top 10 D04 mandates dropping all capabilities and adding back only those required (OWASP, 2024).

**Add to ALL services in `docker-compose.flows.yml`:**
```yaml
cap_drop:
  - ALL
security_opt:
  - no-new-privileges:true
```

**Capabilities required for ML training (none — all flows work with zero capabilities):**
PyTorch, MONAI, CUDA, Optuna, MLflow client, and Prefect client do not require any elevated Linux capabilities. The CUDA GPU access via CDI (`nvidia.com/gpu=all`) does not require `SYS_ADMIN` or `IPC_LOCK` — those are only needed for legacy GPU access methods.

### 10.2 Resource Limits

Unconstrained training containers can exhaust host RAM/CPU, taking down co-located MLflow, PostgreSQL, and Prefect — a DoS condition (OWASP D07, 2024). `--pids-limit` prevents fork bombs from PyTorch DataLoader workers (unconstrained `num_workers`).

**Recommended limits per flow type:**

```yaml
# In docker-compose.flows.yml, per service:
services:
  train:
    mem_limit: 32g
    memswap_limit: 32g
    cpus: 8.0
    pids_limit: 2048  # DataLoader workers need high PID count
    shm_size: 4g      # PyTorch DataLoader shared memory (default 64m is too small)

  hpo:
    mem_limit: 16g
    memswap_limit: 16g
    cpus: 4.0
    pids_limit: 1024

  data:
    mem_limit: 8g
    cpus: 4.0
    pids_limit: 512

  analyze:
    mem_limit: 16g
    cpus: 4.0
    pids_limit: 512

  dashboard:
    mem_limit: 4g
    cpus: 2.0
    pids_limit: 256

  qa:
    mem_limit: 4g
    cpus: 2.0
    pids_limit: 256
```

**Important:** `mem_limit` controls CPU RAM only — GPU VRAM limits require NVIDIA MIG or CUDA MPS. `shm_size: 4g` is critical for PyTorch DataLoader — the default 64 MB shared memory causes "Bus error" crashes with `num_workers > 0`.

### 10.2b CRITICAL: shm_size for PyTorch/MONAI DataLoader

**This is the most operationally critical missing setting.** Docker containers get `/dev/shm` as a tmpfs with a **64 MiB default**. PyTorch DataLoader workers use shared memory to pass tensor batches between workers and the main process. With `num_workers > 0`, each worker holds `batch_size × sample_bytes` in shared memory.

The error: `RuntimeError: DataLoader worker (pid X) is killed by signal: Bus error` — `SIGBUS` raised when a `mmap`-backed write to `/dev/shm` exceeds the 64 MiB limit (PyTorch issue #5040).

**Formula for 3D MONAI patches:**
```
required_shm = num_workers × batch_size × sample_bytes × 2   (×2 for double-buffering)

# MiniVess patch (256×256×128, float32): 256×256×128×4 = 32 MB
# With num_workers=4, batch=2: 4 × 2 × 32 MB × 2 = 512 MB  ← exceeds 64 MiB default
```

**Setting in `docker-compose.flows.yml`:**
```yaml
services:
  train:
    shm_size: "8g"    # 8 GB covers any realistic 3D MONAI workload
  hpo:
    shm_size: "8g"
```

**⚠️ CRITICAL: `docker compose run` does NOT inherit `shm_size` from the service definition.** Scripts must pass `--shm-size 8g` explicitly:
```bash
docker compose -f deployment/docker-compose.flows.yml run --rm -T \
  --shm-size 8g \
  train </dev/null
```

### 10.2c BuildKit Cache Mounts for uv (Build Speed)

Without BuildKit cache mounts, every `pyproject.toml` change re-downloads the entire PyTorch + MONAI stack (4–8 GB, 10–20 minutes). Cache mounts persist the uv download cache across builds without including it in the image layer.

**Add to `Dockerfile.base` (multi-stage builder stage):**
```dockerfile
# syntax=docker/dockerfile:1   ← required first line for BuildKit syntax

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_DEV=1

# Phase A: install deps only (cached when pyproject.toml/uv.lock unchanged)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

# Phase B: install project (only invalidated when src/ changes)
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev
```

`UV_COMPILE_BYTECODE=1`: pre-compiles `.py` → `.pyc` (faster container startup).
`UV_LINK_MODE=copy`: required when cache mount is on a different filesystem than the venv.
`--frozen`: treats `uv.lock` as immutable — fails if `pyproject.toml` and `uv.lock` diverge (equivalent to `npm ci`). Use instead of `--no-install-project` alone.

**⚠️ CI gotcha:** `--mount=type=cache` does not persist across ephemeral GitHub Actions runners. Cache mounts benefit local development rebuilds; CI always does a cold build.

### 10.3 Read-Only Root Filesystem

Implementing `read_only: true` prevents any process from writing to the container layer. All legitimate writes must happen on explicitly declared volumes (SEI CMU, 2025). Requires careful enumeration of writable paths.

```yaml
# Non-GPU flows (safe to make fully read-only):
services:
  data:
    read_only: true
    volumes:
      - raw_data:/app/data/raw:ro
      - data_cache:/app/data/processed    # writable
      - configs_splits:/app/configs/splits  # writable
      - mlruns_data:/app/mlruns            # writable
    tmpfs:
      - /tmp:size=256m,noexec,nosuid       # process temp

  dashboard:
    read_only: true
    volumes:
      - mlruns_data:/app/mlruns:ro
      - logs_data:/app/logs:ro
      - output_data:/app/outputs           # writable for figures
    tmpfs:
      - /tmp:size=256m,noexec,nosuid
```

**GPU flows (train, hpo):** Read-only filesystem is compatible with CUDA if the following tmpfs mounts are declared:
- `/tmp` — CUDA kernel compilation cache
- `~/.cache` or `${TORCH_CACHE_DIR}` — PyTorch compilation cache (can be a volume instead)

### 10.4 Seccomp Profiles

Seccomp restricts which kernel syscalls a process can invoke (Kubernetes Docs, 2024). The recommended workflow:

```bash
# Step 1: Audit profile (log all, block nothing)
cat > profiles/audit.json << 'EOF'
{"defaultAction": "SCMP_ACT_LOG"}
EOF

# Step 2: Run each flow with audit profile to discover syscalls
docker compose -f deployment/docker-compose.flows.yml run --rm \
  --security-opt seccomp=profiles/audit.json \
  train

# Step 3: Extract syscalls from audit log
sudo grep "SYSCALL" /var/log/audit/audit.log | \
  awk '{for(i=1;i<=NF;i++) if($i~/^syscall=/) print $i}' | \
  sort -u > observed_syscalls_train.txt

# Step 4: Build allowlist profile (see CUDA syscall list below)
# Step 5: Test and iterate
```

**CUDA-required syscalls to include in allowlist:**
```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": ["SCMP_ARCH_X86_64"],
  "syscalls": [{
    "names": [
      "read", "write", "open", "openat", "openat2", "close",
      "stat", "fstat", "lstat", "lseek", "mmap", "mprotect",
      "munmap", "brk", "rt_sigaction", "rt_sigprocmask",
      "ioctl", "pread64", "pwrite64", "readv", "writev",
      "access", "pipe", "select", "sched_yield", "mremap",
      "msync", "madvise", "shmget", "shmat", "shmctl",
      "dup", "dup2", "pause", "nanosleep", "getitimer",
      "alarm", "setitimer", "getpid", "socket", "connect",
      "accept", "sendto", "recvfrom", "sendmsg", "recvmsg",
      "bind", "listen", "getsockname", "getpeername",
      "socketpair", "setsockopt", "getsockopt",
      "clone", "fork", "vfork", "execve", "exit", "wait4",
      "kill", "uname", "fcntl", "flock", "fsync",
      "getcwd", "chdir", "mkdir", "rmdir", "creat",
      "link", "unlink", "rename", "chmod", "fchmod",
      "getuid", "getgid", "geteuid", "getegid",
      "getppid", "getpgrp", "setsid", "setuid", "setgid",
      "futex", "sched_getaffinity", "sched_setaffinity",
      "epoll_create", "epoll_create1", "epoll_ctl", "epoll_wait",
      "epoll_pwait", "clock_gettime", "clock_getres",
      "clock_nanosleep", "exit_group", "tgkill",
      "openat", "mkdtemp", "prlimit64",
      "perf_event_open",
      "mlock", "munlock",
      "process_vm_readv", "process_vm_writev",
      "getrandom", "memfd_create", "statx"
    ],
    "action": "SCMP_ACT_ALLOW"
  }]
}
```

**Critical:** Seccomp is silently ignored when `privileged: true` is set. Never use `--privileged` for GPU access — use CDI (`nvidia.com/gpu=all`) as already implemented.

### 10.5 Docker Daemon Hardening

Independent of container-level hardening, the Docker daemon requires its own configuration (CIS Benchmark Sections 1–3):

```json
{
  "userns-remap": "default",
  "log-driver": "local",
  "log-opts": {
    "max-size": "10m",
    "max-file": "5"
  },
  "live-restore": true,
  "userland-proxy": false,
  "no-new-privileges": true
}
```

| Setting | Purpose | CIS Ref |
|---------|---------|---------|
| `userns-remap: "default"` | Remaps container root to host UID 65534 | 2.8 |
| `no-new-privileges: true` | Global default for all containers | 2.17 |
| Log rotation (local driver) | Prevents log-based disk exhaustion | 2.12 |
| `live-restore: true` | Containers survive daemon restart | 2.13 |

**⚠️ OMITTED: `icc: false`** — CIS 2.1 recommends disabling inter-container communication at the daemon level. However, this project requires extensive inter-container communication (MLflow ↔ MinIO, flows ↔ PostgreSQL, flows ↔ MLflow, etc.) all routed via `minivess-network`. Setting `icc: false` at the daemon level would break the entire stack without adding explicit Docker network policies for every service pair. For this architecture, network segmentation is enforced at the Docker network level (`minivess-network`) rather than the daemon level. Do not add `icc: false` to `daemon.json` without also configuring explicit `--link` or network policy rules.

**⚠️ NOTE on `userns-remap`:** With user namespace remapping active, the container's UID 1000 maps to a high host UID (e.g., 165536). Named Docker volumes owned by UID 1000 inside the container are owned by UID 165536 on the host. The existing `chown -R 1000:1000` volume-init commands and `--build-arg UID=$(id -u)` workaround both interact with this in non-obvious ways. Test thoroughly on dev machine before enabling in production.

---

## 11. MLSecOps-Specific Considerations

### 11.1 CVE-2025-23266: NVIDIAScape — Critical Container Escape

**This is the most important immediate security action.**

| Field | Details |
|-------|---------|
| CVE ID | CVE-2025-23266 |
| CVSS Score | **9.0 (Critical)** |
| Vector | AV:L/AC:L/PR:L/UI:N/S:C/C:H/I:H/A:H |
| Disclosed | July 17, 2025 |
| Discoverer | Wiz Research (2025) |
| Affected | NVIDIA Container Toolkit ≤ 1.17.7; GPU Operator ≤ 25.3.1 |
| Fixed | Container Toolkit v1.17.8; GPU Operator v25.3.2 |

**Attack:** The `enable-cuda-compat` OCI hook inherits container env vars. A malicious image sets `LD_PRELOAD` to a crafted `.so`. When the privileged `nvidia-ctk` hook executes with host filesystem access, it loads the rogue library, granting host root (Wiz Research, 2025; NVIDIA PSB, 2025).

```bash
# Verify immediately:
nvidia-ctk --version  # Must be >= 1.17.8

# Emergency mitigation if upgrade not possible:
sudo tee -a /etc/nvidia-container-toolkit/config.toml << 'EOF'
[features]
disable-cuda-compat-lib-hook = true
EOF
```

**CVE-2025-23267 (CVSS 8.5):** Companion vulnerability — symlink following in `update-ldcache` hook, data tampering and DoS. Fixed in same versions (NVIDIA PSB, 2025).

### 11.2 Model Supply Chain Security

| Risk | Mitigation |
|------|-----------|
| HuggingFace weight provenance | Use `hf_hub.snapshot_download()` with hash verification; prefer official org repos |
| VesselFM data leakage | Already documented as test-set exclusion; verify VesselFM repo is from official source |
| ONNX model signing | `cosign sign` at export time; verify at BentoML serving load |
| DVC data integrity | DVC SHA256 checksums already tracked; add verification step in Data Engineering flow |

### 11.3 Pydantic AI Agent Security

| Concern | Mitigation |
|---------|-----------|
| Unrestricted tool access | Explicit `tools` allowlist in every Pydantic AI agent |
| Docker socket in agent container | Never mount `/var/run/docker.sock` in flow containers |
| LLM rate runaway | LiteLLM proxy per-agent rate limits |
| Audit gaps | Langfuse traces all agent actions with tool inputs/outputs |
| Agent-generated code execution | Route through Prefect task functions with STOP protocol |

### 11.4 OpenSSF MLSecOps Framework

The OpenSSF MLSecOps Whitepaper (OpenSSF, 2025) defines a pipeline security framework:

| OpenSSF Control | MinIVess Implementation |
|----------------|------------------------|
| SLSA Build L1 | Add `--attest type=provenance` to `docker build` |
| SLSA Build L2 | GitHub Actions `docker/attest-build-provenance` (manual dispatch, respects Rule #21) |
| Sigstore/cosign | `cosign sign` on release images; verify at deployment |
| OpenSSF Scorecard | Monthly manual workflow dispatch |
| Signed data artifacts | DVC + GPG signing for training data manifests |
| Model registry signing | `cosign sign` on ONNX models pushed to BentoML store |

---

## 12. Compliance Frameworks

### 12.1 NIST SP 800-190

Key mandates (NIST, 2017):
- Scan images for known CVEs; patch promptly ❌ (not yet)
- Principle of least privilege ⚠️ (non-root done; capabilities not dropped)
- Separate workloads by sensitivity ✅ (per-flow isolation)
- Deploy as immutable instances ✅ (images rebuilt, not updated in place)
- Monitor for anomalous runtime behavior ❌ (no IDS)
- Isolate container network traffic ✅ (minivess-network)

### 12.2 CIS Docker Benchmark v1.8.0

**Tool:** `docker/docker-bench-security` (open-source, Apache 2.0).

| Section | Scope | Status |
|---------|-------|--------|
| 1. Host Configuration | Docker host OS | Not audited |
| 2. Docker Daemon | dockerd config | Not hardened |
| 3. Docker Daemon Logging | Log config | Not configured |
| 4. Container Images (Dockerfile) | Image controls | Partially compliant |
| 5. Container Runtime | Runtime flags | Not configured |
| 6. Docker Security Ops | Ongoing processes | Not formalized |

### 12.3 OWASP Docker Top 10

| # | Control | MinIVess Status |
|---|---------|----------------|
| D01 | Secure User Mapping | ✅ Non-root implemented |
| D02 | Patch and Update Images | ❌ No automated rebuild on CVE |
| D03 | Network Segmentation | ✅ External Docker network |
| D04 | Secure Defaults | ❌ No capability dropping |
| D05 | Security Contexts | ❌ No seccomp/AppArmor |
| D06 | Protect Secrets | ⚠️ `.env.example` good; vault not used |
| D07 | Resource Limits | ❌ None configured |
| D08 | Share Host Resources | ✅ No Docker socket in flows |
| D09 | Immutable Containers | ✅ Rebuild, not update |
| D10 | Logging | ⚠️ Grafana/Prometheus; no external SIEM |

---

## 13. Vulnerability Scanning Tooling

### 13.1 Trivy (Recommended — Free, Zero Infrastructure)

Trivy (Aqua Security) is a single-binary scanner requiring no server (Trivy Docs, 2024). Best fit for research lab CI/CD gates.

```bash
# Install
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

# Scan NVIDIA base (discover baseline CVEs)
trivy image nvidia/cuda:12.6.3-runtime-ubuntu24.04

# CI gate: fail on Critical CVEs only (High produces too many false positives initially)
trivy image --exit-code 1 --severity CRITICAL --ignore-unfixed \
  minivess-base:latest

# SBOM generation (per release)
trivy image --format spdx-json --output sbom/minivess-base-sbom.json \
  minivess-base:latest

# just target:
scan:
    @for flow in base train data analyze deploy dashboard qa hpo post_training; do \
        echo "Scanning minivess-$$flow..."; \
        trivy image --exit-code 0 --severity CRITICAL,HIGH minivess-$$flow:latest; \
    done
```

### 13.2 Docker Scout (Free — 1 Repo Continuous Tracking)

| Plan | Repos | Notes |
|------|-------|-------|
| Docker Personal (free) | 1 repo | Continuous cloud tracking |
| Local analysis | Unlimited | On-demand, no tracking |

```bash
docker scout cves minivess-base:latest     # Local, free, unlimited
docker scout quickview minivess-base:latest
```

Sufficient for a research lab using one primary training image in continuous tracking mode.

### 13.3 Docker Bench for Security

CIS Benchmark audit tool — run periodically on host:

```bash
git clone https://github.com/docker/docker-bench-security.git
cd docker-bench-security
sudo sh docker-bench-security.sh 2>&1 | tee cis-audit-$(date +%Y%m%d).txt
```

### 13.4 Comparison Matrix

| Tool | Cost | Infrastructure | CVE DB | SBOM | CI | Best For |
|------|------|---------------|--------|------|-----|----------|
| Trivy | Free | None | NVD + OS | ✅ | Excellent | CI gates, SBOM |
| Docker Scout | Free (1 repo) | Cloud | Docker | Limited | Good | Quick checks |
| Docker Bench | Free | Host | N/A | ❌ | Poor | CIS audit |
| Grype | Free | None | NVD + OS | ✅ | Good | Trivy alternative |
| Chainguard | Paid | Managed | Proprietary | ✅ | Excellent | Zero-CVE commercial |

---

## 14. Conclusion and Recommendations

### 14.1 Recommended Architecture: H3 + H4 Combined

**For GPU flows (train, hpo, post\_training, deploy):**
- Base: `nvidia/cuda:12.6.3-runtime-ubuntu24.04` (Layer 1)
- Builder stage: `nvidia/cuda:12.6.3-devel-ubuntu24.04` (multi-stage H4)
- Runtime stage: `nvidia/cuda:12.6.3-runtime-ubuntu24.04` without build tools (H3 + H4)
- Systematic hardening: cap\_drop, no-new-privileges, resource limits, seccomp

**For non-GPU flows (data, dashboard, qa, biostatistics, annotation):**
- Consider DHI (H1) or Chainguard Wolfi base as an upgrade path once the GPU flows are stabilized
- Short-term: same NVIDIA CUDA base (inert without GPU device reservation) — simpler to maintain a single base

**Why not H2 (Chainguard CUDA) now?** Commercial pricing unknown; musl/glibc compatibility with PyTorch + MONAI CUDA stack unvalidated; Early Access only. Re-evaluate in 6 months.

### 14.2 Phased Implementation Plan

**Phase 1 — Critical Safety (Day 1)**

1. **Verify NVIDIA Container Toolkit ≥ 1.17.8:**
   ```bash
   nvidia-ctk --version
   ```
   If below 1.17.8, update before any training run.

2. **Rotate all default credentials** in `.env` (not `.env.example`):
   ```bash
   LANGFUSE_SECRET=$(openssl rand -hex 32)
   LANGFUSE_SALT=$(openssl rand -hex 32)
   MINIO_ROOT_PASSWORD=$(openssl rand -base64 24)
   GRAFANA_PASSWORD=$(openssl rand -base64 16)
   ```

3. **Add comprehensive `.dockerignore`** (see Section 5.3) — prevents `.env` from entering build context.

**Phase 2 — Dockerfile and Compose Fixes (Sprint 1, < 1 week)**

4. **Multi-stage `Dockerfile.base`:** Move uv, git, curl, pip to builder stage; runtime stage has only `python3 python3-venv`.

5. **Fix uv.lock glob:** `COPY pyproject.toml uv.lock ./` (remove `*`).

6. **Add OCI labels** to `Dockerfile.base` with build-arg injection.

7. **Custom MLflow image** (`Dockerfile.mlflow`): bake psycopg2-binary + boto3 in at build time; remove pip install from compose entrypoint.

8. **Pin MinIO image tag:** Replace `minio/minio:latest` with specific release.

9. **Add `cap_drop: [ALL]` and `security_opt: [no-new-privileges:true]`** to all flow services in `docker-compose.flows.yml`.

10. **Add resource limits** (mem\_limit, cpus, pids\_limit, shm\_size) to all flow services.

11. **Add `minio-init` service** for automatic bucket creation (fixes #528 long-term).

12. **Add `volume-init` compose service** or document `make init-volumes` target (fixes #529 pattern).

**Phase 3 — Defense in Depth (Sprint 2, < 1 month)**

13. **Seccomp profiles per flow:** Run audit profiles → build per-flow allowlists → deploy.

14. **Read-only root filesystem** for non-GPU flows with explicit tmpfs declarations.

15. **Restrict port bindings** to `127.0.0.1:` on all compose services.

16. **Restrict `MLFLOW_SERVER_ALLOWED_HOSTS`** to specific hostnames.

17. **Add `HEALTHCHECK`** to per-flow Dockerfiles for `depends_on: condition: service_healthy`.

18. **Add `depends_on` with health conditions** in `docker-compose.flows.yml`.

19. **Fix `MODEL_CACHE_HOST_PATH` fallback** — remove hardcoded `/home/petteri/` path.

20. **Run `docker/docker-bench-security`** → document findings → create issues for gaps.

**Phase 4 — MLSecOps Maturity (Ongoing)**

21. **Trivy scan `just scan` target** — scan all images before deployment.

22. **SBOM generation** at image build time; store alongside MLflow run artifacts.

23. **cosign image signing** for releases (manual GitHub Actions dispatch, respects Rule #21).

24. **Model weight hash verification** in Data Engineering flow (DVC check).

25. **Per-flow MinIO credentials** (scoped bucket policies instead of root user).

26. **Evaluate Chainguard CUDA/PyTorch** images when production deployment budget is available.

### 14.3 GPU-Specific Security Summary

| Control | Status | Action |
|---------|--------|--------|
| CDI GPU access (no `--privileged`) | ✅ Correct | Maintain |
| NVIDIA CTK ≥ 1.17.8 | ⚠️ Unverified | **Verify immediately** |
| No seccomp for GPU flows | ❌ | Profile and deploy per-flow allowlist |
| GPU VRAM limits | ❌ (not possible via compose) | Use NVIDIA MIG for multi-tenant |
| CUDA cache writable volume | ⚠️ | Declare explicit tmpfs or volume |

### 14.4 Prefect + Pydantic AI Architecture Summary

| Component | Security Concern | Recommendation |
|-----------|-----------------|----------------|
| Prefect server (port 4200) | Unauthenticated API | Bind to `127.0.0.1`; add Prefect API key auth in production |
| Prefect Docker worker | Docker socket access | Limit to worker only; consider rootless Docker |
| Pydantic AI agents | Tool execution scope | Explicit `tools` allowlist; no Docker socket |
| MLflow server | `ALLOWED_HOSTS: "*"` | Restrict to minivess-network hostnames |
| PostgreSQL | Shared credentials across services | Move to per-service credentials |
| MinIO | Root credentials shared | Implement scoped bucket policies |
| Langfuse | Agent traces with LLM I/O | Ensure network-isolated; rotate salt/secret |
| BentoML serving | Unauthenticated inference endpoint | Add API key auth for production; output rate limiting |

---

## 15. Implementation Checklist

### Day 1 (Critical)
- [ ] `nvidia-ctk --version` → must be ≥ 1.17.8
- [ ] Rotate: LANGFUSE\_SECRET, LANGFUSE\_SALT, MINIO\_ROOT\_PASSWORD, GRAFANA\_PASSWORD
- [ ] Add comprehensive `.dockerignore` (Section 5.3)

### Sprint 1 (< 1 week)
- [ ] Convert `Dockerfile.base` to multi-stage build (builder: devel, runner: runtime)
- [ ] Remove git, curl, pip from runtime stage
- [ ] Fix `uv.lock*` → `uv.lock`
- [ ] Add OCI labels with build-arg injection
- [ ] Create `Dockerfile.mlflow` with baked-in psycopg2-binary + boto3
- [ ] Pin MinIO image to specific release tag
- [ ] Add `cap_drop: [ALL]` + `security_opt: [no-new-privileges:true]` to all flows
- [ ] Add mem\_limit, cpus, pids\_limit, shm\_size to all flows
- [ ] Add `minio-init` compose service for bucket auto-creation
- [ ] Document `make init-volumes` or add volume-init compose service

### Sprint 2 (< 1 month)
- [ ] Generate seccomp audit profiles for each flow
- [ ] Implement per-flow seccomp allowlists
- [ ] Add `read_only: true` to non-GPU flows
- [ ] Restrict port bindings to `127.0.0.1:`
- [ ] Restrict `MLFLOW_SERVER_ALLOWED_HOSTS`
- [ ] Add `HEALTHCHECK` to per-flow Dockerfiles
- [ ] Add `depends_on` with health conditions in flows compose
- [ ] Fix `MODEL_CACHE_HOST_PATH` fallback
- [ ] Run docker-bench-security → document → create issues
- [ ] Add `just scan` Trivy target

### Production readiness (ongoing)
- [ ] SBOM generation at build time
- [ ] cosign image signing (manual dispatch)
- [ ] Model weight hash verification in Data Engineering flow
- [ ] Per-flow MinIO bucket policies
- [ ] Evaluate Chainguard CUDA images
- [ ] Rootless Docker daemon evaluation

---

## 16. References

Appsecco (2024). *Top 10 Docker Hardening Best Practices*. Appsecco Blog. Retrieved 2026-03-09 from https://appsecco.com/blog/top-10-docker-hardening-best-practices

Aqua Security (2024). *Trivy: Comprehensive and versatile security scanner*. Trivy Documentation. Retrieved 2026-03-09 from https://trivy.dev/

Chainguard (2025). *Announcing Early Access to Chainguard's CUDA Optimized Images*. Chainguard Blog. Retrieved 2026-03-09 from https://www.chainguard.dev/unchained/announcing-early-access-to-chainguards-cuda-optimized-images

Chainguard (2025). *Chainguard PyTorch Image Overview*. Chainguard Images Directory. Retrieved 2026-03-09 from https://images.chainguard.dev/directory/image/pytorch/overview

CIS Security (2024). *CIS Docker Benchmark v1.8.0*. Center for Internet Security. Retrieved 2026-03-09 from https://www.cisecurity.org/benchmark/docker

Cloud Security Alliance (CSA) (2025). *The Hidden Security Threats Lurking in Your Machine Learning Pipeline*. CSA Blog, September 11, 2025. Retrieved 2026-03-09 from https://cloudsecurityalliance.org/blog/2025/09/11/the-hidden-security-threats-lurking-in-your-machine-learning-pipeline

Dark Reading (2025). *Exposed Docker Daemons Fuel DDoS Botnet*. Dark Reading. Retrieved 2026-03-09 from https://www.darkreading.com/cyber-risk/exposed-docker-daemons-fuel-ddos-botnet

dev.to (2024). *7 Tips for Docker Security Hardening on Production Servers*. dev.to. Retrieved 2026-03-09 from https://dev.to/ramer2b58cbe46bc8/7-tips-for-docker-security-hardening-on-production-servers-21kl

DISA/DoD CyberCom (2021). *DevSecOps Enterprise Container Hardening Guide v1.2*. Department of Defense. Retrieved 2026-03-09 from https://dl.dod.cyber.mil/wp-content/uploads/devsecops/pdf/Final_DevSecOps_Enterprise_Container_Hardening_Guide_1.2.pdf

Docker (2024). *Docker Container Security Overview*. Docker Documentation. Retrieved 2026-03-09 from https://docs.docker.com/engine/security/

Docker (2024). *Docker Hardened Images: CIS Docker Benchmark Core Concepts*. Docker Documentation. Retrieved 2026-03-09 from https://docs.docker.com/dhi/core-concepts/cis/

Docker (2024). *Docker Bench for Security*. GitHub. Retrieved 2026-03-09 from https://github.com/docker/docker-bench-security

Docker Press Release (2025). *Docker Makes Hardened Images Free, Open, and Transparent for Everyone*, December 22, 2025. Retrieved 2026-03-09 from https://www.docker.com/press-release/docker-hardened-images-free-open-and-transparent-for-everyone/

Docker (2026). *Hardened Images Are Free. Now What?* Docker Blog (CISO Mark Lechner), February 10, 2026. Retrieved 2026-03-09 from https://www.docker.com/blog/hardened-images-free-now-what/

InfoQ (2025). *Docker Makes Hardened Images Free in Container Security Shift*, December 2025. Retrieved 2026-03-09 from https://www.infoq.com/news/2025/12/docker-hardened-images/

Kubernetes Docs (2024). *Restrict a Container's Syscalls with seccomp*. Kubernetes Documentation. Retrieved 2026-03-09 from https://kubernetes.io/docs/tutorials/security/seccomp/

minivess-mlops/minivess-mlops (2026). *GitHub Issue #527: fix(minio): MinIO service name unreachable from flow containers*. Retrieved 2026-03-09.

minivess-mlops/minivess-mlops (2026). *GitHub Issue #528: fix(minio): mlflow-artifacts bucket not auto-created*. Retrieved 2026-03-09.

minivess-mlops/minivess-mlops (2026). *GitHub Issue #529: fix(docker): data\_cache volume empty*. Retrieved 2026-03-09.

minivess-mlops/minivess-mlops (2026). *GitHub Issue #530: fix(docker): GPU CDI access broken in run\_debug.sh*. Retrieved 2026-03-09.

minivess-mlops/minivess-mlops (2026). *GitHub Issue #531: fix(docker): docker compose run consumes while-loop stdin*. Retrieved 2026-03-09.

NIST (2017). *SP 800-190: Application Container Security Guide*. National Institute of Standards and Technology. Retrieved 2026-03-09 from https://nvlpubs.nist.gov/nistpubs/specialpublications/nist.sp.800-190.pdf

Northcode (2025). *Docker Images: Best Practices for Production*. Northcode Blog. Retrieved 2026-03-09 from https://www.northcode.fi/article/dockerimages

NVIDIA PSB (2025). *Security Bulletin: NVIDIA Container Toolkit — July 2025*. NVIDIA Product Security, July 17, 2025. Retrieved 2026-03-09 from https://nvidia.custhelp.com/app/answers/detail/a_id/5659/~/security-bulletin:-nvidia-container-toolkit---july-2025

OpenSSF (2025). *Visualizing Secure MLOps (MLSecOps): A Practical Guide for Building Robust AI/ML Pipeline Security*. OpenSSF Whitepaper, August 5, 2025. Dell Technologies and Ericsson. Retrieved 2026-03-09 from https://openssf.org/blog/2025/08/05/visualizing-secure-mlops-mlsecops-a-practical-guide-for-building-robust-ai-ml-pipeline-security/

OWASP (2024). *Docker Top 10*. OWASP Foundation. Retrieved 2026-03-09 from https://owasp.org/www-project-docker-top-10/

OWASP (2024). *Docker Security Cheat Sheet*. OWASP Cheat Sheet Series. Retrieved 2026-03-09 from https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html

SEI CMU (2025). *An Introduction to Hardening Docker Images*. Software Engineering Institute, Carnegie Mellon University Blog. Retrieved 2026-03-09 from https://www.sei.cmu.edu/blog/an-introduction-to-hardening-docker-images/

SentinelOne (2024). *Docker Container Security Best Practices*. SentinelOne Cybersecurity 101. Retrieved 2026-03-09 from https://www.sentinelone.com/cybersecurity-101/cloud-security/docker-container-security-best-practices/

SentinelOne (2024). *Docker Container Security Scanners*. SentinelOne Cybersecurity 101. Retrieved 2026-03-09 from https://www.sentinelone.com/cybersecurity-101/cloud-security/docker-container-security-scanner/

Trend Micro (2024). *What is Container Security?* Trend Micro. Retrieved 2026-03-09 from https://www.trendmicro.com/en/what-is/container-security.html

vCluster Research (2025). *vNode: Container-Native Isolation — Securing AI Workloads*, 2025. Retrieved 2026-03-09 from https://www.vcluster.com/blog/vnode-container-native-isolation-securing-ai-workloads

Wiz Research (2025). *NVIDIAScape: NVIDIA AI Vulnerability CVE-2025-23266*, July 2025. Retrieved 2026-03-09 from https://www.wiz.io/blog/nvidia-ai-vulnerability-cve-2025-23266-nvidiascape

---

*This report was produced through multi-source synthesis from 20+ primary sources including governmental security guidance, industry security vendor research, open-source tool documentation, CVE disclosures, and direct codebase analysis of the MinIVess MLOps Docker configuration.*
