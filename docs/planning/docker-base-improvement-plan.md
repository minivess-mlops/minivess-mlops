---
title: "Docker Base Image Improvement Plan"
status: implemented
created: ""
---

# Docker Base Image Improvement Plan

## Status: COMPLETE

## Problem Statement

The current Docker setup has **two separate base images** that duplicate dependency
management:

1. `Dockerfile.base` — `FROM python:3.12-slim` (Debian, CPU-only, ~200MB)
2. `Dockerfile.base-gpu` — `FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04` (Ubuntu, CUDA)

Both install uv, create the `minivess` user, copy `pyproject.toml`, run `uv sync`,
and copy `src/` + `configs/`. This means:

- **Two places to maintain** when adding system packages or changing Python deps
- Different OS bases (Debian vs Ubuntu) create subtle incompatibilities
- `uv sync` resolves different dependency sets because PyTorch with CUDA extras
  only installs in the GPU base

## Target Architecture: 3-Layer Hierarchy

```
Layer 1: nvidia/cuda:12.6.3-runtime-ubuntu24.04     ← upstream, never modified
    │
Layer 2: minivess-base:latest                        ← ONE base, all shared deps
    │    (CUDA + Python 3.12 + uv + uv sync + src/ + configs/)
    │
    ├── Dockerfile.acquisition   (+ scripts/, CMD)
    ├── Dockerfile.data          (+ scripts/, CMD)
    ├── Dockerfile.train         (+ scripts/, env vars, CMD, GPU device)
    ├── Dockerfile.post_training (+ scripts/, CMD)
    ├── Dockerfile.analyze       (+ scripts/, CMD)
    ├── Dockerfile.deploy        (+ scripts/, CMD)
    ├── Dockerfile.dashboard     (+ scripts/, CMD)
    ├── Dockerfile.qa            (+ scripts/, CMD)
    └── Dockerfile.annotation    (+ scripts/, env vars, CMD)
```

### Key Design Decisions

1. **Single CUDA base for ALL flows** — CUDA runtime libs add ~300MB but eliminate
   the dual-base maintenance burden. CPU-only flows simply don't request GPU devices
   in docker-compose. The CUDA runtime is inert without `--gpus` / device reservations.

2. **One `uv sync` for everyone** — All flows share the same Python dependency set.
   No flow-specific extras at the base level. If a flow needs additional packages,
   it can `uv pip install` in its own Dockerfile (but this should be rare and
   eventually migrated to the base).

3. **Flow Dockerfiles are THIN** — Only `COPY scripts/`, `ENV`, and `CMD`. No
   `apt-get`, no `uv sync`, no `uv pip install`. System deps and Python packages
   belong exclusively in the base.

4. **Layer caching** — The base image changes only when:
   - `pyproject.toml` or `uv.lock` changes (Python deps)
   - `Dockerfile.base` itself changes (system packages)
   - `src/` or `configs/` changes (application code — still cached for deps layer)

## Implementation Tasks

### Completed (this PR)

- [x] Created `Dockerfile.base-gpu` (CUDA + Python 3.12 + uv + deps)
- [x] Rewrote `Dockerfile.train` to inherit from `minivess-base-gpu:latest`
- [x] Cleaned `Dockerfile.data` — removed redundant empty `apt-get`
- [x] Cleaned `Dockerfile.acquisition` — removed redundant `apt-get install git`
- [x] Updated `docker-build-gate.yml` CI to build base-gpu before train
- [x] Added tests: `test_base_gpu_dockerfile_exists`, `test_train_inherits_gpu_base`,
  `test_base_gpu_has_cuda`, `test_flow_dockerfiles_no_apt_or_uv`

### Completed (single-base unification)

- [x] **Merged `Dockerfile.base` and `Dockerfile.base-gpu` into ONE `Dockerfile.base`**
  - Uses `nvidia/cuda:12.6.3-runtime-ubuntu24.04` as the single upstream base
  - Installs Python 3.12 (Ubuntu 24.04 native), uv, git, curl
  - Runs `uv sync --no-dev --no-install-project`
  - Copies `src/` and `configs/`
  - Deleted `Dockerfile.base-gpu` (superseded)

- [x] **ALL flow Dockerfiles use `FROM minivess-base:latest`**
  - All 9 flows (acquisition, data, train, post_training, analyze, deploy,
    dashboard, qa, annotation) inherit from the single base

- [x] **Updated CI (`docker-build-gate.yml`)**
  - Builds one base image, then one flow image
  - No base-gpu references

- [x] **Updated tests**
  - `test_base_has_cuda` — verifies single base has CUDA
  - `test_flow_dockerfiles_no_apt_or_uv` — enforces thin flows
  - Removed base-gpu specific tests

- [x] **Documented the 3-layer pattern**
  - Dockerfile.base header comments describe the hierarchy
  - This planning document

### Future Optimization (P2 — not blocking)

- [ ] **Multi-stage build for smaller CPU images** — If the ~300MB CUDA overhead
  matters (e.g., for CI runners), consider a multi-stage build that produces both
  a CUDA and non-CUDA variant from one Dockerfile using build args.
- [ ] **Docker layer caching in CI** — GHA cache for the base image build
- [ ] **Pre-built base in container registry** — Push `minivess-base:latest` to
  GHCR so flow images can pull without building locally

## Risk: PyTorch CUDA vs CPU Resolution

When `uv sync` runs on a CUDA base image, it may resolve PyTorch with CUDA support
(larger download, ~2GB). CPU-only flows would then carry unused CUDA PyTorch libs.
This is acceptable — the tradeoff is:

- **Pro:** Single dependency set, one `uv.lock`, no resolution conflicts
- **Con:** ~1-2GB larger CPU flow images (acceptable for dev/staging)
- **Mitigation (P2):** Multi-stage build with `--extra cpu` / `--extra gpu` if needed

## References

- PR #461 — Docker enforcement (fix/prefect-docker-policing branch)
- CLAUDE.md Rule #17 (Prefect flows required)
- CLAUDE.md Rule #18 (Docker volume mounts)
- v0.1 pattern: `Dockerfile.base` → `Dockerfile.app` layering
