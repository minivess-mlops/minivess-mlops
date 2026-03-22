# RunPod as Quick Dev Environment via SkyPilot (Non-Docker)

> **Status**: Active implementation (2026-03-14)
> **Priority**: P1 — Enables rapid GPU iteration without Docker overhead
> **Related Issues**: #681 (Lambda Labs for staging/prod), #680 (SAM3 FP16 dtype)
> **Branch**: `feat/skypilot-runpod-gpu-offloading`

---

## 1. Problem Statement

Our three-environment model (CLAUDE.md Design Goal #2) defines:

| Environment | Docker | Compute | Purpose |
|-------------|--------|---------|---------|
| **dev** | Docker-free | GPU | Fast iteration, quick experiments |
| **staging** | Docker Compose | Local GPU in container | Integration testing |
| **prod** | Docker + SkyPilot | Cloud spot / on-prem K8s | Full pipeline |

**The dev environment does not exist yet.** Only staging (Docker-mandatory) and prod are
implemented. This means every GPU experiment — even a quick 2-epoch sanity check — requires
building and pulling a 21.4 GB Docker image.

### Why RunPod for Dev?

[RunPod pods ARE Docker containers, not VMs](../planning/runpod-vs-lambda-vs-rest-for-skypilot-with-docker.md).
This makes RunPod **architecturally incompatible** with Docker-mandatory staging/prod
workflows (no Docker-in-Docker), but **ideal for dev**:

- No Docker overhead — the pod IS the runtime
- RunPod provides pre-cached PyTorch images (~7-9 GB, instant start vs 40-60 min for our 21.4 GB image)
- Cheapest GPU cloud: A4000 $0.16/hr, RTX 4090 $0.44/hr
- SkyPilot manages provisioning, SSH, and teardown automatically

### Architecture: Dev vs Staging/Prod

```
DEV (RunPod):
  RunPod pod = PyTorch container (pre-cached, instant)
  → setup: installs uv + clones repo + installs deps + pulls data
  → run: python -m minivess.orchestration.flows.train_flow
  → No Docker-in-Docker, no Prefect, direct execution
  → MINIVESS_ALLOW_HOST=1, PREFECT_DISABLED=1

STAGING/PROD (Lambda Labs — separate implementation):
  Lambda VM = Ubuntu VM with Docker preinstalled
  → SkyPilot pulls our Docker image inside the VM
  → Full Docker-per-flow isolation, Prefect orchestration
  → Production pipeline as designed
```

---

## 2. Design

### 2.1 Base Image Selection

Use RunPod's pre-cached PyTorch image instead of our custom 21.4 GB image:

| Option | Image | Size | Pre-cached? | Setup time |
|--------|-------|------|-------------|------------|
| A (chosen) | SkyPilot default (`skypilot:gpu-ubuntu-2204`) | ~3 GB | YES | ~8-12 min (Python 3.13 + uv install) |
| B | Our `ghcr.io/petteriteikari/minivess-base:latest` | 21.4 GB | NO | 20-60 min (GHCR pull) |
| C | `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` | ~8 GB | Likely | ~5-8 min (Python 3.13 + project deps) |

**Option A** is chosen: uses SkyPilot's default GPU image (pre-cached, instant start).
Python 3.13 is installed via deadsnakes PPA (our `pyproject.toml` requires `==3.13.*`),
then `uv` installs all project deps. This avoids the 21.4 GB Docker image pull entirely.

**Note**: RunPod PyTorch images ship Python 3.11, but we need 3.13. The setup installs
Python 3.13 via deadsnakes PPA (~30-60s) and uses `uv` to create a venv with 3.13.

### 2.2 Setup Phase

Unlike the Docker-based YAMLs where setup is "data only" (all deps baked in),
the dev YAML installs everything in `setup:`:

```
1. Install uv (fast Python package manager)
2. Clone repo from GitHub (or mount via SkyPilot file_mounts)
3. uv sync --all-extras (install all dependencies)
4. Configure DVC + pull training data from UpCloud S3
5. Pre-cache model weights (SAM3/VesselFM from HuggingFace)
```

Total setup time: ~5-8 minutes (vs 40-60 min for Docker image pull).

### 2.3 File Strategy

Two options for getting source code onto the pod:

| Strategy | Pros | Cons |
|----------|------|------|
| **`workdir: .` (chosen)** | SkyPilot rsyncs project to `~/sky_workdir`, re-syncs on each `sky exec` | Slightly larger upload |
| `git clone` | Always latest code | Needs git auth, adds latency |

Using `workdir: .` — SkyPilot rsync's the project directory to the pod before
`setup:` runs. Commands execute from this directory. A `.skyignore` file excludes
`data/`, `mlruns/`, `.git/`, `outputs/` from sync. Source is re-synced on `sky exec`.

### 2.4 Escape Hatches

The dev environment uses the same escape hatches as the smoke test:

| Escape Hatch | Value | Justification |
|--------------|-------|---------------|
| `MINIVESS_ALLOW_HOST=1` | Skip Docker context check | Dev = no Docker by design |
| `PREFECT_DISABLED=1` | Skip Prefect orchestration | Dev = direct execution |

These are the same approved escape hatches used in pytest (RC12 in CLAUDE.md).

### 2.5 Cost Model

| GPU | $/hr (on-demand) | $/hr (spot) | 2-epoch smoke | 50-epoch DynUNet | 50-epoch SAM3 |
|-----|------------------|-------------|---------------|-----------------|---------------|
| A4000 (16 GB) | $0.16 | $0.10 | ~$0.02 | ~$0.40 | ~$0.80 |
| RTX 4090 (24 GB) | $0.44 | $0.22 | ~$0.04 | ~$0.88 | ~$1.76 |
| RTX 3090 (24 GB) | $0.22 | $0.14 | ~$0.02 | ~$0.44 | ~$0.88 |
| A40 (48 GB) | $0.39 | $0.25 | ~$0.04 | ~$0.78 | ~$1.56 |

**Auto-stop**: `idle_minutes_to_autostop=10` prevents forgotten pods from burning credits.

---

## 3. Implementation Plan

### Phase 1: SkyPilot YAML + Launcher (This Session)

1. Create `deployment/skypilot/dev_runpod.yaml`
   - Uses `runpod/pytorch` pre-cached image (no custom Docker)
   - `setup:` installs uv, syncs deps, pulls DVC data
   - `run:` executes training directly
   - Supports `MODEL_FAMILY` and `EXPERIMENT` overrides

2. Create `scripts/launch_dev_runpod.py`
   - Loads `.env`, resolves MLflow URIs
   - Launches via `sky.launch()` Python API
   - Supports `--model`, `--spot`, `--experiment` flags

3. Add Makefile targets:
   - `make dev-gpu MODEL=dynunet` — launch dev environment
   - `make dev-gpu-stop` — tear down

### Phase 2: Verification

4. Launch with `dynunet` (simplest model, no HF auth needed)
5. Verify: MLflow experiment appears on UpCloud MLflow server
6. Test with `sam3_vanilla` (needs HF auth, FP16 cast verified)

### Phase 3: Developer Workflow Integration

7. Add to `justfile`: `just dev-gpu sam3_vanilla`
8. Update `deployment/CLAUDE.md` with dev environment docs
9. Update CLAUDE.md three-environment table

---

## 4. SkyPilot YAML Design

```yaml
# Key differences from smoke_test_gpu.yaml:
#
# smoke_test_gpu.yaml (Docker-based):
#   image_id: docker:ghcr.io/petteriteikari/minivess-base:latest  (21.4 GB, slow pull)
#   setup: data pull only (all deps in image)
#   Purpose: verify Docker pipeline works on cloud
#
# dev_runpod.yaml (this file, non-Docker):
#   image_id: docker:runpod/pytorch:2.4.0-...  (pre-cached, instant start)
#   setup: install uv + deps + data pull
#   Purpose: quick GPU iteration without Docker overhead
```

### Environment Variables

Same set as smoke test + production YAMLs, but computed at runtime:
- `EXPERIMENT` computed in `run:` block (SkyPilot doesn't resolve intra-env `${}`)
- `MLFLOW_TRACKING_URI` resolved by launcher script (same as `launch_smoke_test.py`)

---

## 5. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| `uv sync` takes too long on RunPod | Use `--no-dev` flag + cache `.venv` via SkyPilot storage mounts |
| RunPod region has no GPUs | Wide fallback list: A4000, RTX4090, RTX3090, A40, L40S |
| Source code too large for rsync | Use `.skyignore` to exclude `data/`, `mlruns/`, `.git/`, `outputs/` |
| HF model download during GPU time | Pre-cache in `setup:` (before `run:` starts billing GPU) |
| Pod preempted (spot) | Use on-demand for dev (default), spot flag optional |

---

## 6. Success Criteria

- [ ] `make dev-gpu MODEL=dynunet` launches and completes 2-epoch smoke test
- [ ] MLflow experiment visible on UpCloud MLflow server
- [ ] Total time from launch to training start: < 10 minutes
- [ ] No Docker image pull (uses pre-cached RunPod image)
- [ ] Works with all model families: dynunet, sam3_vanilla, sam3_hybrid

---

## 7. Relationship to Other Environments

```
┌─────────────────────────────────────────────────────────────────┐
│ MinIVess Three-Environment Model (Updated 2026-03-14)          │
├──────────┬────────────┬──────────────┬──────────────────────────┤
│ Env      │ Docker     │ Compute      │ Orchestration            │
├──────────┼────────────┼──────────────┼──────────────────────────┤
│ dev      │ None       │ RunPod       │ Direct (PREFECT_DISABLED)│
│          │            │ (SkyPilot)   │                          │
├──────────┼────────────┼──────────────┼──────────────────────────┤
│ staging  │ Docker     │ Lambda Labs  │ Prefect (Docker workers) │
│          │ Compose    │ (SkyPilot)   │                          │
├──────────┼────────────┼──────────────┼──────────────────────────┤
│ prod     │ Docker +   │ Lambda Labs  │ Prefect (Docker workers) │
│          │ SkyPilot   │ + AWS/GCP    │                          │
└──────────┴────────────┴──────────────┴──────────────────────────┘
```
