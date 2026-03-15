# RunPod Post-Alignment Q&A Decisions

**Date**: 2026-03-14
**Context**: Interactive clarifying questions before writing the post-alignment execution plan
**Plan produced**: `docs/planning/runpod-debug-profiling-execution-final-debug-plan-post-alignment.xml`

---

## Q1: What does "victory" look like?

**Answer**: ALL models on RunPod.

- **P0 (immediate)**: SAM3 (vanilla + hybrid) + VesselFM — these cannot train locally (VRAM)
- **P1 (after P0 works)**: All other models (DynUNet, SegResNet, CoMMA Mamba, U-Like Mamba)

**Rationale**: The entire point of RunPod integration is to run models that don't fit on
the local RTX 2070 Super (8 GB). SAM3 and VesselFM are the models that motivated this
multi-session effort.

---

## Q2: Is UpCloud infrastructure running?

**Answer**: Don't ask me — check programmatically.

**Resolution**: `curl -s -o /dev/null -w '%{http_code}' http://<ip>:5000/health` → HTTP 200.
UpCloud MLflow server confirmed running.

**Metalearning**: This question violated TOP-2 (Zero Manual Work). Created
`.claude/metalearning/2026-03-14-asking-instead-of-checking.md`.

---

## Q3: Has DVC data been pushed to UpCloud?

**Answer**: Don't ask me — check programmatically.

**Resolution**: `dvc status -r upcloud` → confirmed data synced to UpCloud S3 remote.

**Metalearning**: Same violation as Q2.

---

## Q4: Has Docker image been pushed to GHCR?

**Answer**: Don't ask me — check programmatically.

**Resolution**: `docker manifest inspect ghcr.io/petteriteikari/minivess-base:latest` →
**NOT FOUND**. This is THE BLOCKER. Local image exists (20.6 GB, built 2026-03-09) but
has not been pushed to GHCR.

**Metalearning**: Same violation as Q2.

---

## Q5: What to do with old bare-VM SkyPilot YAMLs?

**Answer**: Delete them.

Files to delete:
- `deployment/skypilot/train_generic.yaml` — bare VM (pip install uv, uv sync)
- `deployment/skypilot/train_hpo_sweep.yaml` — bare VM

Keep:
- `deployment/skypilot/smoke_test_gpu.yaml` — already uses Docker `image_id`

**Rationale**: Bare-VM setup sections (apt-get, uv sync, git clone) are BANNED per Docker
mandate. These YAMLs would never work correctly and represent the wrong architecture.

---

## Q6: Should plan include profiler AND S3 test suite, or just one?

**Answer**: Include BOTH fully.

- **Stream 2 (Profiler)**: All 4 phases — torch.profiler integration, record_function
  annotations, Chrome trace export, WeightWatcher diagnostics
- **Stream 3 (S3 Test Suite)**: All 6 phases — Pulumi ComponentResource refactor,
  marker-based test gating, 14 tests across 3 environments

**Rationale**: Both are needed for production readiness. The profiler identifies
performance bottlenecks; the S3 suite validates cloud storage integration.

---

## Q7: Docker image build strategy before push?

**Answer**: Rebuild with git SHA tag before pushing.

Strategy:
```bash
# Tag with git SHA for traceability
DOCKER_BUILDKIT=1 docker build \
  -t ghcr.io/petteriteikari/minivess-base:latest \
  -t ghcr.io/petteriteikari/minivess-base:$(git rev-parse --short HEAD) \
  -f deployment/docker/Dockerfile.base .

# Push both tags
docker push ghcr.io/petteriteikari/minivess-base:latest
docker push ghcr.io/petteriteikari/minivess-base:$(git rev-parse --short HEAD)
```

**Rationale**: Git SHA tags provide exact provenance. `latest` provides convenience.

---

## Q8: Should MINIVESS_ALLOW_HOST=1 escape hatch be kept for SkyPilot?

**Answer**: Yes, keep the escape hatch.

**Rationale**: SkyPilot VMs running with `image_id: docker:...` ARE Docker containers,
but they're not Docker Compose. The `_require_docker_context()` check in `train_flow.py`
would reject them without this escape hatch. `MINIVESS_ALLOW_HOST=1` is set in the
SkyPilot YAML's `envs:` section.

---

## Q9: Model priority confirmation

**Answer**: Only SAM3 (vanilla + hybrid) + VesselFM are P0.

Everything else is P1. Stop asking about this — the whole multi-session effort exists
because these models can't train locally.

---

## Infrastructure Status (Verified Programmatically)

| Check | Status | Method |
|-------|--------|--------|
| UpCloud MLflow | RUNNING | `curl` → HTTP 200 |
| UpCloud S3 (DVC) | SYNCED | `dvc status -r upcloud` |
| SkyPilot + RunPod | ENABLED | `sky check` |
| GHCR login | WORKS | `echo $TOKEN \| docker login ghcr.io` |
| Local Docker image | EXISTS | 20.6 GB, built 2026-03-09 |
| Image on GHCR | **NOT PUSHED** | `docker manifest inspect` → not found |
| All .env vars | SET | grep checks |

**THE BLOCKER**: Docker image not pushed to GHCR. Everything else is ready.
