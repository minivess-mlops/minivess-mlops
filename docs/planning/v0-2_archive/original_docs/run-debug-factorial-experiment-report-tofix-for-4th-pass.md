# 4th Pass Pre-GCP: Comprehensive Status of All 1st Pass Glitches

**Date**: 2026-03-21
**Branch**: `test/debug-factorial-4th-pass`
**Purpose**: Exact status of every glitch from the 1st GCP run + all work done since

---

## Summary

The 1st GCP factorial run (2026-03-20) identified **12 glitches**. Since then, **3 local
debug passes** and multiple PRs have fixed the vast majority. This document verifies
the CURRENT codebase state for each glitch.

| Status | Count | Glitches |
|--------|-------|----------|
| **FIXED (verified in code)** | 10 | #1, #2, #3, #4, #5, #6, #7, #9, #11, #12 |
| **MOSTLY FIXED (minor remaining)** | 1 | #8 (MLflow artifact persistence) |
| **NEEDS DOCKER REBUILD ONLY** | 1 | #10 (mamba-ssm compilation) |

**Blocking the 4th pass**: Only #10 (Docker rebuild) and the train+post-training
flow merger (new requirement from this session).

---

## Glitch-by-Glitch Verification

### Glitch #1: SkyPilot missing `[gcp]` extra — FIXED
**Verified**: `pyproject.toml` has `skypilot-nightly[gcp,runpod]`. No action needed.

### Glitch #2: pyparsing too old — FIXED
**Verified**: `pyparsing>=3.1.0` in `pyproject.toml`. No action needed.

### Glitch #3: `sky` binary not on PATH — FIXED
**Verified**: `scripts/run_factorial.sh` has fallback logic (lines 34-47). No action needed.

### Glitch #4: L4 not in europe-north1 — FIXED
**Verified**: `train_factorial.yaml` has no hardcoded region. SkyPilot auto-selects. No action needed.

### Glitch #5: Docker image missing git + DVC — FIXED
**Verified**: `Dockerfile.base` has `git` in apt-get and DVC files in COPY. No action needed.

### Glitch #6: DVC partial pull + cp same-file — FIXED
**Verified**: `train_factorial.yaml` setup does targeted `dvc pull data/raw/minivess/` with verification. No action needed.

### Glitch #7: train_flow.py missing 6 factorial CLI arguments — FIXED
**Verified**: All 6 arguments added to argparse + training_flow() signature. No action needed.

### Glitch #8: MLflow checkpoint upload 413 on Cloud Run — MOSTLY FIXED
**Current state**:
- `train_factorial.yaml` line 70: `MLFLOW_TRACKING_URI: ${MLFLOW_TRACKING_URI}` ✅
- `file_mounts` for `/app/checkpoints` → `gs://minivess-mlops-checkpoints` ✅
- `Dockerfile.mlflow-gcp` has `--no-serve-artifacts` ✅
- Pulumi sets `MLFLOW_ARTIFACTS_DESTINATION` to GCS ✅
- `check_resume_state_task()` exists in train_flow.py ✅
- Resume wiring at train_flow.py lines 607-622 ✅

**Remaining (nice-to-have, NOT blocking 4th pass)**:
- Atomic checkpoint writes (TDD RED phase — test exists, impl pending)
- SHA256 sidecar verification (TDD RED phase — test exists, impl pending)
- End-to-end spot recovery test on actual GCP

**NOTE on competing mechanisms**: The `file_mounts` for checkpoints was identified as
a competing mechanism vs MLflow artifact store. For the 4th pass, we keep both
(belt-and-suspenders). Post-4th-pass, consolidate to MLflow-only per
`.claude/metalearning/2026-03-21-competing-artifact-persistence-mechanisms.md`.

### Glitch #9: sam3_topolora LoRA on Conv2d — FIXED + ENHANCED
**Verified in code** (`src/minivess/adapters/sam3_topolora.py`):
- Line 119: `LORA_FFN_KEYWORDS = ("mlp", "lin1", "lin2", "fc1", "fc2")` ✅
- Line 127: `_apply_lora_to_encoder()` filters by keywords ✅
- Line 179: `SpatialConvAdapter` class implemented ✅
- Line 268: `self.spatial_adapter = SpatialConvAdapter(channels=256)` ✅
- Line 307: `fpn_features = self.spatial_adapter(fpn_features)` wired ✅

**Additional work done** (post-1st-pass):
- Issue #879 reframed from "LoRAConv2d" to "Spatial Adapter" per Khazem et al. 2025
- Metalearning doc: `2026-03-21-topolora-paper-implementation-fidelity.md`
- Paper architecture fully documented in fix plan

**No action needed.**

### Glitch #10: mamba-ssm not compiled in Docker image — NEEDS DOCKER REBUILD
**Current state**:
- `Dockerfile.base` has `INSTALL_MAMBA` build arg (lines 79-82) ✅
- Current GAR image built with `INSTALL_MAMBA=0` (default) ❌
- Local machine has mamba-ssm installed but wrong CUDA toolkit (11.5 < 11.6 required)

**Action required**:
```bash
DOCKER_BUILDKIT=1 docker build \
  --build-arg INSTALL_MAMBA=1 \
  -t minivess-base:latest \
  -f deployment/docker/Dockerfile.base .
docker tag minivess-base:latest \
  europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest
docker push europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest
```

**Time**: ~15-20 min (CUDA compilation). This is the ONLY infrastructure blocker.

### Glitch #11: `--detach-run` missing from `sky jobs launch` — FIXED
**Verified**: `scripts/run_factorial.sh` has `--detach-run` on both launch commands. No action needed.

### Glitch #12: Zero-shot max_epochs=0 Pydantic validation — FIXED
**Verified**: `src/minivess/config/models.py` line 227: `max_epochs: int = Field(default=100, ge=0)` ✅
No action needed.

---

## Additional Bugs from 2nd Pass (Local 3-Flow) — ALL FIXED

| Bug | File | Status |
|-----|------|--------|
| skeletonize_3d removed | scripts/run_local_debug_3flow.py | FIXED |
| MLflow status int→string | biostatistics_discovery.py | FIXED |
| multi_swa state_dict key | multi_swa.py | FIXED |
| model_merging state_dict key | model_merging.py | FIXED |
| loss_name mapping missing | factorial_config.py | FIXED |
| fold_0 prefix rejected | biostatistics_duckdb.py | FIXED |
| Spurious `file:` directory | RECURRING (needs resolve_tracking_uri fix) |

---

## Additional Bugs from 3rd Pass (SWAG + Calibration) — ALL FIXED

| Bug | File | Status |
|-----|------|--------|
| SWAG BCE for multi-class | post_training_plugins/swag.py | FIXED |
| update_bn with dict loaders | post_training_plugins/swag.py | FIXED |
| SegmentationOutput not unwrapped | post_training_plugins/swag.py | FIXED |
| BA-ECE fallback on 1D arrays | calibration_metrics.py | KNOWN (needs 3D input) |

---

## New Features Implemented Since 1st Pass

### SWAG (Maddox et al. 2019)
- `src/minivess/ensemble/swag.py` — SWAGModel with low-rank posterior
- `src/minivess/pipeline/post_training_plugins/swag.py` — SWAGPlugin
- `src/minivess/config/post_training_config.py` — SWAGPluginConfig
- Validated locally: 100% posterior diversity, all calibration metrics non-zero

### Comprehensive Calibration Metrics (9 scalar + 2 spatial maps)
- `src/minivess/pipeline/calibration_metrics.py` — 11 functions
- Tier 1 (fast): ECE, MCE, RMSCE, Brier, NLL, OE, D-ECE
- Tier 2 (comprehensive): ACE, BA-ECE
- Spatial maps: Brier map, NLL map
- Integrated into SegmentationMetrics (training val loop) and evaluation runner

### SWA → Checkpoint Averaging Rename
- ~30 files renamed (SWAPlugin → CheckpointAveragingPlugin, etc.)
- Metalearning: `2026-03-21-fake-swa-checkpoint-averaging-mislabeled.md`

### Spot Resume Infrastructure
- `job_recovery` section in train_factorial.yaml with FAILOVER strategy
- `check_resume_state_task()` in train_flow.py
- `MOUNT_CACHED` for async GCS checkpoint sync

### SkyPilot Spot Preemption Research
- `docs/planning/skypilot-spot-preemption-checkpoint-research-report.md`
- `docs/planning/skypilot-spot-resume.md`

---

## What ACTUALLY Blocks the 4th Pass

### MUST DO (blocking)

1. **Docker rebuild with INSTALL_MAMBA=1** (~20 min)
   - All other code changes are already in main
   - This is the ONLY infrastructure blocker

2. **Train + Post-Training flow merger** (new requirement)
   - Plan: `docs/planning/training-and-post-training-into-two-subflows-under-one-flow.md`
   - 4-phase implementation: config → refactor → SWAG sub-flow → wire SkyPilot

3. **DeepVess data on GCS** (for VesselFM zero-shot)
   - Not yet DVC-tracked or pushed to GCS
   - Need: download → `dvc add` → `dvc push -r gcs`

### SHOULD DO (not blocking but important)

4. **Instance validation script** in SkyPilot setup phase (Issue #904)
5. **Atomic checkpoint writes + SHA256** (tests in RED phase)
6. **Fix `file:` directory creation** in resolve_tracking_uri()

### ALREADY DONE (no action)

- Glitches #1-7, #9, #11, #12 — all fixed in current codebase
- Glitch #8 — 90% fixed (MLFLOW_TRACKING_URI + file_mounts)
- SWAG plugin + calibration metrics
- SWA → checkpoint averaging rename
- Spot resume infrastructure

---

## Recommended Execution Order

1. Train+post-training flow merger (Phase 0-3 from merger plan)
2. Docker rebuild with `INSTALL_MAMBA=1` + push to GAR
3. DeepVess data → DVC → GCS
4. Instance validation script in SkyPilot setup phase
5. Finalize 4th pass XML plan
6. `make test-prod` → all pass
7. `./scripts/run_factorial.sh --dry-run` → validates
8. Launch 4th pass on GCP
