# Learnings

Accumulated discoveries from TDD iterations. Persists across sessions.

## 2026-02-25 — OOM Root Cause: Spacingd Resampling

- **Discovery**: MiniVess volume mv02 has 4.97 um voxel spacing. Resampling to 1.0 um isotropic creates 2545x2545x305 arrays (~8 GB each), causing terminal OOM crashes.
- **Resolution**: Set `voxel_spacing=(0,0,0)` to skip resampling. Train at native resolution. Patches handle shape consistency.

## 2026-02-25 — MiniVess Dataset Characteristics

- **Discovery**: 70 volumes, heterogeneous shapes (min Z ~5 voxels), spacings range 0.31-4.97 um/voxel. Total ~4.5 GB at native resolution.
- **Resolution**: Patch Z dimension must be <= min_z (5). DynUNet needs divisor of 8 → patch_z=4 (not 8 which would exceed some volumes). SpatialPadd before RandCropByPosNegLabeld.

## 2026-02-25 — FAILURE: Training Launched Without Multi-Metric Config

- **Discovery**: Launched `train_monitored.py` directly instead of through `run_experiment.py`, bypassing the YAML checkpoint config. Only val_loss checkpoints were saved; best-by-val_dice and best-by-val_f1 model weights were lost despite metrics being recorded.
- **Resolution**: ALWAYS use `run_experiment.py --config` as the entry point for training. Never bypass the YAML-driven pipeline. See `docs/planning/failure-metalearning-001-training-launch.md`.

## 2026-02-25 — ThreadDataLoader vs DataLoader

- **Discovery**: Multiprocessing DataLoader with num_workers>0 causes fork memory duplication. With 70 cached volumes, each worker duplicates the cache.
- **Resolution**: Use MONAI ThreadDataLoader with num_workers=0 (main thread). CacheDataset with runtime_cache=True provides progressive caching without init spike.

## 2026-03-02 — MONAI Deploy SDK silently missing (#254)
- **Discovery**: `monai-deploy-app-sdk` was never added to `pyproject.toml`. The code at `monai_deploy_app.py` uses duck-typed Protocol stubs that pass without the real SDK. Tests pass silently because they test the stubs, not the real SDK.
- **Root Cause**: Someone implemented the duck-typing fallback as a convenience, then the actual SDK dependency was never installed. No warning is logged when the fallback activates.
- **Resolution**: Issue #254 opened. Fix: add as optional dependency, log warning on fallback, mark tests with `pytest.mark.skipif` with clear message. Pattern to avoid: never silently bypass a missing dependency — always warn.

## 2026-03-07 — SAM3 Stub Permanently Removed; VRAM Enforcement Added

- **Discovery**: `_StubSam3Encoder` (and `_StubFPNNeck`, `_StubSam3Decoder`) produced
  valid-looking training output from random weights. On 2026-03-02 a training run on stub
  weights completed and produced apparently meaningful metrics. The error went undetected
  until manual inspection. This is a "cosmetic success" anti-pattern — worse than a crash.
- **Resolution**:
  - All stub classes removed permanently from `sam3_backbone.py`, `sam3_decoder.py`.
  - `use_stub` parameter removed from all SAM3 adapter `__init__` methods.
  - `_auto_stub_sam3()` removed from `model_builder.py`.
  - GPU VRAM ≥16 GB enforced via `check_sam3_vram()` in new `sam3_vram_check.py`.
  - 10 AST-based enforcement tests in `tests/unit/adapters/test_no_sam3_stub.py` will
    catch any future stub regression at CI level — they check source files without
    importing the classes.
  - Tests that needed `use_stub=True` migrated to `pytest.mark.skipif(not _sam3_package_available())`.
- **Lesson**: Any "convenient" stub that produces valid-looking outputs from random weights
  is a trap. If CI needs to run without real weights, use `pytest.mark.skipif` to skip
  the tests entirely — never silently substitute random-weight computation.
- **Real VRAM requirements** (corrected from original 3.0-7.5 GB stub estimates):
  - V1 Vanilla: ≥16 GB (full ViT-32L must load even when frozen)
  - V2 TopoLoRA: ≥18 GB
  - V3 Hybrid: ≥22 GB

## 2026-03-09 — Docker Infra: 12 Failures on First Staging Run (Issues #524–#532)

- **Discovery**: First full staging run (train → post_training → analyze in Docker) hit 12 distinct infrastructure failures, all fixable but requiring systematic debugging. Root causes documented here for other projects.
- **Failures and resolutions**:
  1. Docker named volumes start root-owned → one-time `chown -R 1000:1000` with project-prefixed names (`deployment_*`)
  2. Docker Compose project name prefix (`deployment_`) must be used for volume names in `docker run` commands
  3. MLflow 3.x YAML folded scalar bug: `>` preserves newlines → `mlflow server` gets no args → binds 127.0.0.1 → Fixed with list entrypoint + `>-`
  4. Wrong MLflow security env var: `MLFLOW_ALLOWED_HOSTS` ignored → correct is `MLFLOW_SERVER_ALLOWED_HOSTS`
  5. Cross-compose service name resolution fails → use container names (`minivess-minio`) not service names (`minio`) on external networks
  6. MinIO bucket not auto-created → must create `mlflow-artifacts` bucket manually before first run
  7. Data volume empty → must copy dataset into `deployment_data_cache` before training
  8. GPU CDI vs nvidia-runtime: `--runtime nvidia` requires daemon config. Use CDI: `devices: ["nvidia.com/gpu=all"]`
  9. `docker compose run` in while loop consumes herestring stdin → only first model trains. Fix: `readarray` + `for` loop
  10. Bash variable expansion into Python heredoc: `statuses = ${ARRAY[@]}` → SyntaxError. Fix: temp JSON file
  11. MLflow tag value `None` → TypeError in protobuf. Fix: filter None before passing tags
  12. Base image must be rebuilt with `--no-cache` after `src/` changes; train Dockerfile only copies `scripts/`
- **Persistence**: Full catalog saved in `.claude/projects/.../memory/docker-infra-learnings.md`

## 2026-03-29 — CRITICAL: Proposed local launcher hack instead of Docker+Prefect

- **Discovery**: When asked to run Phase 3 mini-experiment training, Claude proposed `MINIVESS_ALLOW_HOST=1` shortcuts and "local launcher scripts" instead of the Docker+Prefect pipeline. This is the 7th+ documented instance of this anti-pattern.
- **Resolution**: Created CLAUDE.md Rule #33 (Docker+Prefect non-negotiable, zero bypass), metalearning doc, P1 Issue #971. The correct answer is ALWAYS `docker compose run --shm-size 8g train`.

## 2026-03-29 — P0: Hardcoded MLFLOW_TRACKING_URI in docker-compose.flows.yml

- **Discovery**: `docker-compose.flows.yml` line 20 hardcoded `http://minivess-mlflow:5000` instead of `${MLFLOW_TRACKING_URI:-...}`, silently ignoring the DagsHub URI in `.env`. The AST guard only scans Python for `alpha=0.05` — it doesn't scan YAML.
- **Resolution**: Fixed to `${MLFLOW_TRACKING_URI:-http://minivess-mlflow:${MLFLOW_PORT:-5000}}`. Created P0 Issue #972 (hardcoded URLs) and #973 (AST guard must scan ALL values + YAML).

## 2026-03-29 — Biostatistics Flow: Phase 0-8 implementation

- **Discovery**: Implemented stratified within-fold permutation test, BCa/percentile adaptive bootstrap, hierarchical gatekeeping, DuckDB-only data loading, JSON sidecar models, R data export, Nature Protocols compliance generators. 184 tests passing.
- **Resolution**: Synthetic fixture DuckDB with known effects validates the full statistical pipeline. Real training deferred to Docker+Prefect pipeline (correctly — no shortcuts).

## 2026-03-30 — Observability: 4-pass journey from dead code to production

- **Discovery**: Pass 1-2 wrote 5 Python modules but only imported them (dead code). Pass 3 wired context managers into all 15 flows + 77 @task hooks + Docker HEALTHCHECK + LGTM/DCGM compose services. Pass 4 threaded event_logger into SegmentationTrainer epoch loop, added CPU healthcheck, wired stall detection, updated KG.
- **Resolution**: Rule #34 added: "Import ≠ Done — code must be CALLED + DEPLOYED + OBSERVABLE." AST enforcement test verifies every @flow body has `with` context manager. 10/10 flow services have HEALTHCHECK. Trainer calls log_epoch_complete() each epoch. 7086 tests passing.
- **Key lesson**: Writing code ≠ shipping functionality. Import tests are necessary but NOT sufficient. Every observability feature must produce OBSERVABLE OUTPUT verifiable by `docker logs` or dashboard.
