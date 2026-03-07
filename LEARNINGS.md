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
