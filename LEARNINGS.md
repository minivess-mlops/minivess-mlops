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
