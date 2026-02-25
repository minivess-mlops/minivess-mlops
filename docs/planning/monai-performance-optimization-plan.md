# MONAI Performance Optimization Plan for MiniVess

**Created**: 2026-02-25
**Status**: Planning complete, implementation pending
**Context**: Terminal crashes during DynUNet training traced to OOM from `Spacingd` resampling outlier volumes. This plan documents the proper MONAI-native approach instead of ad-hoc fixes.

## Root Cause Analysis

### Crash Chain
1. `Spacingd(pixdim=(1.0, 1.0, 1.0))` resamples volume `mv02.nii.gz` (spacing 4.97 μm) from (512, 512, 61) to (2545, 2545, 305) = **7.9 GB per volume**
2. Multiple such volumes loaded via `CacheDataset` + DataLoader workers → process RSS reaches 43+ GB
3. Linux OOM killer terminates `python3` (pid 461914, total-vm: 73 GB, anon-rss: 43 GB)
4. `gnome-terminal-server.service` fails with `oom-kill` → terminal window disappears
5. All training state lost (no crash-resistant checkpointing)

### Dataset Characteristics
| Property | Value |
|----------|-------|
| Volumes | 70 image/label pairs |
| XY dimensions | Always 512×512 |
| Z slices | 5–110 (highly variable) |
| Voxel spacing (in-plane) | 0.31–4.97 μm |
| Voxel spacing (Z) | 1.0–10.0 μm |
| Compressed size | 940 MB |
| Uncompressed float32 (native) | 3.2 GB images + 3.2 GB labels = 6.4 GB total |
| Per-volume range | 5.2–115.3 MB (native resolution) |
| Outlier volumes | mv02 (4.97 μm → 7.9 GB at 1mm), mv10 (2.48 μm → 2.0 GB at 1mm) |

### Spacing Distribution
```
(0.99, 0.99, 5.0):  31 volumes  ← mode
(0.62, 0.62, 5.0):  20 volumes
(0.99, 0.99, 2.0):   8 volumes
(0.62, 0.62, 10.0):  5 volumes
(4.97, 4.97, 5.0):   1 volume   ← outlier (mv02)
(2.48, 2.48, 5.0):   1 volume   ← outlier (mv10)
(others):             4 volumes
```

### System Resources
| Resource | Capacity |
|----------|----------|
| RAM | 64 GB |
| Swap | 17 GB (was full from previous crashes) |
| GPU | NVIDIA RTX 2070 Super, 8 GB VRAM |
| Disk | 1.1 TB free |

## MONAI-Native Solution Architecture

### Key Insight
The problem is NOT that 70 volumes don't fit in 64 GB RAM. At native resolution, the entire dataset is only ~6.4 GB. The problem is `Spacingd` creating enormous resampled intermediates for outlier volumes. MONAI provides built-in solutions for this.

### Phase 1: Immediate Fix (Current — Working)

**Status**: Implemented and validated. Debug smoke test passes.

| Component | Before (OOM) | After (Fixed) |
|-----------|-------------|---------------|
| `Spacingd` | `pixdim=(1.0, 1.0, 1.0)` always | Disabled via `(0,0,0)` sentinel |
| `voxel_spacing` default | `(1.0, 1.0, 1.0)` | `(0.0, 0.0, 0.0)` (native) |
| Train cache_rate | 0.5 | 0.2 (conservative) |
| Val cache_rate | 1.0 | 0.5 (conservative) |
| `num_workers` | 4 (gpu_low) | 2 (capped) |
| Val inference | Full-volume forward | Sliding window (roi=patch_size) |
| Padding (train) | None | `SpatialPadd` to patch_size |
| Padding (val) | None | `DivisiblePadd(k=8)` |
| Memory cleanup | None | `gc.collect()` + `torch.cuda.empty_cache()` between folds |
| Monitoring | None | `SystemMonitor` tracking RAM/GPU/process RSS |
| Checkpointing | None | Per-fold JSON checkpoint for crash recovery |

**Validation**: Peak RAM 19.3 GB, peak GPU 3.5 GB, process RSS stable at 2.5–3.4 GB.

### Phase 2: Optimal MONAI Configuration (Next)

Based on thorough MONAI documentation review:

#### 2a. CacheDataset with `runtime_cache=True`

**Current**: `CacheDataset(cache_rate=0.2, num_workers=0)`
**Optimal**: `CacheDataset(cache_rate=1.0, runtime_cache=True, num_workers=4)`

Rationale:
- At native resolution, all 70 volumes fit in ~6.4 GB RAM (10% of system RAM)
- `runtime_cache=True` fills cache lazily during epoch 1 (no initialization spike)
- `num_workers=4` for cache initialization only (MONAI CacheDataset parameter, separate from DataLoader workers)
- Epochs 2+ enjoy full cache speed — deterministic transforms computed only once

Memory estimate:
```
70 volumes × ~65 MB average (image+label, float32, after LoadImaged+EnsureChannelFirst+Normalize)
= ~4.5 GB cached data
+ ~3-4 GB OS/Python overhead
= ~8 GB total RAM usage (well within 64 GB)
```

#### 2b. Transform Chain Optimization

**Deterministic transforms** (cached once by CacheDataset):
```python
LoadImaged(keys=keys)
EnsureChannelFirstd(keys=keys)
# Spacingd — see Phase 3 for lazy resampling approach
NormalizeIntensityd(keys=ik, nonzero=True)
SpatialPadd(keys=keys, spatial_size=config.patch_size)
```

**Stochastic transforms** (applied each epoch, NOT cached):
```python
RandRotate90d(keys=keys, prob=0.3, spatial_axes=(0, 1))
RandFlipd(keys=keys, prob=0.3, spatial_axis=0)
RandFlipd(keys=keys, prob=0.3, spatial_axis=1)
RandFlipd(keys=keys, prob=0.3, spatial_axis=2)
RandCropByPosNegLabeld(
    keys=keys, label_key=lk,
    spatial_size=config.patch_size,
    pos=1, neg=1, num_samples=4,
)
```

CacheDataset automatically splits at the first `Randomizable` transform. Our ordering is already correct — all deterministic transforms come before the first `Rand*` transform.

#### 2c. ThreadDataLoader instead of DataLoader

**Current**: `torch.utils.data.DataLoader(num_workers=2)` — forks processes, duplicates memory
**Optimal**: `monai.data.ThreadDataLoader(num_workers=0)` — threading, no fork overhead

Rationale:
- With `cache_rate=1.0`, all deterministic transforms are cached in RAM
- Only lightweight random augmentations run at batch time
- Threading avoids the copy-on-write memory duplication from multiprocessing fork
- `num_workers=0` (main thread) is sufficient since cached data access is fast
- Known thread-safety issue with `RandCropByPosNegLabeld` (MONAI #8080) — `num_workers=0` avoids this

#### 2d. SlidingWindowInfererAdapt for Inference

**Current**: Custom `SlidingWindowInferenceRunner` wrapping `monai.inferers.sliding_window_inference`
**Optimal**: Use `monai.inferers.SlidingWindowInfererAdapt`

Benefits:
- Automatic GPU OOM recovery (falls back to CPU stitching)
- Remembers which volume sizes caused OOM and proactively uses CPU for similar sizes
- `sw_device="cuda"` for model inference, `device=None` for adaptive stitching

```python
from monai.inferers import SlidingWindowInfererAdapt

inferer = SlidingWindowInfererAdapt(
    roi_size=patch_size,
    sw_batch_size=4,
    overlap=0.25,
    mode="gaussian",
    sw_device=torch.device("cuda"),
    device=None,  # adaptive: GPU first, CPU fallback
)
```

### Phase 3: Re-enabling Spacingd with Lazy Resampling (Future)

MONAI's lazy resampling (available since v1.2) solves the Spacingd OOM without disabling resampling:

```python
Compose(transforms, lazy=True)
```

**How it works**:
1. `Spacingd` records the resampling as a pending affine operation (no data created)
2. `RandCropByPosNegLabeld` determines the crop region
3. Only the crop region is resampled — the full 8 GB volume is **never materialized**
4. For the outlier mv02: instead of 7.9 GB, only the 96×96×24 patch region is resampled (~70 KB)

**Caveats**:
- Marked "experimental" in MONAI docs (but actively developed since v1.2)
- `NormalizeIntensityd` is NOT a spatial transform and requires materialized data — must come AFTER lazy spatial transforms, OR the spatial transforms must be finalized before normalization
- Need careful transform ordering: spatial (lazy) → crop (triggers resample) → intensity
- CacheDataset interaction: cached items store pending operations, not resampled data

**Recommended transform chain with lazy resampling**:
```python
Compose([
    LoadImaged(keys=keys),
    EnsureChannelFirstd(keys=keys),
    Spacingd(keys=keys, pixdim=target_spacing, mode=("bilinear", "nearest")),
    SpatialPadd(keys=keys, spatial_size=config.patch_size),
    RandRotate90d(keys=keys, prob=0.3, spatial_axes=(0, 1)),
    RandFlipd(keys=keys, prob=0.3, spatial_axis=0),
    RandFlipd(keys=keys, prob=0.3, spatial_axis=1),
    RandFlipd(keys=keys, prob=0.3, spatial_axis=2),
    RandCropByPosNegLabeld(
        keys=keys, label_key=lk,
        spatial_size=config.patch_size,
        pos=1, neg=1, num_samples=4,
    ),
    # Spatial operations finalized here ↑ — lazy ops are fused and applied
    NormalizeIntensityd(keys=ik, nonzero=True),  # Needs materialized data
], lazy=True)
```

**Target spacing options**:
- Median spacing: `(0.99, 0.99, 5.0)` — minimal resampling for 31/70 volumes
- Mode spacing: same as median in this dataset
- Skip Z-resampling: `(1.0, 1.0, None)` — only resample XY, keep native Z

**Implementation priority**: Phase 3 is lower priority than Phase 2. Native-resolution training (Phase 1) is valid for the loss-variation experiment. Lazy resampling matters more for the eventual production pipeline.

### Phase 4: Pre-Training System Benchmark (Future)

Before running expensive multi-loss sweeps, run a quick benchmark to validate:

```python
# 1. Data loading throughput
# Time: load all 70 volumes, measure cache fill time, epoch iteration time
# Target: < 30 seconds for cache fill, < 5 seconds per epoch iteration

# 2. GPU memory footprint
# Measure: model size, training batch size, validation sliding window
# Target: < 6 GB VRAM during training, < 7 GB during inference

# 3. CPU-GPU transfer rate
# Measure: batch transfer time with/without pin_memory
# Target: < 10ms per batch

# 4. Memory stability across folds
# Measure: RSS at start/end of each fold, check for monotonic growth
# Target: RSS returns to baseline (±200 MB) after each fold cleanup
```

## Implementation Order

1. **Phase 1** (DONE): Immediate OOM fix — skip Spacingd, reduce cache, add monitoring
2. **Phase 2a**: Switch to `CacheDataset(cache_rate=1.0, runtime_cache=True)` — safe since native resolution is small
3. **Phase 2c**: Switch to `ThreadDataLoader(num_workers=0)` — eliminates fork overhead
4. **Phase 2d**: Switch to `SlidingWindowInfererAdapt` — auto OOM recovery
5. **Phase 2b**: Verify transform chain ordering is optimal for caching
6. **Phase 4**: Pre-training benchmark script
7. **Phase 3**: Lazy resampling (when MONAI lazy mode is more mature, or for production pipeline)

## References

- [MONAI CacheDataset docs](https://docs.monai.io/en/stable/data.html#cachedataset)
- [MONAI ThreadDataLoader docs](https://docs.monai.io/en/stable/data.html#threaddataloader)
- [MONAI SlidingWindowInfererAdapt](https://docs.monai.io/en/stable/inferers.html#slidingwindowinfereradapt)
- [MONAI Lazy Resampling](https://docs.monai.io/en/stable/lazy_resampling.html)
- [MONAI Fast Training Guide](https://github.com/Project-MONAI/tutorials/blob/main/acceleration/fast_model_training_guide.md)
- [MONAI #8080: ThreadDataLoader thread-safety](https://github.com/Project-MONAI/MONAI/issues/8080)
- [MONAI #6626: GPU memory leak with transforms](https://github.com/Project-MONAI/MONAI/issues/6626)
- [docs/planning/dynunet-evaluation-plan.xml](dynunet-evaluation-plan.xml) — Crash forensics and execution plan
