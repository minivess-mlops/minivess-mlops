# Consolidated DevEx + Training Evaluation Plan

**Status**: ACTIVE
**Branch**: `feat/experiment-evaluation`
**Created**: 2026-02-25
**Consolidates**: [dynunet-evaluation-plan.xml](dynunet-evaluation-plan.xml) | [devex-and-prefect-execution-plan.xml](devex-and-prefect-execution-plan.xml) | [prefect-and-devex-profiling-optimizations.md](prefect-and-devex-profiling-optimizations.md)

## Target Hardware Constraints

All phases must run safely on a **low-end desktop GPU setup**:

| Resource | Budget | Hard Limit | Rationale |
|----------|--------|------------|-----------|
| **GPU VRAM** | 6 GB usable | 8 GB total (RTX 2070 Super) | Common desktop GPU; keep 2 GB headroom for display |
| **System RAM** | 24 GB usable | 32 GB total | Common desktop config; keep 8 GB for OS + desktop |
| **Process RSS** | < 16 GB | - | Single training process limit |
| **Peak GPU** | < 6 GB | 8 GB | Leave room for display server + monitoring |
| **Swap** | 0 GB expected | - | Don't rely on swap; warn if stale swap detected |

## Crash Prevention Checklist

Every training phase includes these guards:

- [ ] Process RSS monitored every 10s (abort at 24 GB for 32 GB target)
- [ ] GPU VRAM monitored via nvidia-smi (abort at 7 GB)
- [ ] Sliding window inference for validation (never full-volume forward pass)
- [ ] ThreadDataLoader (no fork memory duplication)
- [ ] CacheDataset with runtime_cache=True (progressive, no init spike)
- [ ] gc.collect() + torch.cuda.empty_cache() between every fold
- [ ] Native resolution (no Spacingd resampling — avoids 7.9 GB outlier volumes)
- [ ] CheckpointManager JSON for crash recovery

## Progress Tracking

| Phase | Status | Issue | Description |
|-------|--------|-------|-------------|
| **A** | NOT_STARTED | [#62](https://github.com/minivess-mlops/minivess-mlops/issues/62) | Dataset profiler: scan volumes, compute DatasetProfile |
| **B** | NOT_STARTED | [#63](https://github.com/minivess-mlops/minivess-mlops/issues/63) | Adaptive compute profiles: hardware + dataset-aware |
| **C** | NOT_STARTED | [#64](https://github.com/minivess-mlops/minivess-mlops/issues/64) | Patch validation guards: pre-training checks |
| **D** | NOT_STARTED | [#65](https://github.com/minivess-mlops/minivess-mlops/issues/65) | Prefect compatibility layer + orchestration module |
| **E** | NOT_STARTED | [#66](https://github.com/minivess-mlops/minivess-mlops/issues/66) | Preflight system: environment validation |
| **F** | NOT_STARTED | [#67](https://github.com/minivess-mlops/minivess-mlops/issues/67) | Experiment runner: single-command YAML-driven |
| **G** | NOT_STARTED | [#68](https://github.com/minivess-mlops/minivess-mlops/issues/68) | DynUNet dice_ce full 3-fold training |
| **H** | NOT_STARTED | [#69](https://github.com/minivess-mlops/minivess-mlops/issues/69) | DynUNet loss sweep (cbdice, cldice, warp) |
| **I** | NOT_STARTED | [#70](https://github.com/minivess-mlops/minivess-mlops/issues/70) | Cross-loss comparison + MLflow registry |
| **J** | NOT_STARTED | [#71](https://github.com/minivess-mlops/minivess-mlops/issues/71) | Model profile YAMLs + VRAM benchmarks |

## Dependency Graph

```
Phase A (Profiler) ──┬──→ Phase B (Adaptive Profiles) ──┬──→ Phase F (Experiment Runner)
                     ├──→ Phase C (Patch Validation) ────┤
                     │                                    ├──→ Phase G (dice_ce training)
Phase D (Prefect) ───┘                                   │        ↓
                                                          │    Phase H (loss sweep)
Phase E (Preflight) ──────────────────────────────────────┘        ↓
                                                               Phase I (comparison)
Phase B ──→ Phase J (Model Profiles)

CRITICAL PATH: A → B → F → G → H → I
```

---

## Phase A: Dataset Profiler (TDD)

**Goal**: Scan all volumes before training to compute min/max/median shapes, spacing
distributions, anisotropy flags, and safe patch sizes per model family.

### Memory Budget for 32 GB Desktop

MiniVess: 70 volumes × ~65 MB avg = ~4.5 GB total. On a 32 GB machine with 24 GB
available, cache_rate should be capped at `min(1.0, 24 * 0.6 / 4.5)` = 1.0 (fits).
For a 1000-volume dataset at 65 MB/vol = 65 GB, cache_rate = `14.4 / 65` = 0.22.

### Tests (RED phase)

```python
# tests/v2/unit/test_data_profiler.py
class TestVolumeScanner:
    def test_scan_single_volume_returns_stats()      # shape, spacing, intensity
    def test_scan_dataset_computes_min_max_shape()    # min_z=5 for MiniVess
    def test_anisotropy_detected()                    # spacing varies >2x
    def test_safe_patch_z_lte_min_z()                 # patch_z <= 5 (or 4 for div-by-4)
    def test_outlier_spacing_flagged()                # mv02 at 4.97μm
    def test_total_size_gb_accurate()                 # ~4.5 GB for MiniVess

class TestSafePatchComputation:
    def test_dynunet_patch_divisible_by_8()           # 4 levels → 2^3=8
    def test_segresnet_patch_divisible_by_8()         # 4 stages → 2^3=8
    def test_patch_reduced_for_small_z()              # z=5 → patch_z=4 (nearest div-8 below)
    def test_patch_xy_unconstrained_for_512()         # XY always 512 in MiniVess
```

### Implementation

- `src/minivess/data/profiler.py`: `VolumeStats`, `DatasetProfile`, `scan_volume()`, `scan_dataset()`, `compute_safe_patch_sizes()`
- Uses NIfTI header for shape/spacing (fast, no full load) + nibabel for intensity stats

---

## Phase B: Adaptive Compute Profiles (TDD)

**Goal**: Replace static profile dictionaries with hardware-detected + dataset-constrained
adaptive computation. Target: 8 GB VRAM / 32 GB RAM desktop runs without OOM.

### 32 GB RAM Budget

```
OS + desktop:          8 GB
Dataset cache (1.0):   4.5 GB  (MiniVess at native res)
Model + optimizer:     0.5 GB
Training overhead:     2 GB
──────────────────────
Total:                15 GB → 17 GB headroom
Abort threshold:      24 GB (process RSS)
```

### 8 GB VRAM Budget

```
Model weights:         0.3 GB
Optimizer states:      0.6 GB (AdamW)
Activations (AMP):    ~2 GB   (depends on patch size)
Sliding window buf:    0.5 GB
──────────────────────
Total:                ~3.5 GB → 4.5 GB headroom
Abort threshold:       7 GB
```

### Tests (RED phase)

```python
# tests/v2/unit/test_adaptive_profiles.py
class TestHardwareDetection:
    def test_detect_returns_budget()                  # All fields populated
    def test_no_gpu_detected()                        # Mock nvidia-smi failure → gpu_vram_mb=0
    def test_8gb_gpu_returns_gpu_low()                # 8192 MB → gpu_low tier

class TestAdaptiveComputation:
    def test_patch_constrained_by_dataset()           # min_z=5 → z=4
    def test_patch_divisible_by_model_divisor()       # DynUNet → div by 8
    def test_cache_rate_adaptive_to_ram()             # 32GB → rate=1.0 for MiniVess
    def test_cache_rate_reduced_for_large_dataset()   # 100GB dataset → rate=0.14
    def test_batch_size_reduced_when_vram_tight()     # 8GB GPU → batch=2 max
    def test_auto_profile_name()                      # Returns "auto_gpu_low_dynunet"

class TestManualOverride:
    def test_explicit_profile_bypasses_auto()         # --compute gpu_low → static
    def test_explicit_patch_size_overrides()           # --patch-size 64x64x8 → exact
```

### Implementation

- `src/minivess/config/adaptive_profiles.py`: `HardwareBudget`, `detect_hardware()`, `compute_adaptive_profile()`
- Reads `/proc/meminfo` for RAM, `nvidia-smi --query-gpu` for GPU, `/proc/loadavg` for CPU
- Model-specific VRAM estimation via `configs/model_profiles/*.yaml`

---

## Phase C: Patch Validation Guards (TDD)

**Goal**: Pre-training validation that catches patch/dataset/memory mismatches BEFORE
training starts. Fail fast, not 30 minutes into epoch 5.

### Tests (RED phase)

```python
# tests/v2/unit/test_patch_validation.py
class TestPatchDatasetCompatibility:
    def test_patch_fits_all_volumes()                 # patch <= min_shape per dim
    def test_rejects_oversized_patch_z()              # patch_z=16 with min_z=5 → error
    def test_patch_divisible_by_model()               # DynUNet needs div-by-8
    def test_rejects_non_divisible_patch()             # 96x96x23 → error (23 % 8 != 0)

class TestMemoryBudget:
    def test_cache_fits_ram()                         # cached_size < 70% available
    def test_rejects_oversized_cache()                # 100GB dataset, rate=1.0, 32GB RAM → error
    def test_vram_estimate_within_budget()            # batch × patch → VRAM < 80% GPU
    def test_rejects_large_batch_on_small_gpu()       # batch=8 on 8GB GPU → error

class TestDefaults:
    def test_no_default_resampling()                  # voxel_spacing == (0,0,0)
```

### Implementation

- `src/minivess/data/validation.py`: Pure validation functions (no side effects)
- Called by preflight system AND experiment runner AND as standalone tests

---

## Phase D: Prefect Compatibility Layer (TDD)

**Goal**: Adopt foundation-PLR `_prefect_compat.py` pattern. Prefect is required but
`PREFECT_DISABLED=1` works for CI/testing.

### Tests (RED phase)

```python
# tests/v2/unit/test_prefect_compat.py
class TestPrefectCompat:
    def test_noop_task_preserves_function()
    def test_noop_flow_preserves_function()
    def test_prefect_disabled_env_var()
    def test_get_run_logger_fallback()
    def test_decorated_function_callable()
```

### Implementation

- `src/minivess/orchestration/__init__.py`
- `src/minivess/orchestration/_prefect_compat.py`
- Add `prefect>=3.0.11` to `pyproject.toml` core dependencies

---

## Phase E: Preflight System (TDD)

**Goal**: Automated environment validation that runs before every experiment.
Replaces all manual checks (swap, GPU, disk, data).

### Tests (RED phase)

```python
# tests/v2/unit/test_preflight.py
class TestPreflight:
    def test_all_check_categories_present()
    def test_gpu_detection()
    def test_ram_check_warns_below_16gb()
    def test_disk_space_warns_below_20gb()
    def test_swap_health_warns_above_5gb()
    def test_environment_detection()                  # local/docker/cloud/ci
    def test_critical_failure_raises()                # no data → fail
    def test_non_critical_warns()                     # stale swap → warn only
```

### Implementation

- `scripts/preflight.py`: `run_preflight()`, check functions, `PreflightResult`

---

## Phase F: Experiment Runner + YAML Config (TDD)

**Goal**: `just experiment --config configs/experiments/dynunet_losses.yaml` — single command.

### Tests (RED phase)

```python
# tests/v2/unit/test_experiment_runner.py
class TestExperimentRunner:
    def test_parse_experiment_yaml()
    def test_dry_run_validates_without_training()
    def test_auto_profile_triggers_detection()
    def test_explicit_profile_overrides_auto()
    def test_resume_finds_checkpoint()
    def test_debug_overrides_applied()
```

### Implementation

- `scripts/run_experiment.py`: Orchestrates preflight → profile → train_monitored.py → report
- `configs/experiments/dynunet_losses.yaml`: Experiment definition
- New justfile recipes: `experiment`, `experiment-debug`, `preflight`, `experiment-dry-run`

---

## Phase G: DynUNet dice_ce Full 3-Fold Training

**Goal**: Baseline training with the validated pipeline on 32 GB RAM / 8 GB GPU target.

### Command

```bash
uv run python scripts/train_monitored.py \
  --compute gpu_low --loss dice_ce \
  --experiment-name dynunet_loss_variation \
  --log-dir logs/dice_ce_full_$(date +%Y%m%d_%H%M%S) \
  --memory-limit-gb 24 \
  --monitor-interval 10
```

### Success Criteria

- All 3 folds complete: train + MetricsReloaded evaluation
- Peak process RSS < 16 GB (safe for 32 GB machines)
- Peak GPU < 6 GB (safe for 8 GB cards)
- MLflow: centreline_dsc, dsc, measured_masd with bootstrap CIs per fold
- No OOM, no swap usage, no memory warnings

### Monitoring

- System metrics CSV every 10s
- Process RSS tracked (abort at 24 GB)
- GPU VRAM tracked (abort at 7 GB)
- Checkpoint saved after each fold (crash-recoverable)

---

## Phase H: DynUNet Loss Sweep (3 Additional Losses)

**Goal**: Run cbdice, dice_ce_cldice, warp as separate processes (not comma-separated —
avoids memory accumulation across losses).

### Commands (run sequentially, one at a time)

```bash
# 1. cbdice
uv run python scripts/train_monitored.py \
  --compute gpu_low --loss cbdice \
  --experiment-name dynunet_loss_variation \
  --log-dir logs/cbdice_full_$(date +%Y%m%d_%H%M%S) \
  --memory-limit-gb 24

# 2. dice_ce_cldice
uv run python scripts/train_monitored.py \
  --compute gpu_low --loss dice_ce_cldice \
  --experiment-name dynunet_loss_variation \
  --log-dir logs/cldice_full_$(date +%Y%m%d_%H%M%S) \
  --memory-limit-gb 24

# 3. warp
uv run python scripts/train_monitored.py \
  --compute gpu_low --loss warp \
  --experiment-name dynunet_loss_variation \
  --log-dir logs/warp_full_$(date +%Y%m%d_%H%M%S) \
  --memory-limit-gb 24
```

### Success Criteria

- Each loss: 3 folds × training + evaluation
- Same memory constraints as Phase G
- All logged to same MLflow experiment for comparison

---

## Phase I: Cross-Loss Comparison + MLflow Model Registry

**Goal**: Compare all 4 loss functions, register best model.

### Tasks

1. Query MLflow for all 4 × 3 = 12 completed runs
2. Build comparison table: loss × metric × fold (with CIs)
3. Paired bootstrap significance testing across losses
4. Register best-loss model in MLflow Model Registry
5. Tag with git SHA, DVC hash, config snapshot, frozen dependencies
6. Transition best model to "Staging" stage

### Expected Output

```
Loss Function    | DSC (mean ± CI)      | clDSC (mean ± CI)   | MASD (mean ± CI)
─────────────────|──────────────────────|──────────────────────|───────────────────
dice_ce          | 0.XXX [0.XXX, 0.XXX] | ...                  | ...
cbdice           | ...                  | ...                  | ...
dice_ce_cldice   | ...                  | ...                  | ...
warp             | ...                  | ...                  | ...
```

---

## Phase J: Model Profile YAMLs + VRAM Benchmarks

**Goal**: Create per-model memory profiles so adaptive compute works accurately.

### Tasks

1. Create `configs/model_profiles/dynunet.yaml` with empirically measured VRAM
2. Create `configs/model_profiles/segresnet.yaml`
3. Create `configs/model_profiles/vista3d.yaml`
4. Create `configs/model_profiles/example_custom.yaml` (template for researchers)
5. Benchmark VRAM: run DynUNet with varying patch sizes, record peak GPU memory

### Benchmarking Protocol

```bash
# For each patch size, run 1-epoch debug training, record peak VRAM
for patch in "64x64x4" "96x96x8" "128x128x16" "128x128x32"; do
  uv run python scripts/train_monitored.py \
    --compute gpu_low --loss dice_ce --debug \
    --patch-size $patch \
    --log-dir logs/vram_bench_${patch}
done
```

---

## Future Work (Next Branch/PR)

These are planned but NOT on this branch:

| Item | Issue |
|------|-------|
| Prefect Flow 1: Data Engineering | Part of #65 expansion |
| Prefect Flow 2: Training wrapper | Part of #65 expansion |
| Prefect Flow 3: Model Analysis & Ensembling | #61 dependency |
| Prefect Flow 4: Deployment & Serving | #61 dependency |
| Docker training image (Dockerfile.train) | Separate PR |
| Prefect server Docker Compose profile | Separate PR |
| DVC → Prefect trigger | Separate PR |
| Ensemble exploration (model soup, voting) | After Phase I |
| Multi-agent LangGraph personas | #61 (P1 future) |
