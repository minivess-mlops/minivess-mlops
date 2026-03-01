# MLflow Robustifying Plan: Comprehensive Reproducibility for Academic Publication

**Status:** v2 (reviewed by 3 agents) — 2026-02-28
**Branch:** `feat/architecture-logging-and-half-width`
**Context:** Post-mortem on reproducibility gaps discovered during half-width DynUNet experiment setup

---

## 1. Problem Statement

Training runs in `dynunet_loss_variation_v2` (4 losses x 3 folds x 100 epochs, ~25 hours)
were completed without logging **critical** reproducibility metadata:

| Category | Currently Logged | Missing |
|----------|-----------------|---------|
| **Architecture** | model_family, in/out_channels | filters, kernel_size, strides, norm, deep_supervision |
| **Environment** | (post-hoc tags only) | Python, PyTorch, MONAI, CUDA, cuDNN versions at run start |
| **Hardware** | (post-hoc tags only) | GPU model, VRAM, RAM, CPU at run start |
| **Training** | LR, epochs, optimizer, scheduler, seed | weight_decay, grad_clip, warmup_epochs, patch_size, training_time |
| **Data** | (nothing) | dataset name, volume count, volume IDs, split sizes |
| **Git** | (post-hoc tag only) | commit hash, branch, dirty state at run start |
| **Dependencies** | (never called) | uv pip freeze as artifact |
| **Loss config** | loss name as tag | loss as param, component weights |
| **System metrics** | (custom monitor, not in MLflow) | GPU/CPU/memory via MLflow native |
| **Run lifecycle** | (none) | FAILED/KILLED status on crashes |
| **Cross-flow links** | (none) | upstream_training_run_id in eval runs |
| **Run description** | (none) | mlflow.note.content for humans |

This means: **a third party cannot determine from MLflow alone what model architecture, software environment, or data was used for any existing run**. The only way to infer model width was forensically from checkpoint weight tensor shapes.

### Root Causes

1. `ExperimentTracker._log_config()` only logs 12 of 17+ `TrainingConfig` fields
2. `log_git_hash()`, `log_frozen_deps()`, `log_hydra_config()`, `log_split_file()`, `log_model_info()` all exist but are **never called** in the training pipeline
3. No system info collection at training time (only post-hoc via `mlruns_enhancement.py`)
4. No dataset metadata logging, no volume identifiers per fold
5. No MLflow system metrics enabled (`log_system_metrics=True`)
6. No run failure status handling (crashed runs show as FINISHED)
7. No cross-flow traceability (evaluation runs don't link back to training runs)
8. Analysis flow and EvaluationRunner bypass `ExperimentTracker` entirely
9. `architecture_params` not wired through YAML -> pipeline (fixed in this branch)

---

## 2. Audit: Current vs Required Logging

### 2.1 What ExperimentTracker Already Has (But Doesn't Call)

| Method | What It Does | Called During Training? |
|--------|-------------|----------------------|
| `_log_config()` | Logs 12 training params | Yes (auto in `start_run`) |
| `log_model_info(model)` | Logs model config + trainable_parameters | **NO** |
| `log_git_hash()` | Logs git commit as tag | **NO** |
| `log_frozen_deps()` | Logs `uv pip freeze` as artifact | **NO** |
| `log_hydra_config(dict)` | Logs resolved config YAML artifact | **NO** |
| `log_split_file(path)` | Logs split JSON as artifact | **NO** |
| `log_test_set_hash(paths)` | Logs test set SHA-256 | **NO** |
| `log_evaluation_results()` | Logs MetricsReloaded CIs | Yes (in eval) |
| `log_pyfunc_model()` | Logs pyfunc model artifact | Yes (in analysis flow) |
| `log_post_training_tags()` | Logs best_* metric tags | Yes (partially) |

**Five fully implemented methods are never called during training.** This is the primary gap.

### 2.2 What's Completely Missing

| Category | What's Needed | MLflow API | Priority |
|----------|-------------|-----------|----------|
| **Run lifecycle** | FAILED/KILLED status on crashes | `mlflow.end_run(status="FAILED")` | P0 |
| **Run description** | Human-readable note | `mlflow.set_tag("mlflow.note.content", ...)` | P0 |
| **Experiment tags** | Project/dataset/type on experiment | `mlflow.set_experiment_tags(...)` | P0 |
| System metrics | GPU/CPU/memory/disk monitoring | `mlflow.start_run(log_system_metrics=True)` | P0 |
| Python version | `3.13.x` | `mlflow.log_param("sys_python_version", ...)` | P0 |
| PyTorch version | `2.x.y+cuXXX` | `mlflow.log_param("sys_torch_version", ...)` | P0 |
| MONAI version | `1.5.x` | `mlflow.log_param("sys_monai_version", ...)` | P0 |
| CUDA version | `12.x` | `mlflow.log_param("sys_cuda_version", ...)` | P0 |
| cuDNN version | `8900` | `mlflow.log_param("sys_cudnn_version", ...)` | P0 |
| OS info | `Linux 6.8.0-90-generic` | `mlflow.log_param("sys_os", ...)` | P0 |
| GPU model | `NVIDIA GeForce RTX 2070 SUPER` | `mlflow.log_param("sys_gpu_model", ...)` | P0 |
| GPU VRAM | `8192 MB` | `mlflow.log_param("sys_gpu_vram_mb", ...)` | P0 |
| Total RAM | `64.0 GB` | `mlflow.log_param("sys_total_ram_gb", ...)` | P0 |
| Hostname | `machine-name` | `mlflow.log_param("sys_hostname", ...)` | P0 |
| Weight decay | `1e-5` | `mlflow.log_param("weight_decay", ...)` | P0 |
| Gradient clip | `1.0` | `mlflow.log_param("gradient_clip_val", ...)` | P0 |
| Warmup epochs | `5` | `mlflow.log_param("warmup_epochs", ...)` | P0 |
| Patch size | `(96, 96, 24)` | `mlflow.log_param("patch_size", ...)` | P0 |
| Training time | `14523.7` (seconds) | `mlflow.log_param("training_time_seconds", ...)` | P0 |
| Training device | `cuda:0` | `mlflow.log_param("device", ...)` | P0 |
| Trainable params | `1234567` (as param, not just metric) | `mlflow.log_param("trainable_parameters", ...)` | P0 |
| Cache rate | `1.0` | `mlflow.log_param("cache_rate", ...)` | P1 |
| Num workers | `2` | `mlflow.log_param("num_workers", ...)` | P1 |
| Dataset name | `minivess` | `mlflow.log_param("data_dataset_name", ...)` | P0 |
| N train volumes | `46` | `mlflow.log_param("data_n_train_volumes", ...)` | P0 |
| N val volumes | `24` | `mlflow.log_param("data_n_val_volumes", ...)` | P0 |
| Train volume IDs | `mv01,mv03,...` | `mlflow.log_param("data_train_volume_ids", ...)` | P0 |
| Val volume IDs | `mv02,mv05,...` | `mlflow.log_param("data_val_volume_ids", ...)` | P0 |
| Fold IDs | `0,1,2` | `mlflow.log_param("fold_ids", ...)` | P0 |
| Compute profile | `gpu_low` | `mlflow.log_param("compute_profile", ...)` | P1 |
| **Cross-flow link** | Training run ID in eval runs | `mlflow.log_param("upstream_training_run_id", ...)` | P0 |
| **Model signature** | Input/output tensor schema | `mlflow.pyfunc.log_model(..., signature=sig)` | P1 |
| Volume shape range | `(512,512,5)-(512,512,110)` | `mlflow.log_param("data_shape_range", ...)` | P1 |
| Loss components | `0.5*cbDice + 0.5*dice_ce_cldice` | `mlflow.log_param("loss_config", ...)` | P1 |
| MONAI config dump | Full `monai.config.print_config()` | `mlflow.log_artifact(...)` | P1 |
| Training log file | Copy of loguru/Python log | `mlflow.log_artifact(...)` | P1 |
| Dataset profile JSON | Volume stats, shape distributions | `mlflow.log_artifact(...)` | P1 |
| Per-volume metrics | Dice/clDice/MASD per volume | `mlflow.log_table(...)` | P1 |
| Model summary | `torchinfo.summary()` output | `mlflow.log_artifact(...)` | P2 |
| Training curves | matplotlib figure | `mlflow.log_figure(...)` | P2 |

### 2.3 Architectural Decision: `mlflow.pytorch.autolog()` NOT Used

`mlflow.pytorch.autolog()` provides full functionality only with **PyTorch Lightning**
(`pytorch_lightning.LightningModule`). This project uses vanilla PyTorch training loops
with MONAI, so autolog would only capture TensorBoard `add_scalar` calls (which we don't
use). All logging is explicit via `ExperimentTracker`.

### 2.4 Comparison with foundation-PLR Reference Implementation

foundation-PLR (`/home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR`) logs:

| What | How | File |
|------|-----|------|
| CPU model | `/proc/cpuinfo` parse + fallback | `src/log_helpers/system_utils.py` |
| Total RAM (GB) | `psutil.virtual_memory().total` | `src/log_helpers/system_utils.py` |
| OS + kernel | `platform.system()` + `platform.release()` | `src/log_helpers/system_utils.py` |
| Python version | `platform.python_version()` | `src/log_helpers/system_utils.py` |
| Library versions | `np.__version__`, `torch.__version__`, etc. | `src/log_helpers/system_utils.py` |
| Git commit | `subprocess: git rev-parse HEAD` | `src/log_helpers/system_utils.py` |
| Full Hydra config | YAML artifact in `config/` | `src/log_helpers/mlflow_utils.py` |
| Hydra log file | `.log` file as artifact | `src/log_helpers/hydra_utils.py` |
| Upstream run IDs | Params for traceability | `src/classification/classifier_log_utils.py` |
| Upstream metrics | Copied from parent runs as params | `src/classification/classifier_log_utils.py` |
| Training time | Wall-clock seconds as param | `src/classification/xgboost_cls/xgboost_main.py` |
| Subject codes | Train/test identifiers as params | `src/classification/classifier_log_utils.py` |
| `num_params` | As param (not metric) | `src/imputation/imputation_log_artifacts.py` |
| Artifact validation | Smoke test before long runs | `tests/mlflow_tests.py` |
| MLflow housekeeping | Batch rename, clear deleted, fix bad runs | `src/log_helpers/mlflow_tools/` |
| Retrain-or-skip | Check if run exists before training | `src/log_helpers/retrain_or_not.py` |
| Run context dict | Capture active run metadata for cross-flow | `src/log_helpers/mlflow_utils.py` |
| Config-driven logging | `log_system_metrics` from YAML config | `configs/mlflow_config/*.yaml` |

**Key patterns we adopt:**
- `sys_` prefix for system params (we use underscore `_`, not slash `/`, to avoid metric naming conflicts)
- All critical data as searchable **params** (not just tags/artifacts)
- Full config as YAML artifact for exact reproduction
- Upstream run IDs for cross-flow traceability
- Training time as explicit param
- Volume/subject identifiers per split
- Artifact store validation before long runs
- Training log file as artifact

---

## 3. Implementation Plan

### Phase 0: Dependencies + System Info Module (P0)

**Step 0.0: Install dependencies**
```bash
uv add psutil nvidia-ml-py
```
`psutil` is required for MLflow system metrics and RAM detection.
`nvidia-ml-py` is required for GPU metrics. Both degrade gracefully on CPU-only.

**Step 0.1: Create `src/minivess/observability/system_info.py`**

Replicates foundation-PLR's `system_utils.py` pattern but adapted for our stack:

```
Functions:
- get_system_params() -> dict[str, str]
    Returns: sys_python_version, sys_os, sys_os_kernel, sys_hostname,
             sys_total_ram_gb, sys_cpu_model
    Fallback chain for CPU: platform.processor() -> /proc/cpuinfo -> "unknown"

- get_library_versions() -> dict[str, str]
    Returns: sys_torch_version, sys_monai_version, sys_torchio_version,
             sys_torchmetrics_version, sys_cuda_version, sys_cudnn_version,
             sys_mlflow_version, sys_numpy_version
    Each wrapped in try/except -> "not_installed"

- get_gpu_info() -> dict[str, str]
    Returns: sys_gpu_count, sys_gpu_model, sys_gpu_vram_mb, sys_gpu_driver_version
    Guarded by torch.cuda.is_available() -> returns {"sys_gpu_count": "0"} if no GPU

- get_git_info() -> dict[str, str]
    Returns: sys_git_commit, sys_git_commit_short, sys_git_branch, sys_git_dirty
    Handles detached HEAD: sys_git_branch = "HEAD (detached at <commit_short>)"
    Handles no .git directory: returns {"sys_git_commit": "unknown"}

- get_all_system_info() -> dict[str, str]
    Combines all above into single dict
```

**Step 0.2: Deprecate duplicate functions in `mlruns_enhancement.py`**

`mlruns_enhancement.py` has `get_software_versions()` (line 70-105) and `get_hardware_spec()`
(line 107-145) which overlap with `system_info.py`. These become thin wrappers calling
`system_info.py` functions, marked as deprecated.

**Tests:** `tests/v2/unit/test_system_info.py` (~15 tests)
- Test each function returns expected keys and non-empty values
- Test graceful degradation when libraries missing (mock `import_module`)
- Test graceful degradation when GPU unavailable (mock `torch.cuda.is_available`)
- Test graceful degradation when no `.git` directory (mock `subprocess.run`)
- Test git detached HEAD state (mock `subprocess.run`)
- Test CPU model detection fallback chain
- Test all values are strings (MLflow params must be str/float/int)

### Phase 1: Enhance ExperimentTracker (P0)

**Modify `src/minivess/observability/tracking.py`**

#### 1.1 Fix run lifecycle handling (CRITICAL)

Currently, if training crashes, `mlflow.start_run()` context manager exits and marks
the run as FINISHED. Failed runs are indistinguishable from successful runs.

```python
@contextmanager
def start_run(self, *, run_name=None, tags=None):
    with mlflow.start_run(
        run_name=run_name,
        tags=merged_tags,
        log_system_metrics=self.config.log_system_metrics,  # config-driven
    ) as run:
        self._run_id = run.info.run_id
        try:
            self._log_config()
            self._log_system_info()      # NEW: auto-called
            self._log_git_hash_safe()     # NEW: safe wrapper
            self._log_frozen_deps_safe()  # NEW: safe wrapper
            self._log_run_description()   # NEW: human-readable note
            yield run.info.run_id
        except Exception:
            mlflow.set_tag("run_status", "FAILED")
            mlflow.set_tag("error_type", type(e).__name__)
            mlflow.end_run(status="FAILED")
            raise
```

All auto-called logging methods wrapped in try/except to prevent metadata collection
failure from aborting training:

```python
def _log_system_info(self) -> None:
    """Log system info. Never raises — logs warning on failure."""
    try:
        mlflow.log_params(get_all_system_info())
    except Exception:
        logger.warning("Failed to log system info", exc_info=True)
```

#### 1.2 Fix param key collision between `_log_config()` and `log_model_info()`

`_log_config()` logs `model_family`, `model_name`, `in_channels`, `out_channels`.
`log_model_info()` calls `model.get_config().to_dict()` which returns keys including
`family` and `name`. MLflow throws `MlflowException` if same key logged with different
value. Fix: `log_model_info()` skips keys already logged by `_log_config()`, only adds
new keys like `trainable_parameters` (as param, not just metric), `arch_*` extras.

#### 1.3 Complete `_log_config()` with ALL TrainingConfig fields

Currently logs 12 params, missing 5 from `TrainingConfig` (models.py:113-129):
- `weight_decay`, `warmup_epochs`, `gradient_clip_val`, `gradient_checkpointing`,
  `early_stopping_patience`

#### 1.4 Add experiment-level tags

```python
def _set_experiment_tags(self) -> None:
    mlflow.set_experiment_tags({
        "project": "minivess",
        "mlflow.note.content": self.config.experiment_description or "",
    })
```

#### 1.5 Add human-readable run description

```python
def _log_run_description(self) -> None:
    mlflow.set_tag(
        "mlflow.note.content",
        f"{self.config.model.name} with {loss_name} loss, "
        f"{self.config.training.max_epochs} epochs, "
        f"{self.config.model.architecture_params.get('filters', 'default')} filters"
    )
```

#### 1.6 Make `log_system_metrics` config-driven

Add `log_system_metrics: bool = True` to `ExperimentConfig`. In CI tests, set to `False`.
This avoids unnecessary overhead and `psutil`/`pynvml` requirements in test environments.

#### 1.7 Fix temp file leak in `log_frozen_deps()` and `log_hydra_config()`

Both use `tempfile.NamedTemporaryFile(delete=False)` but never clean up. Add cleanup
in `finally` block after `mlflow.log_artifact()`.

#### 1.8 Add artifact store validation

Before starting a long training run, verify the artifact store is writable:
```python
def _validate_artifact_store(self) -> None:
    """Smoke test: write/delete a test artifact to verify store is writable."""
    with tempfile.NamedTemporaryFile(suffix=".test", delete=False) as f:
        f.write(b"test")
        test_path = f.name
    try:
        mlflow.log_artifact(test_path, "test")
        # If this fails, we'd rather know now than after 25 hours
    finally:
        Path(test_path).unlink(missing_ok=True)
```

**Tests:** Extend `tests/v2/unit/test_observability.py` (~12 tests)
- Test system info params appear in run
- Test run failure status is set correctly (FAILED on exception)
- Test run KILLED status on abort
- Test `_log_config()` includes all 17 TrainingConfig fields
- Test param key collision between `_log_config()` and `log_model_info()` does not raise
- Test `trainable_parameters` logged as param (not just metric)
- Test experiment-level tags are set
- Test run description is human-readable
- Test `log_system_metrics` config flag is respected
- Test temp file cleanup after `log_frozen_deps()`
- Test graceful degradation when `log_system_info()` fails
- Test artifact store validation

### Phase 2: Wire Logging into Training Pipeline (P0)

**Modify `scripts/train_monitored.py`**

#### 2.1 Wire model info logging

Problem: Model is created inside `run_fold_safe()` (line ~412), not accessible in the
`start_run()` context (line ~704). Solution: On the first fold, call
`tracker.log_model_info(model)` inside `run_fold_safe()`, passing the tracker.

#### 2.2 Wire remaining existing methods

After `tracker.start_run()`:
1. `tracker.log_split_file(args.splits_file)` — split JSON artifact
2. `tracker.log_hydra_config(config_snapshot)` — full experiment config as YAML

Problem: `run_experiment.py` passes individual CLI args, not the full config dict.
Solution: Serialize experiment config to JSON, pass path via `--experiment-config-path` arg.
`train_monitored.py` reads and logs it as artifact.

#### 2.3 Log fold-level data

Since current architecture is **one MLflow run per loss function** (all folds in one run),
log comma-separated fold info:
- `fold_ids`: `"0,1,2"`
- `data_train_volume_ids`: `"mv01,mv03,mv04,..."` (per fold, as artifact)
- `data_val_volume_ids`: `"mv02,mv05,..."` (per fold, as artifact)
- `data_n_train_volumes`: `46`
- `data_n_val_volumes`: `24`

Log per-fold split details as JSON artifact: `splits/fold_details.json`.

#### 2.4 Log additional params

- `loss_name` as param (currently only a tag)
- `compute_profile` name and actual `patch_size`
- `device` (`cuda:0`, `cpu`)
- `cache_rate`, `num_workers` (passed from `train_monitored.py` locals)

#### 2.5 Log training time

```python
import time
start_time = time.monotonic()
# ... training loop ...
training_time = time.monotonic() - start_time
mlflow.log_param("training_time_seconds", round(training_time, 1))
```

#### 2.6 Log training log file as artifact

After training completes, copy the loguru/Python log to MLflow artifacts:
```python
mlflow.log_artifact(str(log_dir / "train.log"), "logs")
```

**Modify `scripts/run_experiment.py`**
1. Already passes `architecture_params` (fixed in this branch)
2. Serialize full experiment config to temp JSON, pass path as CLI arg
3. Log `experiment_config_path` as param

**Tests:** Extend `tests/v2/unit/test_experiment_runner.py` (~8 tests)
- Test model info logged on first fold only
- Test split file logged as artifact
- Test config snapshot logged as artifact
- Test fold-level params logged correctly
- Test loss_name logged as param (not just tag)
- Test training_time_seconds logged as param
- Test log file logged as artifact
- Test device param logged

### Phase 3: Cross-Flow Traceability (P0)

**Modify `src/minivess/orchestration/flows/analysis_flow.py`**

The analysis flow creates runs via `mlflow.start_run()` directly (lines 278, 309),
bypassing `ExperimentTracker`. It and `EvaluationRunner` (line 261) must be updated.

#### 3.1 Add `upstream_run_id` logging

When evaluation/analysis runs are created, log the training run ID:
```python
mlflow.log_param("upstream_training_run_id", training_run_id)
mlflow.log_param("upstream_training_experiment", training_experiment_name)
```

#### 3.2 Add run context propagation utility

Create `get_run_context() -> dict` that captures `{run_id, experiment_name, run_name,
artifact_uri}` from the active run. This dict is passed via Prefect task results between
flows.

#### 3.3 Add system info to non-training runs

Evaluation and analysis runs also need system info, git hash, and frozen deps.
Factor this into a shared `log_run_environment()` function usable by any flow.

**Tests:** `tests/v2/unit/test_cross_flow_traceability.py` (~6 tests)
- Test upstream_training_run_id logged in evaluation run
- Test get_run_context() returns expected keys
- Test analysis flow logs system info
- Test evaluation runner logs system info

### Phase 4: Dataset Metadata Logging (P1)

**Add `log_dataset_profile()` to ExperimentTracker**

Leverages existing `src/minivess/data/profiler.py`:
- `DatasetProfile` needs a `to_dict()` method (currently not JSON-serializable due to
  `pathlib.Path` in `VolumeStats.path` and frozen dataclass)
- Log as params: `data_dataset_name`, `data_n_volumes`, `data_total_size_gb`,
  `data_shape_range`, `data_spacing_range`
- Log as artifact: `dataset/dataset_profile.json`
- Cache profile to disk (`data/raw/.dataset_profile_cache.json`) to avoid re-scanning
  4.5 GB of NIfTI files on every run (~30-60 sec overhead otherwise)

**MLflow native dataset tracking (P2 stretch):**
```python
import mlflow.data
dataset = mlflow.data.from_numpy(
    features=np.zeros((n_volumes, 1)),
    name="minivess",
    digest=dataset_profile_hash,
)
mlflow.log_input(dataset, context="training")
```

**Per-volume evaluation metrics (P1):**
After evaluation, log per-volume metrics table:
```python
mlflow.log_table(per_volume_df, artifact_file="evaluation/per_volume_metrics.json")
```

**Tests:** Extend `tests/v2/unit/test_observability.py` (~6 tests)
- Test dataset params logged
- Test dataset profile JSON artifact created
- Test profile cache hit (no re-scan)
- Test per-volume metrics table logged

### Phase 5: Model Signatures (P1)

**Modify `log_pyfunc_model()` in `tracking.py`**

Add explicit tensor signature for 3D segmentation models:
```python
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec
import numpy as np

input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 1, -1, -1, -1), "input")])
output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 1, -1, -1, -1), "output")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
# Pass to mlflow.pyfunc.log_model(..., signature=signature)
```

**Tests:** 2 tests for signature presence and schema correctness.

### Phase 6: Retroactive Run Update Script (P0)

**Create `scripts/backfill_mlflow_metadata.py`**

#### CRITICAL SAFETY RULES (from reviewer findings):

1. **`mlflow.log_params()` THROWS on existing params with different values.**
   The script must check existing params and skip keys that already exist.
   Same key + same value = OK (idempotent). Same key + different value = exception.

2. **`mlflow.start_run(run_id=...)` silently changes FAILED/RUNNING to FINISHED.**
   The script must preserve original run status:
   ```python
   client = MlflowClient()
   run = client.get_run(run_id)
   original_status = run.info.status
   with mlflow.start_run(run_id=run_id):
       mlflow.log_params(new_params)
   # Restore original status if not FINISHED
   if original_status != "FINISHED":
       client.set_terminated(run_id, status=original_status)
   ```

3. **System info is from CURRENT environment, not original run-time.**
   Add `sys_backfill_note` param: "System info captured at backfill time (2026-02-28),
   not at original training time. Same machine, software may have been updated."

4. **System metrics (`log_system_metrics=True`) does NOT work with existing run IDs.**
   Known MLflow bug (#10253). Do not attempt to backfill system metrics.

5. **Use absolute path for tracking URI** to prevent creating `mlruns/` in wrong directory.
   ```python
   mlflow.set_tracking_uri(str(Path(__file__).resolve().parents[1] / "mlruns"))
   ```

6. **Skip runs in RUNNING state** (likely crashed mid-training). Do not mark as FINISHED.

#### Backfill content:

For each FINISHED run in `dynunet_loss_variation_v2`:
```python
new_params = {}
# System info (only params not already present)
new_params.update({k: v for k, v in system_info.items() if k not in existing_params})
# Missing training config fields
new_params.update({k: v for k, v in training_params.items() if k not in existing_params})
# Dataset metadata
new_params.update({k: v for k, v in dataset_params.items() if k not in existing_params})
# Loss function as param (from existing tag)
if "loss_name" not in existing_params:
    new_params["loss_name"] = existing_tags.get("loss_function", "unknown")
# Backfill note
new_params["sys_backfill_note"] = "Backfilled 2026-02-28, same machine"
```

**Tests:** `tests/v2/unit/test_backfill_metadata.py` (~10 tests)
- Test backfill adds new params without touching existing ones
- Test backfill is idempotent (running twice doesn't throw)
- Test backfill preserves FAILED/RUNNING run status
- Test backfill skips RUNNING runs
- Test backfill adds `sys_backfill_note`
- Test backfill uses absolute tracking URI
- Test param collision handling (existing key + same value = OK)
- Test param collision handling (existing key + different value = skip with warning)
- All tests use isolated `tmp_path` MLflow backends, never real `mlruns/`

### Phase 7: CLAUDE.md Update (P0)

**Add MLflow section to CLAUDE.md** describing:
1. How MLflow works in this repo (local filesystem backend, no server)
2. Actual experiment names (not aspirational)
3. What gets logged automatically (params, metrics, artifacts, system metrics)
4. Naming conventions: `sys_` prefix for system, `arch_` for architecture, `data_` for dataset
5. Cross-flow traceability pattern (upstream_run_id)
6. Run lifecycle (FINISHED, FAILED, KILLED)
7. How to add new params/metrics/artifacts
8. How to retroactively update existing runs
9. Key files

### Phase 8: MLflow Housekeeping Scripts (P2)

**Create `scripts/mlflow_tools/` directory:**
- `clear_deleted_runs.py` — permanently remove soft-deleted runs, reclaim disk
- `batch_rename_artifact_uri.py` — fix paths when mlruns moves between machines
- `check_run_integrity.py` — verify all runs have required params/tags

These are operational tools, not blocking training. Modeled after
`foundation-PLR/src/log_helpers/mlflow_tools/`.

---

## 4. Param Naming Convention

Standardized prefixes for all MLflow params. **NOTE:** foundation-PLR uses `sys/` (slash)
prefix. We deliberately use `sys_` (underscore) to avoid conflicts with MLflow's metric
naming convention (e.g. `train/loss`). Slash in param names can confuse MLflow UI filtering.

| Prefix | Category | Example |
|--------|----------|---------|
| (none) | Training hyperparams | `learning_rate`, `batch_size`, `max_epochs` |
| `arch_` | Model architecture | `arch_filters`, `arch_deep_supervision` |
| `sys_` | System/environment | `sys_python_version`, `sys_gpu_model` |
| `data_` | Dataset metadata | `data_n_volumes`, `data_shape_range` |
| `loss_` | Loss function config | `loss_name`, `loss_weights` |
| `eval_` | Evaluation config | `eval_bootstrap_n`, `eval_ci_level` |
| `upstream_` | Cross-flow links | `upstream_training_run_id` |

**Param value length limit:** MLflow params have a 500-character limit. Long values
(volume ID lists, config strings) must be truncated or logged as artifacts instead.

---

## 5. Tag Naming Convention

Tags are for filtering/grouping, not for structured data:

| Tag | Purpose | Example |
|-----|---------|---------|
| `loss_function` | Primary filter | `cbdice_cldice` |
| `loss_type` | Backward compat alias | `cbdice_cldice` |
| `num_folds` | CV structure | `3` |
| `started_at` | ISO timestamp | `2026-02-27T07:53:04+00:00` |
| `model_family` | Architecture family | `dynunet` |
| `model_name` | Architecture name | `dynunet` |
| `git_commit` | Version control | `abc123def...` |
| `compute_profile` | Hardware tier | `gpu_low` |
| `experiment_group` | Logical grouping | `loss_ablation_v2` |
| `champion_*` | Champion selection | See champion_tagger.py |
| `run_status` | Lifecycle (FAILED/KILLED) | `FAILED` |
| `error_type` | Exception class on failure | `RuntimeError` |
| `mlflow.note.content` | Human-readable description | `DynUNet cbdice_cldice 100ep` |

---

## 6. Artifact Directory Layout

Standard artifact paths within each MLflow run:

```
artifacts/
  checkpoints/          # Model checkpoints (.pth)
    best_val_loss.pth
    best_val_compound_masd_cldice.pth
    last.pth
  config/               # Experiment configuration
    resolved_config.yaml
    experiment_config.json
  environment/          # Software environment
    frozen_deps.txt
    monai_config.txt
    system_info.json
  splits/               # Data splits
    3fold_seed42.json
    fold_details.json   # Per-fold volume IDs
  dataset/              # Dataset metadata
    dataset_profile.json
  logs/                 # Training logs
    train.log
  history/              # Training history
    metric_history.json
  evaluation/           # Evaluation results
    per_volume_metrics.json
  model/                # Pyfunc model (analysis flow)
    MLmodel
    model_config.json
    checkpoint.pth
```

---

## 7. Metrics Naming Convention

### Per-Epoch Training Metrics (step = epoch)

| Metric | Description | Logged By |
|--------|-------------|-----------|
| `train_loss` | Training loss | `trainer.fit()` |
| `val_loss` | Validation loss | `trainer.fit()` |
| `train_dice` | Training Dice | `trainer.fit()` |
| `val_dice` | Validation Dice | `trainer.fit()` |
| `val_f1_foreground` | Validation F1 | `trainer.fit()` |
| `learning_rate` | Current LR (scheduler) | `trainer.fit()` |
| `val_cldice` | Centerline Dice (every 5 epochs) | `trainer.fit()` |
| `val_masd` | Mean Avg Surface Distance (every 5 epochs) | `trainer.fit()` |
| `val_compound_masd_cldice` | Primary metric (every 5 epochs) | `trainer.fit()` |

### Post-Training Evaluation Metrics (step = 0)

| Metric | Description | Logged By |
|--------|-------------|-----------|
| `eval_fold{N}_{metric}` | Per-fold point estimate | `log_evaluation_results()` |
| `eval_fold{N}_{metric}_ci_lower` | CI lower bound | `log_evaluation_results()` |
| `eval_fold{N}_{metric}_ci_upper` | CI upper bound | `log_evaluation_results()` |
| `eval_fold{N}_{metric}_ci_level` | CI level (0.95) | `log_evaluation_results()` |
| `trainable_parameters` | Model parameter count | `log_model_info()` |

### System Metrics (auto-logged by MLflow, step = wall time)

MLflow logs **12 system metrics** when `log_system_metrics=True`:

| Metric | Description |
|--------|-------------|
| `system/cpu_utilization_percentage` | CPU usage |
| `system/system_memory_usage_megabytes` | RAM usage |
| `system/system_memory_usage_percentage` | RAM percentage |
| `system/gpu_utilization_percentage` | GPU compute |
| `system/gpu_memory_usage_megabytes` | GPU VRAM |
| `system/gpu_memory_usage_percentage` | GPU VRAM percentage |
| `system/gpu_power_usage_watts` | GPU power |
| `system/gpu_power_usage_percentage` | GPU power percentage |
| `system/network_receive_megabytes` | Network in |
| `system/network_transmit_megabytes` | Network out |
| `system/disk_usage_megabytes` | Disk usage |
| `system/disk_available_megabytes` | Disk available |

**Caveat:** System metrics require `psutil` (CPU/RAM/disk/network) and `nvidia-ml-py`
(GPU). GPU metrics silently skipped if NVIDIA drivers not installed. Metrics poll every
10 seconds in a background thread.

**Interaction with custom SystemMonitor:** Both `log_system_metrics` and our custom
`SystemMonitor` (scripts/system_monitor.py) poll GPU stats. This is acceptable — MLflow
writes to the run's metrics, SystemMonitor writes to a CSV. No conflict, just redundant
GPU queries (negligible overhead).

---

## 8. Retroactive Backfill Strategy

For the 9 existing runs in `dynunet_loss_variation_v2`:

| What | How | Status |
|------|-----|--------|
| `arch_filters` | Filesystem write to `params/arch_filters` | DONE (this branch) |
| System info (`sys_*`) | `mlflow.start_run(run_id=...)` + `log_params()` | TODO |
| Missing training params | `mlflow.start_run(run_id=...)` + `log_params()` | TODO |
| Dataset metadata (`data_*`) | `mlflow.start_run(run_id=...)` + `log_params()` | TODO |
| Loss function as param | Copy from tag `loss_function` to param `loss_name` | TODO |
| Volume IDs per fold | Extract from splits JSON, log as param/artifact | TODO |
| Training time | Not recoverable (not logged originally) | SKIPPED |
| System metrics (GPU/CPU) | **Cannot backfill** (MLflow bug #10253) | SKIPPED |
| `sys_backfill_note` | Metadata about backfill provenance | TODO |
| `loss_type` tag alias | Already done via `mlruns_enhancement.py` | DONE |
| Software versions tags | Already done via `mlruns_enhancement.py` | DONE |
| Hardware spec tags | Already done via `mlruns_enhancement.py` | DONE |
| Git commit tag | Already done via `mlruns_enhancement.py` | DONE |

**Critical safety notes:**
- `mlflow.log_params()` THROWS if a param key already exists with a DIFFERENT value
- `mlflow.start_run(run_id=...)` silently changes FAILED/RUNNING status to FINISHED
- System info captured at backfill time may differ from original run-time
- System metrics CANNOT be backfilled to existing runs (MLflow bug #10253)

---

## 9. Dependencies to Add

```bash
# MUST be done before any implementation (Phase 0, Step 0)
uv add psutil nvidia-ml-py

# psutil: CPU/memory/disk metrics, required by MLflow log_system_metrics
# nvidia-ml-py: GPU utilization/memory/power metrics via pynvml
```

Both degrade gracefully on CPU-only environments (CI runners, Docker without GPU).

---

## 10. Testing Strategy

Each phase includes tests BEFORE implementation (TDD):

| Phase | Test File | Test Count (est.) |
|-------|-----------|-------------------|
| Phase 0 | `tests/v2/unit/test_system_info.py` | 15 |
| Phase 1 | `tests/v2/unit/test_observability.py` (extend) | 12 |
| Phase 2 | `tests/v2/unit/test_experiment_runner.py` (extend) | 8 |
| Phase 3 | `tests/v2/unit/test_cross_flow_traceability.py` | 6 |
| Phase 4 | `tests/v2/unit/test_observability.py` (extend) | 6 |
| Phase 5 | `tests/v2/unit/test_observability.py` (extend) | 2 |
| Phase 6 | `tests/v2/unit/test_backfill_metadata.py` | 10 |
| Phase 7 | N/A (documentation) | 0 |
| Phase 8 | N/A (P2 tools) | 0 |

Total: **~59 new tests**

### Critical test scenarios (reviewer-identified):
- Param collision: `_log_config()` + `log_model_info()` both called in one run
- Backfill idempotency: run backfill twice without throwing
- Backfill on RUNNING runs: must not mark as FINISHED
- System info with no GPU: returns sensible defaults
- System info with no `.git`: returns "unknown"
- `log_system_metrics=True` with no `psutil`: MLflow degrades gracefully
- Temp file cleanup after `log_frozen_deps()` and `log_hydra_config()`
- GPU-dependent tests: `@pytest.mark.skipif(not torch.cuda.is_available())`
- All backfill tests use isolated `tmp_path` MLflow backends, never real `mlruns/`

---

## 11. Success Criteria

After implementation, every MLflow training run must have:

1. **Reproducible environment** — exact Python, PyTorch, MONAI, CUDA versions logged
2. **Reproducible architecture** — all model architecture params (filters, strides, etc.)
3. **Reproducible data** — dataset name, volume count, volume IDs, split sizes, patch size
4. **Reproducible training** — ALL hyperparams (LR, weight_decay, grad_clip, warmup, etc.)
5. **Traceable code** — git commit hash, branch, dirty state, frozen dependencies
6. **Observable hardware** — GPU/CPU/memory utilization via MLflow system metrics
7. **Full config artifact** — complete experiment YAML for exact reproduction
8. **Queryable params** — all critical info as MLflow params (not just tags/artifacts)
9. **Lifecycle tracking** — FINISHED, FAILED, KILLED status correctly set
10. **Cross-flow links** — evaluation runs link back to training runs
11. **Human-readable** — run description via `mlflow.note.content`
12. **Training time** — wall-clock seconds logged as param
13. **Volume identifiers** — exact volumes per fold logged for reproducibility

A third party should be able to answer from MLflow alone:
- "What GPU was this trained on?" -> `sys_gpu_model`
- "What model width was used?" -> `arch_filters`
- "What loss function and weights?" -> `loss_name`, `loss_config`
- "How many volumes per fold?" -> `data_n_train_volumes`, `data_n_val_volumes`
- "Which volumes were in the validation set?" -> `data_val_volume_ids`
- "What software stack?" -> `sys_python_version`, `sys_torch_version`, `sys_monai_version`
- "What code version?" -> `sys_git_commit`, `sys_git_branch`, `sys_git_dirty`
- "Did this run crash?" -> `run.info.status == "FAILED"`
- "Which training run produced this evaluation?" -> `upstream_training_run_id`
- "How long did training take?" -> `training_time_seconds`

---

## 12. Execution Order

```
0. uv add psutil nvidia-ml-py
1. Phase 0: system_info.py module (RED -> GREEN -> VERIFY)
2. Phase 1: ExperimentTracker enhancements (RED -> GREEN -> VERIFY)
3. Phase 2: Training pipeline wiring (RED -> GREEN -> VERIFY)
4. Phase 3: Cross-flow traceability (RED -> GREEN -> VERIFY)
5. Phase 7: CLAUDE.md update (documentation)
6. Phase 6: Retroactive backfill script (run on existing mlruns)
7. Phase 4: Dataset metadata (RED -> GREEN -> VERIFY)
8. Phase 5: Model signatures (RED -> GREEN -> VERIFY)
9. Commit, push, PR
10. THEN launch half-width training (with proper logging!)
11. Phase 8: Housekeeping scripts (P2, separate PR)
```

**Rationale for order:**
- Dependencies (step 0) before anything else
- Core system info (Phase 0) needed by all subsequent phases
- ExperimentTracker (Phase 1) is the central piece
- Training wiring (Phase 2) makes logging actually happen
- Cross-flow (Phase 3) is P0 for analysis flow integrity
- CLAUDE.md (Phase 7) before backfill so the patterns are documented
- Backfill (Phase 6) after all param schemas are finalized
- Dataset metadata (Phase 4) and signatures (Phase 5) are P1
- Housekeeping (Phase 8) is P2, separate PR

---

## 13. Design Decision: Nested Runs for Folds

**Current architecture:** One MLflow run per loss function. All 3 folds share a single run.
Fold metrics are logged with epoch-step offsets (fold 0: steps 0-99, fold 1: steps 100-199).

**Nested runs alternative:** Parent run = loss function, child runs = individual folds.
Each child has its own metric history, params, and artifacts.

**Decision: DEFER to separate PR.** Nested runs would be cleaner but require significant
refactoring of `train_monitored.py`, the analysis flow, and DuckDB extraction. The current
architecture works (fold-level data is distinguishable via step offsets and the analysis
flow already handles it). Nested runs are a P1 improvement for a future PR.

For now, fold-level info is logged as:
- `fold_ids` param: `"0,1,2"`
- `splits/fold_details.json` artifact: full per-fold volume lists
- Metrics use step offsets within the single run

---

## 14. CLAUDE.md MLflow Section (Draft v2)

To be added to CLAUDE.md:

```markdown
## MLflow Tracking Architecture

### Backend
- **Local filesystem** backend: `mlruns/` directory (no server required)
- Tracking URI: `mlruns` (set via code, resolved to absolute path)
- Each run stores params, metrics, tags, artifacts as plain files
- Run lifecycle: FINISHED (success), FAILED (exception), KILLED (abort)

### Experiments (Actual)
| Experiment | Purpose | Created By |
|-----------|---------|-----------|
| `dynunet_loss_variation_v2` | Training runs (loss ablation, 4 losses x 3 folds) | `train_monitored.py` |
| `dynunet_half_width_v1` | Training runs (width ablation) | `train_monitored.py` |
| `minivess_evaluation` | Evaluation runs (ensembles + analysis) | `analysis_flow.py` |

### Param Prefixes
- (none): Training hyperparams (learning_rate, batch_size, training_time_seconds)
- `arch_`: Model architecture (arch_filters, arch_deep_supervision)
- `sys_`: System/environment (sys_python_version, sys_gpu_model)
- `data_`: Dataset metadata (data_n_volumes, data_train_volume_ids)
- `loss_`: Loss function config (loss_name, loss_weights)
- `eval_`: Evaluation config (eval_bootstrap_n, eval_ci_level)
- `upstream_`: Cross-flow links (upstream_training_run_id)
NOTE: We use `sys_` (underscore) not `sys/` (slash) to avoid metric naming conflicts.

### Automatic Logging (ExperimentTracker.start_run())
On every `start_run()`, automatically logs:
- All TrainingConfig params (17 fields including weight_decay, warmup_epochs)
- Architecture params from ModelConfig.architecture_params (arch_ prefix)
- System info: Python, PyTorch, MONAI, CUDA, cuDNN, OS, GPU, RAM (sys_ prefix)
- Git commit hash, branch, dirty state
- Frozen dependencies artifact (uv pip freeze)
- Human-readable run description (mlflow.note.content)
- MLflow system metrics (GPU/CPU/memory/disk, 12 metrics at 10s intervals)
On failure: sets run status to FAILED with error_type tag.

### Cross-Flow Traceability
- Training runs -> `upstream_training_run_id` logged in evaluation runs
- Analysis flow propagates run context between Prefect tasks
- `get_run_context()` captures {run_id, experiment, artifact_uri}

### Key Files
- `src/minivess/observability/tracking.py` — ExperimentTracker class
- `src/minivess/observability/system_info.py` — System info collection
- `src/minivess/observability/analytics.py` — DuckDB analytics over runs
- `src/minivess/pipeline/mlruns_enhancement.py` — Post-hoc tag enhancement
- `src/minivess/pipeline/champion_tagger.py` — Champion model tagging
- `src/minivess/pipeline/duckdb_extraction.py` — DuckDB extraction from mlruns
- `scripts/backfill_mlflow_metadata.py` — Retroactive run update tool

### Autolog Decision
`mlflow.pytorch.autolog()` is NOT used. It only works with PyTorch Lightning.
This project uses vanilla PyTorch + MONAI. All logging is explicit.
```

---

## 15. Review Findings Summary

This plan was reviewed by 3 specialized agents. Key findings incorporated:

### From MLflow API reviewer:
- Added run lifecycle handling (FAILED/KILLED status) — was completely missing
- Added experiment-level tags
- Added human-readable run descriptions (mlflow.note.content)
- Corrected system metrics list from 5 to 12
- Documented `log_system_metrics` bug with existing run IDs (#10253)
- Added model signatures to pyfunc logging
- Added mlflow.log_table() for per-volume metrics
- Documented autolog decision
- Noted 500-char param value limit

### From foundation-PLR comparison reviewer:
- Added upstream run ID linking for cross-flow traceability (CRITICAL gap)
- Added training time as explicit param
- Added artifact store validation (smoke test before long runs)
- Made `log_system_metrics` config-driven, not hardcoded
- Added volume identifiers per fold (not just counts)
- Added training log file as artifact
- Added MLflow housekeeping scripts (Phase 8)
- Added `trainable_parameters` as both param AND metric
- Documented `sys_` vs `sys/` prefix deviation
- Added run context propagation utility

### From implementation reviewer:
- Identified `log_model_info()` param key collision with `_log_config()` (would crash)
- Identified model not accessible in `start_run()` context (architecture gap)
- Identified config snapshot not wired through CLI args
- Identified fold_id cannot be static param (varies within run)
- Identified temp file leak in `log_frozen_deps()` and `log_hydra_config()`
- Identified backfill risks: param overwrite throws, status change, wrong environment
- Identified analysis flow bypasses ExperimentTracker
- Identified `cache_rate` not in DataConfig
- Identified `DatasetProfile` not JSON-serializable
- Increased test count from 38 to ~59
- Added GPU test skip guards
- Added isolated tmp_path requirement for backfill tests
