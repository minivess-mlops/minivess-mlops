# Multi-Metric Downstream Double-Check

> **Created:** 2026-02-26
> **Purpose:** Ensure ALL artifacts are saved during training so we NEVER need to re-train
> **Status:** PRE-TRAINING AUDIT — Must resolve before launching v2 training

---

## 0. Downstream Consumer Inventory

The training pipeline produces artifacts consumed by 4 independent Prefect flows
using **MLflow as the inter-flow contract**:

| Prefect Flow | Consumes | Produces | Serving |
|-------------|----------|----------|---------|
| **Flow 2: Training** | Raw data, configs | Checkpoints, metrics, predictions | — |
| **Flow 3: Analysis** | Checkpoints, predictions, metrics | Reports, comparisons, ensemble models | MLflow local serving |
| **Flow 4: Deployment** | Registered models | Served endpoints, MONAI Deploy packages | BentoML + MONAI Deploy |

Additionally:
- **MLflow Model Registry** — Champion/challenger aliases, model versioning, promotion workflow
- **MLflow Model Evaluation** (`mlflow.evaluate()`) — Post-training evaluation framework
- **MONAI Deploy** (`monai-deploy`) — Clinical deployment packaging in Flow 4
- **BentoML** — Production-grade serving in Flow 4 (already uses `bentoml.mlflow.get()`)
- **Ensemble construction** — Load multiple checkpoints (cross-fold, cross-loss) for combination

---

## 1. Current State — What Training Saves Now

### Per Fold (3 folds per loss):

| Artifact | Saved to MLflow? | Saved to disk? | Contents |
|----------|:---:|:---:|---------|
| `best_val_loss.pth` | YES (as raw artifact) | /tmp (lost on reboot) | model + optimizer + scheduler + metadata |
| `best_val_dice.pth` | NO | /tmp (lost on reboot) | model + optimizer + scheduler + metadata |
| `best_val_f1_foreground.pth` | NO | /tmp (lost on reboot) | model + optimizer + scheduler + metadata |
| `last.pth` | NO | /tmp (lost on reboot) | model + optimizer + scheduler + metadata |
| `metric_history.json` | NO | /tmp (lost on reboot) | per-epoch metrics + wall time |
| Training metrics (per epoch) | YES (mlflow.log_metrics) | — | train_loss, val_loss, dice, f1, lr |
| Evaluation metrics (post-train) | YES (mlflow.log_metrics) | — | clDice, DSC, MASD with 95% CI |
| Config params | YES (mlflow.log_params) | — | model, training, compute settings |
| Tags | YES (mlflow.set_tag) | — | loss_function, model_family, git_commit |
| Frozen deps | YES (artifact) | — | uv pip freeze output |
| Resolved config | YES (artifact) | — | YAML config snapshot |
| Split file | YES (artifact) | — | Train/val volume IDs |

### What's Missing (per fold):

| Artifact | Why Needed | Impact if Missing |
|----------|-----------|-------------------|
| **`best_val_dice.pth` in MLflow** | Ensemble by best-dice checkpoint | Must re-train to get this checkpoint |
| **`best_val_f1_foreground.pth` in MLflow** | Ensemble by best-F1 checkpoint | Must re-train |
| **`best_val_cldice.pth`** | NEW: best by centreline Dice | Not computed, must add & train |
| **`best_val_masd.pth`** | NEW: best by boundary distance | Not computed, must add & train |
| **`best_val_compound_masd_cldice.pth`** | NEW: best by compound metric | Not computed, must add & train |
| **`last.pth` in MLflow** | Resume training, late-epoch analysis | Lost on /tmp cleanup |
| **`metric_history.json` in MLflow** | Post-hoc training curve analysis | Lost on /tmp cleanup |
| **Validation predictions (numpy)** | Compute ANY new metric without re-inference | Impossible to compute new metrics post-hoc |
| **Soft predictions (probabilities)** | Ensemble averaging, calibration | Must re-run inference for ensembles |
| **`mlflow.pyfunc.log_model()` call** | Model Registry, serving, MONAI Deploy | Models cannot be registered or served |
| **ModelSignature** | Input/output schema for BentoML/MLflow serving | Serving breaks on type mismatches |
| **Input example** | ONNX export, model documentation | Manual reconstruction needed |
| **GradScaler state_dict** | Resume mixed-precision training | Training resume gives different results |
| **Model summary (torchinfo)** | Architecture documentation | Must re-create from code |

---

## 2. CRITICAL GAPS

### GAP 1: Only Primary Metric Checkpoint Uploaded to MLflow

**Current code** (`trainer.py:403-406`):
```python
if self.tracker is not None:
    self.tracker.log_artifact(best_path, artifact_path="checkpoints")
```

This runs for EVERY improved metric — so all `best_*.pth` files ARE uploaded. **This gap is less severe than initially feared**, but needs verification.

**However**, `last.pth` and `metric_history.json` are NOT uploaded.

### GAP 2: No `log_model()` — Models Are Not Servable

**Current:** Checkpoints logged as raw `.pth` artifacts via `log_artifact()`.
**Problem:** Raw `.pth` files cannot be:
- Loaded via `mlflow.pytorch.load_model()`
- Registered in Model Registry
- Served via `mlflow models serve`
- Consumed by `bentoml.mlflow.get()`
- Packaged by MONAI Deploy

**Fix needed:** Add `mlflow.pyfunc.log_model()` with:
- `MiniVessServingModel(PythonModel)` wrapper
- `ModelSignature` with tensor specs: `(-1, 1, -1, -1, -1)` → `(-1, 2, -1, -1, -1)`
- `input_example`: small `(1, 1, 8, 16, 16)` numpy array
- `artifacts`: checkpoint path + model config
- `pip_requirements`: torch, monai, numpy
- `metadata`: roi_size, num_classes, model_family

### GAP 3: No Validation Predictions Saved

**Problem:** Without saved predictions, computing new metrics requires re-running
sliding window inference on all validation volumes. This takes ~10-15 minutes per fold.

**Fix:** Save compressed numpy arrays after evaluation:
```python
np.savez_compressed(
    f"predictions/fold_{fold}/vol_{name}.npz",
    hard_pred=pred.astype(np.uint8),     # ~0.3 MB compressed
    soft_pred=probs.astype(np.float16),  # ~2 MB compressed
)
```

**Storage cost:** ~300 MB for full experiment (4 losses × 3 folds × 23 volumes/fold).
Negligible compared to checkpoint storage (~4 GB).

### GAP 4: `register_model()` Uses Deprecated `stage` Parameter

**Current** (`tracking.py:243-260`):
```python
def register_model(self, model_name, *, stage="Staging"):
    model_uri = f"runs:/{self._run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)
```

**Problem:** MLflow deprecated stages in v2.9. Aliases are the replacement.
**Fix:** Use `champion`/`challenger` aliases:
```python
client.set_registered_model_alias(model_name, "challenger", mv.version)
```

### GAP 5: New Metrics Not Computed During Training

**Current validation:** TorchMetrics (val_dice, val_f1_foreground) — fast, GPU, approximate.
**Missing:** MetricsReloaded (val_cldice, val_masd) — exact, CPU, requires full-volume inference.

Without computing these during training, we cannot:
- Save `best_val_cldice.pth` (best by centreline topology)
- Save `best_val_masd.pth` (best by boundary accuracy)
- Save `best_val_compound_masd_cldice.pth` (best by compound metric)

**This is the primary reason we MUST re-train** — previous runs did not compute
these metrics at all during validation, so no checkpoints exist for them.

---

## 3. Checkpoint Count Analysis

### Previous Training (v1): 3 tracked metrics

| Tracked Metric | Direction | Checkpoint File |
|---------------|-----------|-----------------|
| val_loss | minimize | best_val_loss.pth |
| val_dice | maximize | best_val_dice.pth |
| val_f1_foreground | maximize | best_val_f1_foreground.pth |

**Per fold:** 3 best_*.pth + last.pth + metric_history.json = 5 files (~338 MB)
**Per loss (3 folds):** 15 files (~1 GB)
**Full experiment (4 losses):** 60 files (~4 GB)

### Proposed Training (v2): 5+ tracked metrics

| Tracked Metric | Direction | Checkpoint File | NEW? |
|---------------|-----------|-----------------|------|
| val_loss | minimize | best_val_loss.pth | no |
| val_dice | maximize | best_val_dice.pth | no |
| val_cldice | maximize | best_val_cldice.pth | **YES** |
| val_masd | minimize | best_val_masd.pth | **YES** |
| val_compound_masd_cldice | maximize | best_val_compound_masd_cldice.pth | **YES** |

**Per fold:** 5 best_*.pth + last.pth + metric_history.json + predictions/ = ~7 files + predictions (~470 MB)
**Per loss (3 folds):** 21 files + predictions (~1.6 GB)
**Full experiment (4 losses):** 84 files + predictions (~6.7 GB)

### Total Storage Budget

| Component | Size | Location |
|-----------|------|----------|
| Checkpoints (84 .pth files) | ~5.7 GB | MLflow artifacts |
| Predictions (compressed npz) | ~0.3 GB | MLflow artifacts |
| Metric histories (12 JSON) | ~0.5 MB | MLflow artifacts |
| Logged model (pyfunc per run) | ~0.1 GB | MLflow model |
| Training metrics/params/tags | ~0.1 GB | MLflow backend |
| **Total** | **~6.2 GB** | MLflow |

Acceptable for a research project. Dropbox sync handles backup.

---

## 4. MLflow Serving Requirements

### 4.1 Analysis Flow (Flow 3) — MLflow Local Serving

The Analysis flow loads checkpoints for:
- Post-hoc evaluation with new metrics
- Cross-loss comparison
- Paired bootstrap statistical tests
- Ensemble construction and evaluation

**Requires:**
- `mlflow.pyfunc.load_model("runs:/{run_id}/model")` — load any checkpoint
- `mlflow.evaluate()` — evaluate on test set with MetricsReloaded
- `SlidingWindowInferenceRunner` wrapped in the pyfunc `predict()` method

### 4.2 Deployment Flow (Flow 4) — BentoML + MONAI Deploy

**BentoML** (`deployment/bento/service.py`):
- Already uses `bentoml.mlflow.get("minivess-segmentor:latest")`
- Requires: registered model in MLflow with `log_model()` (currently broken)
- Input spec: `NumpyNdarray(shape=(-1, 1, -1, -1, -1), dtype="float32")`

**MONAI Deploy** (future):
- Packages model as a MAP (MONAI Application Package)
- Needs: ONNX export OR PyTorch model + transforms + inference config
- Will read from MLflow model registry
- Requires: model weights + inference transforms + sliding window config

### 4.3 ModelSignature Definition

```python
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
import numpy as np

signature = ModelSignature(
    inputs=Schema([
        TensorSpec(np.dtype(np.float32), (-1, 1, -1, -1, -1), name="images"),
    ]),
    outputs=Schema([
        TensorSpec(np.dtype(np.float32), (-1, 2, -1, -1, -1), name="logits"),
    ]),
)
```

### 4.4 PyFunc Serving Wrapper

```python
class MiniVessServingModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load model from checkpoint artifact
        # Build SlidingWindowInferenceRunner
        ...

    def predict(self, context, model_input, params=None):
        # Run sliding window inference
        # Return soft predictions (probabilities)
        ...
```

This wrapper:
- Encapsulates model loading + sliding window inference
- Makes the model servable via `mlflow models serve`
- Compatible with BentoML's `bentoml.mlflow.get()`
- Can be packaged into MONAI Deploy MAP

---

## 5. MLflow Model Evaluation

Per user requirement, we use `mlflow.evaluate()` after training to:
- Evaluate each checkpoint against the validation/test set
- Compute MetricsReloaded metrics
- Store evaluation results as MLflow metrics with confidence intervals
- Enable MLflow UI comparison across runs

The existing `EvaluationRunner` and `ExperimentTracker.log_evaluation_results()`
handle this. The key addition is that we must evaluate **every best_*.pth checkpoint**,
not just the primary one.

---

## 6. MLflow Model Registry Tags

For downstream selection (ensemble construction, deployment promotion):

| Tag | Set When | Example Value |
|-----|----------|---------------|
| `loss_type` | Training start | `"dice_ce"` |
| `fold_id` | Training start | `"0"` |
| `model_family` | Training start | `"dynunet"` |
| `git_commit` | Training start | `"d543e52..."` |
| `checkpoint_metric` | Per checkpoint | `"val_compound_masd_cldice"` |
| `best_val_dice` | Post-training | `"0.843"` |
| `best_val_cldice` | Post-training | `"0.904"` |
| `best_val_masd` | Post-training | `"2.31"` |
| `is_best_for_loss` | Cross-loss comparison | `"true"` |
| `is_best_overall` | Cross-loss comparison | `"true"` |
| `ensemble_candidate` | Ensemble selection | `"true"` |

**Registered model aliases** (replace deprecated stages):
- `champion` — current best model in production
- `challenger` — candidate for A/B testing
- Per-loss aliases: `best-dice-ce`, `best-cbdice`, etc.

---

## 7. Artifact Completeness Checklist

**All items must be true before training v2 starts:**

### Training Artifacts (per fold)
- [ ] All tracked metric checkpoints uploaded to MLflow (not just primary)
- [ ] `last.pth` uploaded to MLflow
- [ ] `metric_history.json` uploaded to MLflow
- [ ] GradScaler state_dict included in checkpoints
- [ ] Validation predictions saved as compressed .npz and uploaded

### MLflow Model (per run)
- [ ] `mlflow.pyfunc.log_model()` called with serving wrapper
- [ ] ModelSignature defined (tensor specs for 3D segmentation)
- [ ] Input example saved (small numpy array)
- [ ] pip_requirements specified (torch, monai, numpy)
- [ ] Model metadata (roi_size, num_classes, model_family)

### MLflow Tags (per run)
- [ ] `loss_type`, `fold_id`, `model_family`, `git_commit` set
- [ ] `checkpoint_metric` tag on each logged checkpoint
- [ ] Post-training best metric values logged as tags

### Configuration
- [ ] 5 tracked metrics in YAML (val_loss, val_dice, val_cldice, val_masd, val_compound)
- [ ] cbdice_cldice loss registered in factory (replaces warp)
- [ ] MetricsReloaded computed during validation (clDice + MASD)
- [ ] Compound metric formula implemented and tested

---

## 8. What Can Wait (NOT Needed Before Training)

These are downstream tasks that consume artifacts but don't affect what we save:

| Task | Prefect Flow | When | Why It Can Wait |
|------|-------------|------|-----------------|
| Register model in registry | Analysis (Flow 3) | Post-training | Just reads existing artifacts |
| Champion/challenger promotion | Analysis (Flow 3) | Post-training | Tags + aliases on existing versions |
| BentoML serving | Deployment (Flow 4) | Separate PR | Reads from registry |
| MONAI Deploy packaging | Deployment (Flow 4) | Separate PR | Reads from registry |
| Ensemble construction | Analysis (Flow 3) | Post-training | Loads from saved checkpoints |
| `mlflow.evaluate()` framework | Analysis (Flow 3) | Post-training | Uses saved predictions |
| Cross-loss comparison report | Analysis (Flow 3) | Post-training | Reads MLflow metrics |
| ONNX export | Deployment (Flow 4) | Separate PR | Loads checkpoint + input example |

**The key principle:** Save everything now; consume it later. Storage is cheap; re-training is expensive.

---

## 9. Implementation Order

1. **Add `mlflow.pyfunc.log_model()` to trainer** — makes models servable (GAP 2)
2. **Upload ALL checkpoints to MLflow** — verify current code already does this (GAP 1)
3. **Upload `last.pth` + `metric_history.json`** to MLflow (GAP 1)
4. **Save validation predictions** as compressed .npz (GAP 3)
5. **Add GradScaler state** to checkpoint dict
6. **Implement MetricsReloaded validation metrics** (val_cldice, val_masd) — Phase 2 of XML plan
7. **Implement compound metric** (val_compound_masd_cldice) — Phase 2 of XML plan
8. **Add cbdice_cldice loss** — Phase 1 of XML plan
9. **Update YAML config** with 5 tracked metrics — Phase 3 of XML plan
10. **Run training** — Phase 4 of XML plan

Steps 1-5 are **pre-requisites** that must be added to the XML plan BEFORE the
existing Phase 1.
