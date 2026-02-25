# Multi-Validation Metric Tracking Plan

**Status**: ACTIVE
**Branch**: `feat/experiment-evaluation`
**Created**: 2026-02-25
**Depends on**: Phases A-J of [consolidated plan](consolidated-devex-training-evaluation-plan.md) (COMPLETE)

## User Prompt (verbatim)

> Did you benchmark and document the time for each epoch at each loss? Could we next plan
> how to run the experiment for as long as we have the training converging then? And
> previously in an older project we had "multi-metric tracking" which we could implement as
> feature as well. In practice this mean that in the older pytorch repos you typically had
> if check if validation loss/metric had improved and then you saved that .pt(h) file to
> disk with all the metric .deepcopy() for this from the full metric dict that we are
> tracking. But we could make this more flexible that if we track "the best model" then for
> example using DSC, clDSC, centerline boundary Dice (cbDice), MASD and val loss we would
> get 5 best models from a single training run instead of the conventional method of only
> saving one best model? Does this make sense? We could for example make the hypothesis
> that for ensemble, the individual models should be saved based on the validation loss and
> not a validation metric (which was the case in some older paper for Google Research that
> I cannot find know which I think studied creating ensembles for small-data cases and they
> showed that it was better to have individual models of the ensemble optimized for
> cross-entropy which gave the best ensembled AUROC compared to the case of saving best
> models based on validation AUROC). This obviously should be controlled in the .yaml
> config in Hydra and we should now use the multiple validation metrics to track the model
> improvement, and the researchers can later just opt-in to use just one tracking metric,
> but this should be a default feature. And this enables richer post-hoc analysis if we
> have all the different metrics saved when actually running "traditional training", and
> maybe LoRA finetuning could use a similar approach?

---

## Training Time Benchmarks (1 epoch, debug mode, 2 folds, 10 volumes)

| Loss Function | Avg Epoch Time | Total Session | Peak GPU | Peak RSS |
|---------------|---------------|---------------|----------|----------|
| **dice_ce** | ~12 sec | 58 sec | 3,453 MB | 2.5 GB |
| **cbdice** | ~12 sec | 111 sec | 3,955 MB | 2.3 GB |
| **dice_ce_cldice** | ~13 sec | 125 sec | 5,410 MB | 2.5 GB |
| **warp** | ~12 sec | 117 sec | 3,443 MB | 2.2 GB |

**Estimated full training time** (100 epochs, 3 folds, 70 volumes):
- Per-epoch with full dataset: ~5-8 min (extrapolated from 10-vol debug at ~12 sec)
- Per-fold: ~8-13 hours at 100 epochs
- Per-loss (3 folds): ~24-40 hours
- All 4 losses: ~4-7 days
- **Recommendation**: Start with 100 epochs, monitor convergence live via MLflow UI

---

## Problem Statement

### Current State (7 gaps)

1. **Best model = lowest val_loss only** — `trainer.py:fit()` compares `val_result.loss < self._best_val_loss`
2. **No metric-based checkpoint selection** — val Dice and F1 are computed each epoch but never drive saves
3. **Single checkpoint file** — only `best_model.pth`, overwritten each improvement
4. **No minimum delta** — any epsilon improvement (1e-10) counts as new best
5. **No minimum epochs** — early stopping can fire at epoch `patience + 1`
6. **Post-training MetricsReloaded metrics disconnected** — clDSC, DSC, MASD only computed after training on the saved best checkpoint
7. **No epoch-level metric history persisted** — metrics are logged to MLflow but not stored alongside checkpoints

### Desired State

- **N best models from one training run** (1 per tracked metric) instead of 1 best model
- Each checkpoint is **self-contained**: model weights + full metric dict for that epoch
- Researchers can **opt in to single-metric tracking** via YAML config
- **Richer post-hoc analysis**: compare "best by val_loss" vs "best by DSC" vs "best by clDSC"
- **Ensemble hypothesis testing**: proper scoring rule (loss) vs task metric (DSC) for member selection

---

## Literature Support

### Proper Scoring Rules for Model Selection

**Lakshminarayanan et al. (2017)** — "Simple and Scalable Predictive Uncertainty Estimation
using Deep Ensembles" (Google Brain, NeurIPS 2017). Key argument: NLL (cross-entropy) is a
**proper scoring rule** — the expected score is maximized only when the predicted distribution
matches the true distribution. Training and selecting models by NLL/cross-entropy produces
better-calibrated predictions than selecting by accuracy or AUROC. Ensemble diversity emerges
naturally from different random initializations trained with NLL.

**Implication for minivess-mlops**: The "best by val_loss" checkpoint may produce better
ensemble members than "best by DSC" — because val_loss is a proper scoring rule that
preserves calibration, while DSC is a threshold-dependent metric that ignores prediction
confidence. This is a testable hypothesis enabled by multi-metric tracking.

### Checkpoint Ensembles

**Chen & Lundberg (2017)** — "Checkpoint Ensembles: Ensemble Methods from a Single Training
Process" (arXiv 1710.03282). Shows that checkpoints from a single training run can form
effective ensembles, outperforming minimum-validation model selection. The key insight:
intermediate checkpoints represent diverse models that cover different regions of the loss
landscape.

**Huang et al. (2017)** — "Snapshot Ensembles: Train 1, Get M for Free" (ICLR 2017).
Uses cyclic learning rate schedules to collect model snapshots at multiple local minima
during a single training run. M snapshots form an M-member ensemble at zero extra training
cost.

### Implications for Multi-Metric Tracking

With multi-metric tracking, we can test multiple hypotheses post-hoc:

| Hypothesis | Ensemble Selection | Expected Outcome |
|-----------|-------------------|-----------------|
| H1: Proper scoring rule wins | Best by val_loss | Better calibration, possibly better ensemble DSC |
| H2: Task metric wins | Best by DSC | Higher individual DSC but possibly overfit |
| H3: Topology-aware wins | Best by clDSC | Better vessel connectivity preservation |
| H4: Diversity matters | One per metric | Maximum diversity → best ensemble |
| H5: Checkpoint ensemble | Top-K by val_loss across epochs | Snapshot-style without cyclic LR |

---

## Design: Self-Contained Metric Checkpoints

### Current Checkpoint (what exists)

```python
# trainer.py — saves only weights
model.save_checkpoint(best_path)  # → best_model.pth (torch state_dict only)
```

### Proposed: MetricCheckpoint (self-contained)

```python
@dataclass
class MetricCheckpoint:
    """Self-contained checkpoint with all metrics at save time."""
    epoch: int
    model_state_dict: dict           # torch state dict
    optimizer_state_dict: dict       # for resume
    scheduler_state_dict: dict       # for resume
    metrics: dict[str, float]        # ALL tracked metrics at this epoch
    metric_name: str                 # which metric triggered this save
    metric_value: float              # the triggering metric's value
    metric_direction: str            # "minimize" or "maximize"
    train_loss: float
    val_loss: float
    wall_time_sec: float             # epoch wall time
    config_snapshot: dict            # frozen config at save time
```

### File Layout (per fold)

```
checkpoints/
├── fold_0/
│   ├── best_val_loss.pth           # MetricCheckpoint: best by val_loss (minimize)
│   ├── best_dsc.pth                # MetricCheckpoint: best by Dice (maximize)
│   ├── best_centreline_dsc.pth     # MetricCheckpoint: best by clDSC (maximize)
│   ├── best_measured_masd.pth      # MetricCheckpoint: best by MASD (minimize)
│   ├── best_val_dice.pth           # MetricCheckpoint: best by in-training Dice (maximize)
│   ├── last.pth                    # MetricCheckpoint: final epoch (always saved)
│   └── metric_history.json         # Full epoch-by-epoch metric dict
├── fold_1/
│   └── ...
```

Each `.pth` file is a **complete MetricCheckpoint** — you can load any checkpoint and
immediately know all metric values at the epoch it was saved, which metric triggered
the save, and the full training config.

### Metric History (per fold)

```json
{
  "epochs": [
    {
      "epoch": 1,
      "wall_time_sec": 312.5,
      "train_loss": 0.6251,
      "val_loss": 0.6242,
      "val_dice": 0.0688,
      "val_f1_foreground": 0.0712,
      "learning_rate": 1.01e-04,
      "checkpoints_saved": ["best_val_loss"]
    },
    {
      "epoch": 2,
      "wall_time_sec": 308.1,
      "train_loss": 0.5891,
      "val_loss": 0.5810,
      "val_dice": 0.1250,
      "val_f1_foreground": 0.1302,
      "learning_rate": 1.00e-04,
      "checkpoints_saved": ["best_val_loss", "best_val_dice"]
    }
  ]
}
```

---

## Configuration Design (YAML + Hydra-zen)

### Experiment YAML (`configs/experiments/dynunet_losses.yaml`)

```yaml
experiment_name: dynunet_loss_variation
model: dynunet
losses:
  - dice_ce
  - cbdice
  - dice_ce_cldice
  - warp
compute: auto
data_dir: data/raw
num_folds: 3
max_epochs: 100
seed: 42

# === NEW: Multi-metric tracking ===
checkpoint:
  # Metrics to track for "best model" checkpointing.
  # Each entry creates a separate best_<metric>.pth file.
  # Default: all available metrics tracked (researchers can reduce to one).
  tracked_metrics:
    - name: val_loss
      direction: minimize    # proper scoring rule — default primary
      patience: 15           # per-metric early stopping patience
    - name: val_dice
      direction: maximize
      patience: 20
    - name: val_f1_foreground
      direction: maximize
      patience: 20

  # Global early stopping: stop when ALL tracked metrics have exhausted patience.
  # Alternative: "any" stops when any single metric exhausts patience.
  early_stopping_strategy: all   # "all" | "any" | "primary"

  # Primary metric for --resume and MLflow model registry.
  # Must be one of tracked_metrics[].name.
  primary_metric: val_loss

  # Minimum improvement delta (avoids saving on trivial changes).
  min_delta: 1e-4

  # Minimum epochs before early stopping can trigger.
  min_epochs: 10

  # Save last.pth every epoch (for crash recovery).
  save_last: true

  # Save full metric history as JSON alongside checkpoints.
  save_history: true

# === Memory constraints (unchanged) ===
memory_limit_gb: 24
monitor_interval: 10
```

### Opt-in Single Metric (researcher override)

```yaml
# Researcher who only wants conventional val_loss tracking:
checkpoint:
  tracked_metrics:
    - name: val_loss
      direction: minimize
      patience: 15
  early_stopping_strategy: primary
  primary_metric: val_loss
```

---

## Improvements over foundation-PLR

### foundation-PLR Pattern (current)

```python
# Checkpoint saves based on val_loss ONLY
if eval_dicts["test"]["average_loss"] < best_validation_loss:
    best_validation_loss = eval_dicts["test"]["average_loss"]
    improved_loss = True
    # Save checkpoint

# Task metric tracked but NOT used for saving
if eval_dicts["outlier_test"]["best_metric"] > best_validation_metric:
    best_validation_metric = eval_dicts["outlier_test"]["best_metric"]
    # No checkpoint save
```

**Problems**:
- Best-loss checkpoint may not be best-metric checkpoint
- No epoch-level MLflow streaming (only end-of-training logging)
- No metric history persisted alongside checkpoints
- Checkpoints contain only model weights, not metrics
- No per-metric early stopping

### minivess-mlops Pattern (proposed)

```python
# Multi-metric tracker manages N independent "best" trackers
for metric_tracker in self.metric_trackers:
    current_value = epoch_metrics[metric_tracker.name]
    if metric_tracker.has_improved(current_value):
        save_metric_checkpoint(
            path=checkpoint_dir / f"best_{metric_tracker.name}.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=epoch_metrics,       # ALL metrics at this epoch
            triggered_by=metric_tracker.name,
            triggered_value=current_value,
        )
        metric_tracker.reset_patience()
    else:
        metric_tracker.increment_patience()

# Global early stopping check
if self.should_stop(strategy=config.early_stopping_strategy):
    break
```

**Improvements**:
- N checkpoints per fold (one per tracked metric) + last.pth
- Each checkpoint self-contained with full metric dict
- Per-metric patience (DSC may converge slower than val_loss)
- Epoch-level metric history saved as JSON
- MLflow epoch-level streaming (metrics + wall time)
- Config-driven: YAML controls everything
- Testable ensemble hypothesis: compare "best by loss" vs "best by DSC"

---

## Implementation Plan

### Phase 1: MultiMetricTracker (core logic)

**Files**:
- `src/minivess/pipeline/multi_metric_tracker.py` — NEW
- `tests/v2/unit/test_multi_metric_tracker.py` — NEW

**Components**:
- `MetricDirection` enum: `MINIMIZE`, `MAXIMIZE`
- `MetricTracker` dataclass: name, direction, patience, best_value, patience_counter, min_delta
- `MultiMetricTracker` class: manages N MetricTrackers, determines global early stopping
- `MetricCheckpoint` dataclass: self-contained checkpoint with model + metrics + config
- `save_metric_checkpoint()` / `load_metric_checkpoint()` functions
- `MetricHistory` class: epoch-by-epoch accumulator with JSON serialization

**Tests** (TDD):
```
test_single_metric_improvement_detected()
test_multi_metric_independent_tracking()
test_per_metric_patience_independent()
test_early_stop_strategy_all()
test_early_stop_strategy_any()
test_early_stop_strategy_primary()
test_min_delta_filters_trivial_improvement()
test_min_epochs_prevents_early_stop()
test_checkpoint_is_self_contained()
test_metric_history_json_round_trip()
test_maximize_direction_correct()
test_minimize_direction_correct()
```

### Phase 2: Integrate into SegmentationTrainer

**Files** (modify existing):
- `src/minivess/pipeline/trainer.py` — refactor `fit()` to use MultiMetricTracker
- `src/minivess/config/models.py` — add `CheckpointConfig` to `TrainingConfig`
- `tests/v2/unit/test_trainer_multi_metric.py` — NEW

**Changes to `fit()`**:
- Replace `self._best_val_loss` + `self._patience_counter` with `MultiMetricTracker`
- Save `MetricCheckpoint` instead of bare `state_dict`
- Save `metric_history.json` after each epoch
- Log wall time per epoch to MLflow
- Add `save_last: true` support

**Backward compatibility**: If no `checkpoint` config is provided, default to single
`val_loss` tracking (current behavior).

### Phase 3: Experiment YAML + Hydra Integration

**Files**:
- `configs/experiments/dynunet_losses.yaml` — update with checkpoint config
- `scripts/run_experiment.py` — pass checkpoint config to trainer
- `scripts/train_monitored.py` — same

### Phase 4: Run Convergence Training

**Goal**: Train all 4 losses for enough epochs to converge. Monitor live via MLflow.

**Protocol**:
1. Start with `max_epochs: 200`, `patience: 30` per metric
2. Run dice_ce first (baseline), monitor in MLflow
3. If converged by epoch 60-80, adjust `max_epochs` for remaining losses
4. Run cbdice, dice_ce_cldice, warp sequentially (separate processes)
5. After all 4 complete: run Phase I comparison with production checkpoints
6. Fix final `max_epochs` for publication

**Resource budget** (per loss, 3 folds):
- Estimated: ~24-40 hours per loss (100 epochs × 3 folds × ~5-8 min/epoch)
- Total: ~4-7 days for all 4 losses
- Memory: within 8 GB VRAM / 32 GB RAM budget (verified in debug runs)

### Phase 5: Post-hoc Ensemble Analysis

**Goal**: Test the proper scoring rule hypothesis.

**Protocol**:
1. For each loss × each fold: load `best_val_loss.pth` and `best_dsc.pth`
2. Run sliding window inference + MetricsReloaded on both
3. Compare: is "best by loss" better or worse for ensemble DSC?
4. Ensemble strategies: mean, majority vote, greedy soup
5. Results go into the manuscript comparison table

---

## LoRA Finetuning Applicability

Multi-metric tracking applies directly to LoRA finetuning (SAMv3, VISTA-3D):

- **Same MetricCheckpoint format** — LoRA adapters are small (~16 MB) so saving N
  checkpoints per metric is cheap
- **LoRA-specific metrics**: may want to track adapter-specific metrics
  (e.g., LoRA rank utilization, effective rank from WeightWatcher)
- **Transfer learning hypothesis**: for finetuning, val_loss-based selection may be
  even more important because the pretrained backbone provides good features and
  the head should be selected for calibration (proper scoring rule), not task metric

---

## Dependency Graph

```
Phase 1 (MultiMetricTracker)
    ↓
Phase 2 (Trainer integration)
    ↓
Phase 3 (YAML config)
    ↓
Phase 4 (Convergence training) — days of GPU time
    ↓
Phase 5 (Post-hoc ensemble analysis)
```

## References

- lakshminarayanan2017deep: Lakshminarayanan, Pritzel, Blundell. "Simple and Scalable
  Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS 2017.
- chen2017checkpoint: Chen, Lundberg. "Checkpoint Ensembles: Ensemble Methods from a
  Single Training Process." arXiv:1710.03282, 2017.
- huang2017snapshot: Huang, Li, Pleiss, Liu, Hopcroft, Weinberger. "Snapshot Ensembles:
  Train 1, Get M for Free." ICLR 2017.
