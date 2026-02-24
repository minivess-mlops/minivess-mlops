# Optuna HPO Integration — Implementation Plan (Issue #43)

## Current State
- `optuna>=4.7` in pyproject.toml (not yet used)
- `SegmentationTrainer.fit()` returns `{"best_val_loss": float, ...}`
- `TrainingConfig` has all HPO-relevant fields (lr, batch_size, optimizer, etc.)
- `ExperimentTracker` handles MLflow logging
- No Optuna study/trial integration

## Architecture

### New Module: `src/minivess/pipeline/hpo.py`

1. **SearchSpace** — Dataclass defining parameter ranges for Optuna
2. **create_study()** — Factory for Optuna study with MedianPruner
3. **build_trial_config()** — Converts Optuna trial suggestions to TrainingConfig
4. **run_hpo()** — Orchestrator: creates study, runs optimization, returns best params

### Integration Points
- Objective function calls `SegmentationTrainer.fit()` inside each trial
- MedianPruner stops bad trials based on intermediate val_loss
- MLflow logging via Optuna's MLflowCallback (optional)
- Results accessible as `study.best_params` / `study.best_value`

### Search Space (DynUNet HPO defaults)
- learning_rate: [1e-5, 1e-2] (log scale)
- weight_decay: [1e-6, 1e-3] (log scale)
- batch_size: [1, 4]
- optimizer: {adamw, sgd}
- warmup_epochs: [0, 10]
- gradient_clip_val: [0.5, 2.0]

## New Files
- `src/minivess/pipeline/hpo.py`

## Modified Files
- `src/minivess/pipeline/__init__.py` — add HPO exports

## Test Plan
- `tests/v2/unit/test_hpo.py` (~15 tests)
