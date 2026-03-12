# Pipeline — Training, Inference, Evaluation

## Loss Functions (`loss_functions.py`)

18 loss functions with automatic warning suppression system. Default: `cbdice_cldice`
(CbDiceClDiceLoss) — chosen by dynunet_loss_variation_v2 experiment (0.906 clDice).

Loss registry pattern:
- `LOSS_REGISTRY: dict[str, Callable]` maps string names to constructors
- Config specifies `loss_name: str` → resolved at runtime via registry
- NEVER hardcode loss-specific logic — use registry dispatch

## Metrics (`metrics.py`)

Metrics Reloaded-aligned evaluation suite (Maier-Hein 2024, Nature Methods).
Per-fold and ensemble-level metrics. Bootstrap confidence intervals.

## Comparison Table (`comparison.py`)

Cross-loss comparison with bootstrap CI. Produces Markdown + LaTeX tables.
Used by Analysis Flow for ensemble evaluation.

## Drift Detection (`drift_detection.py`, `embedding_drift.py`)

- Evidently DataDriftPreset for feature-level drift
- Kernel MMD (scipy permutation_test) for embedding drift
- Reports persisted as MLflow artifacts

## Key Rules

- NO `import re` for metric name parsing — use `str.split()` (see regex ban)
- All metric names follow `{phase}_{fold}_{metric}` convention parsed by `str.split("_")`
- Loss functions are GENERIC — config-driven, never task-specific
- Trainer val_interval sentinel: `val_interval > max_epochs` = "never validate"
