# Ensemble — Builder and Strategies

## 4 Ensemble Strategies

| Strategy | Description | Requires |
|----------|------------|----------|
| `per_loss_single_best` | Best fold per loss | 1 loss, 3 folds |
| `all_loss_single_best` | Best fold across all losses | 4+ losses, 3 folds |
| `per_loss_all_best` | All folds above threshold, per loss | 1 loss, all qualifying |
| `all_loss_all_best` | All qualifying folds across all losses | All losses, all qualifying |

## Run Discovery

`discover_training_runs_raw()` searches MLflow for training runs:
- Filters by experiment name
- Reads `loss_function` tag (standardized — train_flow.py logs this correctly since PR #871)
- **Hard gate at line 221**: requires `eval_fold2_dsc` metric — debug runs will fail
  unless `require_eval_metrics=False` is passed

## Checkpoint Loading

`load_checkpoint()` loads model weights from MLflow artifact URIs.

**CRITICAL BUG**: Currently returns random weights silently if state_dict doesn't match.
Must be fixed to raise an error. Random weights produce plausible-looking but meaningless
predictions — worse than a crash.

## Files

| File | Purpose |
|------|---------|
| `builder.py` | EnsembleBuilder — discover runs, build ensembles |
| `strategies.py` | Strategy implementations |
| `voting.py` | Soft/hard voting aggregation |

## Rules

- All ensemble building goes through MLflow run discovery — no filesystem traversal
- Tag key is `loss_function` (standardized — guard test: `test_loss_tag_consistency.py`)
- Never return random weights — always raise on mismatch
- Debug scenarios need configurable eval metric gates
