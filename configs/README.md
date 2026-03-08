# MinIVess Configuration System

## Architecture: Hydra-zen as Single Source of Truth

All experiment configuration flows through **Hydra-zen composition**:

```
configs/base.yaml (entry point + defaults list)
    ↓ resolve defaults
configs/{data,model,training,checkpoint}/*.yaml (config groups)
    ↓ merge experiment override
configs/experiment/*.yaml (experiment-specific values)
    ↓ apply CLI/Hydra overrides
compose_experiment_config() → resolved dict
    ↓ log to MLflow
config/resolved_config.yaml (MLflow artifact — THE record of what ran)
```

The resolved config dict logged to MLflow is the **single source of truth**.
Not the CLI args. Not the env vars. Not the YAML file alone. The artifact.

## Directory Structure

| Directory | Purpose | Consumed By |
|-----------|---------|-------------|
| `base.yaml` | Hydra entry point: defaults list + top-level overrides | `compose_experiment_config()` |
| `data/` | Dataset config group (data_dir, splits, normalization) | Hydra defaults |
| `model/` | Model selection group (dynunet, sam3_vanilla, etc.) | Hydra defaults |
| `training/` | Training hyperparams group (seed, compute, epochs) | Hydra defaults |
| `checkpoint/` | Checkpoint strategy group (metrics, patience, early stopping) | Hydra defaults |
| `experiment/` | **Complete experiment definitions** (22+ configs) | `compose_experiment_config(experiment_name=...)` |
| `experiments/` | Shell-level grid sweep configs (legacy) | `train_all_hyperparam_combos.sh` |
| `model_profiles/` | Model VRAM/memory profiles (not Hydra) | `load_model_profile()` in train_flow |
| `method_capabilities.yaml` | Implemented models + losses (not Hydra) | `capability_discovery.py` |
| `metric_registry.yaml` | Metric definitions (not Hydra) | Evaluation code |
| `splits/` | Fold split JSON files | `load_fold_splits_task()` |
| `hpo/` | Optuna HPO search space configs | `hpo_engine.py` |
| `deployment/` | Dynaconf deployment env configs | `settings.py` |

## How Hydra Composition Works

### 1. Base Config (`base.yaml`)

```yaml
defaults:
  - data: minivess        # → loads configs/data/minivess.yaml
  - model: dynunet         # → loads configs/model/dynunet.yaml
  - training: default      # → loads configs/training/default.yaml
  - checkpoint: standard   # → loads configs/checkpoint/standard.yaml
  - _self_                 # → base.yaml values applied last

experiment_name: unnamed
losses:
  - dice_ce
debug: false
```

### 2. Experiment Override (`configs/experiment/dynunet_e2e_debug.yaml`)

```yaml
# @package _global_
defaults:
  - override /checkpoint: lightweight  # swap checkpoint group

experiment_name: dynunet_e2e_debug
model: dynunet
losses:
  - cbdice_cldice
max_epochs: 10
```

### 3. Composition API

```python
from minivess.config.compose import compose_experiment_config

# Default config (base + groups):
cfg = compose_experiment_config()

# With experiment override:
cfg = compose_experiment_config(experiment_name="dynunet_e2e_debug")

# With additional Hydra overrides:
cfg = compose_experiment_config(
    experiment_name="dynunet_e2e_debug",
    overrides=["max_epochs=5", "model=sam3_vanilla"],
)
```

### 4. Resolved Config → MLflow

The resolved dict is logged as an MLflow artifact (`config/resolved_config.yaml`)
via `ExperimentTracker.log_hydra_config()`. This artifact contains EVERY parameter
that was active during the run — the complete, deterministic record.

## Debug Experiment Configs

Debug configs use the **same system** as all other experiments. They are standard
Hydra experiment YAMLs with the `debug_` prefix convention:

| Config | Purpose | Duration |
|--------|---------|----------|
| `debug_single_model.yaml` | Fastest smoke test (1 model, 1 fold, 1 epoch) | ~2 min |
| `debug_all_models.yaml` | All 8GB-compatible models | ~30 min |
| `debug_full_pipeline.yaml` | Train → Post-Training → Analysis | ~15 min |
| `debug_multi_loss.yaml` | Multiple losses for ensemble testing | ~15 min |

Usage via Docker:
```bash
docker compose -f deployment/docker-compose.flows.yml run --rm \
  -e EXPERIMENT=debug_single_model train

# With Hydra override:
docker compose -f deployment/docker-compose.flows.yml run --rm \
  -e EXPERIMENT=debug_single_model \
  -e HYDRA_OVERRIDES="max_epochs=5,model=sam3_vanilla" \
  train
```

Usage via shell wrapper:
```bash
./scripts/run_debug.sh --experiment debug_all_models
./scripts/run_debug.sh --experiment debug_single_model --override max_epochs=5
```

## Rules (CLAUDE.md #23)

1. **NEVER** create parallel config systems (custom merge scripts, separate Pydantic models)
2. **NEVER** hardcode hyperparameters in Python — parameterize via Hydra YAML
3. **NEVER** bypass Hydra with argparse dicts in flow files
4. **ALWAYS** log resolved config to MLflow via `tracker.log_hydra_config()`
5. Debug configs are `configs/experiment/debug_*.yaml` — NOT a separate directory

## Key Files

| File | Purpose |
|------|---------|
| `src/minivess/config/compose.py` | Hydra Compose API bridge + manual fallback |
| `src/minivess/observability/tracking.py` | `log_hydra_config()` (lines 327-350) |
| `configs/base.yaml` | Hydra entry point |
| `configs/experiment/*.yaml` | 22+ experiment configs (including debug_*) |
