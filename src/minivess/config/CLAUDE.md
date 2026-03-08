# Config — Hydra-zen Composition

## Single Source of Truth (CLAUDE.md Rule #23)

ALL experiment configuration flows through Hydra-zen composition:

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

## Key Function

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

## Config Groups

| Group | Directory | Default |
|-------|-----------|---------|
| data | `configs/data/` | minivess |
| model | `configs/model/` | dynunet |
| training | `configs/training/` | default |
| checkpoint | `configs/checkpoint/` | standard |

## Debug Experiment Configs

Debug configs are **standard Hydra experiment YAMLs** in `configs/experiment/` with
the `debug_` prefix. NOT a separate directory. NOT a separate format.

| Config | Purpose | Duration |
|--------|---------|----------|
| `debug_single_model.yaml` | 1 model, 1 fold, 1 epoch | ~2 min |
| `debug_all_models.yaml` | 5 models, 3 folds, 2 epochs | ~30 min |
| `debug_full_pipeline.yaml` | Train + Post-Training + Analysis | ~15 min |
| `debug_multi_loss.yaml` | 3 losses for ensemble testing | ~15 min |

## Files

| File | Purpose |
|------|---------|
| `compose.py` | `compose_experiment_config()` — Hydra Compose API bridge |
| `adaptive_profiles.py` | Hardware detection + compute profiles |
| `model_profiles.py` | ModelProfile loader from YAML |
| `deploy_config.py` | DeployConfig + ChampionCategory enum |

## BANNED Patterns

- Parallel config systems (custom merge scripts, separate Pydantic models)
- Hardcoded hyperparameters in Python
- `argparse` dicts that bypass Hydra composition
- `configs/debug/` directory (debug configs go in `configs/experiment/`)
- Single-key override YAML files (use Hydra overrides instead)
