# Training Pipeline Design

## Implementation Files

| File | Purpose |
|------|---------|
| `src/minivess/orchestration/flows/train_flow.py` | Training flow definition |
| `src/minivess/pipeline/trainer.py` | Training loop with MONAI |
| `src/minivess/config/compose.py` | compose_experiment_config() |
| `src/minivess/observability/tracking.py` | ExperimentTracker |

## Docker Execution

```bash
docker compose --env-file .env -f deployment/docker-compose.flows.yml run --rm \
  --shm-size 8g \
  -e EXPERIMENT=debug_single_model \
  train
```

## Config Flow

```
EXPERIMENT env var
  → compose_experiment_config(experiment_name)
  → resolved dict (all Hydra defaults + overrides)
  → training_flow(config_dict=resolved)
  → per-fold: trainer.train() + ExperimentTracker.log_hydra_config()
  → MLflow artifact: config/resolved_config.yaml
```

## Volume Mounts

| Mount | Purpose | Mode |
|-------|---------|------|
| data_cache:/app/data | Training data | ro |
| configs_splits:/app/configs/splits | Fold definitions | ro |
| checkpoint_cache:/app/checkpoints | Model weights | rw |
| mlruns_data:/app/mlruns | Experiment tracking | rw |
| logs_data:/app/logs | Training CSV/JSONL | rw |

## Key Invariants

- shm_size=8g required (MONAI DataLoader uses /dev/shm for IPC)
- GPU reservation via CDI: `devices: ["nvidia.com/gpu=all"]`
- val_interval sentinel: `val_interval > max_epochs` → skip all validation
