# Observability -- Tracking, Analytics, LLM Tracing

## ExperimentTracker (`tracking.py`)

Central class for MLflow experiment tracking. Auto-logs:
- All TrainingConfig fields (17 params) with `train/` prefix
- Architecture params (`arch/` prefix)
- System info (`sys/` prefix): Python, PyTorch, MONAI, CUDA, GPU, RAM
- Git commit hash, branch, dirty state
- Data augmentation pipeline description (`data/augmentation_pipeline`)

Key functions:
- `start_run()` -- creates MLflow run, logs all params
- `log_hydra_config()` -- logs resolved Hydra config as MLflow artifact
- `resolve_tracking_uri()` -- reads MLFLOW_TRACKING_URI from env (no fallbacks!)

## Metric Keys (`metric_keys.py`)

Single source of truth for ALL MLflow key names. Issue #790.
- `MetricKeys` class: canonical slash-prefix key constants
- Greenfield: MIGRATION_MAP and normalize functions deleted (no legacy runs)

## Slash-Prefix Convention (MLflow 2.11+)

All metric and param keys use slash (`/`) separators for auto-grouping in MLflow UI:

| Group | Prefix | Examples |
|-------|--------|----------|
| Training | `train/` | `train/loss`, `train/dice` |
| Validation | `val/` | `val/loss`, `val/dice`, `val/cldice` |
| Optimizer | `optim/` | `optim/lr`, `optim/grad_scale` |
| Gradient | `grad/` | `grad/norm_mean`, `grad/clip_count` |
| GPU | `gpu/` | `gpu/utilization_pct`, `gpu/temp_c` |
| Profiling | `prof/` | `prof/first_epoch_seconds`, `prof/val_seconds` |
| Model | `model/` | `model/family`, `model/trainable_params` |
| Architecture | `arch/` | `arch/filters`, `arch/strides` |
| System | `sys/` | `sys/gpu_model`, `sys/torch_version` |
| Data | `data/` | `data/n_volumes`, `data/augmentation_pipeline` |
| Cost | `cost/` | `cost/total_usd`, `cost/setup_fraction` |
| Estimate | `est/` | `est/total_cost`, `est/total_hours` |
| Setup | `setup/` | `setup/uv_sync_seconds`, `setup/total_seconds` |
| Config | `cfg/` | `cfg/project_name`, `cfg/dvc_remote` |
| Fold | `fold/` | `fold/0/best_val_loss`, `fold/n_completed` |
| VRAM | `vram/` | `vram/peak_mb` |
| Inference | `infer/` | `infer/latency_ms_per_volume` |
| Checkpoint | `checkpoint/` | `checkpoint/size_mb` |
| Benchmark | `bench/` | `bench/gpu_model`, `bench/{model}/vram_peak_mb` |
| Evaluation | `eval/` | `eval/fold0/dice`, `eval/minivess/all/cldice` |

CI encoding: `val/dice/ci95_lo`, `val/dice/ci95_hi`

## DuckDB Analytics (`analytics.py`)

In-process SQL analytics over MLflow runs. Reads mlruns/ directory.

## Langfuse Tracing (`langfuse_tracing.py`)

Self-hosted Langfuse for LLM call tracing and cost tracking.

## Braintrust Eval (`braintrust_eval.py`)

Braintrust AutoEvals for agent quality gates.

## Key Rules

- `mlflow.pytorch.autolog()` is NOT used -- all logging is explicit via ExperimentTracker
- NEVER use `os.environ.get("MLFLOW_TRACKING_URI", "mlruns")` -- use `resolve_tracking_uri()`
- ALL metric/param keys MUST use slash-prefix convention (see table above)
- Import key names from `metric_keys.MetricKeys` -- never hardcode strings
- Greenfield project: no backward compat layer needed
