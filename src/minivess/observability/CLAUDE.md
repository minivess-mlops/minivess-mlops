# Observability — Tracking, Analytics, LLM Tracing

## ExperimentTracker (`tracking.py`)

Central class for MLflow experiment tracking. Auto-logs:
- All TrainingConfig fields (17 params)
- Architecture params (arch_ prefix)
- System info (sys_ prefix): Python, PyTorch, MONAI, CUDA, GPU, RAM
- Git commit hash, branch, dirty state

Key functions:
- `start_run()` — creates MLflow run, logs all params
- `log_hydra_config()` — logs resolved Hydra config as MLflow artifact
- `resolve_tracking_uri()` — reads MLFLOW_TRACKING_URI from env (no fallbacks!)

## DuckDB Analytics (`analytics.py`)

In-process SQL analytics over MLflow runs. Reads mlruns/ directory.

## Langfuse Tracing (`langfuse_tracing.py`)

Self-hosted Langfuse for LLM call tracing and cost tracking.

## Braintrust Eval (`braintrust_eval.py`)

Braintrust AutoEvals for agent quality gates.

## Key Rules

- `mlflow.pytorch.autolog()` is NOT used — all logging is explicit via ExperimentTracker
- NEVER use `os.environ.get("MLFLOW_TRACKING_URI", "mlruns")` — use `resolve_tracking_uri()`
- Param prefixes: (none)=training, arch_=architecture, sys_=system, data_=dataset,
  loss_=loss, eval_=evaluation, upstream_=cross-flow
- We use `sys_` (underscore) NOT `sys/` (slash) to avoid metric naming conflicts
