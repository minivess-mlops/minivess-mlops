# MLflow Tracking Integration Plan

**Issue**: #37 — Wire MLflow tracking into v2 training loop
**Date**: 2026-02-24
**Status**: Draft → Implementation

---

## 1. Problem Statement

MLflow is in `pyproject.toml`, Docker Compose, and listed as resolved in the PRD, but
the v2 `SegmentationTrainer` does not log anything to MLflow. The `ExperimentTracker`
class exists but is completely disconnected from the training loop. The tracking URI
is hardcoded to `http://localhost:5000`, making it impossible to use without a running
MLflow server.

**Key gap**: `SegmentationTrainer.fit()` → logs to Python `logger.info()` only, never
calls `ExperimentTracker`.

---

## 2. Current State

| Component | File | Status |
|-----------|------|--------|
| ExperimentTracker | `observability/tracking.py` | Scaffolded, hardcoded URI, disconnected from trainer |
| RunAnalytics | `observability/analytics.py` | Scaffolded, hardcoded URI, untested |
| SegmentationTrainer | `pipeline/trainer.py` | Full loop, NO MLflow calls |
| Dynaconf settings | `configs/deployment/settings.toml` | Has `mlflow_tracking_uri` but unused |
| Docker Compose | `deployment/docker-compose.yml` | MLflow service on port 5000 |
| Tests | — | Zero observability tests |

**Key bugs in current code**:
- `tracking.py:21` hardcodes `_DEFAULT_TRACKING_URI = "http://localhost:5000"` — tests
  will fail without a running server
- `analytics.py:12` same hardcoded URI
- `ExperimentTracker` is never instantiated anywhere in the pipeline code

---

## 3. Architecture: Flexible MLflow Backend

MLflow's Python API natively supports multiple backend/artifact configurations via the
tracking URI:

| Scenario | `tracking_uri` | Backend Store | Artifact Store |
|----------|----------------|---------------|----------------|
| Local file | `file:///path/to/mlruns` or `./mlruns` | Local filesystem | Local filesystem |
| SQLite local | `sqlite:///mlflow.db` | SQLite DB | Local filesystem |
| Docker dev | `http://localhost:5000` | PostgreSQL (via Docker) | MinIO S3 |
| DagsHub | `https://dagshub.com/user/repo.mlflow` | DagsHub managed | DagsHub managed |
| Self-hosted AWS | `http://mlflow.internal:5000` | PostgreSQL (RDS) | S3 bucket |
| Managed MLflow | `https://mlflow.company.com` | Managed DB | Managed storage |

**Design**: Accept tracking URI from (in priority order):
1. `MLFLOW_TRACKING_URI` env var (standard MLflow convention)
2. Dynaconf `settings.mlflow_tracking_uri`
3. Constructor `tracking_uri` parameter
4. Default: `mlruns` (local file — works without any server)

This means tests run against a local `mlruns/` directory, no server needed.

---

## 4. Implementation Plan

### Task T1: Make ExperimentTracker backend-flexible

**Changes to `observability/tracking.py`**:
- Change default from `http://localhost:5000` to `mlruns` (local file)
- Add `resolve_tracking_uri()` function that checks env var → Dynaconf → default
- ExperimentTracker constructor uses `resolve_tracking_uri()` when no explicit URI given

**Changes to `observability/analytics.py`**:
- Same default change
- RunAnalytics also uses `resolve_tracking_uri()`

**Tests** (`test_observability.py`):
- `test_resolve_tracking_uri_default` — returns "mlruns" when no env/config
- `test_resolve_tracking_uri_env_override` — MLFLOW_TRACKING_URI env var wins
- `test_resolve_tracking_uri_explicit` — explicit param wins
- `test_tracker_uses_local_backend` — ExperimentTracker works with file backend
- `test_tracker_start_run_creates_experiment` — creates experiment + run locally
- `test_tracker_log_epoch_metrics` — metrics show up in MLflow run
- `test_tracker_log_model_info` — model config logged as params
- `test_tracker_log_artifact` — file artifact logged
- `test_tracker_log_test_set_hash` — SHA256 tag recorded

### Task T2: Wire ExperimentTracker into SegmentationTrainer

**Changes to `pipeline/trainer.py`**:
- Add optional `tracker: ExperimentTracker | None = None` parameter to `__init__`
- In `fit()`: if tracker is present, log epoch metrics (train_loss, val_loss, lr)
  at each step
- After fit: log best_val_loss as final metric
- Keep trainer working without tracker (backward compatible)

**Tests** (`test_observability.py`):
- `test_trainer_logs_to_mlflow` — fit() with tracker → MLflow run has metrics
- `test_trainer_works_without_tracker` — fit() without tracker still works (existing behavior)
- `test_trainer_logs_learning_rate` — LR logged at each epoch

### Task T3: Log model artifacts (checkpoint + ONNX export)

**Changes to `pipeline/trainer.py`**:
- After saving best checkpoint, log it as MLflow artifact if tracker is present
- After fit completes, optionally export ONNX and log as artifact

**Tests** (`test_observability.py`):
- `test_trainer_logs_checkpoint_artifact` — best checkpoint appears in MLflow artifacts
- `test_trainer_fit_with_checkpoint_dir` — fit with both checkpoint_dir and tracker

### Task T4: DuckDB analytics over local MLflow runs

**Tests** (`test_observability.py`):
- `test_analytics_load_experiment_runs` — load runs from file-backed MLflow
- `test_analytics_query_sql` — SQL query against runs DataFrame
- `test_analytics_cross_fold_summary` — cross-fold stats
- `test_analytics_top_models` — top-N selection

### Task T5: Integration test — train → track → analyze

**Tests** (`test_observability.py`):
- `test_train_track_analyze_roundtrip` — create model → fit with tracker → verify
  MLflow run → load into DuckDB analytics → query

---

## 5. Execution Order

```
T1 (flexible backend) → T2 (wire into trainer) → T3 (artifacts) → T4 (analytics) → T5 (integration)
```

---

## 6. Out of Scope

- MLflow Model Registry integration (separate from basic tracking)
- Hyperparameter tuning with Optuna + MLflow (issue #40)
- Grafana dashboards for MLflow metrics
- Docker Compose changes (already has MLflow service)
