# PR-1: FinOps & Infrastructure Timing — Implementation Plan

**Issues**: #683, #747, #717, #735
**Branch**: `feat/finops-infrastructure-timing`
**Base**: `main`
**Estimated tests**: 20-25 new tests (on top of 23 existing)
**TDD Skill**: `.claude/skills/self-learning-iterative-coder/SKILL.md` v2.1.0

---

## Architecture Overview

```
train_flow.py
  |
  +-- log_infrastructure_timing()          # existing: logs setup_* params to MLflow
  +-- log_cost_analysis()                  # existing: logs cost_* metrics to MLflow
  +-- [NEW] log_epoch0_cost_estimate()     # T2: after epoch-0, predict total cost
  +-- [NEW] log_timing_jsonl_artifact()    # T6: wire dead code into MLflow artifact
  |
  v
MLflow (internal contract)
  |
  +-- cost_total_usd, cost_effective_gpu_rate, cost_setup_fraction, ...
  +-- estimated_total_cost, estimated_total_hours, cost_per_epoch
  |
  v
prometheus_metrics.py (shared Gauges)       # T1: BentoML custom metrics module
  |
  +-- minivess_training_cost_total_usd
  +-- minivess_training_cost_per_epoch_usd
  +-- minivess_training_gpu_utilization
  +-- minivess_training_setup_fraction
  +-- minivess_training_effective_gpu_rate_usd
  +-- minivess_training_estimated_total_cost_usd
  |
  v
BentoML /metrics endpoint                  # Prometheus scrapes via prometheus.yml
  |
  v
Grafana: infrastructure-cost.json          # T3: 6-8 panels dashboard
  |
  v
RunAnalytics (DuckDB)                      # T4: cost aggregation methods
  +-- cost_by_model_family()
  +-- cost_trends()
  +-- break_even_analysis()
```

**Data flow**: Training logs to MLflow (source of truth) -> prometheus_metrics.py reads
from MLflow and exposes via prometheus_client Gauges -> BentoML /metrics serves them ->
Prometheus scrapes BentoML -> Grafana dashboards visualize.

---

## What Exists (40-50% done)

| Component | File | Status |
|-----------|------|--------|
| `parse_setup_timing()` | `src/minivess/observability/infrastructure_timing.py` | DONE, tested |
| `compute_cost_analysis()` | same | DONE, tested (9 cost_* keys) |
| `log_cost_analysis()` | same | DONE, tested (MLflow metrics at step=0) |
| `estimate_cost_from_first_epoch()` | same | DONE, tested (5 tests) — but NEVER called from train_flow |
| `generate_timing_jsonl()` | same | DONE, tested — but DEAD CODE (never called) |
| `get_hourly_rate_usd()` | same | DONE, tested (reads INSTANCE_HOURLY_USD env) |
| `log_infrastructure_timing()` | same | DONE, tested (called from train_flow line 796) |
| `train_flow.py` cost logging | `src/minivess/orchestration/flows/train_flow.py` | DONE (lines 810-830) |
| `RunAnalytics` | `src/minivess/observability/analytics.py` | DONE (load_experiment_runs, query, cross_fold_summary, top_models) |
| `ghost_cleanup.py` | `src/minivess/observability/ghost_cleanup.py` | DONE (find_ghost_runs, cleanup_ghost_runs) |
| Prometheus config | `deployment/prometheus/prometheus.yml` | DONE (scrapes BentoML at minivess-bento:3000/metrics) |
| BentoML service | `src/minivess/serving/bento_service.py` | DONE (SegmentationService + OnnxSegmentationService) |
| INSTANCE_HOURLY_USD | `.env.example` line 211 | DONE |
| 23 existing tests | `tests/v2/unit/test_infrastructure_timing.py`, `tests/v2/unit/observability/test_infrastructure_timing_prediction.py` | PASSING |

**What's missing** (this plan):

| Component | Task | Issue |
|-----------|------|-------|
| `prometheus_metrics.py` shared Gauges module | T1 | #747 |
| Wire `estimate_cost_from_first_epoch()` into train_flow after epoch-0 | T2 | #717 |
| Grafana `infrastructure-cost.json` dashboard | T3 | #747 |
| `RunAnalytics.cost_by_model_family()`, `.cost_trends()`, `.break_even_analysis()` | T4 | #735 |
| Prefect `cleanup_stale_runs_task` (maintenance flow) | T5 | #683 |
| Wire `generate_timing_jsonl()` into train_flow as MLflow artifact | T6 | #683 |
| KG decision node: `cost_tracking_strategy` | Phase 0 | — |

---

## Phase 0: Knowledge Graph Prerequisite

**Goal**: Create the `cost_tracking_strategy` decision node before implementation starts.

**File**: `knowledge-graph/decisions/L5-operations/cost-tracking-strategy.yaml`

```yaml
id: cost_tracking_strategy
title: "Cost Tracking Strategy"
domain: operations
status: accepted
date: 2026-03-17
context: >
  Cloud GPU training incurs per-hour costs. Need visibility into
  setup overhead fraction, effective GPU rate, break-even epochs,
  and per-model-family cost trends.
decision: >
  Prometheus export via BentoML custom metrics (training-time counter).
  Shared metrics module at src/minivess/observability/prometheus_metrics.py.
  Cost computed in train_flow -> logged to MLflow -> exposed via prometheus_client
  Gauges -> Prometheus scrapes BentoML /metrics -> Grafana dashboards.
  DuckDB analytics for multi-run cost aggregation.
alternatives_rejected:
  - "MLflow-native Prometheus exporter (mlflow-export-import) — adds external dependency, less control"
  - "Standalone finops.py module — cost methods belong on RunAnalytics, not a separate class"
  - "Standalone cleanup script — cleanup belongs as Prefect maintenance task"
references:
  - "#683: Infrastructure timing"
  - "#747: Prometheus metrics for training cost"
  - "#717: Epoch-0 cost prediction"
  - "#735: Cost aggregation analytics"
```

**Action**: Create this file, add `cost_tracking_strategy` to `knowledge-graph/domains/operations.yaml` decisions list.

**Commit message**: `chore(kg): add cost_tracking_strategy decision node (Phase 0, PR-1)`

---

## Task T1: Prometheus Metrics Shared Module

**Issue**: #747
**File**: `src/minivess/observability/prometheus_metrics.py`
**Test file**: `tests/v2/unit/observability/test_prometheus_metrics.py`
**Marker**: (none) — staging tier, no model loading
**Dependencies**: None

### TDD Spec

#### RED Phase — Write failing tests first

```
tests/v2/unit/observability/test_prometheus_metrics.py
```

**Test 1**: `test_module_exports_gauges` — importing the module provides at least 6 Gauge objects (the metrics listed in the architecture diagram).

**Test 2**: `test_gauges_have_correct_names` — each Gauge name starts with `minivess_training_` prefix to avoid collision with BentoML built-in metrics.

**Test 3**: `test_update_cost_gauges_from_dict` — `update_cost_gauges(cost_dict)` sets all Gauge values from a dict matching `compute_cost_analysis()` output keys. Verify via `gauge._value.get()` or `prometheus_client.generate_latest()`.

**Test 4**: `test_update_cost_gauges_missing_keys` — passing a dict with missing keys logs a warning but does not raise. Gauges for missing keys retain their previous value.

**Test 5**: `test_update_estimated_cost_gauges` — `update_estimated_cost_gauges(estimate_dict)` sets Gauges from `estimate_cost_from_first_epoch()` output dict.

**Test 6**: `test_generate_latest_contains_metrics` — `prometheus_client.generate_latest()` output contains all registered metric names as byte strings.

#### GREEN Phase — Implement

Create `src/minivess/observability/prometheus_metrics.py`:

- Import `prometheus_client` (Gauge, generate_latest).
- Define module-level Gauge instances:
  - `minivess_training_cost_total_usd` (Gauge, "Total training cost in USD")
  - `minivess_training_cost_per_epoch_usd` (Gauge, "Cost per epoch in USD")
  - `minivess_training_gpu_utilization` (Gauge, "GPU utilization fraction")
  - `minivess_training_setup_fraction` (Gauge, "Setup fraction of total cost")
  - `minivess_training_effective_gpu_rate_usd` (Gauge, "Effective GPU rate in USD/hr")
  - `minivess_training_estimated_total_cost_usd` (Gauge, "Estimated total cost from epoch-0")
- Function `update_cost_gauges(cost: dict[str, float]) -> None`: maps `compute_cost_analysis()` keys to Gauges.
- Function `update_estimated_cost_gauges(estimate: dict[str, float]) -> None`: maps `estimate_cost_from_first_epoch()` keys to Gauges.
- Use `from __future__ import annotations` at top.
- Use `logging.getLogger(__name__)` for warnings on missing keys.

#### VERIFY

```bash
MINIVESS_ALLOW_HOST=1 uv run pytest tests/v2/unit/observability/test_prometheus_metrics.py -x -q
uv run ruff check src/minivess/observability/prometheus_metrics.py
uv run mypy src/minivess/observability/prometheus_metrics.py
```

#### Commit message

`feat(observability): T1 — prometheus_metrics.py shared Gauges module (#747)`

---

## Task T2: Wire Epoch-0 Cost Estimate into train_flow

**Issue**: #717
**File**: `src/minivess/orchestration/flows/train_flow.py` (modify `train_one_fold_task`)
**Test file**: `tests/v2/unit/observability/test_infrastructure_timing_prediction.py` (extend) + `tests/v2/unit/orchestration/test_train_flow_cost_estimate.py` (new)
**Marker**: (none) — staging tier, mocked MLflow
**Dependencies**: T1 (prometheus_metrics.py must exist)

### TDD Spec

#### RED Phase

**Test file**: `tests/v2/unit/orchestration/test_train_flow_cost_estimate.py`

**Test 7**: `test_epoch0_cost_logged_to_mlflow` — After `trainer.fit()` returns with `training_time_seconds > 0` and `max_epochs > 1`, verify `mlflow.log_metrics` is called with keys `estimated_total_cost`, `estimated_total_hours`, `cost_per_epoch`.

**Test 8**: `test_epoch0_cost_skipped_for_single_epoch` — When `max_epochs == 1`, cost estimate is NOT logged (no point predicting from the only epoch).

**Test 9**: `test_epoch0_cost_updates_prometheus_gauges` — After epoch-0 cost estimate, `update_estimated_cost_gauges()` is called with the estimate dict.

**Test 10**: `test_epoch0_cost_logged_with_correct_values` — Given known `epoch_seconds`, `max_epochs`, `num_folds`, `hourly_rate_usd`, verify the logged values match `estimate_cost_from_first_epoch()` output.

#### GREEN Phase

In `train_one_fold_task()`, after `trainer.fit()` returns:

1. Extract `epoch_seconds` from `fit_result` (the trainer already tracks per-epoch timing).
2. If `max_epochs > 1` and `epoch_seconds > 0`:
   a. Call `estimate_cost_from_first_epoch(epoch_seconds, max_epochs, num_folds, hourly_rate_usd)`.
   b. Log estimate dict to MLflow via `mlflow.log_metrics(estimate, step=0)`.
   c. Call `update_estimated_cost_gauges(estimate)` from `prometheus_metrics.py`.
3. If `max_epochs == 1` or `epoch_seconds` is missing, skip silently.

The `hourly_rate_usd` comes from `get_hourly_rate_usd()` (already reads INSTANCE_HOURLY_USD env).

**Key constraint**: The estimate uses `fit_result.get("epoch_0_seconds")` or derives from `training_time_seconds / max_epochs` as fallback. The trainer must expose per-epoch timing for this to be accurate. If the trainer does not provide epoch-0 time directly, use `training_time_seconds / actual_epochs_completed` as the per-epoch estimate.

#### VERIFY

```bash
MINIVESS_ALLOW_HOST=1 uv run pytest tests/v2/unit/orchestration/test_train_flow_cost_estimate.py -x -q
MINIVESS_ALLOW_HOST=1 uv run pytest tests/v2/unit/observability/test_infrastructure_timing_prediction.py -x -q
uv run ruff check src/minivess/orchestration/flows/train_flow.py
uv run mypy src/minivess/orchestration/flows/train_flow.py
```

#### Commit message

`feat(finops): T2 — wire epoch-0 cost estimate into train_flow (#717)`

---

## Task T3: Grafana Infrastructure Cost Dashboard

**Issue**: #747
**File**: `deployment/grafana/dashboards/infrastructure-cost.json`
**Test file**: `tests/v2/unit/deployment/test_grafana_infrastructure_cost.py`
**Marker**: (none) — staging tier, JSON validation only
**Dependencies**: T1 (metric names must match)

### TDD Spec

#### RED Phase

**Test file**: `tests/v2/unit/deployment/test_grafana_infrastructure_cost.py`

**Test 11**: `test_dashboard_json_is_valid` — File loads as valid JSON, has `dashboard.panels` list, `dashboard.uid` == `"minivess-infrastructure-cost"`.

**Test 12**: `test_dashboard_has_required_panels` — At least 6 panels exist with titles containing keywords: "Cost", "GPU Utilization", "Setup Fraction", "Effective Rate", "Estimated Cost", "Cost Trend" (or similar).

**Test 13**: `test_panel_queries_reference_correct_metrics` — Every panel target `expr` references a metric starting with `minivess_training_` (matching prometheus_metrics.py Gauge names).

**Test 14**: `test_dashboard_tags_include_finops` — `dashboard.tags` includes both `"minivess"` and `"finops"`.

#### GREEN Phase

Create `deployment/grafana/dashboards/infrastructure-cost.json` with 6-8 panels:

| Panel ID | Title | Type | Prometheus Query | Grid Position |
|----------|-------|------|-----------------|---------------|
| 1 | Total Training Cost (USD) | stat | `minivess_training_cost_total_usd` | (0,0, w=6, h=4) |
| 2 | Estimated Total Cost (USD) | stat | `minivess_training_estimated_total_cost_usd` | (6,0, w=6, h=4) |
| 3 | Effective GPU Rate (USD/hr) | gauge | `minivess_training_effective_gpu_rate_usd` | (12,0, w=6, h=4) |
| 4 | Cost Per Epoch (USD) | stat | `minivess_training_cost_per_epoch_usd` | (18,0, w=6, h=4) |
| 5 | GPU Utilization | gauge | `minivess_training_gpu_utilization` | (0,4, w=12, h=6) |
| 6 | Setup Overhead Fraction | gauge | `minivess_training_setup_fraction` | (12,4, w=12, h=6) |
| 7 | Cost Over Time | timeseries | `minivess_training_cost_total_usd` | (0,10, w=24, h=8) |
| 8 | Estimated vs Actual Cost | timeseries | `minivess_training_cost_total_usd` and `minivess_training_estimated_total_cost_usd` | (0,18, w=24, h=8) |

Follow the same JSON schema as `deployment/grafana/dashboards/bentoml-requests.json`:
- `schemaVersion: 39`
- `overwrite: true`
- `timezone: "utc"`

#### VERIFY

```bash
MINIVESS_ALLOW_HOST=1 uv run pytest tests/v2/unit/deployment/test_grafana_infrastructure_cost.py -x -q
uv run ruff check tests/v2/unit/deployment/test_grafana_infrastructure_cost.py
```

#### Commit message

`feat(grafana): T3 — infrastructure-cost.json dashboard (6-8 panels) (#747)`

---

## Task T4: RunAnalytics Cost Aggregation Methods

**Issue**: #735
**File**: `src/minivess/observability/analytics.py` (extend `RunAnalytics`)
**Test file**: `tests/v2/unit/observability/test_analytics_cost.py`
**Marker**: (none) — staging tier, DuckDB in-memory only
**Dependencies**: None (operates on DataFrames, not live MLflow)

### TDD Spec

#### RED Phase

**Test file**: `tests/v2/unit/observability/test_analytics_cost.py`

Build a fixture that creates a `pd.DataFrame` mimicking `load_experiment_runs()` output with columns: `run_id`, `run_name`, `status`, `start_time`, `end_time`, `param_model_family`, `metric_cost_total_usd`, `metric_cost_effective_gpu_rate`, `metric_cost_setup_fraction`, `metric_cost_break_even_epochs`, `metric_cost_epochs_to_amortize_setup`.

**Test 15**: `test_cost_by_model_family` — `RunAnalytics.cost_by_model_family(runs_df)` returns DataFrame with columns `model_family`, `total_cost_usd_mean`, `total_cost_usd_std`, `effective_rate_mean`, grouped by `param_model_family`.

**Test 16**: `test_cost_by_model_family_empty` — Empty DataFrame returns empty result, no crash.

**Test 17**: `test_cost_trends` — `RunAnalytics.cost_trends(runs_df)` returns DataFrame ordered by `start_time` with columns `run_id`, `start_time`, `metric_cost_total_usd`, `cumulative_cost_usd`.

**Test 18**: `test_break_even_analysis` — `RunAnalytics.break_even_analysis(runs_df)` returns DataFrame with `model_family`, `avg_break_even_epochs`, `avg_epochs_to_amortize`.

**Test 19**: `test_cost_by_model_family_multiple_families` — Given runs from 3 model families (dynunet, sam3_vanilla, segresnet), result has exactly 3 rows.

#### GREEN Phase

Add three methods to `RunAnalytics` in `src/minivess/observability/analytics.py`:

```python
def cost_by_model_family(self, runs_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cost metrics by model family.

    Returns DataFrame with mean/std of cost_total_usd and effective_gpu_rate
    grouped by param_model_family.
    """

def cost_trends(self, runs_df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative cost trend over time.

    Returns DataFrame ordered by start_time with cumulative_cost_usd column.
    """

def break_even_analysis(self, runs_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate break-even and amortization metrics by model family.

    Returns DataFrame with avg_break_even_epochs and avg_epochs_to_amortize
    grouped by param_model_family.
    """
```

All three use `self.register_dataframe()` + `self.conn.execute()` (DuckDB SQL) for consistency with existing methods.

#### VERIFY

```bash
MINIVESS_ALLOW_HOST=1 uv run pytest tests/v2/unit/observability/test_analytics_cost.py -x -q
uv run ruff check src/minivess/observability/analytics.py
uv run mypy src/minivess/observability/analytics.py
```

#### Commit message

`feat(analytics): T4 — RunAnalytics cost aggregation methods (#735)`

---

## Task T5: Prefect Maintenance Cleanup Task

**Issue**: #683
**File**: `src/minivess/orchestration/flows/maintenance_flow.py` (new)
**Test file**: `tests/v2/unit/orchestration/test_maintenance_flow.py`
**Marker**: (none) — staging tier, mocked MLflow
**Dependencies**: None (uses existing `ghost_cleanup.py`)

### TDD Spec

#### RED Phase

**Test file**: `tests/v2/unit/orchestration/test_maintenance_flow.py`

**Test 20**: `test_cleanup_stale_runs_task_exists` — `cleanup_stale_runs_task` is importable from `minivess.orchestration.flows.maintenance_flow` and is a Prefect `@task`.

**Test 21**: `test_cleanup_stale_runs_dry_run` — With mocked `MlflowClient`, `cleanup_stale_runs_task(dry_run=True)` calls `find_ghost_runs()` and `cleanup_ghost_runs(dry_run=True)`, returns result dict with `would_clean` count.

**Test 22**: `test_cleanup_stale_runs_wet_run` — With mocked client, `cleanup_stale_runs_task(dry_run=False)` calls `cleanup_ghost_runs(dry_run=False)`, returns result dict with `cleaned` count.

**Test 23**: `test_maintenance_flow_exists` — `maintenance_flow` is importable and is a Prefect `@flow`.

**Test 24**: `test_maintenance_flow_runs_cleanup` — `maintenance_flow()` calls `cleanup_stale_runs_task` at least once.

#### GREEN Phase

Create `src/minivess/orchestration/flows/maintenance_flow.py`:

```python
"""Maintenance Prefect flow — periodic cleanup of stale MLflow runs.

Flow 6: Runs as a scheduled maintenance task. Finds and cleans up
ghost runs (RUNNING runs that are likely orphaned from crashed containers
or preempted spot instances).

Issue: #683
"""
from __future__ import annotations

import logging
from typing import Any

from prefect import flow, task

from minivess.observability.ghost_cleanup import cleanup_ghost_runs, find_ghost_runs
from minivess.observability.tracking import resolve_tracking_uri

logger = logging.getLogger(__name__)


@task(name="cleanup-stale-runs")
def cleanup_stale_runs_task(
    *,
    experiment_name: str = "minivess_training",
    max_age_hours: float = 24.0,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Find and clean up stale RUNNING MLflow runs.

    Parameters
    ----------
    experiment_name:
        MLflow experiment to search for ghost runs.
    max_age_hours:
        Maximum age for a run to be considered potentially active.
    dry_run:
        If True, only report what would be cleaned.

    Returns
    -------
    Dict with cleanup results (cleaned/would_clean/errors counts).
    """
    ...  # implementation wraps find_ghost_runs + cleanup_ghost_runs


@flow(name="maintenance")
def maintenance_flow(
    *,
    experiment_name: str = "minivess_training",
    dry_run: bool = True,
) -> dict[str, Any]:
    """Maintenance flow — periodic MLflow cleanup.

    Designed for Prefect scheduling (e.g., daily at 03:00 UTC).
    """
    ...  # calls cleanup_stale_runs_task
```

Register `FLOW_NAME_MAINTENANCE = "maintenance"` in `src/minivess/orchestration/constants.py`.

#### VERIFY

```bash
MINIVESS_ALLOW_HOST=1 PREFECT_DISABLED=1 uv run pytest tests/v2/unit/orchestration/test_maintenance_flow.py -x -q
uv run ruff check src/minivess/orchestration/flows/maintenance_flow.py
uv run mypy src/minivess/orchestration/flows/maintenance_flow.py
```

#### Commit message

`feat(orchestration): T5 — Prefect maintenance cleanup task (#683)`

---

## Task T6: Wire generate_timing_jsonl() into MLflow Artifacts

**Issue**: #683
**File**: `src/minivess/orchestration/flows/train_flow.py` (modify)
**Test file**: `tests/v2/unit/orchestration/test_train_flow_timing_artifact.py`
**Marker**: (none) — staging tier, mocked MLflow
**Dependencies**: None

### TDD Spec

#### RED Phase

**Test file**: `tests/v2/unit/orchestration/test_train_flow_timing_artifact.py`

**Test 25**: `test_timing_jsonl_logged_as_artifact` — After training completes with setup_seconds > 0, verify `mlflow.log_text()` or `mlflow.log_artifact()` is called with artifact path containing `timing/timing_report.jsonl`.

**Test 26**: `test_timing_jsonl_content_is_valid` — The JSONL content passed to `mlflow.log_text()` contains at least one line with `"phase":"cost"` and `"operation":"cost_summary"`.

**Test 27**: `test_timing_jsonl_skipped_without_timing_file` — When no timing_setup.txt exists (local runs), no JSONL artifact is logged.

#### GREEN Phase

In `training_flow()`, inside the MLflow run context (after cost analysis logging, around line 830), add:

```python
# Log timing JSONL artifact (#683 — wire dead code)
if _setup_seconds > 0 or _total_training_seconds > 0:
    from minivess.observability.infrastructure_timing import generate_timing_jsonl

    timing_jsonl = generate_timing_jsonl(
        setup_durations=_setup_durations if _setup_seconds > 0 else {},
        training_seconds=_total_training_seconds,
        epoch_count=max_epochs * len(fold_results),
        hourly_rate_usd=get_hourly_rate_usd(),
    )
    mlflow.log_text(timing_jsonl, "timing/timing_report.jsonl")
```

This wires the previously dead `generate_timing_jsonl()` function into the actual pipeline.

#### VERIFY

```bash
MINIVESS_ALLOW_HOST=1 uv run pytest tests/v2/unit/orchestration/test_train_flow_timing_artifact.py -x -q
uv run ruff check src/minivess/orchestration/flows/train_flow.py
uv run mypy src/minivess/orchestration/flows/train_flow.py
```

#### Commit message

`feat(observability): T6 — wire generate_timing_jsonl() as MLflow artifact (#683)`

---

## Ralph-Loop Monitoring Checkpoint

**When**: After all 6 tasks are DONE (or at least T1 + T3 complete).
**Purpose**: Verify the full Prometheus -> Grafana pipeline works end-to-end.

### Checkpoint Steps

1. **Docker Compose smoke test** — Verify Prometheus scrapes BentoML /metrics:
   ```bash
   docker compose --env-file .env -f deployment/docker-compose.yml --profile monitoring up -d
   # Wait for services
   curl -s http://localhost:9090/api/v1/targets | python -m json.tool | grep bentoml
   ```

2. **Verify Grafana dashboard provisioning** — Check infrastructure-cost.json is loaded:
   ```bash
   curl -s -u "$GRAFANA_USER:$GRAFANA_PASS" http://localhost:3000/api/dashboards/uid/minivess-infrastructure-cost | python -m json.tool | head -5
   ```

3. **Verify Prometheus metrics exist** (even if zero):
   ```bash
   curl -s http://localhost:9090/api/v1/query?query=minivess_training_cost_total_usd | python -m json.tool
   ```

4. **Staging test suite passes** — All new + existing tests:
   ```bash
   make test-staging
   ```

5. **Ralph-loop assessment**: Record in `state/tdd-state.json`:
   - Total tests passing
   - Total tests skipped
   - Any regressions in existing 23 tests
   - Prometheus /metrics confirmed accessible: yes/no
   - Grafana dashboard loaded: yes/no

### Acceptance Criteria for Ralph-Loop Checkpoint

| Check | Pass Condition |
|-------|---------------|
| New tests | All 20-25 tests pass (0 failures, 0 errors) |
| Existing tests | 23 existing infrastructure_timing tests still pass |
| Lint | `ruff check src/ tests/` clean |
| Type check | `mypy src/minivess/observability/ src/minivess/orchestration/` clean |
| Prometheus scrape | `curl localhost:9090/api/v1/targets` shows bentoml job UP |
| Grafana dashboard | `curl localhost:3000/api/dashboards/uid/minivess-infrastructure-cost` returns 200 |
| Metric names | `prometheus_client.generate_latest()` contains all 6 `minivess_training_*` gauges |

---

## Task Execution Order

```
Phase 0: KG decision node (no code, no tests)
   |
   v
T1: prometheus_metrics.py (independent)
   |
   +---> T2: Wire epoch-0 estimate (depends on T1 for Gauge updates)
   +---> T3: Grafana dashboard (depends on T1 for metric names)
   |
T4: RunAnalytics cost methods (independent of T1-T3)
T5: Maintenance flow (independent of T1-T4)
T6: Wire timing JSONL (independent of T1-T5)
   |
   v
Ralph-Loop Checkpoint (after all tasks)
```

**Parallelizable**: T4, T5, T6 are independent and can run in any order.
**Sequential**: T2 and T3 depend on T1 (metric names and module must exist).

---

## TDD State File Integration

Before starting, reset `state/tdd-state.json` for this plan:

```json
{
  "plan_file": "docs/planning/pr1-finops-infrastructure-timing-plan.md",
  "plan_version": "1.0",
  "execution_mode": "autonomous",
  "current_task_id": null,
  "current_phase": null,
  "inner_iteration": 0,
  "tasks": {
    "T1": { "status": "TODO", "phase": null, "iterations": 0, "issue_ref": "#747" },
    "T2": { "status": "TODO", "phase": null, "iterations": 0, "issue_ref": "#717" },
    "T3": { "status": "TODO", "phase": null, "iterations": 0, "issue_ref": "#747" },
    "T4": { "status": "TODO", "phase": null, "iterations": 0, "issue_ref": "#735" },
    "T5": { "status": "TODO", "phase": null, "iterations": 0, "issue_ref": "#683" },
    "T6": { "status": "TODO", "phase": null, "iterations": 0, "issue_ref": "#683" }
  },
  "convergence": {
    "reached": false,
    "tasks_done": 0,
    "tasks_total": 6,
    "tasks_stuck": 0
  },
  "session_inner_iterations": 0,
  "session_start": null
}
```

---

## Git Commit Sequence

| Order | Commit Message | Files Changed |
|-------|---------------|---------------|
| 0 | `chore(kg): add cost_tracking_strategy decision node (Phase 0, PR-1)` | `knowledge-graph/decisions/L5-operations/cost-tracking-strategy.yaml`, `knowledge-graph/domains/operations.yaml` |
| 1 | `feat(observability): T1 — prometheus_metrics.py shared Gauges module (#747)` | `src/minivess/observability/prometheus_metrics.py`, `tests/v2/unit/observability/test_prometheus_metrics.py` |
| 2 | `feat(finops): T2 — wire epoch-0 cost estimate into train_flow (#717)` | `src/minivess/orchestration/flows/train_flow.py`, `tests/v2/unit/orchestration/test_train_flow_cost_estimate.py` |
| 3 | `feat(grafana): T3 — infrastructure-cost.json dashboard (6-8 panels) (#747)` | `deployment/grafana/dashboards/infrastructure-cost.json`, `tests/v2/unit/deployment/test_grafana_infrastructure_cost.py` |
| 4 | `feat(analytics): T4 — RunAnalytics cost aggregation methods (#735)` | `src/minivess/observability/analytics.py`, `tests/v2/unit/observability/test_analytics_cost.py` |
| 5 | `feat(orchestration): T5 — Prefect maintenance cleanup task (#683)` | `src/minivess/orchestration/flows/maintenance_flow.py`, `tests/v2/unit/orchestration/test_maintenance_flow.py`, `src/minivess/orchestration/constants.py` |
| 6 | `feat(observability): T6 — wire generate_timing_jsonl() as MLflow artifact (#683)` | `src/minivess/orchestration/flows/train_flow.py`, `tests/v2/unit/orchestration/test_train_flow_timing_artifact.py` |
| 7 | `chore: ralph-loop checkpoint — verify Prometheus/Grafana pipeline` | `state/tdd-state.json` |

---

## Test Tier Assignment

All new tests belong to **staging tier** (no model loading, no slow, no integration):

| Test File | Tests | Tier | Marker |
|-----------|-------|------|--------|
| `tests/v2/unit/observability/test_prometheus_metrics.py` | 6 | staging | (none) |
| `tests/v2/unit/orchestration/test_train_flow_cost_estimate.py` | 4 | staging | (none) |
| `tests/v2/unit/deployment/test_grafana_infrastructure_cost.py` | 4 | staging | (none) |
| `tests/v2/unit/observability/test_analytics_cost.py` | 5 | staging | (none) |
| `tests/v2/unit/orchestration/test_maintenance_flow.py` | 5 | staging | (none) |
| `tests/v2/unit/orchestration/test_train_flow_timing_artifact.py` | 3 | staging | (none) |
| **Total** | **27** | | |

All tests mock MLflow/Prefect (no live services required). All use `tmp_path` fixture where filesystem is needed. All pass under `make test-staging`.

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| `prometheus_client` not in dependencies | Verify with `uv run python -c "import prometheus_client"`. If missing, `uv add prometheus_client`. |
| BentoML /metrics port mismatch | `prometheus.yml` scrapes `minivess-bento:3000`. Verify BentoML actually serves on 3000 (not 3333 from .env). May need to align. |
| Gauge name collisions with BentoML built-ins | All custom metrics use `minivess_training_` prefix. BentoML uses `bentoml_` prefix. No collision. |
| `RunAnalytics` DuckDB schema sensitivity | Tests build explicit DataFrames with known columns. No dependence on live MLflow. |
| `maintenance_flow.py` import collisions | Use unique flow name `"maintenance"` registered in constants.py. |
| `fit_result` missing epoch timing keys | T2 implementation uses `.get()` with fallback to `training_time_seconds / epochs`. |

---

## Scope Boundary (Explicitly Excluded)

- **#751 (Docker pull optimization)** — MOVED OUT of PR-1 per user decision.
- **BentoML service code changes** — prometheus_metrics.py is a standalone module. BentoML already has a `/metrics` endpoint via `prometheus_client`. Our custom Gauges are registered in the same process registry and will appear automatically.
- **Prometheus alert rules for cost** — Future work. This PR adds metrics and dashboards only.
- **Cloud cost API integration** — (e.g., RunPod billing API, GCP cost export). Future work. This PR uses `INSTANCE_HOURLY_USD` env var.
- **Multi-GPU cost accounting** — Future work. Current implementation assumes single-GPU cost.
