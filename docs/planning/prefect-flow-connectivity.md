# Prefect Flow Connectivity — MLflow as Inter-Flow Contract

**Branch**: `fix/prefect-flow-connectivity`
**Date**: 2026-03-09
**Status**: Planning (v2 — incorporates Dashboard correction, foundation-PLR analysis, extensible topology)

---

## 1. The Problem

The debug run on 2026-03-09 showed all 7 flows returning `OK` — but that was a lie.
Post-training and Analysis ran but did nothing:

```
post-training: 4 plugins skipped ("No checkpoint paths provided", "got 0 checkpoints")
analyze:       0 models evaluated, no champion, no comparison table, no figures
```

The pipeline plumbing is broken at the **train → post-training → analyze** seam.
The flows execute cleanly as Prefect tasks but carry no real data between them.

### 1.1 Root Causes (Diagnosed from Code)

| Problem | Location | Evidence |
|---------|----------|---------|
| Train flow never logs `checkpoint_dir` path to MLflow | `train_flow.py:604` | `checkpoint_dir = checkpoint_base / "fold_{n}"` — written to disk only, never to MLflow |
| Post-training calls `find_upstream_safely` but gets `run_id` only — no checkpoint location | `post_training_flow.py:273-279` | Even if upstream found, checkpoints not in MLflow |
| Analysis hardcodes `experiment_name="minivess_training"` — breaks in debug | `analysis_flow.py:1698` | Debug uses `_DEBUG` suffix; no `_resolve_experiment_name()` utility exists |
| `flow_status: completed` tag is set but `FLOW_COMPLETE` is never queried with checkpoint artifact path | `flow_contract.py:log_flow_completion` | `flow_artifacts` param exists but callers never pass checkpoint paths |
| No `_DEBUG` suffix on experiment names | all flows | Debug runs pollute production experiment namespace |
| Data Engineering never writes `splits_path` tag | `data_flow.py` | `find_upstream_safely(experiment_name="minivess_data")` in train_flow always returns None |
| `FlowContract.find_upstream_run()` does not propagate debug suffix | `flow_contract.py` | Callers hardcode `experiment_name="minivess_training"` |

---

## 2. Overarching Design Principles

The following principles govern all connectivity decisions in this document.
These are derived from the TOP-1 principle in CLAUDE.md and from analysis of
the `foundation-PLR` reference implementation.

### 2.1 MLflow is the ONLY Inter-Flow Communication Channel

Flows communicate exclusively through MLflow. No shared filesystem paths (beyond
volume-mounted artifact directories that MLflow references by tag). No direct
function calls between flow containers. No Prefect artifacts used as inter-flow
data (only for observability).

### 2.2 Config-Driven Flow Dispatch (from foundation-PLR)

`foundation-PLR`'s `pipeline_PLR.py` uses a clean pattern for extensible flow
execution:

```python
# Config-driven — add new flows by editing YAML, not Python
if prefect_flows["POST_TRAINING"]:
    flow_post_training(cfg)
if prefect_flows["SURVIVAL_ANALYSIS"]:
    flow_survival_analysis(cfg)
if prefect_flows["CLASSIFICATION"]:
    flow_classification_on_mask(cfg)
```

This means a researcher adds a new flow by:
1. Adding a YAML key under `prefect_flows:`
2. Implementing the flow function
3. Adding the `if prefect_flows[...]` dispatch in `pipeline.py`

No existing flow code changes. This is the target pattern for MinIVess.

### 2.3 Structured Run Name Encoding (from foundation-PLR)

`foundation-PLR` encodes pipeline state into MLflow run names:

```python
# Run name encodes the pipeline configuration — parseable by downstream flows
run_name = f"{classifier}__{featurization}__{imputation}__{outlier_detection}"
```

MinIVess adaptation:

```python
# Training run name — encodes model, experiment config, seed
run_name = f"{model_name}__{experiment_name}__{seed}"

# Post-training run name — extends training run name with variant
run_name = f"{base_run_name}__swa"
run_name = f"{base_run_name}__calibrated"
run_name = f"{base_run_name}__conformal"
run_name = f"{base_run_name}__merged"
```

Downstream flows parse via `str.split("__")` — **no regex** (banned by CLAUDE.md rule 16).

### 2.4 Two-Block Design: Extraction → Analysis (from foundation-PLR)

`foundation-PLR` cleanly separates:
- **Extraction block**: MLflow → DuckDB (computation-only, no visualization)
- **Analysis block**: DuckDB → figures/tables (visualization-only, no computation)

MinIVess should adopt this structure in the Analysis flow to decouple heavy evaluation
from reporting:
- `analysis_flow.py` Block 1: Load all runs from MLflow → evaluate → write to DuckDB
- `analysis_flow.py` Block 2: DuckDB → comparison tables, figures, champion selection

### 2.5 `define_sources_for_flow()` Pattern (from foundation-PLR)

`foundation-PLR`'s 1585-line `define_sources_for_flow(cfg, prev_experiment_name, task)`
function is the most valuable pattern. It centralizes ALL MLflow querying for a flow in
one function that returns resolved artifact URIs. MinIVess equivalent:

```python
def define_sources_for_flow(
    flow_name: str,
    *,
    debug_suffix: str = "",
    min_metric: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Resolve all upstream MLflow artifacts for a given flow.

    Returns
    -------
    Dict with keys: run_ids, checkpoint_paths, splits_path, config_artifact, etc.
    Raises RuntimeError if a required upstream source is not available.
    """
```

This function is the `FlowContract`'s natural home. It replaces ad-hoc `find_upstream_safely`
calls scattered across flow files.

---

## 3. Full Flow Connectivity Architecture

The following diagram shows the intended information flow. MLflow is the **only**
inter-flow communication channel. Flows are independent Docker containers.

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              MLFLOW CONTRACT LAYER                                   │
│                                                                                      │
│  Experiment: minivess_data[_DEBUG]           (Data Engineering writes)               │
│  Experiment: minivess_training[_DEBUG]       (Train + Post-training write)           │
│  Experiment: minivess_analysis[_DEBUG]       (Analysis writes)                       │
│  Experiment: minivess_biostatistics[_DEBUG]  (Biostatistics writes)                  │
│  Experiment: minivess_hpo[_DEBUG]            (HPO writes)                            │
│  Model Registry: minivess-champion           (Analysis registers, Deploy reads)      │
└──────────────────────────────────────────────────────────────────────────────────────┘

Data Acquisition ──→ (filesystem only: raw_data volume, no MLflow)

Data Engineering ──→ mlflow: minivess_data[_DEBUG]
    writes: dataset_hash, n_volumes, dvc_commit, splits_path tag
    ↓
    [FlowContract: flow_status=FLOW_COMPLETE + splits_path tag]

Modelling (Train) ──reads──→ minivess_data[_DEBUG] (latest FINISHED run, splits_path tag)
    writes: minivess_training[_DEBUG]
    ├── parent run: n_folds_completed, model_name, experiment config artifact
    └── per-fold child runs: checkpoint_dir tag, fold_id tag, best_val_loss metric
    ↓
    [FlowContract: FLOW_COMPLETE + checkpoint_dir_{fold_n} tags on parent run]

[OPTIONAL PARALLEL BRANCHES — config-driven, do not block main chain:]
├── HPO Flow ──→ minivess_hpo[_DEBUG] (parallel search over minivess_training)
├── Survival Analysis ──→ minivess_survival[_DEBUG] (uses same checkpoints, new labels)
└── Classification ──reads──→ minivess_training[_DEBUG] → (sequential, uses seg output as input)

Post-training ──reads──→ minivess_training[_DEBUG] (latest FINISHED parent run → fold runs)
    reads: checkpoint_dir tags from each fold run → resolves .ckpt files on volume
    writes: SAME experiment (minivess_training[_DEBUG]), new sibling runs with __ suffix:
      {base}__swa, {base}__calibrated, {base}__conformal, {base}__merged
    ↓
    [FlowContract: FLOW_COMPLETE on each post-training sibling run]

Analysis ──reads──→ minivess_training[_DEBUG] (ALL runs: base + post-training variants)
    Block 1 (Extraction): Evaluate all model × fold × variant permutations → DuckDB
    Block 2 (Analysis): DuckDB → comparison tables, champion selection, figures
    writes: minivess_analysis[_DEBUG]
      - one run per evaluated model/ensemble
      - champion registered in Model Registry
    ↓
    [FlowContract: FLOW_COMPLETE + champion_run_id tag + champion_model_uri tag]

Biostatistics ──reads──→ minivess_analysis[_DEBUG] (latest FINISHED run)
    reads: comparison table artifact, all evaluation run metrics
    writes: minivess_biostatistics[_DEBUG]
      run_name = "paper_comparison_{YYYY-MM-DD}"
      artifacts: comparison CSV, LaTeX tables, PDF figures, Parquet export
    ↓
    [FlowContract: FLOW_COMPLETE + artifact URIs]
    ↓
    Notifies Dashboard (via Prefect event / polling — NOT a hardcoded trigger)

Deployment ──reads──→ Model Registry (champion from Analysis)
    reads: mlflow_analysis[_DEBUG] → champion_model_uri tag → Model Registry
    deploys via BentoML
    no MLflow writes

Dashboard ──reads──→ 23 independent sources (NOT just Biostatistics)
    See Section 4 for full architecture.
    Auto-updates: Prefect completion events + independent scheduled polling per source.

Data Annotation ──reads──→ BentoML served endpoint (from Deployment)
    no MLflow reads or writes
```

---

## 4. Dashboard: 23-Source Unified Integration Hub

**IMPORTANT**: The Dashboard is NOT a simple downstream of Biostatistics. It is an
independent React+FastAPI+D3.js application (issue #331) that integrates 23 data sources
and displays 8 live monitoring sections. It auto-updates from TWO independent channels:
1. Prefect flow completion events (near-real-time on pipeline activity)
2. Independent scheduled polling per source (Grafana, Evidently, Posthog, etc. update
   even when no Prefect flows are running)

### 4.1 Architecture (Issue #331)

```
┌─────────────────────────────────────────────────────┐
│  React + Vite + TypeScript + D3.js Frontend         │
│  Port: UI=3002                                      │
│                                                     │
│  8 Sections: Data | Models | Post-Training |        │
│    Analysis | Biostatistics | Deployment |          │
│    Drift | Trust                                    │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP / WebSocket
┌──────────────────────▼──────────────────────────────┐
│  FastAPI Backend                                    │
│  Port: API=8090                                     │
│  12 route groups — one per source cluster          │
│                                                     │
│  Service Adapters: connect(), query(), cache()      │
│  per data source — fully pluggable                  │
└──────────────────────┬──────────────────────────────┘
                       │ Adapters
┌─────── 23 Data Sources ────────────────────────────┐
│ MLflow         Prefect        Grafana               │
│ DVC            Evidently      whylogs               │
│ BentoML        Prometheus     Langfuse              │
│ Optuna         DuckDB         Posthog               │
│ GitHub         Label Studio   MONAI Deploy          │
│ Biostatistics  OpenLineage    Deepchecks            │
│ WeightWatcher  netcal         MAPIE                 │
│ Cleanlab       Braintrust                           │
└─────────────────────────────────────────────────────┘
```

### 4.2 Update Architecture (No Hardcoded Trigger List)

The Dashboard MUST NOT have a hardcoded list of flows that trigger it. Instead:

**Channel 1: Prefect flow completion events**
```python
# dashboard_flow.py subscribes to Prefect automations:
# - Any flow tagged "minivess-pipeline" emits a completion event
# - Dashboard backend receives webhook → triggers refresh for affected sections
# - Each flow is tagged with which dashboard sections it feeds
# - Adding a new flow = adding a tag, not modifying Dashboard code
```

**Channel 2: Per-source scheduled polling**
```python
# Each source adapter has its own poll interval:
adapters = {
    "grafana":   GrafanaAdapter(poll_interval="5m"),
    "evidently": EvidentlyAdapter(poll_interval="30m"),
    "posthog":   PosthogAdapter(poll_interval="1h"),
    "mlflow":    MLflowAdapter(poll_interval="2m"),
    # ... 19 more adapters
}
# Dashboard backend scheduler: run each adapter.poll() on its own schedule
# Independent of whether any Prefect flow ran
```

**Implementation rule**: When adding a new data source, create a new adapter class
implementing `BaseSourceAdapter(connect, query, cache, poll_interval)`. Do NOT modify
existing adapters or Dashboard routing code.

### 4.3 What Biostatistics Actually Is

Biostatistics is ONE source among 23 in the Dashboard — specifically one of the MLflow
experiments (`minivess_biostatistics`) that the MLflow adapter queries. It is also a
standalone scientific output (paper-ready comparison run). It is NOT the "primary feed"
for the Dashboard. Other sources (Grafana, Evidently, Prefect) update the Dashboard
completely independently of Biostatistics.

---

## 5. Extensible Flow Topology

### 5.1 Supported Topologies

The platform must support three topologies without code changes to core infrastructure:

**Type A: Linear Sequential (default)**
```
Data Engineering → Train → Post-Training → Analysis → Biostatistics → Deploy
```

**Type B: Parallel Independent**
```
Train ─────────────────────────────────────────→ Analysis
      └─→ HPO Flow (separate experiment)         ↑
      └─→ Survival Analysis (new labels, same ckpts) ┘
```
Both branches read checkpoints from `minivess_training[_DEBUG]` and write to their own
experiment. Analysis aggregates ALL sources via `define_sources_for_flow()`.

**Type C: Sequential Dependent (mask-conditioned)**
```
Train → [produces segmentation mask] → Classification Flow → Analysis
```
Classification uses segmentation output as input. It is a NEW flow in the same
experiment chain — does NOT modify Train or Analysis code.

### 5.2 Adding a New Flow (Zero Modification to Existing Flows)

Recipe for adding "Survival Analysis Flow":

```python
# 1. Add to configs/experiment/base.yaml:
prefect_flows:
  SURVIVAL_ANALYSIS: true

# 2. Create src/minivess/orchestration/flows/survival_flow.py:
@flow(name=FLOW_NAME_SURVIVAL)
def survival_flow(...) -> SurvivalResult:
    sources = define_sources_for_flow("survival", debug_suffix=debug_suffix)
    # uses sources["checkpoint_paths"] from minivess_training experiment
    ...
    # writes to minivess_survival[_DEBUG] experiment

# 3. Add dispatch in pipeline.py:
if prefect_flows["SURVIVAL_ANALYSIS"]:
    survival_flow(cfg)

# 4. Add FLOW_NAME_SURVIVAL = "minivess-survival-flow" to constants.py
# 5. Add experiment name to the experiment name map (Section 7 below)
```

No changes to `train_flow.py`, `analysis_flow.py`, or `FlowContract`.

### 5.3 Experiment Name Isolation for Parallel Flows

Parallel flows MUST write to their own experiments. NEVER reuse another flow's experiment:

```
minivess_training[_DEBUG]      → Train + Post-Training only
minivess_hpo[_DEBUG]           → HPO runs only
minivess_survival[_DEBUG]      → Survival Analysis only
minivess_classification[_DEBUG] → Classification on mask only
minivess_analysis[_DEBUG]      → Aggregated evaluation of ALL of the above
```

The Analysis flow is the **single aggregation point** — it reads from multiple upstream
experiments and produces a unified comparison.

---

## 6. Per-Flow MLflow Contract Specification

### 6.1 Data Acquisition Flow
**Current state**: No MLflow integration.
**Target state**: Unchanged — filesystem only. Writes raw data to `raw_data` volume.
**Reads**: Nothing.
**Writes**: Nothing to MLflow.

### 6.2 Data Engineering Flow
**Current state**: Writes to `minivess_data` but `find_upstream_safely` in train_flow
returns None (tag/experiment name mismatch or run missing).
**Target state**: Write one FINISHED run per dataset version.

**Required MLflow writes** (experiment: `minivess_data{debug_suffix}`):
```python
mlflow.log_param("data_hash", dataset_hash)           # sha256 of all volume IDs
mlflow.log_param("data_n_volumes", n_volumes)
mlflow.log_param("data_dvc_commit", dvc_commit)        # git hash of DVC pointer
mlflow.log_param("splits_version", splits_hash)
mlflow.set_tag("flow_name", "data-flow")
mlflow.set_tag("flow_status", "FLOW_COMPLETE")         # queried by train_flow
mlflow.set_tag("splits_path", str(splits_path))        # absolute path on shared volume
```

**Open question — DVC Dataset Registration**:
MLflow 3.x supports `mlflow.log_input(mlflow.data.from_dvc(...))` to register a DVC
dataset entity linked to a run. This enables "which model trained on which dataset version?"
lineage queries (IEC 62304 value).
*Recommendation*: Register it. Phase 1 — not blocking flow connectivity.

### 6.3 Modelling (Train) Flow
**Current state**: Logs metrics per fold but never logs `checkpoint_dir` paths. Post-training
has no way to find checkpoints via MLflow.

**Required MLflow writes** (experiment: `minivess_training{debug_suffix}`):

Parent run (one per `training_flow()` call):
```python
mlflow.set_tag("flow_name", "training-flow")
mlflow.set_tag("flow_status", "FLOW_COMPLETE")         # MISSING — critical for post-training
mlflow.set_tag("model_name", config["model"])          # MISSING
mlflow.set_tag("n_folds", str(n_folds))
mlflow.log_param("n_folds_completed", n_folds)         # already done
mlflow.log_artifact(resolved_config_path)              # already done
```

Per-fold child run (one per fold):
```python
# Run name encodes pipeline state (foundation-PLR pattern):
run_name = f"{model_name}__{experiment_name}__fold{fold_id}"

mlflow.set_tag("checkpoint_dir", str(checkpoint_dir))  # MISSING — critical
mlflow.set_tag("fold_id", str(fold_id))                # MISSING
mlflow.set_tag("flow_name", "training-flow")
mlflow.set_tag("parent_run_id", parent_run_id)         # MISSING — enables find_fold_checkpoints()
mlflow.log_metric("best_val_loss", best_val_loss)      # already done
```

**The critical missing piece**: `checkpoint_dir` tag on each fold run.

**Storage decision — tags vs artifacts** (see Design Decisions section):
Phase 0 uses tags (volume path, no storage duplication).
Phase 3 can upgrade to `mlflow.log_artifact(ckpt_path)` for MinIO-backed portability.

### 6.4 Post-Training Flow
**Current state**: Queries `find_upstream_safely` → gets run_id only. `checkpoint_paths`
defaults to `[]`, all 4 plugins skip.

**Required changes**:
1. Call `define_sources_for_flow("post-training", ...)` → returns `checkpoint_paths` list
2. For each fold's checkpoint: run SWA, calibration, conformal, merging plugins
3. Write sibling runs with `__` suffix naming convention

**MLflow writes** (same experiment: `minivess_training{debug_suffix}`):
```python
# Sibling runs (NOT child runs) — cleaner MLflow UI, independent queryability
run_name = f"{base_run_name}__swa"
run_name = f"{base_run_name}__calibrated"
run_name = f"{base_run_name}__conformal"
run_name = f"{base_run_name}__merged"

mlflow.set_tag("flow_name", "post-training-flow")
mlflow.set_tag("flow_status", "FLOW_COMPLETE")
mlflow.set_tag("post_training_variant", "swa|calibrated|conformal|merged")
mlflow.set_tag("upstream_training_run_id", training_run_id)
mlflow.set_tag("checkpoint_path", str(output_checkpoint_path))
```

### 6.5 Analysis Flow
**Current state**: Hardcodes `experiment_name="minivess_training"`. Evaluates 0 models
because checkpoint tags are absent.

**Required changes** (two-block design from foundation-PLR):
```
Block 1 (Extraction):
  sources = define_sources_for_flow("analysis", debug_suffix=debug_suffix)
  # sources returns: all run_ids from minivess_training[_DEBUG] (base + variants)
  # For each run: resolve checkpoint_path/checkpoint_dir tag → actual .ckpt file
  # Build permutation matrix: model × fold × variant × ensemble_strategy
  # Evaluate each permutation → write metrics to DuckDB

Block 2 (Analysis):
  # DuckDB → comparison tables, champion selection
  # Select champion by primary metric (DSC + clDice composite)
  # Register champion in Model Registry
```

**MLflow writes** (experiment: `minivess_analysis{debug_suffix}`):
```python
# One run per evaluated model/ensemble configuration
run_name = f"{model_name}__fold{fold_id}__{variant}__dsc{dsc:.3f}"

mlflow.set_tag("flow_name", "analysis-flow")
mlflow.set_tag("flow_status", "FLOW_COMPLETE")         # on final summary run only
mlflow.set_tag("champion_run_id", champion_run_id)
mlflow.set_tag("champion_model_uri", champion_model_uri)
# Register champion:
mlflow.register_model(champion_model_uri, "minivess-champion")
```

### 6.6 Biostatistics Flow
**Current state**: Stub — not wired to Analysis output.

**Target state**:
1. `define_sources_for_flow("biostatistics", ...)` → reads `minivess_analysis{debug_suffix}`
2. Download comparison table artifact
3. Aggregate all metrics across models, variants, folds, datasets
4. Write ONE MLflow run per invocation

**MLflow writes** (experiment: `minivess_biostatistics{debug_suffix}`):
```python
run_name = f"paper_comparison_{date.today().isoformat()}"  # dated snapshot
mlflow.set_tag("flow_name", "biostatistics-flow")
mlflow.set_tag("flow_status", "FLOW_COMPLETE")
mlflow.set_tag("upstream_analysis_run_id", analysis_run_id)
mlflow.log_artifact(comparison_csv_path)
mlflow.log_artifact(latex_table_path)
mlflow.log_artifact(pdf_figures_path)
mlflow.log_artifact(parquet_export_path)
```

### 6.7 Deployment Flow
**Current state**: Reads `minivess_training` (hardcoded) — same debug suffix problem.
**Target state**: Read champion from Model Registry (populated by Analysis) + from
`minivess_analysis{debug_suffix}` for the `champion_run_id` tag.

**Reads only** — no MLflow writes:
```python
sources = define_sources_for_flow("deploy", debug_suffix=debug_suffix)
champion_uri = sources["champion_model_uri"]
# Fetch from Model Registry:
client.get_latest_versions("minivess-champion", stages=["Production"])
```

### 6.8 Dashboard Flow
**Current state**: Reads `minivess_biostatistics` (stub).
**Target state**: See Section 4 — Dashboard reads from 23 independent sources via adapters.
MLflow (`minivess_biostatistics`) is ONE of those sources.

### 6.9 Data Annotation Flow
**Target state**: Calls BentoML served endpoint for inference on new data.
No MLflow reads or writes. Output: new labels → Label Studio for annotation review.

### 6.10 HPO Flow (Parallel, Optional)
**Reads**: `minivess_training[_DEBUG]` (best single-model run for warm-starting search space)
**Writes**: `minivess_hpo[_DEBUG]` (one run per Optuna trial)
**Triggered**: Via `prefect_flows["HPO"]: true` in experiment config — NOT automatically
after every training run.

---

## 7. Debug Experiment Naming Convention

**Rule**: All experiment names get a `_DEBUG` suffix in debug runs.
Not the mlrun names (those stay human-readable within the experiment).

```python
# In constants.py or flow_utils.py — single implementation:
import os

def resolve_experiment_name(base_name: str) -> str:
    """Append _DEBUG suffix when MINIVESS_DEBUG_SUFFIX env var is set."""
    suffix = os.environ.get("MINIVESS_DEBUG_SUFFIX", "")
    return f"{base_name}{suffix}"

# Usage in every flow:
experiment_name = resolve_experiment_name("minivess_training")
# → "minivess_training_DEBUG" when MINIVESS_DEBUG_SUFFIX=_DEBUG
# → "minivess_training" in production (env var not set)
```

**Propagation**: `FlowContract.find_upstream_run()` must call `resolve_experiment_name()`
internally — callers should NOT pass the suffix directly (single source of truth in env var).

**Setting the flag**: `run_debug.sh` exports `MINIVESS_DEBUG_SUFFIX=_DEBUG` for all
container environments via `-e MINIVESS_DEBUG_SUFFIX=_DEBUG`.

**Why this matters**:
- `minivess_training` = real production experiment (100 epochs, full data)
- `minivess_training_DEBUG` = smoke test (2 epochs, 5 volumes, code validation only)
- MLflow UI cleanly separates them
- FlowContract queries automatically propagate the same suffix through the whole chain
- Adding a new experiment follows the same pattern: `resolve_experiment_name("minivess_survival")`

---

## 8. FlowContract: Required Changes

### 8.1 Centralize Upstream Discovery in `define_sources_for_flow()`

Replace scattered `find_upstream_safely()` calls with a single structured function
(foundation-PLR pattern — adapted to MinIVess):

```python
# src/minivess/orchestration/flow_contract.py

def define_sources_for_flow(
    self,
    flow_name: str,
    *,
    min_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Resolve all upstream MLflow artifacts for a given flow.

    The debug suffix is read from MINIVESS_DEBUG_SUFFIX env var internally —
    callers do NOT pass it (single source of truth).

    Returns
    -------
    Dict with flow-specific keys. Raises RuntimeError if required source missing.
    """
    if flow_name == "train":
        return self._sources_for_train()
    elif flow_name == "post-training":
        return self._sources_for_post_training()
    elif flow_name == "analysis":
        return self._sources_for_analysis()
    # ... etc. New flows: add a new `elif` branch + `_sources_for_X()` method
```

### 8.2 Add `find_fold_checkpoints()`

```python
def find_fold_checkpoints(
    self,
    *,
    parent_run_id: str,
) -> list[dict[str, Any]]:
    """Find all fold child runs and their checkpoint paths.

    Queries MLflow for all runs tagged with parent_run_id,
    returns checkpoint_dir tag values resolved to actual paths.

    Returns
    -------
    List of dicts: [{fold_id, run_id, checkpoint_dir, available_ckpts}, ...]
    Available ckpts discovered by globbing checkpoint_dir/*.ckpt on volume.
    """
```

### 8.3 `log_flow_completion()` Must Write Checkpoint Metadata

```python
def log_flow_completion(
    self,
    *,
    flow_name: str,
    run_id: str,
    checkpoint_paths: list[Path] | None = None,  # NEW
    checkpoint_dir: Path | None = None,           # NEW
    artifacts: dict[str, str] | None = None,      # artifact_name → path
) -> None:
    """Log flow completion with optional checkpoint and artifact metadata."""
    client = mlflow.tracking.MlflowClient()
    client.set_tag(run_id, "flow_status", "FLOW_COMPLETE")
    client.set_tag(run_id, "flow_name", flow_name)
    if checkpoint_dir:
        client.set_tag(run_id, "checkpoint_dir", str(checkpoint_dir))
    if checkpoint_paths:
        for i, p in enumerate(checkpoint_paths):
            client.set_tag(run_id, f"checkpoint_path_{i}", str(p))
    if artifacts:
        for name, path in artifacts.items():
            client.set_tag(run_id, f"artifact_{name}", path)
```

### 8.4 Debug Suffix — Single Source of Truth

```python
# flow_contract.py class constructor:
class FlowContract:
    def __init__(self, tracking_uri: str | None = None):
        self.tracking_uri = tracking_uri or resolve_tracking_uri()
        self._debug_suffix = os.environ.get("MINIVESS_DEBUG_SUFFIX", "")

    def _resolve_experiment(self, base_name: str) -> str:
        return f"{base_name}{self._debug_suffix}"
```

All internal methods call `self._resolve_experiment(base_name)` — no external suffix passing.

---

## 9. Experiment Name Map

| Flow | Writes To | Reads From |
|------|-----------|------------|
| Data Acquisition | nothing | nothing |
| Data Engineering | `minivess_data{sfx}` | nothing |
| Train (Modelling) | `minivess_training{sfx}` | `minivess_data{sfx}` |
| Post-training | `minivess_training{sfx}` (sibling runs, `__` suffix) | `minivess_training{sfx}` |
| Analysis | `minivess_analysis{sfx}` + Model Registry | `minivess_training{sfx}` (ALL runs incl. post-train) |
| Biostatistics | `minivess_biostatistics{sfx}` | `minivess_analysis{sfx}` |
| Deployment | nothing | Model Registry + `minivess_analysis{sfx}` |
| Dashboard | nothing | 23 independent sources (see Section 4) |
| Data Annotation | nothing | BentoML endpoint |
| HPO | `minivess_hpo{sfx}` | `minivess_training{sfx}` (warm-start) |
| Survival Analysis | `minivess_survival{sfx}` | `minivess_training{sfx}` (checkpoints) |
| Classification | `minivess_classification{sfx}` | `minivess_training{sfx}` (seg masks) |

`{sfx}` = `_DEBUG` when `MINIVESS_DEBUG_SUFFIX=_DEBUG`, `""` in production.

---

## 10. Improvements Beyond foundation-PLR

The `foundation-PLR` reference implementation provides valuable patterns but lacks several
things that MinIVess needs. These are the differences/improvements:

### 10.1 Explicit FlowContract Class (MinIVess improvement)

`foundation-PLR` uses ad-hoc MLflow queries in each flow file (no FlowContract class).
MinIVess has `FlowContract` which centralizes all inter-flow discovery.
**Improvement**: Move ALL `find_upstream_safely()` calls into `FlowContract.define_sources_for_flow()`.
Result: a single file (`flow_contract.py`) is the complete inter-flow API spec.

### 10.2 Debug Suffix Mechanism (MinIVess-specific)

`foundation-PLR` has no concept of debug experiments — it has a single namespace.
**MinIVess addition**: `MINIVESS_DEBUG_SUFFIX=_DEBUG` env var propagated through `FlowContract`
ensures debug and production runs are always namespace-isolated.

### 10.3 Checkpoint Path Tags on Fold Runs (MinIVess-specific)

`foundation-PLR` stores feature vectors in MLflow artifacts (tabular data — small).
Biomedical segmentation checkpoints are 0.5–2 GB each — cannot live in MLflow artifact store.
**MinIVess approach**: Tag the volume path. `checkpoint_dir` tag on each fold run gives
downstream flows a pointer to the actual file without copying it.

### 10.4 MONAI Sliding Window Inference (MinIVess-specific)

`foundation-PLR` uses standard inference. MinIVess uses `sliding_window_inference()` with
model-specific `roi_size` (from `ModelAdapter.get_eval_roi_size()`).
`define_sources_for_flow()` for Analysis must also resolve the `roi_size` per checkpoint.

### 10.5 Multi-Source Dashboard (MinIVess improvement)

`foundation-PLR` has no dashboard concept. MinIVess Dashboard reads 23 sources via adapters.
**Pattern**: Each source has a `BaseSourceAdapter` with `connect()`, `query()`, `cache()`,
`poll_interval`. Adding a new source = adding one adapter class.

---

## 11. Key Design Decisions

### Q1: Tags vs MLflow artifacts for checkpoint paths?

**Option A — Tag the volume path** (Phase 0, recommended):
```python
mlflow.set_tag("checkpoint_dir_fold_0", "/app/checkpoints/fold_0")
```
- Pros: Zero extra storage, checkpoints stay on shared volume
- Cons: Volume path must be consistent across all containers (it is — CDI bind mount)

**Option B — Log as MLflow artifact** (Phase 3):
```python
mlflow.log_artifact("/app/checkpoints/fold_0/best.ckpt")
```
- Pros: Checkpoint lives in MinIO, queryable via `mlflow.artifacts.download_artifacts`
- Cons: Doubles storage; SAM3 ≈ 2 GB per checkpoint

**Decision**: Phase 0 uses Option A (tags). Option B after Model Registry integration.

### Q2: Sibling runs vs child runs for post-training?

**Option A — Sibling runs** (recommended):
- Same experiment, new parent run, `__` suffix naming
- Cleaner MLflow UI, independently queryable
- Analysis can search the whole experiment for `__swa`, `__calibrated` variants

**Option B — Child runs (nested)**:
- MLflow nested run support is limited; querying is more complex
- Harder for Analysis to aggregate

**Decision**: Sibling runs with `__` suffix naming and `upstream_training_run_id` tag.

### Q3: DVC Dataset registration in MLflow?

**Recommendation**: Yes, via `mlflow.log_input(mlflow.data.from_dvc(...))`.
Enables "which model trained on which dataset version?" lineage.
Phase 1 — not blocking flow connectivity.

### Q4: What if upstream flow's run is not FINISHED?

`FlowContract.find_upstream_run` filters `status = 'FINISHED'`. FAILED/KILLED runs are
skipped. Downstream flow waits for a new successful run. No partial result recovery.

### Q5: Should Analysis aggregate HPO + Survival + Classification experiments?

**Yes** — Analysis is the single aggregation point. `define_sources_for_flow("analysis")`
queries ALL configured upstream experiments. Which experiments to aggregate is controlled
by the experiment config YAML (config-driven, not hardcoded).

---

## 12. Implementation Roadmap

### Phase 0 — Fix the Broken Seam (Train → Post-training → Analysis)
**Goal**: Post-training plugins actually execute. Analysis evaluates real models.

- [ ] `train_flow.py`: Log `checkpoint_dir` tag on each fold run
- [ ] `train_flow.py`: Log `flow_status=FLOW_COMPLETE` tag on parent run
- [ ] `train_flow.py`: Add `parent_run_id` tag on each fold run
- [ ] `flow_contract.py`: Add `find_fold_checkpoints()` method
- [ ] `flow_contract.py`: Add `_debug_suffix` from env var (constructor)
- [ ] `post_training_flow.py`: Call `find_fold_checkpoints()` → real paths → plugins
- [ ] `analysis_flow.py`: Query ALL runs in experiment (base + `__` variants)
- [ ] `constants.py`: Add `resolve_experiment_name()` utility
- [ ] `run_debug.sh`: Export `MINIVESS_DEBUG_SUFFIX=_DEBUG`

**Test criteria**: Debug run shows post-training plugins execute (not skipped),
analysis evaluates ≥1 real model, champion is tagged in MLflow.

### Phase 1 — Data Engineering → Train Handoff
- [ ] `data_flow.py`: Write `splits_path` tag on FLOW_COMPLETE
- [ ] `train_flow.py`: Read `splits_path` from upstream data run via FlowContract
- [ ] (Optional) DVC Dataset registration via `mlflow.log_input()`

### Phase 2 — Analysis → Biostatistics → Dashboard
- [ ] `biostatistics_flow.py`: Implement `define_sources_for_flow("biostatistics")`
- [ ] `biostatistics_flow.py`: Write `paper_comparison_{date}` run with artifacts
- [ ] `dashboard_flow.py` (issue #331): Implement 23-source adapter pattern
- [ ] `dashboard_flow.py`: Two-channel update (Prefect events + per-source polling)

### Phase 3 — Deployment Flow Champion Discovery
- [ ] `analysis_flow.py`: Register champion to Model Registry after evaluation
- [ ] `deploy_flow.py`: Read from Model Registry via `define_sources_for_flow("deploy")`
- [ ] (Optional) Upgrade checkpoint storage to MLflow artifact + MinIO

### Phase 4 — Extensible Flow Topology
- [ ] `pipeline.py` (if not exists): Config-driven flow dispatch (foundation-PLR pattern)
- [ ] `configs/base.yaml`: Add `prefect_flows:` section with boolean flags
- [ ] Validate that adding HPO flow requires only: new YAML key + new flow file + one `if` line

---

## 13. Files That Need Changes (Phase 0)

| File | Change |
|------|--------|
| `src/minivess/orchestration/flows/train_flow.py` | Log `checkpoint_dir`, `parent_run_id`, `flow_status=FLOW_COMPLETE` tags |
| `src/minivess/orchestration/flow_contract.py` | Add `find_fold_checkpoints()`, `_debug_suffix` from env, `define_sources_for_flow()` skeleton |
| `src/minivess/orchestration/flows/post_training_flow.py` | Call `find_fold_checkpoints()` → resolve paths → pass to plugins |
| `src/minivess/orchestration/flows/analysis_flow.py` | Use `resolve_experiment_name()`, query ALL runs incl. post-training variants |
| `src/minivess/orchestration/constants.py` | Add `resolve_experiment_name()` utility |
| `scripts/run_debug.sh` | Export `MINIVESS_DEBUG_SUFFIX=_DEBUG` |
| `tests/unit/orchestration/test_flow_connectivity.py` | NEW — unit tests for FlowContract seams |
| `tests/integration/orchestration/test_flow_contract_integration.py` | NEW — end-to-end contract tests with real MLflow |

---

## Appendix A: Original User Prompt (Verbatim — Session 1)

> So let's create fix/prefect-flow-connectivity and I am disappointed then on the bad behavior that we have for our codebase as we have been trying to get the mlflow-prefect working for quite a time but still the basic functionalities are not working. Let's plan to /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/prefect-flow-connectivity.md on how each Flow should obviously be able to query the previous Flow's results . Modelling Flow should be able the query the available DVC versioned datasets from Data Engineering Flow. Post-training should be able to query the the Modelling results from the MLflow, as should Analysis of Mlflow (note that Post-training is using now the same MLflow experiment (same experiment_name) as the Modelling Flow but only adding the suffix to the run_names). The analysis then uses it new experiment_name and creates all the possible permutation names. And the biostatistics module is able to query the previous Flow again (the Analysis Flow) and the Biostatistics Flow then creates a new experiment (name) with one mlrun with some name and a date that essentially creates a comparison of everything for a scientific paper or for a PI/CTO/CEO as in what do we have at the moment (and can also function as a data source for the Dashboard Flow). Then the deployement Flow can read the Analysis MLflow experiment with these functioning as the MFlow contract layer so that they depoyment Flow can read MLflow Model Registry and deploy the "chosen champion" to production via BentoML then. So Modelling Flow starts using MLflow in this scenario with the Deployment Flow not producing any outputs to MLflow but reading the input. It is open for debate and analysis whether Data Engineering Flow needs to output anything to MLflow yet as should the DVC versioned dataset be registered as MLflow dataset? Data Acquisition Flow does not work in anyway at the moment with MLflow. Data annotation needs to mainly access the served BentoML deployment and definitely not write anything to MLflow. Does this as a vision make any sense? The debug then is also a special case in the sense that every experiment name should get a "_DEBUG" suffix (not the mlrun names) so that real experiments with all the epochs and all the data do not get mixed with the debug runs that are as the name implies meant for debugging the code and the "pipeline mechanics". Save my prompt verbatim as an appendix to the .md plan and start planning!

## Appendix B: User Correction on Dashboard (Session 2)

> Are you sure you got this right '...Biostatistics → new experiment...doubles as Dashboard source...' as it does not 'double' as source, but we have multiple interactive d3.js dashboards there in one React frontend (https://github.com/minivess-mlops/minivess-mlops/issues/331) that connects to various datasources automatically, not from hardcoded trigger definitions there. So instead of having them hardcoded there is like a scheduler as well that is getting updates from Grafana, Evidently, Posthog kind of updating the dashboard automatically as well and not just from Prefect flows as I imagine that the dashboard also gets updates from the live system (that is running for serving). The Biostatistics flow is not directly linked to the Dashboard as well, but is more of a standalone thing for scientific reports and the Biostatistics experiment in MLflow is just one of the many data sources for the Dashboard there as well. You had a very good start previously but missed this important information. Also please as a part of this plan add to the CLAUDE.md that TOP-1 principle that we have for this project as well, namely that we are building a highly flexible MLOps architecture for multiphoton experiments as extension to the MONAI ecosystem and we don't want to constrain the expansion. Also please add to the plan that we need to support both the sequential and parallel topologies for flows as we have two types of dependencies: for example Classification on a binary mask that is created with segmentation → this is sequential. But if we have Survival Analysis that you just add independently as a new flow then there is a parallel topology etc. please also explore the foundation-PLR rigorously and don't take now shortcuts or just read 'key segments' of docs. This is a massive and highly important task to get this right now! Improve the plan there to achieve this!
