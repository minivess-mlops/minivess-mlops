---
title: "Intermediate Plan Synthesis v1"
status: reference
created: "2026-03-07"
---

# MinIVess MLOps v2 — Intermediate Plan Synthesis

**Date:** 2026-03-07
**Branch:** feat/flow-biostatistics
**Synthesizes:** All infrastructure docs, all flow files, all metalearning docs, deployment YAML, SkyPilot configs
**Issues addressed:** #367 (Prefect-only execution) and #369 (Docker volume audit)
**Author:** Synthesis via deep read of 40+ source files

---

## Executive Summary

This document synthesizes the current state of the MinIVess MLOps v2 platform across all 9
flows/apps, with particular focus on the two P0 blockers:

- **Issue #367**: Standalone Python scripts (`python scripts/*.py`) are still the de-facto
  training entry point. This is architecturally forbidden. Every training run must go through
  `prefect deployment run`, not a bare Python invocation.

- **Issue #369**: Docker volume mounts are incomplete across all flows. The `analyze`, `deploy`,
  `dashboard`, and `qa` flows declare **zero** volume mounts. The `post_training` flow declares
  only the checkpoint read volume. The result: any artifact produced inside these containers
  is lost when the container exits.

Both issues compound each other: because `train_flow.py` is a stub that delegates to
`train_monitored.py`, checkpoints are written to `tempfile.mkdtemp()` (line 673 of
`scripts/train_monitored.py`), a path that is ephemeral inside Docker. No container exit
survives a checkpoint.

---

## Part 1: Current State Assessment

### 1.1 Flow Inventory and Audit

The platform defines exactly 9 flow entry points (7 numbered flows plus annotation and dashboard):

| # | Flow Name | File | @flow Exists? | Implementation Level |
|---|-----------|------|---------------|---------------------|
| 0 | Acquisition | `acquisition_flow.py` | YES — `run_acquisition_flow` | Substantial: download, convert, provenance |
| 1 | Data Engineering | `data_flow.py` | YES — `run_data_flow` | Substantial: discover, validate, split, external datasets |
| 2 | Training | `train_flow.py` | YES — `training_flow` | **STUB**: delegates to `train_monitored.py` via argparse.Namespace |
| 2.5 | Post-Training | `post_training_flow.py` | YES — `post_training_flow` | Substantial: SWA, merging, calibration, conformal plugins |
| 3 | Analysis | `analysis_flow.py` | YES — `analysis_flow` (large file, 58KB) | Substantial: ensemble, eval, comparison, bootstrap |
| 4 | Deployment | `deploy_flow.py` | YES — `deploy_flow` | Substantial: champion discovery, ONNX, BentoML, promotion |
| 5 | Dashboard | `dashboard_flow.py` | YES — `run_dashboard_flow` | Substantial: 4 sections, markdown+JSON output |
| 6 | QA | `qa_flow.py` | YES — `qa_flow` | Substantial: backend check, ghost runs, param validation |
| - | Annotation | `annotation_flow.py` | YES — `run_annotation_flow` | Substantial: inference, session recording, agreement |

### 1.2 Detailed Audit Per Flow

#### Flow 0: Data Acquisition (`acquisition_flow.py`)

| Dimension | Current State |
|-----------|--------------|
| @flow decorated | YES — `run_acquisition_flow` |
| Inputs | `AcquisitionConfig` (datasets list, output_dir, skip_existing, convert_formats) |
| Outputs | `AcquisitionResult` (per-dataset status, total_volumes, conversion_log, provenance dict) |
| Temp paths | None visible in flow code; `output_dir` comes from config |
| Volume mounts declared | **NONE** — no `acquisition` service in `docker-compose.flows.yml` |
| Volume gaps | Needs: `/data/raw` (output), `/logs` (provenance log) |
| Prefect deployment | Listed in `FLOW_WORK_POOL_MAP` as `cpu-pool`; image `minivess-acquisition:latest` |
| Resume capability | None — downloads are idempotent via `skip_existing=True`; format conversion is not resumable |
| MLflow logging | provenance dict with `acq_` prefix — but NOT yet logged to an MLflow run |

**Critical gap**: The `acquisition` service does not appear in `docker-compose.flows.yml`. The
`Dockerfile.acquisition` exists, and the flow is referenced in `deployments.py`, but there is
no compose service definition for it. Acquisition runs have no volume mount for the output data
directory that the downstream data flow needs to read.

#### Flow 1: Data Engineering (`data_flow.py`)

| Dimension | Current State |
|-----------|--------------|
| @flow decorated | YES — `run_data_flow` |
| Inputs | `data_dir: Path`, `n_folds: int = 3`, `seed: int = 42`, `external_dirs: dict` |
| Outputs | `DataFlowResult` (pairs, validation, splits, provenance) |
| Temp paths | None — all paths from explicit `data_dir` parameter |
| Volume mounts declared | `data_cache:/app/data` (read-write) — ONLY this |
| Volume gaps | No mount for `/app/configs/splits` (output split files), no mount for `/app/mlruns` |
| Prefect deployment | `cpu-pool`, image `minivess-data:latest` |
| Resume capability | `skip_existing` semantics for pair discovery; splits are deterministic from seed |
| MLflow logging | `data_` prefix provenance (n_volumes, n_folds, dataset_hash) — but no MLflow run opened |

**Critical gap**: Split files written by `split_data_task` go to… where? The `DataFlowResult`
returns `splits` as Python objects, but there is no serialization to disk and no volume mount
for a splits output directory. Downstream training flow has no path to read split definitions.

DVC integration is entirely absent from the current `data_flow.py`. The flow discovers pairs
via filesystem traversal, not via `dvc pull`. This means the "DVC versioning as inter-flow
contract" is not yet implemented.

#### Flow 2: Training (`train_flow.py`)

| Dimension | Current State |
|-----------|--------------|
| @flow decorated | YES — `training_flow` |
| Inputs | `loss_name`, `model_family`, `compute`, `debug`, `experiment_name`, `trigger_source` |
| Outputs | Returns dict with `status: "configured"` — NOT actual training results |
| Temp paths | `tempfile.mkdtemp()` in `scripts/train_monitored.py:673` — BANNED |
| Volume mounts declared | `data_cache:/app/data:ro`, `checkpoint_cache:/app/checkpoints` |
| Volume gaps | No `/app/mlruns`, no `/app/logs`, no `/app/configs/splits` |
| Prefect deployment | `gpu-pool`, image `minivess-train:latest`, concurrency 1 |
| Resume capability | `--resume` flag in `train_monitored.py` reads from `checkpoint_dir` — but `checkpoint_dir = tempfile.mkdtemp()` so there is nothing to resume from |
| MLflow logging | Full epoch-level logging via `tracker.log_epoch_metrics(epoch_log, step=epoch+1)` — THIS IS IMPLEMENTED |

**Critical gap**: The `training_flow()` function body explicitly states:
```
"Training flow configured. Use scripts/train_monitored.py for execution."
```
This is the stub antipattern documented in metalearning `2026-03-06-standalone-script-antipattern.md`.
The flow does NOT execute training. It creates an `argparse.Namespace` object and logs a message.

The SkyPilot config `train_generic.yaml` also invokes `uv run python scripts/train_monitored.py`,
which is a second violation of the standalone script ban.

**Positive finding**: Epoch-level loss logging IS implemented in `trainer.py:562`:
```python
self.tracker.log_epoch_metrics(epoch_log, step=epoch + 1)
```
And `tracking.py:222` confirms:
```python
mlflow.log_metrics(prefixed, step=step)
```
So epoch-level train/val loss curves with `step=epoch` are already working in the underlying
trainer. The gap is that this code is only reachable via `train_monitored.py`, not via
`training_flow()`.

#### Flow 2.5: Post-Training (`post_training_flow.py`)

| Dimension | Current State |
|-----------|--------------|
| @flow decorated | YES — `post_training_flow` |
| Inputs | `config`, `checkpoint_paths: list[Path]`, `run_metadata`, `output_dir`, `calibration_data` |
| Outputs | Dict with plugin results, n_plugins_run |
| Temp paths | `output_dir = Path("outputs/post_training")` — relative path, resolved against CWD inside container |
| Volume mounts declared | `checkpoint_cache:/app/checkpoints:ro` — ONLY this |
| Volume gaps | No mount for `/app/outputs` (plugin output), no `/app/mlruns`, no calibration data path |
| Prefect deployment | Listed in `FLOW_IMAGE_MAP` as `minivess-post-training:latest` but NOT in `FLOW_WORK_POOL_MAP` |
| Resume capability | Best-effort plugins; no checkpoint-within-plugin recovery |
| MLflow logging | Plugin metrics returned in dict but NOT logged to MLflow runs |

**Critical gap**: `post_training` is missing from `FLOW_WORK_POOL_MAP` in `deployments.py`.
This means `get_flow_deployment_config("post_training")` returns `cpu-pool` (default fallback)
but with the wrong image. The deployment configuration is inconsistent.

The `output_dir` defaults to a relative path `"outputs/post_training"` — inside Docker this
resolves to `/app/outputs/post_training`, which has no volume mount and will be lost on container
exit.

#### Flow 3: Analysis (`analysis_flow.py`)

| Dimension | Current State |
|-----------|--------------|
| @flow decorated | YES (large file, 58KB) |
| Inputs | MLflow `experiment_name`, `data_dir`, `test_loader_spec`, evaluation config |
| Outputs | ComparisonTable, champion tags, summary report |
| Temp paths | Uses `import re` at line 1033 (`_FOLD_RE`) — BANNED, must fix |
| Volume mounts declared | **ZERO** — the `analyze` service has no volumes section |
| Volume gaps | Needs: `/app/data:ro`, `/app/mlruns` (read+write), `/app/outputs/analysis` |
| Prefect deployment | `cpu-pool`, image `minivess-analyze:latest` |
| Resume capability | MLflow-based: can re-run from existing runs; no within-flow checkpointing |
| MLflow logging | Comprehensive — ComparisonTable, champion tags, bootstrap CI |

**Critical gap**: The analysis flow produces figures (PNG/SVG), comparison tables (MD/TEX),
and Parquet exports. These currently go to `outputs/analysis/` inside the container with no
volume mount — they are lost when the container exits. The outputs currently in the repository
under `outputs/analysis/` were generated by running the flow outside Docker.

The regex violation at line 1033 (`_FOLD_RE = re.compile(r"^(.+)_fold(\d+)$")`) must be fixed
per the regex ban rule. Correct replacement: `name.rsplit("_fold", 1)`.

#### Flow 4: Deployment (`deploy_flow.py`)

| Dimension | Current State |
|-----------|--------------|
| @flow decorated | YES — `deploy_flow` |
| Inputs | `DeployConfig`, `experiment_id: str = "1"` |
| Outputs | `DeployResult` (champions, onnx_paths, bento_tags, artifacts_dir, promotion_results) |
| Temp paths | `config.output_dir / "onnx"` — depends on DeployConfig, not tempfile |
| Volume mounts declared | **ZERO** — the `deploy` service has no volumes section |
| Volume gaps | Needs: `/app/mlruns:ro`, `/app/outputs/deploy` (ONNX + artifacts), BentoML store mount |
| Prefect deployment | `cpu-pool`, image `minivess-deploy:latest` |
| Resume capability | Champion discovery is idempotent; ONNX export is idempotent |
| MLflow logging | Reads mlruns for champion tags; does not write back flow completion marker |

**Critical gap**: ONNX files are explicitly gitignored (CLAUDE.md: "ONNX models belong in MLflow
model registry / BentoML store"). They must be stored on a mounted volume or uploaded to MinIO.
The current flow writes them to `config.output_dir / "onnx"` inside the container with no
persistence. On container exit, all ONNX exports are lost.

BentoML model store path also has no volume mount. After `import_champion_to_bento()`, the model
is in the container-local BentoML store which vanishes on container exit.

#### Flow 5: Dashboard (`dashboard_flow.py`)

| Dimension | Current State |
|-----------|--------------|
| @flow decorated | YES — `run_dashboard_flow` |
| Inputs | `output_dir: Path`, plus data/config/model/pipeline section params |
| Outputs | Markdown report + JSON metadata written to `output_dir` |
| Temp paths | None — `output_dir` is an explicit parameter |
| Volume mounts declared | **ZERO** — the `dashboard` service has no volumes section |
| Volume gaps | Needs: `/app/outputs/dashboard` (report output), `/app/mlruns:ro` |
| Prefect deployment | `cpu-pool`, image `minivess-dashboard:latest` |
| Resume capability | Stateless — always regenerates from MLflow data |
| MLflow logging | Minimal — reads from other flows, does not log to MLflow |

**Critical gap**: `run_dashboard_flow(output_dir=Path("/app/outputs/dashboard"))` with no volume
mount means the markdown report and JSON metadata are discarded on container exit. The "paper
figures" that appeared in `outputs/` were generated outside Docker.

#### Flow 6: QA (`qa_flow.py`)

| Dimension | Current State |
|-----------|--------------|
| @flow decorated | YES — `qa_flow` |
| Inputs | `tracking_uri: str = "mlruns"`, `experiment_ids: list[str]` |
| Outputs | Dict with checks, summary, report (markdown string) |
| Temp paths | None |
| Volume mounts declared | **ZERO** — the `qa` service has no volumes section |
| Volume gaps | Needs: `/app/mlruns:ro` (to read MLflow runs) |
| Prefect deployment | `cpu-pool`, image `minivess-qa:latest` |
| Resume capability | Stateless — always re-checks current state |
| MLflow logging | Reads mlruns; does not write QA results back to MLflow |

**Critical gap**: `qa_flow(tracking_uri="mlruns")` with a relative path resolves to
`/app/mlruns` inside the container. With no volume mount for `/app/mlruns`, the QA flow is
checking an empty directory and reporting "no ghost runs found" — vacuously true.

#### Annotation App (`annotation_flow.py`)

| Dimension | Current State |
|-----------|--------------|
| @flow decorated | YES — `run_annotation_flow` |
| Inputs | `volume: NDArray`, `volume_id: str`, `config: AnnotationFlowConfig`, `reference` |
| Outputs | `AnnotationFlowResult` (response dict, session report, agreement DSC) |
| Temp paths | None |
| Volume mounts declared | No compose service defined for annotation |
| Volume gaps | Needs: BentoML store mount (to load models), `/app/data:ro` (input volumes) |
| Prefect deployment | NOT in `FLOW_WORK_POOL_MAP` or `FLOW_IMAGE_MAP` |
| Resume capability | Stateless |
| MLflow logging | None — annotation sessions not logged to MLflow |

**Critical gap**: The annotation flow depends on a running inference server (either
`RemoteInferenceClient` pointing to the BentoML service, or `LocalInferenceClient` with model
paths). There is no Docker compose service for this flow, no image defined, and no volume
mounts. The flow exists as code but has no Docker infrastructure.

#### Dashboard (same as Flow 5 — covered above)

### 1.3 What Works Today (Verified E2E Pipeline)

The CLAUDE.md (MEMORY.md) documents a verified quasi-E2E pipeline producing 35+ artifacts
from real MiniVess experiments (70 volumes, 4 losses, 3 folds, 100 epochs). What works:

1. **ExperimentTracker** — full epoch-level metric logging with `step=epoch` parameter
2. **ComparisonTable** — cross-loss comparison with paired bootstrap testing
3. **Champion tagging** — MLflow tags for best models per category
4. **DeployResult** — ONNX export, BentoML import, promotion (in non-Docker context)
5. **DuckDB analytics** — post-hoc SQL over mlruns
6. **Conformal UQ** — morphological, distance transform, risk-controlling predictors
7. **Graph topology metrics** — ccDice, Betti number, skeleton recall, CAPE loss

What does NOT work in a Docker context:
1. Training flow — is a stub; real training still goes through `train_monitored.py`
2. All Docker volume mounts for most flows — outputs lost on container exit
3. MLflow as inter-flow contract — only works when flows share a filesystem, not in Docker
4. `post_training` deployment config — missing from `FLOW_WORK_POOL_MAP`
5. Annotation and dashboard compose services — not defined

### 1.4 What Is Definitively Broken

1. **`scripts/train_monitored.py:673`** — `tempfile.mkdtemp()` creates ephemeral checkpoint dir
2. **`training_flow()` body** — returns `{"status": "configured"}` without executing training
3. **SkyPilot `train_generic.yaml`** — calls `uv run python scripts/train_monitored.py` (banned)
4. **SkyPilot `train_hpo_sweep.yaml`** — calls `uv run python scripts/run_hpo.py` (banned)
5. **`analysis_flow.py:1033`** — `import re` with `_FOLD_RE = re.compile(...)` (banned)
6. **`duckdb_extraction.py:364`** — `re.match(r"eval_fold(\d+)_(.+)", metric_name)` (banned)
7. **`docker-compose.flows.yml`** — `analyze`, `deploy`, `dashboard`, `qa` have zero volume mounts
8. **`docker-compose.flows.yml`** — `acquisition` service is missing entirely
9. **`deployments.py`** — `post_training` missing from `FLOW_WORK_POOL_MAP`
10. **`FlowContract`** — `find_upstream_run` uses string-formatted filter expressions, not typed API

---

## Part 2: Docker Isolation Architecture

### 2.1 Volume Contract Specification Per Flow

The core principle (from CLAUDE.md and metalearning `2026-03-06-standalone-script-antipattern.md`):
flows communicate ONLY through MLflow artifacts and Prefect artifacts. No shared filesystem.
All volumes must be explicitly declared.

#### Required Named Volumes (docker-compose level)

```yaml
volumes:
  # Data volumes
  raw_data:           # Raw NIfTI files from acquisition
    driver: local
  data_cache:         # Processed/validated data for training
    driver: local

  # Artifact volumes
  checkpoint_cache:   # Training checkpoints (.pth files) — GPU-written
    driver: local
  mlruns_data:        # MLflow run data (metrics, params, artifacts)
    driver: local
  outputs_analysis:   # Analysis figures, tables, reports
    driver: local
  outputs_deploy:     # ONNX files, bentofiles, deployment artifacts
    driver: local
  outputs_dashboard:  # Dashboard reports, JSON metadata
    driver: local
  bentoml_store:      # BentoML model store (ONNX models imported by deploy flow)
    driver: local
  logs_data:          # Training logs, monitor CSV, JSONL
    driver: local
  configs_splits:     # k-fold split JSON files (written by data flow, read by train)
    driver: local
  post_training_out:  # SWA, merging, calibration artifacts
    driver: local
```

#### Per-Service Volume Mount Specification

| Service | Mount | Mode | Purpose |
|---------|-------|------|---------|
| `acquisition` | `raw_data:/app/data/raw` | rw | Write downloaded NIfTI files |
| `acquisition` | `logs_data:/app/logs` | rw | Acquisition provenance log |
| `data` | `raw_data:/app/data/raw` | ro | Read raw files for discovery |
| `data` | `data_cache:/app/data/processed` | rw | Write validated/profiled data |
| `data` | `configs_splits:/app/configs/splits` | rw | Write k-fold split JSON |
| `data` | `mlruns_data:/app/mlruns` | rw | Log data provenance run |
| `train` | `data_cache:/app/data:ro` | ro | Read training data |
| `train` | `configs_splits:/app/configs/splits:ro` | ro | Read fold splits |
| `train` | `checkpoint_cache:/app/checkpoints` | rw | Write best-metric checkpoints |
| `train` | `mlruns_data:/app/mlruns` | rw | Write experiment metrics |
| `train` | `logs_data:/app/logs` | rw | Write monitor CSV, training.log |
| `post_training` | `checkpoint_cache:/app/checkpoints:ro` | ro | Read training checkpoints |
| `post_training` | `post_training_out:/app/outputs/post_training` | rw | Write SWA/merge artifacts |
| `post_training` | `mlruns_data:/app/mlruns` | rw | Log post-training metrics |
| `post_training` | `data_cache:/app/data:ro` | ro | Read calibration data |
| `analyze` | `checkpoint_cache:/app/checkpoints:ro` | ro | Read checkpoints for eval |
| `analyze` | `data_cache:/app/data:ro` | ro | Read test data |
| `analyze` | `mlruns_data:/app/mlruns` | rw | Read training runs + write analysis |
| `analyze` | `outputs_analysis:/app/outputs/analysis` | rw | Write figures, tables |
| `analyze` | `configs_splits:/app/configs/splits:ro` | ro | Read fold assignments |
| `deploy` | `checkpoint_cache:/app/checkpoints:ro` | ro | Read champion checkpoints |
| `deploy` | `mlruns_data:/app/mlruns:ro` | ro | Read champion tags |
| `deploy` | `outputs_deploy:/app/outputs/deploy` | rw | Write ONNX, bentofiles |
| `deploy` | `bentoml_store:/home/minivess/bentoml` | rw | BentoML model store |
| `dashboard` | `mlruns_data:/app/mlruns:ro` | ro | Read experiment results |
| `dashboard` | `outputs_analysis:/app/outputs/analysis:ro` | ro | Read figures for embedding |
| `dashboard` | `outputs_dashboard:/app/outputs/dashboard` | rw | Write reports, JSON |
| `qa` | `mlruns_data:/app/mlruns:ro` | ro | Read runs for QA checks |
| `qa` | `outputs_dashboard:/app/outputs/dashboard` | rw | Write QA report |

### 2.2 Environment Variables Per Flow

In addition to the common env block (`MLFLOW_TRACKING_URI`, `PREFECT_API_URL`, MinIO creds),
each flow requires specific env vars:

| Service | Additional Env Vars | Purpose |
|---------|--------------------|---------|
| `acquisition` | `ACQUISITION_DATASETS`, `ACQUISITION_OUTPUT_DIR=/app/data/raw` | Control which datasets to download |
| `data` | `DATA_DIR=/app/data/raw`, `SPLITS_OUTPUT_DIR=/app/configs/splits`, `N_FOLDS=3`, `SEED=42` | Data flow params |
| `train` | `CHECKPOINT_DIR=/app/checkpoints`, `LOG_DIR=/app/logs`, `SPLITS_DIR=/app/configs/splits`, `NVIDIA_VISIBLE_DEVICES=all` | Training I/O paths |
| `post_training` | `CHECKPOINT_DIR=/app/checkpoints`, `OUTPUT_DIR=/app/outputs/post_training` | Post-training I/O |
| `analyze` | `CHECKPOINT_DIR=/app/checkpoints`, `ANALYSIS_OUTPUT=/app/outputs/analysis`, `SPLITS_DIR=/app/configs/splits` | Analysis I/O |
| `deploy` | `DEPLOY_OUTPUT_DIR=/app/outputs/deploy`, `BENTOML_HOME=/home/minivess/bentoml` | Deployment paths |
| `dashboard` | `DASHBOARD_OUTPUT=/app/outputs/dashboard`, `ANALYSIS_DIR=/app/outputs/analysis:ro` | Dashboard output |
| `qa` | `MLFLOW_TRACKING_URI=http://minivess-mlflow:5000` (already in common) | QA uses server URI |

### 2.3 GPU Reservation

Only the `train` service requires GPU access. The current configuration is correct:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

The `gpu-pool` work pool in `work-pools.yaml` also correctly specifies `NVIDIA_VISIBLE_DEVICES: all`.
No other flow should have GPU access.

Post-training (SWA, model merging) can run on CPU for inference/weight averaging. If calibration
requires GPU inference, this should be explicitly declared rather than silently inheriting.

### 2.4 Network Policy

All flow services connect to the shared `minivess-network` (external network defined in
`docker-compose.yml`). No direct function calls or shared filesystem between flows. The only
valid communication channels are:

1. **MLflow artifacts** — written by one flow, read by the next via MLflow client
2. **Prefect artifacts** — run IDs, flow names, status reported via Prefect API
3. **Named Docker volumes** — data volumes (raw_data, data_cache, checkpoint_cache) are shared
   but read-only for downstream flows; only the producing flow mounts them read-write

The `FlowContract` class in `flow_contract.py` provides the Python API for this:
- `find_upstream_run()` — discovers the most recent FINISHED run from an upstream flow
- `log_flow_completion()` — tags a run with `flow_name`, `flow_status`, `flow_artifacts`

This must be wired into every flow's entry and exit logic.

---

## Part 3: MLflow as Inter-Flow Contract

### 3.1 How Each Flow Uses MLflow

The standardized param/tag prefixes (from CLAUDE.md):

| Prefix | Category | Written by |
|--------|----------|------------|
| (none) | Training hyperparams | Train flow |
| `arch_` | Model architecture | Train flow |
| `sys_` | System/environment | Train flow (at run start) |
| `data_` | Dataset metadata | Data flow |
| `loss_` | Loss function config | Train flow |
| `eval_` | Evaluation config | Analysis flow |
| `upstream_` | Cross-flow links | Analysis, Deploy flows |
| `acq_` | Acquisition metadata | Acquisition flow |
| `post_` | Post-training metrics | Post-training flow |
| `deploy_` | Deployment metadata | Deploy flow |
| `qa_` | QA check results | QA flow |

### 3.2 The Strict Data Flow

```
Flow 0 (Acquisition)
  └─ writes: raw NIfTI to raw_data volume
  └─ writes: acq_ params to MLflow "minivess_acquisition" experiment

Flow 1 (Data Engineering)
  └─ reads: raw NIfTI from raw_data volume
  └─ writes: data_ params to MLflow "minivess_data" experiment
  └─ writes: split JSON to configs_splits volume
  └─ writes: dataset_hash tag for traceability

Flow 2 (Training)
  └─ reads: upstream data run ID from Flow 1 MLflow
  └─ reads: split JSON from configs_splits volume
  └─ writes: epoch-level metrics (train_loss, val_loss, val_dice, ...) step=epoch
  └─ writes: best-metric checkpoints to checkpoint_cache volume
  └─ writes: training params + arch_ + sys_ to MLflow "minivess_training" experiment
  └─ writes: upstream_data_run_id tag for traceability

Flow 2.5 (Post-Training)
  └─ reads: upstream training run IDs from Flow 2 MLflow
  └─ reads: checkpoints from checkpoint_cache volume (read-only)
  └─ writes: SWA/merged checkpoints to post_training_out volume
  └─ writes: post_ metrics to MLflow "minivess_training" experiment (same experiment)
  └─ writes: upstream_training_run_id tag

Flow 3 (Analysis)
  └─ reads: training run artifacts from Flow 2 MLflow
  └─ reads: checkpoints from checkpoint_cache volume (read-only)
  └─ reads: test data from data_cache volume (read-only)
  └─ writes: eval_ params + comparison metrics to MLflow "minivess_evaluation"
  └─ writes: figures/tables to outputs_analysis volume
  └─ writes: champion tags to training runs in MLflow
  └─ writes: upstream_training_run_id tag in evaluation runs

Flow 4 (Deployment)
  └─ reads: champion tags from MLflow training runs
  └─ reads: checkpoints from checkpoint_cache volume (read-only)
  └─ writes: ONNX files to outputs_deploy volume
  └─ writes: BentoML models to bentoml_store volume
  └─ writes: deploy_ tags to MLflow runs
  └─ writes: audit trail JSON to outputs_deploy volume

Flow 5 (Dashboard)
  └─ reads: all upstream runs from MLflow (all experiments)
  └─ reads: analysis figures from outputs_analysis volume (read-only)
  └─ writes: dashboard report + JSON to outputs_dashboard volume

Flow 6 (QA)
  └─ reads: all runs from MLflow
  └─ writes: QA report to outputs_dashboard volume
  └─ writes: qa_ tags to ghost runs in MLflow

Annotation App
  └─ reads: BentoML models from bentoml_store volume
  └─ reads: raw data volumes for inference inputs
  └─ writes: annotation session to MLflow "minivess_annotation" experiment
```

### 3.3 Cross-Flow Run Linking

The `upstream_training_run_id` tag pattern (already established in CLAUDE.md) must be
consistently applied. The `FlowContract.log_flow_completion()` method provides the primitive.

Each flow should, at the start of its `@flow` function, call:
```python
contract = FlowContract(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
upstream = contract.find_upstream_run(
    experiment_name="minivess_training",
    upstream_flow="train",
)
```

And at completion:
```python
contract.log_flow_completion(
    flow_name="analyze",
    run_id=current_run_id,
    artifacts=["outputs/analysis/comparison_table.md", "outputs/analysis/comparison_table.tex"],
)
```

### 3.4 Epoch-Level Metric Logging — Current State vs Needed

**Current state (WORKING)**: `trainer.py:562` calls `self.tracker.log_epoch_metrics(epoch_log, step=epoch+1)`.
`tracking.py:222` calls `mlflow.log_metrics(prefixed, step=step)`.

The epoch-level loss curves are implemented and working in the underlying trainer. The following
metrics are logged per epoch with `step=epoch`:
- `train_loss`, `val_loss`
- `train_dice`, `val_dice` (and any other metrics in `SegmentationMetrics`)
- `learning_rate`
- `sys_gpu_*` — GPU utilization, memory, temperature

**Needed**: Wire this into the `training_flow()` Prefect task. Currently `run_training()` in
`train_flow.py` creates an `argparse.Namespace` and returns a stub dict. The actual training
loop (which does epoch logging) is never called from the Prefect flow.

**TensorBoard**: No TensorBoard usage found in any file (`SummaryWriter` and `tensorboard`
grep returned no matches). The platform uses MLflow exclusively for metric logging, which is
the correct approach.

### 3.5 Resume on Spot Preemption — Concept and Current State

**Current concept** ("epoch-latest.yaml"): A lightweight YAML file written at the end of each
epoch to the checkpoint volume, containing:
```yaml
epoch: 47
fold: 1
mlflow_run_id: abc123def456
best_val_loss: 0.142
best_val_dice: 0.891
timestamp: 2026-03-07T14:22:11Z
```

**Current state**: NOT IMPLEMENTED. The concept does not exist in any file. `grep epoch-latest`
returns no matches. The current `--resume` flag in `train_monitored.py` reads from
`checkpoint_dir` — but since `checkpoint_dir = tempfile.mkdtemp()`, there is nothing to resume
from after a spot preemption (the container is gone, the tmpdir is gone).

**Required implementation**:
1. Add `epoch_latest.yaml` writer at end of each epoch in `SegmentationTrainer.fit()`
2. Write to the checkpoint_cache volume path (`/app/checkpoints/epoch_latest.yaml`)
3. On `training_flow()` startup, check for `epoch_latest.yaml` in the checkpoint volume
4. If found and `mlflow_run_id` matches an active RUNNING run, resume from that epoch
5. If not found or run is FINISHED/FAILED, start fresh

The MLflow run ID in `epoch_latest.yaml` is the resumption anchor. A new SkyPilot instance
picks it up, calls `mlflow.start_run(run_id=existing_id)`, and continues logging from where
the preempted instance stopped.

---

## Part 4: Cloud-Agnostic Blob Storage (Flow 0 → Flow 1)

### 4.1 Flow 0: Data Acquisition and Raw Storage

The `acquisition_flow.py` downloads datasets to `config.output_dir / dataset_name`. In Docker
this path must be on the `raw_data` named volume (`/app/data/raw`). The MiniVess dataset
(70 volumes, ~4.5GB) is the primary dataset; it requires manual download (no automated
downloader registered). The flow handles this via `DatasetAcquisitionStatus.MANUAL_REQUIRED`
and prints instructions.

For cloud deployment (SkyPilot), the raw data must be pre-staged to S3 (or equivalent) and
mounted via SkyPilot `file_mounts`:
```yaml
file_mounts:
  /app/data/raw:
    source: s3://minivess-data/raw
    mode: COPY
```

### 4.2 Flow 1: DVC Versioning — Current vs Required

**Current state**: `data_flow.py` does NOT use DVC. It calls
`discover_external_test_pairs(data_dir=data_dir, dataset_name="primary")` directly on the
filesystem. There is no `dvc pull` invocation, no DVC cache, and no version tracking.

**Required state**: The data flow should:
1. Accept a DVC revision (git commit hash or tag) as input parameter
2. Run `dvc pull --rev <revision>` to fetch the versioned dataset
3. Record the DVC commit hash as `data_dvc_commit` param in MLflow
4. The `dataset_hash` currently computed from filepath strings should supplement (not replace)
   the DVC version as the canonical data identifier

### 4.3 Cloud-Agnostic Backend Selection

DVC supports multiple remote storage backends. The config should NOT hardcode S3:

```yaml
# configs/dvc/remotes.yaml (proposed)
remotes:
  local:
    url: /app/data/dvc-cache
  s3:
    url: s3://minivess-data/dvc
    endpointurl: ${MLFLOW_S3_ENDPOINT_URL}  # MinIO for local dev
  gcs:
    url: gs://minivess-data/dvc
  azure:
    url: azure://minivess-data/dvc
  hetzner:
    url: s3://minivess-data/dvc
    endpointurl: ${HETZNER_S3_ENDPOINT}
```

The `data_flow.py` should read `DVC_REMOTE` from environment and configure accordingly:
```python
remote = os.environ.get("DVC_REMOTE", "local")
```

### 4.4 How Downstream Flows Consume Versioned Data

Training flow reads data from the `data_cache` Docker volume, which was written by the data
flow. The data version (DVC commit hash) is available in MLflow as `data_dvc_commit` on the
data engineering run. The training flow discovers this via `FlowContract.find_upstream_run()`.

For SkyPilot: the data is fetched via `file_mounts` from S3. The DVC commit hash must be
embedded in the SkyPilot job parameters and logged to MLflow.

---

## Part 5: Prefect Flow Implementation Gaps

### 5.1 Current Stub Level vs Production-Ready Flow

#### Training Flow — Critical Gap

Current `training_flow()` (from `train_flow.py`):
```python
# This is a STUB — it does nothing
results = run_training(config)
return {"status": "configured", "message": "Use scripts/train_monitored.py for execution."}
```

Required production `training_flow()`:
1. Open MLflow run (not done in current stub)
2. Read upstream data run ID from FlowContract
3. Load fold splits from `/app/configs/splits/splits.json`
4. For each fold: call `SegmentationTrainer.fit()` with proper `checkpoint_dir`
5. After each epoch: write `epoch_latest.yaml` to `/app/checkpoints/`
6. After each fold: upload best checkpoint to MLflow artifacts
7. Log `upstream_data_run_id` tag and `flow_name=train` tag
8. Call `FlowContract.log_flow_completion()`

The training logic already exists in `SegmentationTrainer` — it just needs to be called from
the Prefect flow instead of from `train_monitored.py`.

#### Post-Training Flow — Missing Deployment Config

The `post_training` flow is NOT in `FLOW_WORK_POOL_MAP`. Add:
```python
"post_training": "cpu-pool",
```
And image:
```python
"post_training": "minivess-post-training:latest",
```

The `output_dir` default `Path("outputs/post_training")` must change to
`Path(os.environ.get("POST_TRAINING_OUTPUT_DIR", "/app/outputs/post_training"))`.

#### Data Flow — DVC Integration Missing

The data flow must integrate DVC. The `discover_data_task` currently does filesystem traversal
of `data_dir`. This needs to precede or follow a `dvc pull --rev ${DVC_REV}` invocation.

A new task is needed:
```python
@task(name="dvc-pull")
def dvc_pull_task(data_dir: Path, dvc_rev: str | None = None) -> str:
    """Pull DVC-tracked data at optional revision. Returns commit hash."""
```

### 5.2 Entry Points — Shell Scripts Wrapping Prefect

The ONLY valid shell scripts are ones that wrap `prefect deployment run`. Examples:

```bash
# scripts/run_training.sh
#!/usr/bin/env bash
set -euo pipefail

LOSS="${1:-cbdice_cldice}"
MODEL="${2:-dynunet}"
COMPUTE="${3:-auto}"

prefect deployment run 'training-flow/default' \
  --params "{\"loss_name\": \"${LOSS}\", \"model_family\": \"${MODEL}\", \"compute\": \"${COMPUTE}\"}"
```

```bash
# scripts/run_pipeline.sh
#!/usr/bin/env bash
set -euo pipefail

prefect deployment run 'minivess-data/default' && \
prefect deployment run 'training-flow/default' && \
prefect deployment run 'analysis-flow/default'
```

No script may call `uv run python scripts/*.py` for any production workflow.

### 5.3 Deployment YAML Structure

Each flow needs a deployment defined. The pattern (from Prefect 3.x `serve` API or YAML):

```yaml
# deployment/prefect/deployments.yaml (proposed)
deployments:
  - name: default
    flow: minivess.orchestration.flows.train_flow:training_flow
    work_pool:
      name: gpu-pool
    parameters:
      loss_name: cbdice_cldice
      model_family: dynunet
      compute: auto
    schedules: []

  - name: default
    flow: minivess.orchestration.flows.data_flow:run_data_flow
    work_pool:
      name: cpu-pool
    parameters:
      n_folds: 3
      seed: 42
    schedules:
      - cron: "0 2 * * *"  # Nightly data quality check
```

---

## Part 6: Spot Preemption and Resume

### 6.1 SkyPilot Spot Instance Interruption

SkyPilot spot instances can be preempted at any point during training. The current SkyPilot
configs (`train_generic.yaml`, `train_hpo_sweep.yaml`) call `python scripts/train_monitored.py`
— both violations of the script ban AND missing any preemption-recovery logic.

SkyPilot sends a SIGTERM 30 seconds before preemption. This is the resumption opportunity.
The training loop must:
1. Catch SIGTERM via `signal.signal(signal.SIGTERM, handler)`
2. On receipt: flush current epoch metrics to MLflow, write `epoch_latest.yaml`
3. Save current-epoch checkpoint (not just best-metric checkpoint) to `/app/checkpoints/epoch_latest.pth`
4. Exit cleanly — SkyPilot will relaunch the job on a new instance

### 6.2 Checkpoint Saving Strategy

Two types of checkpoints must be persisted to the mounted checkpoint volume:

**Best-metric checkpoints** (existing, needs path fix):
- `best_val_loss.pth` — checkpoint at epoch with lowest val_loss
- `best_val_dice.pth` — checkpoint at epoch with highest val_dice
- `metric_history.json` — complete per-epoch metric record

**Resume checkpoint** (new, needs implementation):
- `epoch_latest.pth` — model state at most recently completed epoch
- `epoch_latest.yaml` — lightweight manifest with epoch number, fold, MLflow run ID

The `SegmentationTrainer.fit()` loop currently saves best-metric checkpoints to `checkpoint_dir`.
With the fix from #369, `checkpoint_dir` will be `Path(os.environ["CHECKPOINT_DIR"]) / f"fold_{fold_id}"`,
which maps to `/app/checkpoints/fold_{fold_id}/` on the mounted volume.

### 6.3 MLflow-Only State

The MLflow run is the canonical state. On spot preemption and relaunch:
1. New instance reads `epoch_latest.yaml` from `/app/checkpoints/epoch_latest.yaml`
2. Extracts `mlflow_run_id` from the YAML
3. Calls `mlflow.start_run(run_id=existing_run_id)` — resumes the existing run
4. Calls `mlflow.log_param("sys_resumed_from_epoch", epoch_number)` for provenance
5. Loads `epoch_latest.pth` as starting model state
6. Continues training from `epoch_number + 1`

MLflow's tracking is append-only — metrics logged at `step=47` on the original instance and
`step=48` on the new instance appear in the same run's metric history. No duplicate detection
needed.

### 6.4 Resume Detection in `training_flow()`

```python
@task(name="check-resume-state")
def check_resume_state(checkpoint_dir: Path) -> dict[str, Any] | None:
    """Check for epoch_latest.yaml to determine if this is a resumed run."""
    latest_path = checkpoint_dir / "epoch_latest.yaml"
    if not latest_path.exists():
        return None
    import yaml  # yaml.safe_load — NOT regex
    with latest_path.open(encoding="utf-8") as f:
        state = yaml.safe_load(f)
    # Validate the referenced MLflow run is still RUNNING
    import mlflow
    run = mlflow.get_run(state["mlflow_run_id"])
    if run.info.status == "RUNNING":
        return state
    return None  # Stale state — start fresh
```

---

## Part 7: Compute Routing

### 7.1 Environment → Infrastructure Mapping

| Environment | Docker | Prefect | GPU | Purpose |
|-------------|--------|---------|-----|---------|
| **dev** | Docker Compose (`docker-compose.yml --profile dev`) | Prefect Server local | Local GPU via Docker | Fast iteration, Prefect UI at localhost:4200 |
| **staging** | Docker Compose (`docker-compose.flows.yml`) | Prefect Server local | docker-compose GPU reservation | Integration testing, real training runs |
| **prod-local** | Docker Compose (`docker-compose.flows.yml`) | Prefect Server + Workers | docker-compose GPU pool | Full pipeline on-prem |
| **prod-cloud** | Docker on SkyPilot VMs | Prefect Server + sky-pool | SkyPilot spot A100 | Cloud training with failover |

### 7.2 Local Dev — Docker Compose Stack

The development stack requires:
1. `docker compose -f deployment/docker-compose.yml --profile dev up` — starts Postgres, MinIO, MLflow, Prefect Server, BentoML
2. Build flow images: `docker compose -f deployment/docker-compose.flows.yml build`
3. Register deployments: `uv run python -m minivess.orchestration.serve` (proposed)
4. Run a flow: `prefect deployment run 'training-flow/default' --params '{"loss_name": "cbdice_cldice"}'`

The `PREFECT_DISABLED=1` env var allows running flows without a Prefect server (for CI), via
the `_prefect_compat.py` no-op decorators.

### 7.3 Staging — Docker Compose With GPU

Identical to dev, but using the `monitoring` profile which adds Prometheus, Grafana, and
the Prefect workers:
```bash
docker compose -f deployment/docker-compose.yml --profile monitoring up
```

The `prefect-worker-cpu` and `prefect-worker-gpu` services start automatically, polling the
Prefect Server for submitted jobs.

### 7.4 Prod — SkyPilot + Prefect sky-pool

The `sky-pool` work pool in `work-pools.yaml` (`type: process`) is designed for SkyPilot
integration. The training flow invocation from SkyPilot should be:

```yaml
# deployment/skypilot/train_generic.yaml (corrected)
run: |
  cd /app
  prefect deployment run 'training-flow/default' \
    --params "{\"loss_name\": \"${LOSS_NAME}\", \"compute\": \"${COMPUTE}\"}"
```

NOT `uv run python scripts/train_monitored.py`.

For NVIDIA MIG (multi-model inference), the inference container can use MIG slices:
```yaml
# deployment/docker-compose.flows.yml (analyze/deploy serve mode)
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
          device_ids: ["MIG-GPU-..."]  # MIG slice ID
```

---

## Part 8: Implementation Roadmap

### 8.1 P0 — Must Fix Before Any Production Run

Priority order reflects dependency chain and severity:

**T-01: Fix `tempfile.mkdtemp()` in `train_monitored.py`** (Issue #369)
- File: `scripts/train_monitored.py:673`
- Fix: Replace `Path(tempfile.mkdtemp()) / f"fold_{fold_id}"` with
  `Path(os.environ.get("CHECKPOINT_DIR", "/app/checkpoints")) / f"fold_{fold_id}"`
- Tests first (TDD): test that checkpoint_dir is not in /tmp when CHECKPOINT_DIR is set

**T-02: Implement `training_flow()` body** (Issue #367)
- File: `src/minivess/orchestration/flows/train_flow.py`
- Fix: Replace stub with actual call to `SegmentationTrainer.fit()`
- Must read: splits from env, open MLflow run, write `epoch_latest.yaml`
- Tests first: test that `training_flow()` creates an MLflow run with expected params

**T-03: Add all missing volume mounts to `docker-compose.flows.yml`** (Issue #369)
- File: `deployment/docker-compose.flows.yml`
- Fix: Add `acquisition` service, add volume mounts for `analyze`, `deploy`, `dashboard`, `qa`
- Tests first: `docker compose config` passes; integration test verifies artifact persistence

**T-04: Fix regex violations** (CLAUDE.md Rule #16)
- File: `src/minivess/orchestration/flows/analysis_flow.py:1033`
- Fix: Replace `_FOLD_RE = re.compile(r"^(.+)_fold(\d+)$")` with `name.rsplit("_fold", 1)`
- File: `src/minivess/pipeline/duckdb_extraction.py:364`
- Fix: Replace `re.match(r"eval_fold(\d+)_(.+)", metric_name)` with `str.split()` approach

**T-05: Fix `deployment/skypilot/train_generic.yaml`** (Issue #367)
- Fix: Replace `uv run python scripts/train_monitored.py` with `prefect deployment run`

**T-06: Add `post_training` to `FLOW_WORK_POOL_MAP`** (Issue #369)
- File: `src/minivess/orchestration/deployments.py`
- Fix: Add `"post_training": "cpu-pool"` to map

### 8.2 P1 — Must Complete Before Paper Submission

**T-07: Add `acquisition` service to `docker-compose.flows.yml`**
- Includes volume mounts for `raw_data` and `logs_data`

**T-08: Implement DVC integration in `data_flow.py`**
- Add `dvc_pull_task()` with configurable remote
- Log DVC commit hash as `data_dvc_commit` to MLflow

**T-09: Implement `epoch_latest.yaml` writer**
- In `SegmentationTrainer.fit()` loop after each epoch
- Format: YAML with epoch, fold, mlflow_run_id, best metrics, timestamp

**T-10: Implement `check_resume_state()` task in `training_flow()`**
- Reads `epoch_latest.yaml` from checkpoint volume
- Uses `yaml.safe_load()` (NOT regex)
- Validates MLflow run is RUNNING before resuming

**T-11: Add MLflow run opening to all flows**
- `data_flow.py`, `acquisition_flow.py`, `post_training_flow.py`, `dashboard_flow.py`, `qa_flow.py`
- Each must open an MLflow run, log its params, and call `FlowContract.log_flow_completion()`

**T-12: Implement `FlowContract` wiring in all flows**
- `find_upstream_run()` at flow start
- `log_flow_completion()` at flow end

**T-13: Fix SkyPilot `train_hpo_sweep.yaml`**
- Replace `uv run python scripts/run_hpo.py` with proper Prefect invocation
- Or create `hpo_flow.py` as a proper Prefect flow

**T-14: Add `annotation` service to `docker-compose.flows.yml`**
- Annotation flow has no Docker infrastructure

**T-15: Implement Prefect deployment YAML**
- `deployment/prefect/deployments.yaml` with all 9 flows
- Or register via Python `serve()` API

### 8.3 P2 — Improvements That Strengthen the Platform

**T-16: Add SIGTERM handler to training loop**
- Write `epoch_latest.yaml` on SIGTERM before exit
- Log "preempted at epoch N" to MLflow

**T-17: Add DVC remote config for cloud-agnostic backends**
- Support local, S3 (MinIO), GCS, Azure, Hetzner

**T-18: Implement annotation flow Docker infrastructure**
- Create `Dockerfile.annotation`
- Add service to `docker-compose.flows.yml`
- Define BentoML store volume mount

**T-19: Add QA report to MLflow**
- `qa_flow()` currently writes report as a string, not to MLflow
- Add `mlflow.log_text(report, "qa_report.md")` at end of QA flow

**T-20: Add deprecation warnings to `train_monitored.py`**
- Print clear warning: "THIS IS NOT THE SUPPORTED ENTRY POINT — use prefect deployment run"
- Do not remove the file (needed for migration reference), but make it non-invitable as default

### 8.4 Dependency Graph

```
T-01 (tempfile fix)
  ↓
T-02 (train_flow implementation)  ← depends on T-01
  ↓
T-09 (epoch_latest.yaml)          ← depends on T-02
T-10 (resume detection)           ← depends on T-09
  ↓
T-16 (SIGTERM handler)            ← depends on T-10

T-03 (docker-compose volumes)
  ↓
T-07 (acquisition service)        ← depends on T-03
T-14 (annotation service)         ← depends on T-03

T-04 (regex violations)           ← independent

T-06 (post_training deploy config) ← independent

T-08 (DVC integration)
  ↓
T-17 (DVC remotes)                ← depends on T-08

T-11 (MLflow run in all flows)
T-12 (FlowContract wiring)        ← depends on T-11

T-15 (Prefect deployment YAML)    ← depends on T-02, T-11, T-12
  ↓
T-05 (SkyPilot fix)               ← depends on T-15
T-13 (HPO SkyPilot fix)           ← depends on T-15
```

---

## Appendix A: Files That Must Change

| File | Issue | Change Required |
|------|-------|-----------------|
| `scripts/train_monitored.py:673` | #369 | Replace `tempfile.mkdtemp()` with env var path |
| `src/minivess/orchestration/flows/train_flow.py` | #367 | Implement real training logic (not stub) |
| `deployment/docker-compose.flows.yml` | #369 | Add volumes for analyze, deploy, dashboard, qa; add acquisition service |
| `src/minivess/orchestration/flows/analysis_flow.py:1033` | regex ban | Replace `re.compile` with `rsplit()` |
| `src/minivess/pipeline/duckdb_extraction.py:364` | regex ban | Replace `re.match` with `str.split()` |
| `deployment/skypilot/train_generic.yaml:50` | #367 | Replace `python scripts/train_monitored.py` with `prefect deployment run` |
| `deployment/skypilot/train_hpo_sweep.yaml:37` | #367 | Replace `python scripts/run_hpo.py` with Prefect invocation |
| `src/minivess/orchestration/deployments.py` | #369 | Add `post_training` to `FLOW_WORK_POOL_MAP` |
| `src/minivess/orchestration/flows/data_flow.py` | DVC gap | Add `dvc_pull_task()` |
| `src/minivess/orchestration/flows/post_training_flow.py` | #369 | Fix `output_dir` relative path to env var |
| `src/minivess/pipeline/trainer.py` | #369 | Add `epoch_latest.yaml` writer |

## Appendix B: Files That Must Be Created

| File | Purpose |
|------|---------|
| `deployment/prefect/deployments.yaml` | Prefect 3.x deployment definitions for all 9 flows |
| `scripts/run_training.sh` | Shell wrapper around `prefect deployment run 'training-flow/default'` |
| `scripts/run_pipeline.sh` | Full pipeline trigger via Prefect deployments |
| `configs/dvc/remotes.yaml` | Cloud-agnostic DVC remote configuration |
| `src/minivess/orchestration/serve.py` | Python script to register all Prefect deployments |

## Appendix C: Constraint Summary (Non-Negotiable Rules)

From CLAUDE.md and metalearning docs, these rules apply to every task in the plan:

1. **`import re` is BANNED** — use `ast.parse()`, `json.loads()`, `yaml.safe_load()`, `str.split()`, `pathlib.Path`
2. **`/tmp` and `tempfile.mkdtemp()` are BANNED** for any artifact that must survive container exit
3. **`python scripts/*.py` is BANNED** as training entry point — use `prefect deployment run`
4. **All Docker outputs must be volume-mounted** — no ephemeral writes that escape the volume declaration
5. **MLflow is the ONLY inter-flow contract** — no shared filesystem, no direct function calls
6. **`from __future__ import annotations`** at top of every Python file
7. **`pathlib.Path()`** always, never string concatenation for paths
8. **`datetime.now(timezone.utc)`** always, never `datetime.now()`
9. **Silent fallback is banned** — missing required components must raise `RuntimeError` with actionable instructions
10. **uv ONLY** for package management — never pip, conda, requirements.txt
11. **TDD mandatory** — tests first, then implementation, never the reverse
12. **No placeholder implementations** — `pass`, `TODO`, `NotImplementedError` are banned in production code
