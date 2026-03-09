# Prefect Flow Connectivity — MLflow as Inter-Flow Contract

**Branch**: `fix/prefect-flow-connectivity`
**Date**: 2026-03-09
**Status**: Planning

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

### Root Causes (Diagnosed from Code)

| Problem | Location | Evidence |
|---------|----------|---------|
| Train flow never logs `checkpoint_dir` path to MLflow as a tag/artifact | `train_flow.py:604` | `checkpoint_dir = checkpoint_base / "fold_{n}"` — written to disk only, never to MLflow |
| Post-training calls `find_upstream_safely` but gets `run_id` only — no checkpoint location | `post_training_flow.py:273-279` | Even if upstream found, checkpoints not in MLflow |
| Analysis hardcodes `experiment_name="minivess_training"` — breaks in debug | `analysis_flow.py:1698` | Debug uses a different name; no `_DEBUG` suffix mechanism |
| `flow_status: completed` tag is set but `FLOW_COMPLETE` is never queried with checkpoint artifact path | `flow_contract.py:log_flow_completion` | `flow_artifacts` param exists but callers never pass checkpoint paths |
| No `_DEBUG` suffix on experiment names | all flows | Debug runs pollute production experiment namespace |
| Data Engineering never writes to MLflow | `data_flow.py` | `find_upstream_safely(experiment_name="minivess_data")` in train_flow always returns None |

---

## 2. Vision: Full Flow Connectivity Architecture

The following diagram shows the intended information flow. MLflow is the **only**
inter-flow communication channel. No shared filesystem paths, no direct function calls.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MLFLOW CONTRACT LAYER                                │
│                                                                             │
│  Experiment: minivess_data[_DEBUG]        (Data Engineering writes)         │
│  Experiment: minivess_training[_DEBUG]    (Train + Post-training write)     │
│  Experiment: minivess_analysis[_DEBUG]    (Analysis writes)                 │
│  Experiment: minivess_biostatistics[_DEBUG] (Biostatistics writes)          │
│  Model Registry: minivess-champion        (Analysis registers, Deploy reads)│
└─────────────────────────────────────────────────────────────────────────────┘

Data Acquisition ──→ (filesystem only, no MLflow output yet)

Data Engineering ──→ mlflow: minivess_data[_DEBUG]
    writes: dataset_hash, n_volumes, dvc_commit, splits_path_tag
    open question: register DVC dataset as MLflow Dataset entity?
    ↓
    [FlowContract: FLOW_COMPLETE tag + splits_path artifact tag]

Modelling (Train) ──reads──→ minivess_data[_DEBUG] (latest FINISHED run)
    writes: minivess_training[_DEBUG]
    writes per-fold run: fold_0_best_val_loss, checkpoint_dir tag, model_name tag
    writes parent run: n_folds_completed, experiment config artifact
    ↓
    [FlowContract: FLOW_COMPLETE + checkpoint_dir_{fold_n} tags]

Post-training ──reads──→ minivess_training[_DEBUG] (latest FINISHED parent run)
    reads: checkpoint_dir_{fold_n} tags → finds checkpoints on volume
    writes: SAME experiment (minivess_training[_DEBUG]), NEW runs with suffix
      run_name = "{base_run_name}.swa", ".calibrated", ".conformal", ".merged"
    ↓
    [FlowContract: FLOW_COMPLETE on each post-training run]

Analysis ──reads──→ minivess_training[_DEBUG] (all FINISHED runs, including post-training)
    reads: ALL runs in experiment (base + swa + calibrated + conformal + merged)
    creates every permutation: single/ensemble × loss × fold × post-processing variant
    writes: minivess_analysis[_DEBUG]
      - one run per model/ensemble evaluated
      - champion registered in Model Registry
    ↓
    [FlowContract: FLOW_COMPLETE + champion_run_id tag]

Biostatistics ──reads──→ minivess_analysis[_DEBUG] (latest FINISHED run)
    reads: comparison table artifact, all evaluation runs
    writes: minivess_biostatistics[_DEBUG]
      - ONE run, name = "paper_comparison_{YYYY-MM-DD}"
      - aggregates everything for PI/CTO/CEO/paper
      - functions as data source for Dashboard Flow
    ↓
    [FlowContract: FLOW_COMPLETE]

Deployment ──reads──→ Model Registry (champion from Analysis)
    reads: champion model URI from mlflow_analysis[_DEBUG] FLOW_COMPLETE tag
    deploys via BentoML
    does NOT write to MLflow (read-only consumer)

Dashboard ──reads──→ minivess_biostatistics[_DEBUG] (latest run)
    reads aggregated comparison, produces figures/reports

Data Annotation ──reads──→ BentoML served endpoint (from Deployment)
    does NOT read/write MLflow at all
```

---

## 3. Per-Flow MLflow Contract Specification

### 3.1 Data Acquisition Flow
**Current state**: No MLflow integration.
**Target state**: Still no MLflow output. Writes raw data to filesystem only.
**Reads**: Nothing.
**Writes**: Nothing to MLflow. Data lands on the `raw_data` volume.

### 3.2 Data Engineering Flow
**Current state**: Writes to MLflow (`minivess_data`) but train_flow's `find_upstream_safely`
returns None because the tag/experiment name doesn't match or no run exists.
**Target state**: Write a single FINISHED run per dataset version.

**Required MLflow writes** (experiment: `minivess_data{debug_suffix}`):
```python
mlflow.log_param("data_hash", dataset_hash)          # sha256 of all volume IDs
mlflow.log_param("data_n_volumes", n_volumes)
mlflow.log_param("data_dvc_commit", dvc_commit)       # git hash of DVC pointer
mlflow.log_param("splits_version", splits_hash)
mlflow.set_tag("flow_name", "data-flow")
mlflow.set_tag("flow_status", "FLOW_COMPLETE")        # queried by train_flow
mlflow.set_tag("splits_path", str(splits_path))       # absolute path on shared volume
```

**Open question — DVC Dataset Registration**:
MLflow 3.x supports `mlflow.data.from_dvc()` to register a DVC dataset entity.
This would make the MLflow UI show which dataset version trained which model.
*Recommendation*: Register it. The overhead is minimal and the lineage value is high.
Defer as Phase 2 — not blocking flow connectivity.

**FlowContract tag written at end**: `flow_status=FLOW_COMPLETE`

### 3.3 Modelling (Train) Flow
**Current state**: Logs metrics per fold but never logs `checkpoint_dir` paths to MLflow.
Post-training therefore has no way to find checkpoints via MLflow.

**Required MLflow writes** (experiment: `minivess_training{debug_suffix}`):

Parent run (one per `training_flow()` call):
```python
mlflow.set_tag("flow_name", "training-flow")
mlflow.set_tag("flow_status", "FLOW_COMPLETE")        # MISSING — must be added
mlflow.set_tag("model_name", config["model"])         # MISSING
mlflow.set_tag("experiment_config", experiment_name)  # already done
mlflow.log_param("n_folds_completed", n_folds)        # already done
mlflow.log_artifact(resolved_config_path)             # already done
```

Per-fold child run (one per fold):
```python
mlflow.set_tag(f"checkpoint_dir", str(checkpoint_dir))   # MISSING — critical
mlflow.set_tag(f"fold_id", str(fold_id))                  # MISSING
mlflow.set_tag("flow_name", "training-flow")
mlflow.log_metric("best_val_loss", best_val_loss)         # already done
```

**The critical missing piece**: `checkpoint_dir` tag on each fold run. Without this,
post-training cannot locate checkpoints via MLflow.

**Alternative approach**: Log checkpoint files as MLflow artifacts (copies them into
MLflow's artifact store on MinIO). This is cleaner but doubles storage. Recommend
tagging the path (volume path stays authoritative) rather than copying. If the
volume is lost, the artifacts are lost either way.

### 3.4 Post-Training Flow
**Current state**: Queries `find_upstream_safely(experiment_name=upstream_exp)` and gets
a run_id — but then has no checkpoint paths because train_flow never wrote them.
`checkpoint_paths` parameter defaults to `[]`, all 4 plugins skip validation.

**Required changes**:
1. After `find_upstream_safely` returns a run_id, query all **child runs** (fold runs)
   from that parent run_id.
2. For each child run, read `checkpoint_dir` tag → resolve to actual `.ckpt` files on volume.
3. Pass resolved checkpoint paths to each plugin.

**MLflow writes** (same experiment as train: `minivess_training{debug_suffix}`):
```python
# New child runs under the SAME parent experiment — suffixed run names
run_name = f"{base_run_name}.swa"           # Stochastic Weight Averaging result
run_name = f"{base_run_name}.calibrated"    # Temperature-scaled model
run_name = f"{base_run_name}.conformal"     # CRC conformal prediction
run_name = f"{base_run_name}.merged"        # Model soup/merging result

mlflow.set_tag("flow_name", "post-training-flow")
mlflow.set_tag("flow_status", "FLOW_COMPLETE")
mlflow.set_tag("post_training_variant", "swa|calibrated|conformal|merged")
mlflow.set_tag("upstream_training_run_id", training_run_id)
mlflow.set_tag("checkpoint_path", str(output_checkpoint_path))  # output artifact
```

### 3.5 Analysis Flow
**Current state**: Queries `minivess_training` (hardcoded) but then `evaluate-all-models`
task finds 0 members because it can't resolve checkpoints from run metadata.
Also: `all_loss_single_best` and `all_loss_all_best` ensembles have 0 members.

**Required changes**:
1. Query ALL runs in `minivess_training{debug_suffix}` — base runs AND post-training suffixed runs.
2. For each run, read `checkpoint_path` or `checkpoint_dir` tag → load model.
3. Build all permutations: `{model} × {fold} × {variant}` × ensembles.
4. Use `{debug_suffix}` in experiment name lookup, not hardcoded `"minivess_training"`.

**MLflow writes** (experiment: `minivess_analysis{debug_suffix}`):
```python
# One run per evaluated model/ensemble configuration
mlflow.set_tag("flow_name", "analysis-flow")
mlflow.set_tag("flow_status", "FLOW_COMPLETE")         # on final summary run
mlflow.set_tag("champion_run_id", champion_run_id)     # for Deploy flow
mlflow.set_tag("champion_model_uri", champion_model_uri)
# Register champion in Model Registry:
mlflow.register_model(champion_model_uri, "minivess-champion")
```

### 3.6 Biostatistics Flow
**Current state**: Exists as a flow stub but not wired to Analysis output.

**Target state**:
1. Query `minivess_analysis{debug_suffix}` — latest FINISHED run.
2. Download comparison table artifact.
3. Aggregate all metrics across models, variants, folds.
4. Write one MLflow run: `minivess_biostatistics{debug_suffix}`.
   - Run name: `paper_comparison_{YYYY-MM-DD}` (date of run, not experiment)
   - Artifacts: comparison CSV, LaTeX tables, PDF figures
   - Functions as data source for Dashboard Flow and for the scientific paper

**MLflow writes** (experiment: `minivess_biostatistics{debug_suffix}`):
```python
run_name = f"paper_comparison_{date.today().isoformat()}"
mlflow.set_tag("flow_name", "biostatistics-flow")
mlflow.set_tag("flow_status", "FLOW_COMPLETE")
mlflow.set_tag("upstream_analysis_run_id", analysis_run_id)
mlflow.log_artifact(comparison_csv_path)
mlflow.log_artifact(latex_table_path)
mlflow.log_artifact(pdf_figures_path)
```

### 3.7 Deployment Flow
**Current state**: Reads `minivess_training` experiment to find champion — same hardcoded
name problem.
**Target state**: Read from Model Registry (which Analysis populated) + from
`minivess_analysis{debug_suffix}` for the `champion_run_id` tag.

**Reads only** (no MLflow writes):
```python
# Read champion from Model Registry:
client.get_latest_versions("minivess-champion", stages=["Production"])
# Or from Analysis FlowContract:
champion_uri = analysis_run.data.tags["champion_model_uri"]
```

Deploy via BentoML. No MLflow writes. This flow is a pure consumer.

### 3.8 Dashboard Flow
**Target state**: Read from `minivess_biostatistics{debug_suffix}` — download artifacts,
render dashboards. No MLflow writes.

### 3.9 Data Annotation Flow
**Target state**: Call BentoML served endpoint for inference. No MLflow reads or writes.

---

## 4. Debug Experiment Naming Convention

**Rule**: All experiment names get a `_DEBUG` suffix in debug runs.
**Not** the mlrun names (those stay human-readable within the experiment).

```python
# In every flow, resolve debug suffix:
def _resolve_experiment_name(base_name: str, *, debug: bool = False) -> str:
    """Append _DEBUG suffix for debug runs."""
    suffix = os.environ.get("MINIVESS_DEBUG_SUFFIX", "")  # e.g. "_DEBUG"
    return f"{base_name}{suffix}"

# Usage:
experiment_name = _resolve_experiment_name("minivess_training", debug=True)
# → "minivess_training_DEBUG"
```

**Setting the flag**: The `run_debug.sh` script sets `MINIVESS_DEBUG_SUFFIX=_DEBUG`
in the container environment via `-e MINIVESS_DEBUG_SUFFIX=_DEBUG`.

**Why this matters**:
- `minivess_training` = real experiment (100 epochs, full data, production quality)
- `minivess_training_DEBUG` = smoke test (2 epochs, 5 volumes, code validation only)
- MLflow UI clearly separates them; no risk of debug metrics polluting real comparisons
- FlowContract queries must always pass the same suffix through the whole chain

---

## 5. FlowContract: What Needs to Change

### 5.1 Missing: `log_flow_completion` must write `checkpoint_dir` tags

```python
# Current — only sets flow_name and flow_status:
def log_flow_completion(self, *, flow_name, run_id, artifacts=None): ...

# Required — must also write structured checkpoint metadata:
def log_flow_completion(
    self,
    *,
    flow_name: str,
    run_id: str,
    checkpoint_paths: list[Path] | None = None,   # NEW
    checkpoint_dir: Path | None = None,            # NEW
    artifacts: list[str] | None = None,
) -> None: ...
```

### 5.2 Missing: `find_upstream_run` must return checkpoint metadata

```python
# Current — returns {run_id, status, tags}
# Required — add method to get ALL fold runs and their checkpoint paths:

def find_fold_checkpoints(
    self,
    *,
    parent_run_id: str,
) -> list[dict]:
    """Find all fold child runs and their checkpoint paths."""
    # Query child runs by parent_run_id tag
    # Return [{fold_id, run_id, checkpoint_dir, checkpoint_path}, ...]
```

### 5.3 Missing: `find_upstream_run` must respect debug suffix

```python
def find_upstream_run(
    self,
    *,
    experiment_name: str,
    upstream_flow: str,
    debug_suffix: str = "",   # NEW — pass "" for prod, "_DEBUG" for debug
    tags: dict | None = None,
) -> dict | None:
    full_experiment_name = f"{experiment_name}{debug_suffix}"
    ...
```

---

## 6. The Experiment Name Map

| Flow | Writes To | Reads From |
|------|-----------|------------|
| Data Acquisition | nothing | nothing |
| Data Engineering | `minivess_data{sfx}` | nothing |
| Train (Modelling) | `minivess_training{sfx}` | `minivess_data{sfx}` |
| Post-training | `minivess_training{sfx}` (new runs, suffixed names) | `minivess_training{sfx}` |
| Analysis | `minivess_analysis{sfx}` + Model Registry | `minivess_training{sfx}` (all runs) |
| Biostatistics | `minivess_biostatistics{sfx}` | `minivess_analysis{sfx}` |
| Deployment | nothing | Model Registry + `minivess_analysis{sfx}` |
| Dashboard | nothing | `minivess_biostatistics{sfx}` |
| Data Annotation | nothing | BentoML endpoint |

`{sfx}` = `_DEBUG` in debug mode, `""` in production.

---

## 7. Implementation Roadmap

### Phase 0 — Fix the Broken Seam (Train → Post-training → Analysis)
**Scope**: Minimum viable fix to make the 3 core flows actually pass real data.

- [ ] `train_flow.py`: Log `checkpoint_dir` tag on each fold run
- [ ] `train_flow.py`: Log `flow_status=FLOW_COMPLETE` tag on parent run
- [ ] `flow_contract.py`: Add `find_fold_checkpoints()` method
- [ ] `post_training_flow.py`: Call `find_fold_checkpoints()` → resolve checkpoint paths
- [ ] `analysis_flow.py`: Query ALL runs in training experiment, build real ensemble members
- [ ] `run_debug.sh`: Pass `MINIVESS_DEBUG_SUFFIX=_DEBUG` to all containers

**Test**: Debug run shows post-training plugins actually execute (not skipped),
analysis evaluates real models, champion is tagged.

### Phase 1 — Data Engineering → Train Handoff
- [ ] `data_flow.py`: Write `splits_path` tag to MLflow on FLOW_COMPLETE
- [ ] `train_flow.py`: Actually read and use `splits_path` from upstream data run
- [ ] (Optional) DVC Dataset registration as MLflow Dataset entity

### Phase 2 — Analysis → Biostatistics → Dashboard Chain
- [ ] `biostatistics_flow.py`: Implement `find_upstream_safely` from `minivess_analysis`
- [ ] `biostatistics_flow.py`: Write `paper_comparison_{date}` run to `minivess_biostatistics`
- [ ] `dashboard_flow.py`: Read from `minivess_biostatistics`

### Phase 3 — Deployment Flow Champion Discovery
- [ ] `analysis_flow.py`: Register champion to Model Registry after evaluation
- [ ] `deploy_flow.py`: Read from Model Registry instead of hardcoded experiment scan

### Phase 4 — Debug Suffix Propagation (All Flows)
- [ ] `_resolve_experiment_name()` utility in `constants.py` or `flow_utils.py`
- [ ] All flows use `_resolve_experiment_name(base, debug=bool)` consistently
- [ ] `run_debug.sh`: Set `MINIVESS_DEBUG_SUFFIX=_DEBUG`
- [ ] `FlowContract.find_upstream_run()`: Accept and propagate `debug_suffix`

---

## 8. Key Design Decisions and Open Questions

### Q1: Should checkpoint files be MLflow artifacts or just tagged paths?

**Option A — Tag the volume path** (recommended for now):
```python
mlflow.set_tag("checkpoint_dir_fold_0", "/app/checkpoints/fold_0")
```
- Pros: Zero extra storage. Checkpoints stay on the shared volume.
- Cons: Volume must be mounted consistently across all flow containers.
  If the volume is lost, checkpoints are gone (but this is already true).

**Option B — Log as MLflow artifact** (cleaner, more portable):
```python
mlflow.log_artifact("/app/checkpoints/fold_0/best.ckpt")
```
- Pros: Checkpoint lives in MinIO artifact store, queryable via `mlflow.artifacts.download_artifacts`.
- Cons: Doubles storage (volume + MinIO). Large checkpoints (SAM3 ~2GB).

**Recommendation**: Phase 0 uses option A (tags). Option B can be added in Phase 3
when deployment needs artifact URIs for Model Registry.

### Q2: Should Data Engineering register datasets in MLflow?

MLflow 3.x supports `mlflow.log_input(mlflow.data.from_dvc(...))` which creates a
dataset entity linked to a run. This enables:
- "Which model ran on which dataset version?" queries
- Lineage tracking aligned with IEC 62304 (SaMD audit trail)

**Recommendation**: Yes, but Phase 1. Not blocking flow connectivity.

### Q3: Should Post-training runs be child runs of the training parent run?

**Option A — Sibling runs** (same experiment, new parent run): Current approach. Cleaner
experiment view in MLflow UI. Post-training result is independently queryable.

**Option B — Child runs** (nested under the training parent run): Shows hierarchy in UI.
But MLflow's nested run support is limited and querying is more complex.

**Recommendation**: Sibling runs with naming convention `{base_name}.{variant}` and
`upstream_training_run_id` tag. Simpler to query.

### Q4: What happens if the upstream flow's run is not FINISHED?

`FlowContract.find_upstream_run` currently filters `status = 'FINISHED'`. If a run
crashed (status = FAILED or KILLED), downstream flows skip it. This is correct behavior.
The flow must be re-run; no partial result recovery.

---

## 9. Files That Need Changes (Phase 0)

| File | Change |
|------|--------|
| `src/minivess/orchestration/flows/train_flow.py` | Log `checkpoint_dir` + `flow_status=FLOW_COMPLETE` tags |
| `src/minivess/orchestration/flow_contract.py` | Add `find_fold_checkpoints()`, respect debug suffix |
| `src/minivess/orchestration/flows/post_training_flow.py` | Call `find_fold_checkpoints()` → real paths |
| `src/minivess/orchestration/flows/analysis_flow.py` | Query all runs including post-training variants |
| `src/minivess/orchestration/constants.py` | Add `_resolve_experiment_name()` utility |
| `scripts/run_debug.sh` | Export `MINIVESS_DEBUG_SUFFIX=_DEBUG` |
| `configs/experiment/debug_all_models.yaml` | Set `debug: true` flag for suffix resolution |
| `tests/v2/unit/test_flow_connectivity.py` | NEW — tests for FlowContract seams |
| `tests/integration/orchestration/test_flow_contract_integration.py` | NEW — end-to-end contract tests |

---

## Appendix: Original User Prompt (Verbatim)

> So let's create fix/prefect-flow-connectivity and I am disappointed then on the bad behavior that we have for our codebase as we have been trying to get the mlflow-prefect working for quite a time but still the basic functionalities are not working. Let's plan to /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/prefect-flow-connectivity.md on how each Flow should obviously be able to query the previous Flow's results . Modelling Flow should be able the query the available DVC versioned datasets from Data Engineering Flow. Post-training should be able to query the the Modelling results from the MLflow, as should Analysis of Mlflow (note that Post-training is using now the same MLflow experiment (same experiment_name) as the Modelling Flow but only adding the suffix to the run_names). The analysis then uses it new experiment_name and creates all the possible permutation names. And the biostatistics module is able to query the previous Flow again (the Analysis Flow) and the Biostatistics Flow then creates a new experiment (name) with one mlrun with some name and a date that essentially creates a comparison of everything for a scientific paper or for a PI/CTO/CEO as in what do we have at the moment (and can also function as a data source for the Dashboard Flow). Then the deployement Flow can read the Analysis MLflow experiment with these functioning as the MFlow contract layer so that they depoyment Flow can read MLflow Model Registry and deploy the "chosen champion" to production via BentoML then. So Modelling Flow starts using MLflow in this scenario with the Deployment Flow not producing any outputs to MLflow but reading the input. It is open for debate and analysis whether Data Engineering Flow needs to output anything to MLflow yet as should the DVC versioned dataset be registered as MLflow dataset? Data Acquisition Flow does not work in anyway at the moment with MLflow. Data annotation needs to mainly access the served BentoML deployment and definitely not write anything to MLflow. Does this as a vision make any sense? The debug then is also a special case in the sense that every experiment name should get a "_DEBUG" suffix (not the mlrun names) so that real experiments with all the epochs and all the data do not get mixed with the debug runs that are as the name implies meant for debugging the code and the "pipeline mechanics". Save my prompt verbatim as an appendix to the .md plan and start planning!
