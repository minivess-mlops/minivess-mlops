# Cloud E2E Verification Plan (Task 3.2)

## Purpose

After the debug factorial experiment completes, verify the full cloud pipeline:
train → analysis → biostatistics. Confirms MLflow artifact storage, inter-flow
communication, and statistical analysis work end-to-end on GCP.

## Prerequisites

- All 34 factorial jobs SUCCEEDED (DynUNet + MambaVesselNet + SAM3)
- `pulumi up` deployed for #878 (GCS artifact store fix)
- MLflow Cloud Run instance accessible

## Verification Steps

### Step 1: Query MLflow for completed runs

```bash
uv run python -c "
import mlflow
mlflow.set_tracking_uri('$MLFLOW_TRACKING_URI')
client = mlflow.MlflowClient()
exp = client.get_experiment_by_name('minivess_training')
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string=\"attributes.status = 'FINISHED'\",
)
print(f'{len(runs)} FINISHED runs')
for r in runs[:5]:
    print(f'  {r.info.run_id}: {r.data.tags.get(\"loss_function\", \"?\")}')
"
```

**Expected**: 16+ FINISHED runs (8 DynUNet + 8 MambaVesselNet minimum)

### Step 2: Verify MLflow artifacts stored in GCS

```bash
gsutil ls gs://minivess-mlops-mlflow-artifacts/ | head -10
```

**Expected**: Artifact directories for each run (no 413 errors)

### Step 3: Trigger analysis flow

```bash
# Via Docker (production path)
docker compose -f deployment/docker-compose.flows.yml run analysis-flow \
  --env UPSTREAM_EXPERIMENT=minivess_training \
  --env MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI

# Or via SkyPilot
sky jobs launch deployment/skypilot/analysis.yaml
```

**Expected**: Ensemble built, evaluation metrics logged, champion registered

### Step 4: Trigger biostatistics flow

```bash
docker compose -f deployment/docker-compose.flows.yml run biostatistics-flow \
  --env MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI
```

**Expected**:
- ANOVA tables generated (one per metric)
- Interaction plots generated
- Variance lollipop charts generated
- Comparison tables with bootstrap CIs
- All artifacts logged to MLflow

### Step 5: Verify all artifacts

```bash
uv run python -c "
import mlflow
mlflow.set_tracking_uri('$MLFLOW_TRACKING_URI')
client = mlflow.MlflowClient()

# Check biostatistics experiment
exp = client.get_experiment_by_name('minivess_biostatistics')
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string=\"attributes.status = 'FINISHED'\",
    max_results=1,
)
if runs:
    r = runs[0]
    artifacts = client.list_artifacts(r.info.run_id)
    print(f'Biostatistics run {r.info.run_id}:')
    for a in artifacts:
        print(f'  {a.path} ({a.file_size} bytes)')
"
```

**Expected artifacts**:
- `figures/` — PNG + SVG for each figure type
- `tables/` — LaTeX .tex files
- `biostatistics.duckdb` — Analysis database
- `parquet/` — Exported data

## Success Criteria

1. All 34 factorial conditions have FINISHED runs in MLflow
2. MLflow artifacts stored in GCS (no local file:// URIs)
3. Analysis flow discovers all runs and builds ensembles
4. Biostatistics flow produces all statistical outputs:
   - Pairwise comparisons with bootstrap CIs
   - ANOVA tables (one per metric)
   - Interaction plots (Model x Loss)
   - Variance lollipop charts
   - Ranking tables
5. All artifacts logged as MLflow artifacts (not local files)
