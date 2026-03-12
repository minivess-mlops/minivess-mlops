# Inter-Flow Contract Design

## Implementation Files

| File | Purpose |
|------|---------|
| `src/minivess/orchestration/flows/train_flow.py` | Sets flow_name tag |
| `src/minivess/orchestration/flows/analysis_flow.py` | Calls find_upstream_run() |
| `src/minivess/observability/tracking.py` | find_upstream_run(), resolve_checkpoint_paths_from_contract() |

## Tag Constants

```python
FLOW_NAME_TRAIN = "training-flow"
FLOW_NAME_ANALYSIS = "analysis-flow"
FLOW_NAME_DEPLOY = "deploy-flow"
FLOW_NAME_DASHBOARD = "dashboard-flow"
```

## Discovery Pattern

```python
upstream_run = find_upstream_run(
    experiment_name="my_experiment",
    flow_name=FLOW_NAME_TRAIN,  # filters by tag
)
checkpoint = resolve_checkpoint_paths_from_contract(upstream_run)
```

## PR History

- PR #589: Fixed find_upstream_run() to filter by flow_name tag
- Bug: was returning most recent FINISHED run regardless of flow type
