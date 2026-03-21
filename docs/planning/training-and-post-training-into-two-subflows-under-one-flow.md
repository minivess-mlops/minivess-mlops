# Plan: Merge Training + Post-Training into One Parent Flow with 2 Sub-Flows

**Date**: 2026-03-21
**Branch**: `test/debug-factorial-4th-pass`
**Status**: PLANNED — awaiting implementation
**Prerequisite for**: 4th pass GCP debug factorial

---

## Architecture

```
CURRENT (two separate flows, two SkyPilot jobs):
================================================
SkyPilot Job A                         SkyPilot Job B (not wired)
┌─────────────────────┐                ┌─────────────────────┐
│ @flow("training")   │   checkpoint   │ @flow("post-train") │
│ └─ train_fold()     │ ────────────►  │ └─ run_plugin()     │
└─────────────────────┘                └─────────────────────┘


TARGET (one parent flow, one SkyPilot job):
==========================================
SkyPilot Job (single GPU session)
┌──────────────────────────────────────────────────────┐
│ @flow("training-flow")  ← PARENT                    │
│ │                                                    │
│ ├─ Sub-flow 1: @flow("training-subflow")             │
│ │  └─ train_one_fold_task() per fold                 │
│ │  └─ Returns: checkpoint, model, train_loader       │
│ │                                                    │
│ ├─ IF post_training_method != "none":                │
│ │                                                    │
│ ├─ Sub-flow 2: @flow("post-training-subflow")        │
│ │  └─ SWAG / checkpoint_avg / etc.                   │
│ │  └─ Logs to MLflow with same tags                  │
│ │                                                    │
│ └─ Returns: combined result                          │
└──────────────────────────────────────────────────────┘
```

**Key insight**: "none" condition is free — just the training sub-flow output. SWAG
shares GPU + DataLoaders with training. No extra SkyPilot job needed.

---

## File Changes

### Modify

| File | Change |
|------|--------|
| `src/minivess/orchestration/flows/train_flow.py` | Refactor into parent + 2 sub-flows. Add `post_training_method` param. |
| `src/minivess/orchestration/constants.py` | Add `FLOW_NAME_TRAINING_SUBFLOW`, `FLOW_NAME_POST_TRAINING_SUBFLOW` |
| `deployment/skypilot/train_factorial.yaml` | Add `POST_TRAINING_METHOD` env var + `--post-training-method` flag |
| `scripts/run_factorial.sh` | Parse `post_training.methods` from factorial YAML |
| `configs/base.yaml` | Add `post_training: none` to defaults list |
| `configs/experiment/debug_factorial.yaml` | Add `post_training.methods: [none]` (debug) |
| `configs/experiment/paper_factorial.yaml` | Add `post_training.methods: [none, swag]` (production) |

### Add

| File | Purpose |
|------|---------|
| `configs/post_training/none.yaml` | Identity — no post-training |
| `configs/post_training/swag.yaml` | SWAG config (swa_lr, swa_epochs, max_rank) |
| `tests/v2/unit/orchestration/test_train_post_training_subflow.py` | 8 tests for merged flow |
| `tests/v2/unit/orchestration/test_post_training_config_hydra.py` | 4 tests for Hydra config |

### Leave Unchanged

| File | Why |
|------|-----|
| `src/minivess/orchestration/flows/post_training_flow.py` | Still functional for Docker-per-flow |
| `src/minivess/pipeline/post_training_plugins/swag.py` | Plugin interface unchanged |
| `src/minivess/pipeline/post_training_plugin.py` | PluginInput/PluginOutput unchanged |

---

## Hydra Config Structure

```yaml
# configs/base.yaml (add to defaults list)
defaults:
  - post_training: none    # ← NEW

# configs/post_training/none.yaml
method: none

# configs/post_training/swag.yaml
method: swag
swag:
  swa_lr: 0.01
  swa_epochs: 10       # debug: 2, production: 10
  max_rank: 20
  n_samples: 30
  update_bn: true
```

Override per-condition: `POST_TRAINING_METHOD=swag` env var or `--post-training-method swag`.

---

## How run_factorial.sh Passes Post-Training Method

**Approach**: Comma-separated methods in single SkyPilot job. The parent flow
iterates internally — SWAG reuses GPU + DataLoaders from training.

```bash
# Parse from factorial YAML
POST_METHODS=$(python3 -c "
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path('${CONFIG_FILE}').read_text(encoding='utf-8'))
methods = cfg.get('post_training', {}).get('methods', ['none'])
print(','.join(methods))
")

# Pass to SkyPilot job (one job handles all methods)
sky jobs launch "${SKYPILOT_YAML}" \
  --env POST_TRAINING_METHOD="${POST_METHODS}" \
  ...other envs...
```

---

## MLflow Artifact Flow

```
Parent flow
├── Sub-flow 1 (training):
│   └── MLflow run with tags:
│       flow_name: "training-flow"
│       post_training_method: "none"
│       model_family, loss_function, fold_id, with_aux_calib
│   └── Artifacts: config/resolved_config.yaml, checkpoints/best_*.pth
│
├── Sub-flow 2 (post-training, if method != "none"):
│   └── MLflow run with tags:
│       flow_name: "post-training-flow"  ← SAME tag as standalone flow
│       post_training_method: "swag"
│       upstream_training_run_id: {parent_run_id}
│       (inherited: model_family, loss_function, fold_id, with_aux_calib)
│   └── Artifacts: post_training/swag_model.pt (via mlflow.log_artifact)
```

Downstream flows (analysis, biostatistics) discover post-training runs via
the same `post_training_method` tag query they already use.

---

## Potential Glitches

| # | Glitch | Risk | Mitigation |
|---|--------|------|------------|
| G1 | SWAG needs train_loader + model in memory | Med | Parent flow passes between sub-flows (same process) |
| G2 | Memory pressure keeping DataLoaders alive | Med | Clear non-active fold DataLoaders between sub-flows |
| G3 | `test_no_cross_flow_imports.py` failure | Low | Sub-flow is co-located in train_flow.py, not imported |
| G4 | Spot preemption during SWAG | Med | Training checkpoint already in MLflow. Resume re-runs SWAG only. |
| G5 | TrainingFlowResult schema change | Med | Add optional fields with defaults. Existing consumers unaffected. |
| G6 | `test_flow_name_tags.py` needs update | Low | Allow train_flow.py to produce both training + post-training tags |

---

## Migration Path (4 phases)

**Phase 0**: Add config infrastructure (no behavior change)
- Create `configs/post_training/{none,swag}.yaml`
- Add to `configs/base.yaml` defaults
- Add constants to `constants.py`
- `make test-staging` — all pass, no behavior change

**Phase 1**: Refactor into parent + sub-flow (no post-training yet)
- Extract training logic into `training_subflow()`
- Parent calls sub-flow and returns result
- `make test-staging` — behavior identical

**Phase 2**: Add post-training sub-flow
- Add `post_training_subflow()` in train_flow.py
- Wire conditional call from parent
- Add 12 new tests
- `make test-staging`

**Phase 3**: Wire SkyPilot + factorial script
- Add env vars to `train_factorial.yaml`
- Update `run_factorial.sh` parsing
- Dry-run test
- `make test-prod`

---

## Test Plan

### New Tests (12)

| Test | Purpose |
|------|---------|
| `test_parent_flow_calls_training_subflow` | "none" → only training sub-flow |
| `test_parent_flow_calls_both_subflows` | "swag" → both sub-flows |
| `test_none_no_post_training_run` | No MLflow post-training run for "none" |
| `test_swag_creates_post_training_run` | MLflow run with correct tags for "swag" |
| `test_result_includes_post_training` | Combined result has post-training info |
| `test_dataloader_passed_to_swag` | SWAG receives train_loader + model |
| `test_method_from_argparse` | `--post-training-method` parsing |
| `test_method_from_env` | `POST_TRAINING_METHOD` env var |
| `test_swag_yaml_loads` | Hydra config loads |
| `test_none_yaml_loads` | Hydra config loads |
| `test_compose_with_swag_override` | Config composition works |
| `test_config_matches_config_class` | YAML values match SWAGPluginConfig |
