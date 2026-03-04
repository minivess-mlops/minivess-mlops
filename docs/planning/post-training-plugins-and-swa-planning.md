# Post-Training Plugin Architecture + Multi-SWA Implementation Plan

**Branch:** `feat/advanced-segmentation-double-check`
**Issues:** #314-#324 (11 issues)
**Status:** IMPLEMENTED (2026-03-04)

## Architecture

### Pipeline Position (Flow 2.5)

```
data(1) -> train(2) -> POST_TRAINING(2.5) -> analyze(3) -> deploy(4) -> dashboard(5) -> qa(6)
```

**Classification:** Best-effort (failure does NOT block analyze).

### Plugin Protocol

```python
@runtime_checkable
class PostTrainingPlugin(Protocol):
    name: str                                    # Unique identifier
    requires_calibration_data: bool              # Needs cal data?
    def validate_inputs(self, pi: PluginInput) -> list[str]: ...
    def execute(self, pi: PluginInput) -> PluginOutput: ...
```

### Task DAG

```
discover-training-runs
    |
    +-- run-swa-plugin          (no cal data needed)
    +-- run-multi-swa-plugin    (no cal data needed)
    +-- run-model-merging       (no cal data needed)
    +-- collect-calibration-data
           |
           +-- run-calibration-plugin
           +-- run-crc-conformal-plugin
           +-- run-conseco-fp-control
    |
aggregate-results -> log-summary
```

## SWA vs SWAG (CRITICAL DISTINCTION)

| Method | Post-Hoc? | What It Does |
|--------|-----------|--------------|
| **SWA** | YES | Average checkpoints from single run |
| **Multi-SWA** | YES | M independent SWA models ensembled |
| **SWAG** | NO | SWA + low-rank covariance (training-time) |
| **Multi-SWAG** | NO | M independent SWAG models |

**SWAG is NOT supported** — requires training-time second-moment collection.

## Implemented Plugins (6)

| Plugin | Wraps | Cal Data? |
|--------|-------|-----------|
| SWA | `model_soup.uniform_swa()` | No |
| Multi-SWA | `model_soup.uniform_swa()` (subsampled) | No |
| Model Merging | `model_merging.{linear,slerp,layer_wise}_merge()` | No |
| Calibration | `calibration.temperature_scale()` + new isotonic/Platt | Yes |
| CRC Conformal | `crc_conformal.CRCPredictor` + `varisco_heatmap()` | Yes |
| ConSeCo FP Control | New threshold/erosion shrinking | Yes |

## Config: `configs/post_training/default.yaml`

Each plugin has `enabled: bool` toggle. Default: SWA, merging, calibration, CRC on;
Multi-SWA and ConSeCo off.

## Files Created

| File | Purpose |
|------|---------|
| `src/minivess/config/post_training_config.py` | Pydantic config model |
| `configs/post_training/default.yaml` | Default YAML config |
| `src/minivess/pipeline/post_training_plugin.py` | Protocol + dataclasses + registry |
| `src/minivess/pipeline/post_training_plugins/__init__.py` | Package |
| `src/minivess/pipeline/post_training_plugins/swa.py` | SWA plugin |
| `src/minivess/pipeline/post_training_plugins/multi_swa.py` | Multi-SWA plugin |
| `src/minivess/pipeline/post_training_plugins/model_merging.py` | Model merging plugin |
| `src/minivess/pipeline/post_training_plugins/calibration.py` | Calibration plugin |
| `src/minivess/pipeline/post_training_plugins/crc_conformal.py` | CRC conformal plugin |
| `src/minivess/pipeline/post_training_plugins/conseco_fp_control.py` | ConSeCo plugin |
| `src/minivess/orchestration/flows/post_training_flow.py` | Prefect Flow 2.5 |
| `deployment/docker/Dockerfile.post_training` | Docker container |

## Files Modified

| File | Change |
|------|--------|
| `src/minivess/orchestration/trigger.py` | Added `post_training` to `_DEFAULT_FLOWS` (best-effort) |
| `deployment/docker-compose.flows.yml` | Added `post_training` service |
| `src/minivess/orchestration/flows/analysis_flow.py` | Added `discover_post_training_models()` task + `include_post_training` param |

## Tests: 96 passing

| Phase | Test File | Count |
|-------|----------|-------|
| 0 | `test_post_training_config.py` | 18 |
| 1 | `test_post_training_plugin.py` | 9 |
| 2 | `test_swa_plugin.py` | 8 |
| 3 | `test_multi_swa_plugin.py` | 6 |
| 4 | `test_merging_plugin.py` | 8 |
| 5 | `test_calibration_plugin.py` | 9 |
| 6 | `test_crc_plugin.py` | 6 |
| 7 | `test_conseco_plugin.py` | 6 |
| 8 | `test_post_training_flow.py` | 7 |
| 9 | `test_flow_trigger.py` | 12 |
| 10 | `test_analysis_flow_post_training.py` | 5 |
| | **Total** | **96** |

## GitHub Issues

| # | Title | Phase |
|---|-------|-------|
| #314 | PostTrainingConfig Pydantic model + YAML | 0 |
| #315 | PostTrainingPlugin protocol + dataclasses | 1 |
| #316 | SWA plugin (wrap model_soup.py) | 2 |
| #317 | Multi-SWA plugin | 3 |
| #318 | Model merging plugin | 4 |
| #319 | Post-hoc calibration plugin | 5 |
| #320 | CRC conformal plugin | 6 |
| #321 | ConSeCo FP control plugin | 7 |
| #322 | post_training_flow.py Prefect flow | 8 |
| #323 | Trigger chain + Docker integration | 9 |
| #324 | Analysis flow post-training model discovery | 10 |

Also closed: #313 (Multi-SWAG — hallucinated, replaced by correct Multi-SWA).
