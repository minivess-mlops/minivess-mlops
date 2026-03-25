# Test Suite Optimization Plan

**Date:** 2026-03-22
**Triggered by:** 34 skips in prod suite, user instruction that skips ARE failures

## Current State

| Tier | Command | Passed | Skipped | Failed | Time |
|------|---------|--------|---------|--------|------|
| **Staging** | `make test-staging` | 5752 | **0** | 0 | 4:47 |
| **Prod** | `make test-prod` | 6063 | **34** | 0 | 6:27 |
| **GPU** | `make test-gpu` | N/A | N/A | N/A | GPU only |
| **Cloud** | (no target) | N/A | N/A | N/A | N/A |

### 34 Prod Skips Breakdown

| Root Cause | Count | Fix |
|-----------|-------|-----|
| MLFLOW_TRACKING_URI not remote | 27 | Parametrize: test local always + remote when available |
| RUNPOD_API_KEY not set | 3 | Move to `make test-runpod` or parametrize |
| SAM3 TopoLoRA >=16 GB VRAM | 1 | Move to gpu_instance/ |
| HF/preflight cloud checks | 3 | Same as MLflow fix |

## Proposed 5-Tier Model (ZERO skips in ALL tiers)

| Tier | Target | What | Skips Allowed | Time Target |
|------|--------|------|---------------|-------------|
| **staging** | `make test-staging` | Unit tests, no model loading, no cloud | **0** | <5 min |
| **prod** | `make test-prod` | Everything except cloud + gpu_instance | **0** | <10 min |
| **cloud** | `make test-cloud` | Cloud MLflow + SkyPilot + DVC remote | **0** | <5 min |
| **gpu** | `make test-gpu` | SAM3, GPU-heavy model tests | **0** | GPU only |
| **runpod** | `make test-runpod` | RunPod-specific SkyPilot tests | **0** | RunPod only |

### Key Change: Prod EXCLUDES cloud tests

Prod currently includes `tests/v2/cloud/` which causes all 34 skips. Fix:
add `--ignore=tests/v2/cloud/` to prod Makefile target. Cloud tests run
separately via `make test-cloud` when credentials are available.

### Cloud Tests: Parametrize for Local + Remote

Per user instruction: "tests should test both local and remote connection."

```python
@pytest.fixture(params=["local", "remote"])
def mlflow_uri(request):
    if request.param == "local":
        return "mlruns"  # Always available
    else:
        uri = os.environ.get("MLFLOW_CLOUD_URI")
        if not uri:
            pytest.skip("MLFLOW_CLOUD_URI not set")
        return uri
```

This way:
- `make test-prod`: runs cloud tests with local URI only (0 skips)
- `make test-cloud`: runs with both local AND remote URIs

### SAM3 VRAM Skip: Move to gpu_instance/

`test_model_builder.py:66` skips because 7.6 GB < 16 GB. This test belongs
in `tests/gpu_instance/` not in the prod suite.

## Implementation Tasks

1. [ ] Add `--ignore=tests/v2/cloud/` to `make test-prod`
2. [ ] Create `make test-cloud` target
3. [ ] Parametrize cloud tests for local+remote
4. [ ] Move SAM3 TopoLoRA VRAM test to gpu_instance/
5. [ ] Add `make test-runpod` target (or fold into test-cloud with RunPod marker)
6. [ ] Verify: ALL tiers have 0 skips

## Acceptance Criteria

- `make test-staging`: 0 skipped
- `make test-prod`: 0 skipped (cloud excluded)
- `make test-cloud`: 0 skipped (both local and remote tested)
- `make test-gpu`: 0 skipped (all GPU tests run on GPU instance)
