# Tests

## Environment Variables for Tests

```bash
MINIVESS_ALLOW_HOST=1    # Bypass Docker gate (_require_docker_context)
PREFECT_DISABLED=1       # No Prefect server required for unit tests
```

Both are **test-only** escape hatches. Never use in scripts or production.

## Test Structure

```
tests/
├── v2/unit/        # Fast isolated unit tests
├── v2/integration/ # Service integration (MLflow, Docker stack)
├── v2/quasi_e2e/   # Parametrized model×loss combinations
├── gpu_instance/   # SAM3 + GPU-heavy tests (EXCLUDED from default collection)
├── unit/           # Legacy unit tests
└── integration/    # Legacy integration tests
```

**`tests/gpu_instance/`** is excluded from default pytest collection via
`collect_ignore_glob` in `conftest.py`. These tests NEVER run on the dev machine
or in CI. They run on external GPU instances (RunPod, intranet servers) via
`make test-gpu`. See #564.

## Three Test Tiers

| Tier | Command | What | Target |
|------|---------|------|--------|
| **Staging** | `make test-staging` | No model loading, no slow, no integration | <3 min |
| **Prod** | `make test-prod` | Everything except `tests/gpu_instance/` | ~5-10 min |
| **GPU** | `make test-gpu` | `tests/gpu_instance/` only | GPU instance |

## Markers

| Marker | Purpose | Excluded from |
|--------|---------|---------------|
| `@pytest.mark.model_loading` | Instantiates PyTorch models (DynUNet, etc.) | staging |
| `@pytest.mark.slow` | Tests >30s | staging |
| `@pytest.mark.integration` | Auto-tagged for `tests/**/integration/` | staging, prod |
| `@pytest.mark.gpu` | Requires CUDA GPU | staging |
| `@pytest.mark.gpu_heavy` | SAM3 forward passes, large models | all (lives in gpu_instance/) |

## Rules

- All test files use `tmp_path` fixture — NEVER `tempfile.mkdtemp()`
- Never import from `scripts/` — those are migration utilities
- Never skip pre-commit hooks in test configurations
- Test data: use `configs/splits/` for fold definitions, mock MLflow for tracking
- Fixtures that need MLflow: use `mlflow.set_tracking_uri(tmp_path / "mlruns")`
- Flow tests must set `LOGS_DIR`, `CHECKPOINT_DIR`, `SPLITS_DIR`, `MLFLOW_TRACKING_URI`
  via `monkeypatch.setenv()` — flows read these from env, not defaults
- SAM3 tests belong in `tests/gpu_instance/` — NEVER in the standard suite

## Cloud Tests (`tests/v2/cloud/`)

Cloud tests require `MLFLOW_CLOUD_*` env vars and auto-skip without them.

```bash
make test-cloud-mlflow   # Run all cloud tests
```

| Test File | Purpose | Marker |
|-----------|---------|--------|
| `test_cloud_mlflow.py` | L2 provider-agnostic MLflow tests | `@cloud_mlflow` |
| `test_skypilot_mlflow.py` | SkyPilot-specific MLflow tests | `@cloud_mlflow` |
| `test_dvc_cloud_pull.py` | DVC pull from UpCloud S3 | `@cloud_mlflow` |
| `test_training_flow_cloud_mlflow.py` | Training flow against remote MLflow | `@cloud_mlflow` |
| `test_cloud_training_artifacts.py` | Post-run artifact verification | `@cloud_mlflow` |

## RunPod GPU Smoke Tests

Manually-triggered end-to-end tests that train on RunPod RTX 4090 via SkyPilot.
**NOT scientifically valid** — infrastructure verification only (2 epochs, 4 volumes).

### Prerequisites

1. Set env vars in `.env`: `DVC_S3_*`, `RUNPOD_API_KEY`, `MLFLOW_CLOUD_*`
2. Install SkyPilot: `uv sync --extra infra && sky check runpod`
3. Push data to UpCloud S3: `uv run python scripts/configure_dvc_remote.py && dvc push -r upcloud`
4. UpCloud MLflow server running (`pulumi up` in `deployment/pulumi/`)

### How to Run

```bash
# 1. Validate prerequisites
make smoke-test-preflight

# 2. Run a single model smoke test (~$0.12, ~10 min)
make smoke-test-gpu MODEL=sam3_vanilla

# 3. Verify results on cloud MLflow
make verify-smoke-test MODEL=sam3_vanilla

# 4. Run all 3 models sequentially (~$0.36, ~30 min)
make smoke-test-all
```

### Cost Model

| Model | VRAM | Duration | Cost |
|-------|------|----------|------|
| SAM3 Vanilla | 2.9 GB (measured) | ~5 min | ~$0.06 |
| SAM3 Hybrid | 7.5 GB (estimated) | ~8 min | ~$0.09 |
| VesselFM | 10 GB (estimated) | ~10 min | ~$0.12 |
| **Total (all 3)** | — | ~31 min | ~$0.36 |

### Troubleshooting

- **SkyPilot RunPod check fails**: `sky check runpod` — verify API key
- **DVC pull fails on RunPod**: Check `DVC_S3_*` vars and bucket accessibility
- **MLflow auth fails**: Verify `MLFLOW_CLOUD_USERNAME`/`PASSWORD` match server config
- **OOM on SAM3 Hybrid**: Expected on 8 GB GPU — requires RTX 4090 (24 GB)
