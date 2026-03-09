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
