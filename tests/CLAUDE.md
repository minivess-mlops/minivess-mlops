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
├── unit/           # Fast, isolated, no external deps (<1s each)
├── integration/    # Service integration (MLflow, Docker)
└── e2e/            # End-to-end pipeline tests
```

## Markers

- `@pytest.mark.slow` — integration tests (>5s each), skipped in staging tier
- `@pytest.mark.gpu` — requires GPU (skipped on CPU-only runners)
- `@pytest.mark.docker` — requires Docker daemon

## Running

```bash
# All tests:
uv run pytest tests/ -x -q

# Staging tier only (fast, for PR readiness):
uv run pytest tests/ -x -q -m "not slow"

# Full suite:
uv run pytest tests/ -x -q --run-slow
```

## Rules

- All test files use `tmp_path` fixture — NEVER `tempfile.mkdtemp()`
- Never import from `scripts/` — those are migration utilities
- Never skip pre-commit hooks in test configurations
- Test data: use `configs/splits/` for fold definitions, mock MLflow for tracking
- Fixtures that need MLflow: use `mlflow.set_tracking_uri(tmp_path / "mlruns")`
