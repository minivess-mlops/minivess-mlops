"""Rule #22 enforcement: container path and HPO vars must be in .env.example.

Tests will FAIL if these variables are missing from .env.example (RED phase).
After adding them, all tests must pass (GREEN phase).

Vars covered:
  DATA_DIR, CHECKPOINT_DIR, LOGS_DIR, SPLITS_DIR   — container-internal artifact paths
  POSTGRES_DB_OPTUNA, OPTUNA_STORAGE_URL            — HPO database (PostgreSQL only)
  REPLICA_INDEX                                     — HPO multi-worker replica
  PREFECT_LOGGING_EXTRA_LOGGERS                     — flow log forwarding

CLAUDE.md Rule #22: ALL config values defined in .env.example FIRST.
Plan: docs/planning/overnight-child-prefect-docker.xml Phase 1
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENV_EXAMPLE = ROOT / ".env.example"


def _env_example_vars() -> set[str]:
    """Parse .env.example and return all defined variable names (no regex)."""
    vars_: set[str] = set()
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, _ = line.partition("=")
            vars_.add(key.strip())
    return vars_


def test_data_dir_in_env_example() -> None:
    assert "DATA_DIR" in _env_example_vars(), (
        "DATA_DIR missing from .env.example — add it as: DATA_DIR=/app/data"
    )


def test_checkpoint_dir_in_env_example() -> None:
    assert "CHECKPOINT_DIR" in _env_example_vars(), (
        "CHECKPOINT_DIR missing from .env.example — add it as: CHECKPOINT_DIR=/app/checkpoints"
    )


def test_logs_dir_in_env_example() -> None:
    assert "LOGS_DIR" in _env_example_vars(), (
        "LOGS_DIR missing from .env.example — add it as: LOGS_DIR=/app/logs"
    )


def test_splits_dir_in_env_example() -> None:
    assert "SPLITS_DIR" in _env_example_vars(), (
        "SPLITS_DIR missing from .env.example — add it as: SPLITS_DIR=/app/configs/splits"
    )


def test_postgres_db_optuna_in_env_example() -> None:
    assert "POSTGRES_DB_OPTUNA" in _env_example_vars(), (
        "POSTGRES_DB_OPTUNA missing from .env.example — add it as: POSTGRES_DB_OPTUNA=optuna"
    )


def test_optuna_storage_url_in_env_example() -> None:
    assert "OPTUNA_STORAGE_URL" in _env_example_vars(), (
        "OPTUNA_STORAGE_URL missing from .env.example — must be a postgresql:// URL"
    )


def test_replica_index_in_env_example() -> None:
    assert "REPLICA_INDEX" in _env_example_vars(), (
        "REPLICA_INDEX missing from .env.example — add it as: REPLICA_INDEX=0"
    )


def test_prefect_logging_extra_loggers_in_env_example() -> None:
    assert "PREFECT_LOGGING_EXTRA_LOGGERS" in _env_example_vars(), (
        "PREFECT_LOGGING_EXTRA_LOGGERS missing from .env.example — "
        "add it as: PREFECT_LOGGING_EXTRA_LOGGERS=minivess"
    )


def test_dockerfile_train_does_not_hardcode_data_dir() -> None:
    """DATA_DIR must not be an ENV in Dockerfile.train — injected by compose (Rule #22)."""
    df = ROOT / "deployment" / "docker" / "Dockerfile.train"
    if not df.exists():
        return
    for line in df.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        assert not (stripped.startswith("ENV") and "DATA_DIR" in stripped), (
            "Dockerfile.train has hardcoded 'ENV DATA_DIR'. "
            "Remove it — DATA_DIR is injected by docker-compose from .env.example."
        )


def test_dockerfile_train_does_not_hardcode_checkpoint_dir() -> None:
    """CHECKPOINT_DIR must not be an ENV in Dockerfile.train."""
    df = ROOT / "deployment" / "docker" / "Dockerfile.train"
    if not df.exists():
        return
    for line in df.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        assert not (stripped.startswith("ENV") and "CHECKPOINT_DIR" in stripped), (
            "Dockerfile.train has hardcoded 'ENV CHECKPOINT_DIR'. "
            "Remove it — CHECKPOINT_DIR is injected by docker-compose from .env.example."
        )


def test_dockerfile_train_does_not_hardcode_logs_dir() -> None:
    """LOGS_DIR must not be an ENV in Dockerfile.train."""
    df = ROOT / "deployment" / "docker" / "Dockerfile.train"
    if not df.exists():
        return
    for line in df.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        assert not (stripped.startswith("ENV") and "LOGS_DIR" in stripped), (
            "Dockerfile.train has hardcoded 'ENV LOGS_DIR'. Remove it."
        )


def test_dockerfile_train_does_not_hardcode_splits_dir() -> None:
    """SPLITS_DIR must not be an ENV in Dockerfile.train."""
    df = ROOT / "deployment" / "docker" / "Dockerfile.train"
    if not df.exists():
        return
    for line in df.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        assert not (stripped.startswith("ENV") and "SPLITS_DIR" in stripped), (
            "Dockerfile.train has hardcoded 'ENV SPLITS_DIR'. Remove it."
        )
