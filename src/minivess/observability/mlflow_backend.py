"""MLflow backend detection and migration utilities.

Provides functions to detect the type of MLflow tracking backend,
warn on local filesystem usage in production, and detect local
mlruns/ directories that should be migrated to a server backend.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def detect_backend_type(tracking_uri: str) -> str:
    """Detect the type of MLflow tracking backend from its URI.

    Parameters
    ----------
    tracking_uri:
        MLflow tracking URI string.

    Returns
    -------
    One of: ``"server"`` (HTTP/HTTPS), ``"database"`` (SQL),
    ``"local"`` (filesystem path).
    """
    uri = tracking_uri.strip()
    if uri.startswith(("http://", "https://")):
        return "server"
    if uri.startswith(("postgresql://", "sqlite://", "mysql://", "mssql://")):
        return "database"
    return "local"


def warn_if_local_backend(
    tracking_uri: str,
    *,
    environment: str = "dev",
) -> None:
    """Emit a warning if using a local filesystem backend in non-dev environments.

    Parameters
    ----------
    tracking_uri:
        MLflow tracking URI string.
    environment:
        Current deployment environment (dev, staging, prod).
    """
    backend_type = detect_backend_type(tracking_uri)
    if backend_type == "local" and environment not in ("dev", "development", "test"):
        logger.warning(
            "MLflow is using a local filesystem backend (%s) in %s environment. "
            "This is not recommended for production. Use a server backend "
            "(e.g., http://mlflow:5000) or database backend "
            "(e.g., postgresql://...) for reliability and collaboration. "
            "See deployment/docker-compose.yml for the MLflow server setup.",
            tracking_uri,
            environment,
        )


def check_local_mlruns(project_root: Path | str) -> dict[str, Any]:
    """Check for existence of local mlruns/ directory.

    Useful for migration detection — identifies projects still using
    local file-based MLflow tracking that should migrate to server backend.

    Parameters
    ----------
    project_root:
        Project root directory to check.

    Returns
    -------
    Dict with:
    - ``has_local_mlruns``: bool
    - ``n_experiments``: int (number of experiment directories)
    - ``mlruns_path``: str (path if exists)
    """
    from pathlib import Path as _Path

    project_root = _Path(project_root)
    mlruns_path = project_root / "mlruns"
    if not mlruns_path.exists():
        return {
            "has_local_mlruns": False,
            "n_experiments": 0,
            "mlruns_path": str(mlruns_path),
        }

    # Count experiment directories (numeric names or named)
    experiments = [
        d for d in mlruns_path.iterdir() if d.is_dir() and d.name != ".trash"
    ]
    return {
        "has_local_mlruns": True,
        "n_experiments": len(experiments),
        "mlruns_path": str(mlruns_path),
    }
