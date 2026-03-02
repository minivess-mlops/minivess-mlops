"""MLflow backend health checking.

Provides a lightweight health check for the MLflow tracking backend,
supporting both filesystem and server (HTTP) backends. Uses only
stdlib ``urllib`` — no new dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HealthCheckResult:
    """Result of an MLflow backend health check.

    Parameters
    ----------
    healthy:
        Whether the backend is reachable and functional.
    backend_type:
        ``"filesystem"`` or ``"server"``.
    message:
        Human-readable status message.
    uri:
        The tracking URI that was checked.
    """

    healthy: bool
    backend_type: str
    message: str
    uri: str


def check_mlflow_health(tracking_uri: str | None = None) -> HealthCheckResult:
    """Check health of the MLflow tracking backend.

    Parameters
    ----------
    tracking_uri:
        Tracking URI to check. If ``None``, resolves via
        :func:`~minivess.observability.tracking.resolve_tracking_uri`.

    Returns
    -------
    HealthCheckResult
    """
    if tracking_uri is None:
        from minivess.observability.tracking import resolve_tracking_uri

        tracking_uri = resolve_tracking_uri()

    if tracking_uri.startswith(("http://", "https://")):
        return _check_server_health(tracking_uri)
    return _check_filesystem_health(tracking_uri)


def _check_server_health(uri: str) -> HealthCheckResult:
    """Ping an MLflow server via HTTP GET."""
    health_url = uri.rstrip("/") + "/health"
    try:
        with urlopen(health_url, timeout=5) as response:  # noqa: S310
            if response.status == 200:  # noqa: PLR2004
                return HealthCheckResult(
                    healthy=True,
                    backend_type="server",
                    message=f"MLflow server healthy at {uri}",
                    uri=uri,
                )
            return HealthCheckResult(
                healthy=False,
                backend_type="server",
                message=f"MLflow server returned status {response.status}",
                uri=uri,
            )
    except Exception as exc:
        return HealthCheckResult(
            healthy=False,
            backend_type="server",
            message=f"MLflow server unreachable: {exc}",
            uri=uri,
        )


def _check_filesystem_health(uri: str) -> HealthCheckResult:
    """Check filesystem backend — create directory if missing."""
    path = Path(uri)
    try:
        path.mkdir(parents=True, exist_ok=True)
        return HealthCheckResult(
            healthy=True,
            backend_type="filesystem",
            message=f"Filesystem backend ready at {path}",
            uri=uri,
        )
    except OSError as exc:
        return HealthCheckResult(
            healthy=False,
            backend_type="filesystem",
            message=f"Cannot create filesystem backend: {exc}",
            uri=uri,
        )
