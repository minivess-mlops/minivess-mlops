from __future__ import annotations

import warnings

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "real_data: requires real MiniVess dataset (not run in CI)"
    )
    config.addinivalue_line(
        "markers",
        "requires_mlflow_server: Tests requiring a running MLflow server"
        " (auto-skipped if unhealthy)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Auto-skip tests marked with requires_mlflow_server when server is down."""
    _mlflow_healthy: bool | None = None

    for item in items:
        if item.get_closest_marker("requires_mlflow_server") is not None:
            # Lazy-evaluate health once per session
            nonlocal_healthy = _mlflow_healthy
            if nonlocal_healthy is None:
                try:
                    from minivess.observability.health import check_mlflow_health

                    result = check_mlflow_health()
                    nonlocal_healthy = result.healthy
                except Exception:
                    nonlocal_healthy = False
                _mlflow_healthy = nonlocal_healthy

            if not _mlflow_healthy:
                item.add_marker(pytest.mark.skip(reason="MLflow server not reachable"))


# Suppress warnings that occur during import of third-party libraries.
# These must be set before the libraries are imported, so pytest's
# filterwarnings config (which applies after collection) is too late.
warnings.filterwarnings(
    "ignore",
    message=".*deprecated.*",
    category=DeprecationWarning,
    module="pyparsing.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module="MetricsReloaded.*",
)
