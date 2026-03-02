"""Environment detection utilities.

Lightweight helpers for querying the active Dynaconf environment
and MLflow backend type without importing heavy dependencies.
"""

from __future__ import annotations


def is_mlflow_server_backend(tracking_uri: str) -> bool:
    """Return ``True`` if the tracking URI points to an HTTP server.

    Parameters
    ----------
    tracking_uri:
        MLflow tracking URI string.
    """
    return tracking_uri.startswith(("http://", "https://"))


def get_active_environment() -> str:
    """Return the active Dynaconf environment name.

    Falls back to ``"default"`` if Dynaconf is unavailable.
    """
    try:
        from minivess.config.settings import get_settings

        return str(getattr(get_settings(), "current_env", "default")).lower()
    except Exception:
        return "default"
