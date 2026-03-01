"""Dynaconf settings singleton for deployment configuration.

Provides a cached Dynaconf instance that reads from
``configs/deployment/settings.toml`` with environment-based overrides
(development.toml, staging.toml, production.toml).

Usage::

    from minivess.config.settings import get_settings

    settings = get_settings()
    print(settings.MLFLOW_TRACKING_URI)

Environment switching via ``ENV_FOR_DYNACONF``::

    ENV_FOR_DYNACONF=staging python -c "from minivess.config.settings import get_settings; print(get_settings().DEBUG)"
"""

from __future__ import annotations

import functools
from pathlib import Path

from dynaconf import Dynaconf

_DEPLOYMENT_DIR = Path(__file__).resolve().parents[3] / "configs" / "deployment"


@functools.lru_cache(maxsize=1)
def get_settings() -> Dynaconf:
    """Return the singleton Dynaconf settings instance.

    Reads from ``configs/deployment/settings.toml`` (base) and
    environment-specific overrides. Secrets from ``.secrets.toml``
    (gitignored) are loaded when present.

    Returns
    -------
    Dynaconf
        Configured settings object.
    """
    return Dynaconf(
        envvar_prefix="MINIVESS",
        settings_files=[
            str(_DEPLOYMENT_DIR / "settings.toml"),
            str(_DEPLOYMENT_DIR / "development.toml"),
            str(_DEPLOYMENT_DIR / "staging.toml"),
            str(_DEPLOYMENT_DIR / "production.toml"),
            str(_DEPLOYMENT_DIR / ".secrets.toml"),
        ],
        environments=True,
        env_switcher="ENV_FOR_DYNACONF",
        load_dotenv=False,
    )


def clear_settings_cache() -> None:
    """Reset the settings singleton (for test isolation)."""
    get_settings.cache_clear()
