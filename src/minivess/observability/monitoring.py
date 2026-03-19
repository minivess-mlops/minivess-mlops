"""Optional Sentry + PostHog monitoring stubs.

Both services are DISABLED by default (empty env vars = no-op).
Both packages are OPTIONAL dependencies — lazy import, never ImportError in CI.

Environment variables (defined in .env.example):
    SENTRY_DSN  — Sentry Data Source Name. Empty = disabled.
    POSTHOG_KEY — PostHog project API key. Empty = disabled.

Usage::

    from minivess.observability.monitoring import init_monitoring

    result = init_monitoring()
    # result == {"sentry": True/False, "posthog": True/False}
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def init_sentry() -> bool:
    """Initialize Sentry error tracking if SENTRY_DSN is set.

    Returns
    -------
    bool
        True if Sentry was initialized, False otherwise (empty DSN or
        missing sentry-sdk package).
    """
    dsn = os.environ.get("SENTRY_DSN", "")
    if not dsn:
        logger.debug("SENTRY_DSN not set — Sentry disabled")
        return False

    try:
        import sentry_sdk

        sentry_sdk.init(dsn=dsn, traces_sample_rate=0.1)
        logger.info("Sentry initialized (DSN=%s...)", dsn[:20])
    except (ImportError, ModuleNotFoundError):
        logger.warning(
            "SENTRY_DSN set but sentry-sdk not installed — Sentry disabled. "
            "Install with: uv add sentry-sdk"
        )
        return False
    else:
        return True


def init_posthog() -> bool:
    """Initialize PostHog product analytics if POSTHOG_KEY is set.

    Returns
    -------
    bool
        True if PostHog was initialized, False otherwise (empty key or
        missing posthog package).
    """
    key = os.environ.get("POSTHOG_KEY", "")
    if not key:
        logger.debug("POSTHOG_KEY not set — PostHog disabled")
        return False

    try:
        import posthog  # noqa: F401

        # PostHog Python SDK uses module-level config
        posthog.project_api_key = key
        posthog.host = os.environ.get("POSTHOG_HOST", "https://app.posthog.com")
        logger.info("PostHog initialized (key=%s...)", key[:8])
    except (ImportError, ModuleNotFoundError):
        logger.warning(
            "POSTHOG_KEY set but posthog not installed — PostHog disabled. "
            "Install with: uv add posthog"
        )
        return False
    else:
        return True


def init_monitoring() -> dict[str, bool]:
    """Initialize all optional monitoring services.

    Convenience wrapper that calls :func:`init_sentry` and
    :func:`init_posthog`. Safe to call unconditionally — disabled
    services are no-ops.

    Returns
    -------
    dict[str, bool]
        Mapping of service name to initialization success.
    """
    return {
        "sentry": init_sentry(),
        "posthog": init_posthog(),
    }
