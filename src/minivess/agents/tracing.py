"""Langfuse + OpenTelemetry tracing for Pydantic AI agents.

Pydantic AI instruments via OpenTelemetry. Langfuse can receive OTEL traces
when configured as an OTEL exporter. This module provides a single
``configure_agent_tracing()`` entry point.

Gracefully degrades: if langfuse or opentelemetry are not installed,
tracing is silently disabled.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_tracing_configured = False


def configure_agent_tracing() -> bool:
    """Configure Pydantic AI agent tracing via Langfuse/OpenTelemetry.

    Call once at application startup (e.g., in the Prefect flow entry point).
    Subsequent calls are no-ops.

    Returns
    -------
    True if tracing was successfully configured, False otherwise.
    """
    global _tracing_configured  # noqa: PLW0603
    if _tracing_configured:
        return True

    try:
        from pydantic_ai import Agent
        from pydantic_ai.agent import InstrumentationSettings

        # Check if Langfuse OTEL exporter is available
        try:
            from langfuse.opentelemetry import LangfuseSpanProcessor

            # Langfuse auto-configures from LANGFUSE_PUBLIC_KEY,
            # LANGFUSE_SECRET_KEY, LANGFUSE_HOST env vars
            processor = LangfuseSpanProcessor()

            from opentelemetry.sdk.trace import TracerProvider

            provider = TracerProvider()
            provider.add_span_processor(processor)

            Agent.instrument_all(InstrumentationSettings(tracer_provider=provider))
            _tracing_configured = True
            logger.info("Agent tracing configured via Langfuse OTEL exporter")
            return True
        except ImportError:
            logger.debug("langfuse not installed — trying bare OTEL")

        # Fallback: bare OpenTelemetry (e.g., for Jaeger, Zipkin)
        try:
            from opentelemetry.sdk.trace import TracerProvider

            provider = TracerProvider()
            Agent.instrument_all(InstrumentationSettings(tracer_provider=provider))
            _tracing_configured = True
            logger.info("Agent tracing configured via OpenTelemetry")
            return True
        except ImportError:
            logger.debug("opentelemetry not installed — tracing disabled")

    except ImportError:
        logger.debug("pydantic-ai not installed — tracing disabled")

    return False


def reset_tracing() -> None:
    """Reset tracing state (for testing)."""
    global _tracing_configured  # noqa: PLW0603
    _tracing_configured = False


def get_tracing_status() -> dict[str, Any]:
    """Return current tracing configuration status.

    Returns
    -------
    Dict with ``enabled``, ``backend``, and ``configured`` keys.
    """
    backend = "none"
    if _tracing_configured:
        try:
            import langfuse  # noqa: F401

            backend = "langfuse"
        except ImportError:
            try:
                import opentelemetry  # noqa: F401

                backend = "opentelemetry"
            except ImportError:
                backend = "unknown"

    return {
        "enabled": _tracing_configured,
        "backend": backend,
        "configured": _tracing_configured,
    }
