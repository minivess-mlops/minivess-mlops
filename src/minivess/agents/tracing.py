"""Langfuse tracing integration for LangGraph agent execution."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _get_langfuse_client() -> Any | None:
    """Get a Langfuse client, returning None if unavailable.

    Returns None if langfuse is not installed, not configured,
    or if the client is disabled (no credentials), allowing
    graceful degradation.
    """
    try:
        from langfuse import Langfuse

        client = Langfuse()
        # Verify the client is functional (has trace method and credentials)
        if not hasattr(client, "trace") or not callable(getattr(client, "trace", None)):
            logger.debug("Langfuse client disabled — no credentials")
            return None
        return client
    except Exception:  # noqa: BLE001
        logger.debug("Langfuse client unavailable — tracing disabled")
        return None


def traced_graph_run(
    graph: Any,
    state: dict[str, Any],
    *,
    trace_name: str = "agent_run",
) -> dict[str, Any]:
    """Run a LangGraph graph with optional Langfuse tracing.

    Parameters
    ----------
    graph:
        Compiled LangGraph StateGraph with .invoke() method.
    state:
        Initial state dict for the graph.
    trace_name:
        Name for the Langfuse trace.

    Returns
    -------
    Final state dict after graph execution.
    """
    client = _get_langfuse_client()

    if client is not None:
        trace = client.trace(name=trace_name)
        logger.info("Langfuse trace started: %s", trace_name)
        try:
            result = graph.invoke(state)
            trace.update(output=str(result.get("status", "unknown")))
            return dict(result)
        except Exception:
            trace.update(output="error")
            raise
    else:
        return dict(graph.invoke(state))
