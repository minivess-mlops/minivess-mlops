"""AG-UI protocol adapter for MinIVess agentic dashboard.

Bridges Pydantic AI agents to the AG-UI streaming protocol used by
CopilotKit React frontend. The adapter translates agent tool calls,
text messages, and state updates into AG-UI Server-Sent Events.

Architecture:
    CopilotKit React SDK → HTTP POST (AG-UI) → this adapter → Pydantic AI Agent
    CopilotKit React SDK ← SSE stream (AG-UI) ← this adapter ← Pydantic AI Agent

References:
    - AG-UI Protocol: https://docs.ag-ui.com/introduction
    - Pydantic AI AG-UI: https://ai.pydantic.dev/ui/ag-ui/
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pydantic_ai import Agent


def create_ag_ui_endpoint(
    agent: Agent[Any, Any],
    *,
    deps: Any | None = None,
) -> Any:
    """Create a FastAPI POST endpoint that handles AG-UI protocol requests.

    Returns an async callable suitable for use as a FastAPI route handler.
    The endpoint accepts AG-UI RunAgentInput via HTTP POST and returns
    a streaming SSE response with AG-UI events.

    Parameters
    ----------
    agent:
        Pydantic AI Agent to wire to the AG-UI protocol.
    deps:
        Optional agent dependencies to inject.

    Returns
    -------
    Async callable that handles AG-UI requests (FastAPI route handler).
    """
    from pydantic_ai.ui.ag_ui import AGUIAdapter

    async def ag_ui_handler(request: Any) -> Any:
        """Handle incoming AG-UI protocol request."""
        from starlette.requests import Request

        if not isinstance(request, Request):
            msg = "Expected Starlette Request object"
            raise TypeError(msg)

        return await AGUIAdapter.dispatch_request(
            request,
            agent=agent,
            deps=deps,
        )

    return ag_ui_handler


def build_ag_ui_app(
    agent: Agent[Any, Any],
    *,
    deps: Any | None = None,
) -> Any:
    """Build a standalone ASGI application for AG-UI protocol handling.

    Returns an AGUIApp instance that can be mounted in a larger ASGI
    application or run standalone with uvicorn.

    Parameters
    ----------
    agent:
        Pydantic AI Agent to expose via AG-UI.
    deps:
        Optional agent dependencies.

    Returns
    -------
    AGUIApp ASGI application.
    """
    from pydantic_ai.ui.ag_ui.app import AGUIApp

    return AGUIApp(agent, deps=deps)
