"""Agent factory — wraps Pydantic AI agents in PrefectAgent for durable execution.

Single factory function ensures consistent retry/timeout/caching behavior
across all agents in the project.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai.durable_exec.prefect import PrefectAgent, TaskConfig

if TYPE_CHECKING:
    from pydantic_ai import Agent

from minivess.agents.config import AgentConfig, load_agent_config


def make_prefect_agent(
    agent: Agent[Any, Any],
    config: AgentConfig | None = None,
) -> PrefectAgent[Any, Any]:
    """Wrap a Pydantic AI Agent for durable Prefect execution.

    Parameters
    ----------
    agent:
        The Pydantic AI Agent to wrap.
    config:
        Agent configuration. Uses load_agent_config() defaults if None.

    Returns
    -------
    PrefectAgent with TaskConfig wired from AgentConfig.
    """
    if config is None:
        config = load_agent_config()

    return PrefectAgent(
        agent,
        model_task_config=TaskConfig(
            retries=config.retries,
            retry_delay_seconds=config.retry_delay_seconds,
            timeout_seconds=config.timeout_seconds,
        ),
        tool_task_config=TaskConfig(
            retries=config.tool_retries,
            retry_delay_seconds=config.tool_retry_delay_seconds,
        ),
    )
