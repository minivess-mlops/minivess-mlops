"""Agent configuration for Pydantic AI agents in the MinIVess pipeline.

Provides AgentConfig with sensible defaults and environment variable overrides.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Configuration for Pydantic AI agents.

    Parameters
    ----------
    model:
        Pydantic AI model identifier (e.g., "anthropic:claude-sonnet-4-6",
        "ollama:qwen2.5-coder:14b", "openai:gpt-4o").
    temperature:
        Sampling temperature for LLM calls.
    retries:
        Number of retries for LLM API calls.
    retry_delay_seconds:
        Exponential backoff delays for LLM retries.
    timeout_seconds:
        Timeout per LLM call.
    tool_retries:
        Number of retries for tool invocations.
    tool_retry_delay_seconds:
        Backoff delays for tool retries.
    """

    model: str = "anthropic:claude-sonnet-4-6"
    temperature: float = 0.0
    retries: int = 3
    retry_delay_seconds: list[float] = Field(default_factory=lambda: [1.0, 2.0, 4.0])
    timeout_seconds: float = 30.0
    tool_retries: int = 2
    tool_retry_delay_seconds: list[float] = Field(default_factory=lambda: [0.5, 1.0])


def load_agent_config() -> AgentConfig:
    """Load AgentConfig with environment variable overrides.

    Environment variables:
        MINIVESS_AGENT_MODEL — override model identifier
        MINIVESS_AGENT_TEMPERATURE — override temperature
        MINIVESS_AGENT_RETRIES — override retry count
        MINIVESS_AGENT_TIMEOUT — override timeout seconds
    """
    overrides: dict[str, Any] = {}

    if model := os.environ.get("MINIVESS_AGENT_MODEL"):
        overrides["model"] = model
    if temp := os.environ.get("MINIVESS_AGENT_TEMPERATURE"):
        overrides["temperature"] = float(temp)
    if retries := os.environ.get("MINIVESS_AGENT_RETRIES"):
        overrides["retries"] = int(retries)
    if timeout := os.environ.get("MINIVESS_AGENT_TIMEOUT"):
        overrides["timeout_seconds"] = float(timeout)

    return AgentConfig(**overrides)
