"""Tests for Phase 0: Agent infrastructure (T-0.1, T-0.2, T-0.3)."""

from __future__ import annotations

import os
from unittest.mock import patch

# ---------------------------------------------------------------------------
# T-0.1: pydantic-ai importability
# ---------------------------------------------------------------------------


def test_pydantic_ai_importable():
    """Verify pydantic_ai.Agent is importable."""
    from pydantic_ai import Agent

    assert Agent is not None


def test_prefect_agent_importable():
    """Verify PrefectAgent is importable from durable_exec."""
    from pydantic_ai.durable_exec.prefect import PrefectAgent, TaskConfig

    assert PrefectAgent is not None
    assert TaskConfig is not None


def test_test_model_importable():
    """Verify TestModel is importable for CI testing."""
    from pydantic_ai.models.test import TestModel

    assert TestModel is not None


# ---------------------------------------------------------------------------
# T-0.2: AgentConfig
# ---------------------------------------------------------------------------


class TestAgentConfig:
    """Tests for AgentConfig Pydantic model."""

    def test_defaults(self):
        from minivess.agents.config import AgentConfig

        cfg = AgentConfig()
        assert cfg.model == "anthropic:claude-sonnet-4-6"
        assert cfg.temperature == 0.0
        assert cfg.retries == 3
        assert cfg.timeout_seconds == 30.0

    def test_custom_values(self):
        from minivess.agents.config import AgentConfig

        cfg = AgentConfig(model="ollama:qwen2.5-coder:14b", temperature=0.7)
        assert cfg.model == "ollama:qwen2.5-coder:14b"
        assert cfg.temperature == 0.7

    def test_env_override(self):
        from minivess.agents.config import load_agent_config

        with patch.dict(os.environ, {"MINIVESS_AGENT_MODEL": "openai:gpt-4o"}):
            cfg = load_agent_config()
            assert cfg.model == "openai:gpt-4o"

    def test_retry_settings(self):
        from minivess.agents.config import AgentConfig

        cfg = AgentConfig()
        assert cfg.retries >= 1
        assert len(cfg.retry_delay_seconds) >= 1
        assert cfg.tool_retries >= 1
        assert len(cfg.tool_retry_delay_seconds) >= 1


# ---------------------------------------------------------------------------
# T-0.3: Agent factory
# ---------------------------------------------------------------------------


class TestAgentFactory:
    """Tests for make_prefect_agent factory."""

    def test_returns_prefect_agent(self):
        from pydantic_ai import Agent
        from pydantic_ai.durable_exec.prefect import PrefectAgent

        from minivess.agents.factory import make_prefect_agent

        agent = Agent("test", name="test-agent")
        pa = make_prefect_agent(agent)
        assert isinstance(pa, PrefectAgent)

    def test_applies_retry_config(self):
        from pydantic_ai import Agent

        from minivess.agents.config import AgentConfig
        from minivess.agents.factory import make_prefect_agent

        cfg = AgentConfig(retries=5, retry_delay_seconds=[2.0, 4.0])
        agent = Agent("test", name="test-agent")
        pa = make_prefect_agent(agent, config=cfg)
        # PrefectAgent was constructed — verify it's usable
        assert pa is not None

    def test_default_config(self):
        from pydantic_ai import Agent

        from minivess.agents.factory import make_prefect_agent

        agent = Agent("test", name="test-agent")
        pa = make_prefect_agent(agent)
        assert pa is not None
