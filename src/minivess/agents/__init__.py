"""Agents — Pydantic AI agents for automated ML workflow decisions.

Public API:
  - AgentConfig, load_agent_config — configuration
  - make_prefect_agent — factory for durable Prefect execution
  - EvalResult, EvalSuite — Braintrust evaluation framework (orthogonal)

Legacy LangGraph code is in agents/_deprecated/ (see #341).
"""

from __future__ import annotations

from minivess.agents.config import AgentConfig, load_agent_config
from minivess.agents.evaluation import (
    EvalResult,
    EvalSuite,
    build_agent_eval_suite,
    build_segmentation_eval_suite,
)
from minivess.agents.factory import make_prefect_agent

__all__ = [
    "AgentConfig",
    "EvalResult",
    "EvalSuite",
    "build_agent_eval_suite",
    "build_segmentation_eval_suite",
    "load_agent_config",
    "make_prefect_agent",
]
