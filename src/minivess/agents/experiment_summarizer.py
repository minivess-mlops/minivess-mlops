"""Experiment summarizer — Pydantic AI agent for analysis_flow.

Replaces DeterministicExperimentSummary with LLM-powered natural-language
experiment summarization. Falls back to deterministic stub when the [agents]
extra is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext

from minivess.agents.config import AgentConfig, load_agent_config
from minivess.agents.factory import make_prefect_agent
from minivess.agents.models import ExperimentSummary


@dataclass
class AnalysisContext:
    """Dependencies injected into the experiment summarizer agent.

    Parameters
    ----------
    experiment_name:
        MLflow experiment name.
    n_models:
        Number of models evaluated.
    best_model:
        Best performing model identifier.
    best_metric_value:
        Best metric value achieved.
    all_results:
        List of result dicts from all models.
    """

    experiment_name: str = ""
    n_models: int = 0
    best_model: str = "unknown"
    best_metric_value: float = 0.0
    all_results: list[dict[str, Any]] | None = None


_SYSTEM_PROMPT = """\
You are an expert ML experiment analyst for a biomedical image segmentation project.
Analyze the provided experiment results and produce a structured summary.

Focus on:
- Which model/loss combination performed best and why
- Key performance differences between configurations
- Actionable recommendations for the next experiment iteration

Be concise, quantitative, and specific. Use metric values in your findings.
"""


def _build_agent(model: str | None = None) -> Agent[AnalysisContext, ExperimentSummary]:
    """Build the experiment summarizer Agent (without PrefectAgent wrapper)."""
    config = load_agent_config()
    model_name = model or config.model

    agent = Agent(
        model_name,
        output_type=ExperimentSummary,
        deps_type=AnalysisContext,
        name="experiment-summarizer",
        system_prompt=_SYSTEM_PROMPT,
    )

    @agent.tool
    def fetch_run_metrics(ctx: RunContext[AnalysisContext]) -> dict[str, Any]:
        """Fetch experiment run metrics from the analysis context."""
        deps = ctx.deps
        results = deps.all_results or []
        return {
            "experiment_name": deps.experiment_name,
            "n_models": deps.n_models,
            "best_model": deps.best_model,
            "best_metric_value": deps.best_metric_value,
            "results": results[:10],  # Limit to avoid token bloat
        }

    return agent


def create_summarizer(
    model: str | None = None,
    config: AgentConfig | None = None,
) -> Any:
    """Create experiment summarizer as PrefectAgent for durable execution.

    Parameters
    ----------
    model:
        Override model identifier. Uses AgentConfig default if None.
    config:
        Override AgentConfig. Uses load_agent_config() if None.

    Returns
    -------
    PrefectAgent wrapping the experiment summarizer.
    """
    agent = _build_agent(model=model)
    return make_prefect_agent(agent, config=config)
