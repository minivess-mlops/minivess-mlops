"""Figure narration agent — Pydantic AI agent for biostatistics_flow.

Replaces DeterministicFigureNarration with LLM-powered paper-quality
caption generation. Falls back to deterministic stub when [agents]
extra is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, RunContext

from minivess.agents.config import AgentConfig, load_agent_config
from minivess.agents.factory import make_prefect_agent
from minivess.agents.models import FigureCaption


@dataclass
class FigureContext:
    """Dependencies injected into the figure narration agent.

    Parameters
    ----------
    figure_type:
        Type of figure (e.g., "box_plot", "heatmap", "bar_chart").
    n_conditions:
        Number of experimental conditions compared.
    primary_metric:
        Primary metric shown in the figure.
    statistical_results:
        Statistical test results (p-values, effect sizes).
    axis_labels:
        Axis labels and ranges.
    """

    figure_type: str = "unknown"
    n_conditions: int = 0
    primary_metric: str = "unknown"
    statistical_results: dict[str, Any] = field(default_factory=dict)
    axis_labels: dict[str, str] = field(default_factory=dict)


_SYSTEM_PROMPT = """\
You are a scientific writing expert specializing in biomedical image analysis papers.
Generate publication-ready figure captions following academic conventions.

Guidelines:
- Start with what the figure shows (e.g., "Comparison of...")
- Include the number of conditions, samples, or folds
- Mention the primary metric and any statistical tests
- Keep captions concise (1-3 sentences)
- Use standard scientific notation for p-values
- Generate accessible alt text for screen readers
"""


def _build_agent(model: str | None = None) -> Agent[FigureContext, FigureCaption]:
    """Build the figure narration Agent (without PrefectAgent wrapper)."""
    config = load_agent_config()
    model_name = model or config.model

    agent = Agent(
        model_name,
        output_type=FigureCaption,
        deps_type=FigureContext,
        name="figure-narrator",
        system_prompt=_SYSTEM_PROMPT,
    )

    @agent.tool
    def get_figure_metadata(ctx: RunContext[FigureContext]) -> dict[str, Any]:
        """Get metadata about the figure being captioned."""
        deps = ctx.deps
        return {
            "figure_type": deps.figure_type,
            "n_conditions": deps.n_conditions,
            "primary_metric": deps.primary_metric,
            "axis_labels": deps.axis_labels,
        }

    @agent.tool
    def get_statistical_context(ctx: RunContext[FigureContext]) -> dict[str, Any]:
        """Get statistical test results for the figure."""
        return ctx.deps.statistical_results

    return agent


def create_figure_narrator(
    model: str | None = None,
    config: AgentConfig | None = None,
) -> Any:
    """Create figure narrator as PrefectAgent for durable execution.

    Parameters
    ----------
    model:
        Override model identifier.
    config:
        Override AgentConfig.

    Returns
    -------
    PrefectAgent wrapping the figure narrator agent.
    """
    agent = _build_agent(model=model)
    return make_prefect_agent(agent, config=config)
