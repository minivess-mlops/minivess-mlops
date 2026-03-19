"""Pydantic AI dashboard agent for the MinIVess agentic dashboard.

Provides a conversational interface for researchers to query MLflow
experiments, aggregate metrics via DuckDB, and retrieve OpenLineage
provenance events. The agent uses Pydantic AI for structured output
and is wired to the AG-UI protocol via the ag_ui_adapter module.

Architecture:
    Dashboard UI → AG-UI → AGUIAdapter → this agent → MLflow/DuckDB/Marquez
    (CopilotKit)                         (Pydantic AI)

The agent is a micro-orchestration component (Pydantic AI) running within
the macro-orchestration layer (Prefect Flow 5 — dashboard_flow).
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from minivess.agents.config import AgentConfig, load_agent_config

logger = logging.getLogger(__name__)


@dataclass
class DashboardContext:
    """Dependencies injected into the dashboard agent.

    Parameters
    ----------
    tracking_uri:
        MLflow tracking URI for experiment queries.
    marquez_url:
        Marquez API URL for OpenLineage event retrieval.
    experiment_name:
        Default MLflow experiment name to query.
    """

    tracking_uri: str = ""
    marquez_url: str = ""
    experiment_name: str = "minivess_training"


class DashboardResponse(BaseModel):
    """Structured response from the dashboard agent.

    Parameters
    ----------
    answer:
        Natural language answer to the researcher's question.
    sources:
        List of data sources consulted (e.g., MLflow run IDs, experiments).
    confidence:
        Confidence in the answer (0.0 = low, 1.0 = high).
    """

    answer: str = Field(description="Natural language answer to the query")
    sources: list[str] = Field(
        default_factory=list,
        description="Data sources consulted (MLflow run IDs, experiment names)",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.8,
        description="Confidence in the answer",
    )


_SYSTEM_PROMPT = """\
You are a research assistant for the MinIVess biomedical image segmentation MLOps platform.
You help PhD researchers query experiment results, compare model performance, and
understand data lineage.

You have access to three tools:
1. query_mlflow_experiments — Query MLflow for experiment runs and metrics
2. aggregate_metrics — Aggregate metrics using DuckDB SQL
3. get_lineage_events — Retrieve OpenLineage provenance events from Marquez

Be concise, quantitative, and reference specific run IDs and metric values.
When comparing models, always mention the loss function and architecture.
"""


def _build_agent(
    model: Any = None,
) -> Agent[DashboardContext, DashboardResponse]:
    """Build the dashboard agent (without PrefectAgent wrapper).

    Parameters
    ----------
    model:
        Override model — either a string identifier or a Model instance
        (e.g., TestModel for testing). Uses AgentConfig default if None.

    Returns
    -------
    Pydantic AI Agent configured for dashboard queries.
    """
    config = load_agent_config()
    resolved_model = model if model is not None else config.model

    agent: Agent[DashboardContext, DashboardResponse] = Agent(
        resolved_model,
        output_type=DashboardResponse,
        deps_type=DashboardContext,
        name="dashboard-agent",
        system_prompt=_SYSTEM_PROMPT,
    )

    @agent.tool
    def query_mlflow_experiments(
        ctx: RunContext[DashboardContext],
        experiment_name: str = "",
    ) -> dict[str, Any]:
        """Query MLflow experiments for runs and metrics.

        Parameters
        ----------
        ctx:
            Agent context with tracking URI.
        experiment_name:
            Experiment name to query. Uses default if empty.

        Returns
        -------
        Dict with experiment info and recent runs.
        """
        deps = ctx.deps
        uri = deps.tracking_uri
        exp_name = experiment_name or deps.experiment_name

        if not uri:
            return {
                "status": "no_tracking_uri",
                "message": "MLflow tracking URI not configured",
            }

        try:
            url = f"{uri.rstrip('/')}/api/2.0/mlflow/experiments/list"
            req = urllib.request.Request(url, method="GET")  # noqa: S310
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
                experiments = data.get("experiments", [])

                matching = [e for e in experiments if e.get("name") == exp_name]
                return {
                    "status": "ok",
                    "experiment_name": exp_name,
                    "n_experiments": len(experiments),
                    "matching_experiments": matching[:5],
                }
        except Exception as exc:  # noqa: BLE001
            logger.debug("MLflow query failed: %s", exc)
            return {
                "status": "error",
                "message": str(exc),
            }

    @agent.tool
    def aggregate_metrics(
        ctx: RunContext[DashboardContext],
        sql_query: str = "",
    ) -> dict[str, Any]:
        """Aggregate experiment metrics using DuckDB SQL.

        Parameters
        ----------
        ctx:
            Agent context (unused in current implementation).
        sql_query:
            SQL query string for DuckDB aggregation. Currently returns
            a placeholder — full DuckDB integration is deferred.

        Returns
        -------
        Dict with aggregation results or status.
        """
        if not sql_query:
            return {
                "status": "no_query",
                "message": "No SQL query provided for aggregation",
            }

        # DuckDB aggregation is a placeholder for now — the full implementation
        # will query MLflow metrics exported to Parquet/DuckDB.
        return {
            "status": "placeholder",
            "message": (
                "DuckDB aggregation not yet connected to MLflow metrics. "
                "Query received: " + sql_query[:200]
            ),
            "sql_query": sql_query[:200],
        }

    @agent.tool
    def get_lineage_events(
        ctx: RunContext[DashboardContext],
        job_name: str = "",
    ) -> dict[str, Any]:
        """Retrieve OpenLineage provenance events from Marquez.

        Parameters
        ----------
        ctx:
            Agent context with Marquez URL.
        job_name:
            Optional job name to filter lineage events.

        Returns
        -------
        Dict with lineage events or status.
        """
        deps = ctx.deps
        marquez_url = deps.marquez_url

        if not marquez_url:
            return {
                "status": "no_marquez_url",
                "message": "Marquez URL not configured — lineage unavailable",
            }

        try:
            endpoint = "api/v1/lineage"
            if job_name:
                endpoint = f"api/v1/jobs/{job_name}/lineage"
            url = f"{marquez_url.rstrip('/')}/{endpoint}"
            req = urllib.request.Request(url, method="GET")  # noqa: S310
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
                return {
                    "status": "ok",
                    "lineage": data,
                }
        except Exception as exc:  # noqa: BLE001
            logger.debug("Marquez query failed: %s", exc)
            return {
                "status": "error",
                "message": str(exc),
            }

    return agent


def create_dashboard_agent(
    model: Any = None,
    config: AgentConfig | None = None,
) -> Agent[DashboardContext, DashboardResponse]:
    """Create the dashboard agent for interactive experiment querying.

    Parameters
    ----------
    model:
        Override model — string identifier or Model instance (e.g.,
        TestModel for testing). Uses AgentConfig default if None.
    config:
        Override AgentConfig. Not used directly for raw Agent creation
        but available for future PrefectAgent wrapping.

    Returns
    -------
    Pydantic AI Agent configured for dashboard queries.
    """
    return _build_agent(model=model)
