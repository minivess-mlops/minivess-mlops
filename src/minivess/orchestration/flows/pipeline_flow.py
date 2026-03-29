"""Pipeline Orchestrator Flow — Docker-native full pipeline execution.

Triggers all flows sequentially via Prefect's run_deployment() API.
Each flow runs in its own Docker container — no Python imports between flows.
MLflow is the sole inter-flow contract for data and artifact exchange.

Pipeline order:
  1. acquisition (core)
  2. data (core)
  3. train (core)
  4. post_training (optional)
  5. analysis (core)
  6. biostatistics (optional)
  7. deploy (core)
  8. dashboard (best-effort)

Core flow failure stops the pipeline. Optional/best-effort flows can fail
without blocking downstream flows. QA was merged into the dashboard health
adapter (#342, PR #567).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prefect import flow, task
from prefect.deployments import run_deployment

if TYPE_CHECKING:
    from prefect.client.schemas.objects import FlowRun

from minivess.orchestration.constants import (
    FLOW_NAME_ACQUISITION,
    FLOW_NAME_ANALYSIS,
    FLOW_NAME_BIOSTATISTICS,
    FLOW_NAME_DASHBOARD,
    FLOW_NAME_DATA,
    FLOW_NAME_DEPLOY,
    FLOW_NAME_PIPELINE,
    FLOW_NAME_POST_TRAINING,
    FLOW_NAME_TRAIN,
)
from minivess.observability.flow_observability import flow_observability_context
from minivess.orchestration.docker_guard import require_docker_context
from minivess.observability.prefect_hooks import create_task_timing_hooks

logger = logging.getLogger(__name__)

_on_complete, _on_fail = create_task_timing_hooks()


@dataclass(frozen=True)
class PipelineStepResult:
    """Result of a single flow step in the pipeline."""

    flow_name: str
    deployment_name: str
    status: str  # "success", "failed", "skipped"
    duration_s: float
    error: str | None = None


@dataclass
class PipelineResult:
    """Result of the full pipeline execution."""

    steps: list[PipelineStepResult] = field(default_factory=list)
    trigger_source: str = "manual"
    core_failed: bool = False

    @property
    def n_succeeded(self) -> int:
        return sum(1 for s in self.steps if s.status == "success")

    @property
    def n_failed(self) -> int:
        return sum(1 for s in self.steps if s.status == "failed")

    @property
    def n_skipped(self) -> int:
        return sum(1 for s in self.steps if s.status == "skipped")


# Pipeline step definitions: (deployment_name, is_core)
# Deployment name format: "{flow-name}/default" matching deployments.yaml
_PIPELINE_STEPS: list[tuple[str, bool]] = [
    (f"{FLOW_NAME_ACQUISITION}/default", True),
    (f"{FLOW_NAME_DATA}/default", True),
    (f"{FLOW_NAME_TRAIN}/default", True),
    (f"{FLOW_NAME_POST_TRAINING}/default", False),
    (f"{FLOW_NAME_ANALYSIS}/default", True),
    (f"{FLOW_NAME_BIOSTATISTICS}/default", False),
    (f"{FLOW_NAME_DEPLOY}/default", True),
    (f"{FLOW_NAME_DASHBOARD}/default", False),
]


@task(name="run-pipeline-step", on_completion=[_on_complete], on_failure=[_on_fail])
def run_pipeline_step(
    deployment_name: str,
    trigger_source: str,
    timeout: int = 86400,
) -> PipelineStepResult:
    """Trigger a single flow via Prefect run_deployment().

    Parameters
    ----------
    deployment_name:
        Prefect deployment name (e.g., "training-flow/default").
    trigger_source:
        What triggered this pipeline (passed to each flow).
    timeout:
        Max seconds to wait for the flow to complete.

    Returns
    -------
    PipelineStepResult with status and timing.
    """
    flow_name = deployment_name.split("/")[0]
    start = time.monotonic()

    try:
        # run_deployment() returns FlowRun in synchronous @task context.
        # The Coroutine variant only applies in async context.
        flow_run: FlowRun = run_deployment(  # type: ignore[assignment]
            name=deployment_name,
            parameters={"trigger_source": trigger_source},
            timeout=timeout,
        )
        duration = time.monotonic() - start

        if flow_run.state and flow_run.state.is_failed():
            return PipelineStepResult(
                flow_name=flow_name,
                deployment_name=deployment_name,
                status="failed",
                duration_s=duration,
                error=f"Flow run {flow_run.id} failed",
            )

        logger.info("Pipeline step '%s' succeeded in %.1fs", flow_name, duration)
        return PipelineStepResult(
            flow_name=flow_name,
            deployment_name=deployment_name,
            status="success",
            duration_s=duration,
        )

    except Exception as exc:
        duration = time.monotonic() - start
        logger.warning("Pipeline step '%s' failed: %s", flow_name, exc)
        return PipelineStepResult(
            flow_name=flow_name,
            deployment_name=deployment_name,
            status="failed",
            duration_s=duration,
            error=str(exc),
        )


@flow(name=FLOW_NAME_PIPELINE)
def pipeline_flow(
    skip_flows: list[str] | None = None,
    trigger_source: str = "manual",
    step_timeout: int = 86400,
) -> dict[str, Any]:
    """Full pipeline orchestrator — triggers all flows via Prefect deployments.

    Each flow runs in its own Docker container. Core flow failure stops the
    pipeline; optional/best-effort flows can fail without blocking.

    Parameters
    ----------
    skip_flows:
        Flow names to skip (e.g., ["post_training", "biostatistics"]).
    trigger_source:
        What triggered this pipeline run.
    step_timeout:
        Max seconds per flow step (default: 24h).

    Returns
    -------
    Dict with pipeline results summary.
    """
    require_docker_context("pipeline")

    logs_dir = Path(os.environ.get("LOGS_DIR", "/app/logs"))
    with flow_observability_context("pipeline", logs_dir=logs_dir) as event_logger:
        if skip_flows is None:
            skip_flows = []

        logger.info(
            "Pipeline started (trigger: %s, skip: %s)",
            trigger_source,
            skip_flows,
        )

        result = PipelineResult(trigger_source=trigger_source)

        for deployment_name, is_core in _PIPELINE_STEPS:
            flow_name = deployment_name.split("/")[0]

            # Skip if requested
            if flow_name in skip_flows:
                result.steps.append(
                    PipelineStepResult(
                        flow_name=flow_name,
                        deployment_name=deployment_name,
                        status="skipped",
                        duration_s=0.0,
                        error="Skipped by user",
                    )
                )
                continue

            # Skip core flows after a core failure
            if result.core_failed and is_core:
                result.steps.append(
                    PipelineStepResult(
                        flow_name=flow_name,
                        deployment_name=deployment_name,
                        status="skipped",
                        duration_s=0.0,
                        error="Skipped due to prior core failure",
                    )
                )
                continue

            # Execute the step
            step_result = run_pipeline_step(
                deployment_name=deployment_name,
                trigger_source=trigger_source,
                timeout=step_timeout,
            )
            result.steps.append(step_result)

            if step_result.status == "failed" and is_core:
                result.core_failed = True
                logger.warning(
                    "Core flow '%s' failed — skipping remaining core flows",
                    flow_name,
                )

        logger.info(
            "Pipeline complete: %d succeeded, %d failed, %d skipped",
            result.n_succeeded,
            result.n_failed,
            result.n_skipped,
        )

        return {
            "trigger_source": trigger_source,
            "n_succeeded": result.n_succeeded,
            "n_failed": result.n_failed,
            "n_skipped": result.n_skipped,
            "core_failed": result.core_failed,
            "steps": [
                {
                    "flow_name": s.flow_name,
                    "status": s.status,
                    "duration_s": s.duration_s,
                    "error": s.error,
                }
                for s in result.steps
            ],
        }


if __name__ == "__main__":
    pipeline_flow()
