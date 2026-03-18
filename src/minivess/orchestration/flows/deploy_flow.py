"""Deploy Flow (Flow 4): BentoML + MLflow + MONAI Deploy.

Orchestrates the deployment pipeline:
1. Discover champion models from MLflow filesystem tags
2. Export champions to ONNX format
3. Import ONNX models into BentoML model store
4. Generate deployment artifacts (bentofile, docker-compose, README)
5. Promote models through the registry (DEVELOPMENT -> PRODUCTION)
6. Optionally package for MONAI Deploy MAP

Each step is a Prefect @task; the flow assembles them with dependency
tracking. Uses Prefect @flow and @task decorators for orchestration.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prefect import flow, get_run_logger, task

from minivess.config.deploy_config import DeployConfig
from minivess.observability.lineage import LineageEmitter, emit_flow_lineage
from minivess.observability.tracking import resolve_tracking_uri
from minivess.orchestration.constants import FLOW_NAME_DEPLOY
from minivess.orchestration.mlflow_helpers import (
    find_upstream_safely,
    log_completion_safe,
)

if TYPE_CHECKING:
    from minivess.pipeline.deploy_champion_discovery import ChampionModel

logger = logging.getLogger(__name__)


def _require_docker_context() -> None:
    """Require Docker container context or MINIVESS_ALLOW_HOST=1."""
    if os.environ.get("MINIVESS_ALLOW_HOST") == "1":
        return
    if os.environ.get("DOCKER_CONTAINER"):
        return
    if Path("/.dockerenv").exists():
        return
    raise RuntimeError(
        "Deploy flow must run inside a Docker container.\n"
        "Run: docker compose -f deployment/docker-compose.flows.yml run deploy\n"
        "Escape hatch for tests: MINIVESS_ALLOW_HOST=1"
    )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DeployResult:
    """Result of the deploy flow.

    Parameters
    ----------
    champions:
        Discovered champion models.
    onnx_paths:
        Category -> ONNX file path mapping.
    bento_tags:
        Category -> BentoML model tag mapping.
    artifacts_dir:
        Directory containing generated artifacts.
    promotion_results:
        Category -> promotion approved (bool) mapping.
    audit_trails:
        List of audit trail dicts.
    """

    champions: list[ChampionModel]
    onnx_paths: dict[str, Path]
    bento_tags: dict[str, str]
    artifacts_dir: Path
    promotion_results: dict[str, bool]
    audit_trails: list[dict[str, Any]] = field(default_factory=list)
    failed_operations: list[str] = field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        """Return a summary dict for logging and reporting."""
        return {
            "num_champions": len(self.champions),
            "onnx_models": {k: str(v) for k, v in self.onnx_paths.items()},
            "bento_models": dict(self.bento_tags),
            "promotions": dict(self.promotion_results),
            "artifacts_dir": str(self.artifacts_dir),
        }


# ---------------------------------------------------------------------------
# Prefect tasks
# ---------------------------------------------------------------------------


@task(name="discover-champions")
def discover_task(
    config: DeployConfig,
    experiment_id: str,
) -> list[ChampionModel]:
    """Discover champion models from MLflow filesystem tags."""
    from minivess.pipeline.deploy_champion_discovery import discover_champions

    log = get_run_logger()
    champions = discover_champions(
        config.mlruns_dir,
        experiment_id,
        categories=list(config.champion_categories),
    )
    log.info("Discovered %d champion(s)", len(champions))
    return champions


@task(name="export-onnx")
def export_task(
    champion: ChampionModel,
    output_dir: Path,
    *,
    opset_version: int = 17,
    input_shape: tuple[int, ...] = (1, 1, 32, 32, 16),
) -> Path:
    """Export a champion model to ONNX format."""
    from minivess.pipeline.deploy_onnx_export import export_champion_to_onnx

    log = get_run_logger()
    onnx_path = export_champion_to_onnx(
        champion, output_dir, opset_version=opset_version, input_shape=input_shape
    )
    log.info("Exported ONNX: %s -> %s", champion.run_id, onnx_path)
    return onnx_path


@task(name="import-bento")
def import_task(
    champion: ChampionModel,
    onnx_path: Path,
) -> str:
    """Import an ONNX model into BentoML model store."""
    from minivess.serving.bento_model_import import import_champion_to_bento

    log = get_run_logger()
    result = import_champion_to_bento(champion, onnx_path)
    log.info("Imported to BentoML: %s", result.tag)
    return result.tag


@task(name="generate-artifacts")
def generate_artifacts_task(
    champions: list[ChampionModel],
    bento_tags: dict[str, str],
    output_dir: Path,
    service_name: str,
) -> Path:
    """Generate deployment artifacts (bentofile, docker-compose, README)."""
    from minivess.serving.deploy_artifacts import (
        generate_bentofile,
        generate_deployment_readme,
        generate_docker_compose,
    )

    log = get_run_logger()
    output_dir.mkdir(parents=True, exist_ok=True)

    models = list(bento_tags.values())
    generate_bentofile(service_name, models, output_dir)

    services = [
        {"name": f"minivess-{cat}", "port": 3000 + i, "model_tag": tag}
        for i, (cat, tag) in enumerate(bento_tags.items())
    ]
    generate_docker_compose(services, output_dir)
    generate_deployment_readme(champions, output_dir)

    log.info("Generated deployment artifacts in %s", output_dir)
    return output_dir


@task(name="promote-model")
def promote_task(
    champion: ChampionModel,
) -> bool:
    """Promote a champion model through the registry."""
    from minivess.observability.model_registry import (
        ModelRegistry,
        PromotionCriteria,
    )
    from minivess.pipeline.deploy_promotion import promote_champion_for_deploy

    log = get_run_logger()
    registry = ModelRegistry()
    criteria = PromotionCriteria(
        min_thresholds={"dsc": 0.0},
    )
    result = promote_champion_for_deploy(champion, registry, criteria)
    log.info(
        "Promotion %s for %s: %s",
        "approved" if result.approved else "rejected",
        champion.category,
        result.reason,
    )
    return bool(result.approved)


# ---------------------------------------------------------------------------
# Deploy flow
# ---------------------------------------------------------------------------


@flow(name=FLOW_NAME_DEPLOY)
def deploy_flow(
    config: DeployConfig | None = None,
    experiment_id: str = "1",
    upstream_analysis_run_id: str | None = None,
    trigger_source: str = "manual",
) -> DeployResult:
    """Orchestrate the full deployment pipeline.

    Parameters
    ----------
    config:
        Deploy flow configuration. If None, built from environment variables
        via ``DeployConfig.from_env()`` (reads MLFLOW_TRACKING_URI).
    experiment_id:
        MLflow experiment ID to search for champions.

    Returns
    -------
    :class:`DeployResult` with all deployment artifacts and status.
    """
    _require_docker_context()

    if config is None:
        config = DeployConfig.from_env()

    log = get_run_logger()
    log.info("Starting deploy flow for experiment %s", experiment_id)

    # Ensure BentoML uses mounted volume instead of container-ephemeral ~/.bentoml
    os.environ.setdefault("BENTOML_HOME", "/home/minivess/bentoml")

    # 1. Discover champions
    champions = discover_task(config, experiment_id)

    if not champions:
        log.warning("No champions found — returning empty result")
        config.output_dir.mkdir(parents=True, exist_ok=True)
        return DeployResult(
            champions=[],
            onnx_paths={},
            bento_tags={},
            artifacts_dir=config.output_dir,
            promotion_results={},
        )

    # 2. Export each champion to ONNX
    onnx_dir = config.output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    onnx_paths: dict[str, Path] = {}
    failed_operations: list[str] = []
    for champion in champions:
        if champion.checkpoint_path is not None:
            try:
                onnx_path = export_task(
                    champion,
                    onnx_dir,
                    opset_version=config.onnx_opset,
                )
                onnx_paths[champion.category] = onnx_path
            except Exception:
                log.exception("ONNX export failed for %s", champion.run_id)
                failed_operations.append(
                    f"onnx_export:{champion.category}:{champion.run_id}"
                )
        else:
            log.warning(
                "No checkpoint for champion %s — skipping ONNX export",
                champion.run_id,
            )

    # 3. Import to BentoML
    bento_tags: dict[str, str] = {}
    for category, onnx_path in onnx_paths.items():
        matching = [c for c in champions if c.category == category]
        if matching:
            try:
                tag = import_task(matching[0], onnx_path)
                bento_tags[category] = tag
            except Exception:
                log.exception("BentoML import failed for %s — skipping", category)

    # 4. Generate deployment artifacts
    artifacts_dir = config.output_dir / "artifacts"
    generate_artifacts_task(
        champions, bento_tags, artifacts_dir, config.bento_service_name
    )

    # 5. Promote models
    promotion_results: dict[str, bool] = {}
    audit_trails: list[dict[str, Any]] = []
    for champion in champions:
        try:
            approved = promote_task(champion)
            promotion_results[champion.category] = approved

            from minivess.pipeline.deploy_promotion import (
                create_deployment_audit_trail,
            )

            trail = create_deployment_audit_trail(
                champion=champion,
                onnx_path=onnx_paths.get(champion.category, config.output_dir / "none"),
                bento_tag=bento_tags.get(champion.category, "unknown"),
                promotion_approved=approved,
            )
            audit_trails.append(trail)
        except Exception:
            log.exception("Promotion failed for %s — skipping", champion.category)

    result = DeployResult(
        champions=champions,
        onnx_paths=onnx_paths,
        bento_tags=bento_tags,
        artifacts_dir=artifacts_dir,
        promotion_results=promotion_results,
        audit_trails=audit_trails,
        failed_operations=failed_operations,
    )

    log.info("Deploy flow complete: %s", result.to_summary())

    # --- FlowContract: tag run and log completion ---
    _tracking_uri = resolve_tracking_uri()
    # Use provided upstream ID or auto-discover from MLflow
    if upstream_analysis_run_id is None:
        upstream = find_upstream_safely(
            tracking_uri=_tracking_uri,
            experiment_name="minivess_training",
            upstream_flow="analyze",
        )
        upstream_analysis_run_id = upstream["run_id"] if upstream else None
    mlflow_run_id: str | None = None
    try:
        import mlflow

        mlflow.set_tracking_uri(_tracking_uri)
        mlflow.set_experiment("minivess_training")
        with mlflow.start_run(
            tags={
                "flow_name": "deploy-flow",
                "upstream_analysis_run_id": upstream_analysis_run_id,
            }
        ) as active_run:
            mlflow_run_id = active_run.info.run_id
    except Exception:
        log.warning("Failed to log deploy_flow to MLflow", exc_info=True)

    # Log flow completion (best-effort, non-blocking) via FlowContract.log_flow_completion()
    log_completion_safe(
        flow_name="deploy-flow",
        tracking_uri=_tracking_uri,
        run_id=mlflow_run_id,
    )

    # OpenLineage lineage emission (Issue #799 — IEC 62304 §8 traceability)
    try:
        _emitter = LineageEmitter(namespace="minivess")
        emit_flow_lineage(
            emitter=_emitter,
            job_name="deploy-flow",
            inputs=[
                {"namespace": "minivess", "name": "champion_model"},
                {"namespace": "minivess", "name": "mlflow_registry"},
            ],
            outputs=[
                {"namespace": "minivess", "name": "onnx_export"},
                {"namespace": "minivess", "name": "bentoml_model"},
            ],
        )
    except Exception:
        log.warning("OpenLineage emission failed (non-blocking)", exc_info=True)

    return result


if __name__ == "__main__":
    deploy_flow()
