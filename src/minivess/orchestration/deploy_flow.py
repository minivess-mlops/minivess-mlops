"""Deploy Flow (Flow 4): BentoML + MLflow + MONAI Deploy.

Orchestrates the deployment pipeline:
1. Discover champion models from MLflow filesystem tags
2. Export champions to ONNX format
3. Import ONNX models into BentoML model store
4. Generate deployment artifacts (bentofile, docker-compose, README)
5. Promote models through the registry (DEVELOPMENT -> PRODUCTION)
6. Optionally package for MONAI Deploy MAP

Each step is a Prefect @task; the flow assembles them with dependency
tracking. Uses ``_prefect_compat.py`` for graceful degradation when
Prefect is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from minivess.orchestration._prefect_compat import flow, get_run_logger, task

if TYPE_CHECKING:
    from pathlib import Path

    from minivess.config.deploy_config import DeployConfig
    from minivess.pipeline.deploy_champion_discovery import ChampionModel

logger = logging.getLogger(__name__)


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


@flow(name="Deploy Pipeline")
def deploy_flow(
    config: DeployConfig,
    experiment_id: str = "1",
) -> DeployResult:
    """Orchestrate the full deployment pipeline.

    Parameters
    ----------
    config:
        Deploy flow configuration.
    experiment_id:
        MLflow experiment ID to search for champions.

    Returns
    -------
    :class:`DeployResult` with all deployment artifacts and status.
    """
    log = get_run_logger()
    log.info("Starting deploy flow for experiment %s", experiment_id)

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
                log.exception("ONNX export failed for %s — skipping", champion.run_id)
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
    )

    log.info("Deploy flow complete: %s", result.to_summary())
    return result
