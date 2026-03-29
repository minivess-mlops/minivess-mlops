"""Dashboard & Reporting Prefect Flow (Flow 5).

Best-effort 5th flow for paper-quality figures, markdown reports,
and metadata export. Failure does not block the core pipeline.

Uses Prefect @flow and @task decorators for orchestration.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prefect import flow, task

from minivess.observability.tracking import resolve_tracking_uri
from minivess.orchestration.constants import (
    EXPERIMENT_TRAINING,
    FLOW_NAME_DASHBOARD,
    resolve_experiment_name,
)
from minivess.observability.flow_observability import flow_observability_context
from minivess.orchestration.docker_guard import require_docker_context

if TYPE_CHECKING:
    from minivess.orchestration.flows.dashboard_sections import (
        ConfigDashboardSection,
        DataDashboardSection,
        EverythingDashboard,
        ModelDashboardSection,
        PipelineDashboardSection,
    )

logger = logging.getLogger(__name__)


@task(name="collect-data-section")
def collect_data_section_task(
    n_volumes: int,
    quality_gate_passed: bool,
    external_datasets: dict[str, Any],
    drift_summary: dict[str, Any] | None = None,
) -> DataDashboardSection:
    """Collect data engineering section for the Everything Dashboard.

    Parameters
    ----------
    n_volumes:
        Number of primary dataset volumes.
    quality_gate_passed:
        Whether the data quality gate passed.
    external_datasets:
        Dict of external dataset name → pair count.
    drift_summary:
        Summary of drift analysis.
    """
    from minivess.orchestration.flows.dashboard_sections import (
        DataDashboardSection,
    )

    return DataDashboardSection(
        n_volumes=n_volumes,
        n_external_datasets=len(external_datasets),
        quality_gate_passed=quality_gate_passed,
        drift_summary=drift_summary or {},
        external_datasets=external_datasets,
    )


@task(name="collect-config-section")
def collect_config_section_task(
    environment: str,
    experiment_config: str,
    model_profile_name: str,
) -> ConfigDashboardSection:
    """Collect configuration section for the Everything Dashboard.

    Parameters
    ----------
    environment:
        Active Dynaconf environment name.
    experiment_config:
        Name of the experiment config file.
    model_profile_name:
        Name of the model profile.
    """
    from minivess.orchestration.flows.dashboard_sections import (
        ConfigDashboardSection,
    )

    # Attempt to read Dynaconf settings (best-effort)
    dynaconf_params: dict[str, Any] = {}
    try:
        from minivess.config.settings import get_settings

        settings = get_settings()
        dynaconf_params = {
            "debug": getattr(settings, "DEBUG", None),
            "project_name": getattr(settings, "PROJECT_NAME", None),
        }
    except Exception:
        logger.debug("Dynaconf unavailable for dashboard config section")

    return ConfigDashboardSection(
        environment=environment,
        dynaconf_params=dynaconf_params,
        experiment_config=experiment_config,
        model_profile_name=model_profile_name,
    )


@task(name="collect-model-section")
def collect_model_section_task(
    architecture_name: str,
    param_count: int,
    onnx_exported: bool,
    champion_category: str,
    loss_name: str,
) -> ModelDashboardSection:
    """Collect model section for the Everything Dashboard."""
    from minivess.orchestration.flows.dashboard_sections import (
        ModelDashboardSection,
    )

    return ModelDashboardSection(
        architecture_name=architecture_name,
        param_count=param_count,
        onnx_exported=onnx_exported,
        champion_category=champion_category,
        loss_name=loss_name,
    )


@task(name="collect-pipeline-section")
def collect_pipeline_section_task(
    flow_results: dict[str, str],
    last_data_version: str,
    last_training_run_id: str,
    trigger_source: str,
) -> PipelineDashboardSection:
    """Collect pipeline status section for the Everything Dashboard."""
    from minivess.orchestration.flows.dashboard_sections import (
        PipelineDashboardSection,
    )

    return PipelineDashboardSection(
        flow_results=flow_results,
        last_data_version=last_data_version,
        last_training_run_id=last_training_run_id,
        trigger_source=trigger_source,
    )


@task(name="collect-drift-section")
def collect_drift_section_task(
    *,
    tracking_uri: str,
    drift_run_id: str,
) -> dict[str, Any]:
    """Collect drift detection summaries for dashboard.

    Reads Tier 1 and Tier 2 drift artifacts from an MLflow run.

    Parameters
    ----------
    tracking_uri:
        MLflow tracking URI.
    drift_run_id:
        Run ID containing drift detection artifacts.

    Returns
    -------
    Dict with drift_tier1, drift_tier2, and status keys.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    result: dict[str, Any] = {"status": "no_drift_data"}
    try:
        artifacts = client.list_artifacts(drift_run_id, "drift_reports")
        artifact_paths = [a.path for a in artifacts]
    except Exception:  # noqa: BLE001
        logger.debug("No drift_reports artifacts for run %s", drift_run_id)
        return result

    if not artifact_paths:
        return result

    result["status"] = "ok"

    # Load Tier 1 summary
    tier1_matches = [p for p in artifact_paths if "tier1_drift_summary" in p]
    if tier1_matches:
        local_path = client.download_artifacts(drift_run_id, tier1_matches[0])
        tier1_data = json.loads(Path(local_path).read_text(encoding="utf-8"))
        result["drift_tier1"] = tier1_data

    # Load Tier 2 summary
    tier2_matches = [p for p in artifact_paths if "tier2_mmd_summary" in p]
    if tier2_matches:
        local_path = client.download_artifacts(drift_run_id, tier2_matches[0])
        tier2_data = json.loads(Path(local_path).read_text(encoding="utf-8"))
        result["drift_tier2"] = tier2_data

    # Load metadata
    metadata_matches = [p for p in artifact_paths if "drift_metadata" in p]
    if metadata_matches:
        local_path = client.download_artifacts(drift_run_id, metadata_matches[0])
        meta_data = json.loads(Path(local_path).read_text(encoding="utf-8"))
        result["drift_metadata"] = meta_data

    return result


def generate_report(
    dashboard: EverythingDashboard,
    output_dir: Path,
) -> Path:
    """Generate a markdown report from the Everything Dashboard.

    Parameters
    ----------
    dashboard:
        The complete dashboard with all sections.
    output_dir:
        Directory for report output.

    Returns
    -------
    Path to the generated markdown report.
    """
    from pathlib import Path as _Path

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "everything_dashboard_report.md"

    lines: list[str] = [
        "# Everything Dashboard",
        "",
        f"**Generated:** {dashboard.generated_at}",
        "",
    ]

    # Data section
    lines.extend(
        [
            "## Data",
            "",
            f"- **Volumes:** {dashboard.data.n_volumes}",
            f"- **External datasets:** {dashboard.data.n_external_datasets}",
            f"- **Quality gate:** {'PASSED' if dashboard.data.quality_gate_passed else 'FAILED'}",
            "",
        ]
    )
    if dashboard.data.drift_summary:
        lines.append("### Drift Summary")
        lines.append("")
        for drift_type, severity in dashboard.data.drift_summary.items():
            lines.append(f"- {drift_type}: {severity}")
        lines.append("")

    # Config section
    lines.extend(
        [
            "## Configuration",
            "",
            f"- **Environment:** {dashboard.config.environment}",
            f"- **Experiment config:** {dashboard.config.experiment_config}",
            f"- **Model profile:** {dashboard.config.model_profile_name}",
            "",
        ]
    )
    if dashboard.config.dynaconf_params:
        lines.append("### Dynaconf Parameters")
        lines.append("")
        for key, val in dashboard.config.dynaconf_params.items():
            lines.append(f"- {key}: {val}")
        lines.append("")

    # Model section
    lines.extend(
        [
            "## Model",
            "",
            f"- **Architecture:** {dashboard.model.architecture_name}",
            f"- **Parameters:** {dashboard.model.param_count:,}",
            f"- **Loss:** {dashboard.model.loss_name}",
            f"- **ONNX exported:** {dashboard.model.onnx_exported}",
            f"- **Champion category:** {dashboard.model.champion_category}",
            "",
        ]
    )

    # Pipeline section
    lines.extend(
        [
            "## Pipeline",
            "",
            f"- **Trigger source:** {dashboard.pipeline.trigger_source}",
            f"- **Data version:** {dashboard.pipeline.last_data_version}",
            f"- **Last training run:** {dashboard.pipeline.last_training_run_id}",
            "",
        ]
    )
    if dashboard.pipeline.flow_results:
        lines.append("### Flow Results")
        lines.append("")
        lines.append("| Flow | Status |")
        lines.append("|------|--------|")
        for flow_name, status in dashboard.pipeline.flow_results.items():
            lines.append(f"| {flow_name} | {status} |")
        lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Everything report saved: %s", report_path)
    return _Path(report_path)


def export_metadata(
    dashboard: EverythingDashboard,
    output_dir: Path,
) -> Path:
    """Export Everything Dashboard metadata as JSON.

    Parameters
    ----------
    dashboard:
        The complete dashboard.
    output_dir:
        Directory for metadata output.

    Returns
    -------
    Path to the JSON metadata file.
    """
    from dataclasses import asdict
    from pathlib import Path as _Path

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "everything_dashboard_metadata.json"

    metadata: dict[str, Any] = {
        "generated_at": dashboard.generated_at,
        "data": asdict(dashboard.data),
        "config": asdict(dashboard.config),
        "model": asdict(dashboard.model),
        "pipeline": asdict(dashboard.pipeline),
    }

    json_path.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Everything metadata saved: %s", json_path)
    return _Path(json_path)


@flow(name=FLOW_NAME_DASHBOARD)
def run_dashboard_flow(
    output_dir: Path | None = None,
    *,
    # Data section params
    n_volumes: int = 0,
    quality_gate_passed: bool = False,
    external_datasets: dict[str, Any] | None = None,
    drift_summary: dict[str, Any] | None = None,
    # Config section params
    environment: str = "unknown",
    experiment_config: str = "",
    model_profile_name: str = "",
    # Model section params
    architecture_name: str = "",
    param_count: int = 0,
    onnx_exported: bool = False,
    champion_category: str = "",
    loss_name: str = "",
    # Pipeline section params
    flow_results: dict[str, str] | None = None,
    last_data_version: str = "",
    last_training_run_id: str = "",
    trigger_source: str = "manual",
) -> dict[str, Any]:
    """Everything Dashboard Flow — expanded Flow 5.

    Collects data, config, model, and pipeline sections into a unified
    dashboard with markdown report and JSON metadata.

    Parameters
    ----------
    output_dir:
        Root directory for all dashboard outputs.
    """
    require_docker_context("dashboard")

    logs_dir = Path(os.environ.get("LOGS_DIR", "/app/logs"))
    with flow_observability_context("dashboard", logs_dir=logs_dir) as event_logger:
        if output_dir is None:
            output_dir = Path(
                os.environ.get("DASHBOARD_OUTPUT_DIR", "/app/outputs/dashboard")
            )
        logger.info("Starting everything dashboard flow → %s", output_dir)

        # Collect sections
        data_section = collect_data_section_task(
            n_volumes=n_volumes,
            quality_gate_passed=quality_gate_passed,
            external_datasets=external_datasets or {},
            drift_summary=drift_summary,
        )
        config_section = collect_config_section_task(
            environment=environment,
            experiment_config=experiment_config,
            model_profile_name=model_profile_name,
        )
        model_section = collect_model_section_task(
            architecture_name=architecture_name,
            param_count=param_count,
            onnx_exported=onnx_exported,
            champion_category=champion_category,
            loss_name=loss_name,
        )
        pipeline_section = collect_pipeline_section_task(
            flow_results=flow_results or {},
            last_data_version=last_data_version,
            last_training_run_id=last_training_run_id,
            trigger_source=trigger_source,
        )

        # Build everything dashboard
        from minivess.orchestration.flows.dashboard_sections import (
            EverythingDashboard,
        )

        dashboard = EverythingDashboard(
            data=data_section,
            config=config_section,
            model=model_section,
            pipeline=pipeline_section,
            generated_at=datetime.now(UTC).isoformat(),
        )

        # Generate outputs
        report_path = generate_report(dashboard=dashboard, output_dir=output_dir)
        metadata_path = export_metadata(dashboard=dashboard, output_dir=output_dir)

        logger.info("Everything dashboard flow complete")

        # --- FlowContract: tag run and log completion ---
        _tracking_uri = resolve_tracking_uri()
        mlflow_run_id: str | None = None
        try:
            import mlflow

            from minivess.orchestration.flow_contract import FlowContract

            mlflow.set_tracking_uri(_tracking_uri)
            mlflow.set_experiment(resolve_experiment_name(EXPERIMENT_TRAINING))
            with mlflow.start_run(tags={"flow_name": "dashboard"}) as active_run:
                mlflow_run_id = active_run.info.run_id

            contract = FlowContract(tracking_uri=_tracking_uri)
            contract.log_flow_completion(
                flow_name="dashboard",
                run_id=mlflow_run_id,
            )
        except Exception:
            logger.warning("Failed to log dashboard_flow to MLflow", exc_info=True)

        return {
            "dashboard": dashboard,
            "report_path": report_path,
            "metadata_path": metadata_path,
            "mlflow_run_id": mlflow_run_id,
        }


if __name__ == "__main__":
    run_dashboard_flow()
