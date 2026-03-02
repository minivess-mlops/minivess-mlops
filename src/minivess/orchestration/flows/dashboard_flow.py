"""Dashboard & Reporting Prefect Flow (Flow 5).

Best-effort 5th flow for paper-quality figures, markdown reports,
and metadata export. Failure does not block the core pipeline.

Uses ``_prefect_compat`` decorators for graceful degradation.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from minivess.orchestration import flow, task
from minivess.pipeline.viz.generate_all_figures import generate_all_figures

if TYPE_CHECKING:
    from pathlib import Path

    from minivess.orchestration.flows.dashboard_sections import (
        ConfigDashboardSection,
        DataDashboardSection,
        EverythingDashboard,
        ModelDashboardSection,
        PipelineDashboardSection,
    )
    from minivess.pipeline.comparison import ComparisonTable

logger = logging.getLogger(__name__)


@task(name="generate-figures")
def generate_figures_task(
    comparison_table: ComparisonTable,
    output_dir: Path,
) -> dict[str, list[str]]:
    """Generate all paper-quality figures from comparison data.

    Parameters
    ----------
    comparison_table:
        Cross-loss comparison results.
    output_dir:
        Directory for figure output.

    Returns
    -------
    Summary dict with 'succeeded' and 'failed' figure name lists.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, list[str]] = generate_all_figures(output_dir=output_dir)
    logger.info(
        "Figure generation: %d succeeded, %d failed",
        len(summary["succeeded"]),
        len(summary["failed"]),
    )
    return summary


@task(name="generate-report")
def generate_report_task(
    comparison_table: ComparisonTable,
    figure_summary: dict[str, list[str]],
    output_dir: Path,
) -> Path:
    """Generate a markdown summary report.

    Parameters
    ----------
    comparison_table:
        Cross-loss comparison results.
    figure_summary:
        Figure generation summary from generate_figures_task.
    output_dir:
        Directory for report output.

    Returns
    -------
    Path to the generated markdown report.
    """
    from pathlib import Path as _Path

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "dashboard_report.md"

    lines: list[str] = [
        "# Dashboard Report",
        "",
        f"**Generated:** {datetime.now(UTC).isoformat()}",
        "",
        "## Loss Function Summary",
        "",
        "| Loss | Metrics |",
        "|------|---------|",
    ]

    for lr in comparison_table.losses:
        metric_strs = []
        for name, summary in lr.metrics.items():
            metric_strs.append(f"{name}: {summary.mean:.3f} ± {summary.std:.3f}")
        lines.append(f"| {lr.loss_name} | {', '.join(metric_strs)} |")

    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"- **Succeeded:** {len(figure_summary['succeeded'])}",
            f"- **Failed:** {len(figure_summary['failed'])}",
            "",
        ]
    )

    if figure_summary["succeeded"]:
        lines.append("### Generated Figures")
        lines.append("")
        for name in figure_summary["succeeded"]:
            lines.append(f"- {name}")

    if figure_summary["failed"]:
        lines.append("")
        lines.append("### Failed Figures")
        lines.append("")
        for name in figure_summary["failed"]:
            lines.append(f"- {name}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Report saved: %s", report_path)
    return _Path(report_path)


@task(name="export-metadata")
def export_metadata_task(
    comparison_table: ComparisonTable,
    figure_summary: dict[str, list[str]],
    output_dir: Path,
) -> Path:
    """Export dashboard metadata as JSON for reproducibility.

    Parameters
    ----------
    comparison_table:
        Cross-loss comparison results.
    figure_summary:
        Figure generation summary.
    output_dir:
        Directory for metadata output.

    Returns
    -------
    Path to the JSON metadata file.
    """
    from pathlib import Path as _Path

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dashboard_metadata.json"

    metadata: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "figures": figure_summary,
        "losses": [
            {
                "name": lr.loss_name,
                "num_folds": lr.num_folds,
                "metrics": {
                    name: {
                        "mean": s.mean,
                        "std": s.std,
                        "ci_lower": s.ci_lower,
                        "ci_upper": s.ci_upper,
                    }
                    for name, s in lr.metrics.items()
                },
            }
            for lr in comparison_table.losses
        ],
    }

    json_path.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Metadata saved: %s", json_path)
    return _Path(json_path)


@flow(name="minivess-dashboard")
def run_dashboard_flow(
    comparison_table: ComparisonTable,
    output_dir: Path,
) -> dict[str, Any]:
    """Dashboard & Reporting Flow (Flow 5, best-effort).

    Generates paper-quality figures, markdown reports, and metadata JSON.
    Designed to be called after the analysis flow completes.

    Parameters
    ----------
    comparison_table:
        Cross-loss comparison results from the analysis flow.
    output_dir:
        Root directory for all dashboard outputs.

    Returns
    -------
    Summary dict with keys: figures, report_path, metadata_path.
    """
    logger.info("Starting dashboard flow → %s", output_dir)

    # Step 1: Generate all figures
    figure_summary = generate_figures_task(
        comparison_table=comparison_table,
        output_dir=output_dir / "figures",
    )

    # Step 2: Generate markdown report
    report_path = generate_report_task(
        comparison_table=comparison_table,
        figure_summary=figure_summary,
        output_dir=output_dir,
    )

    # Step 3: Export metadata
    metadata_path = export_metadata_task(
        comparison_table=comparison_table,
        figure_summary=figure_summary,
        output_dir=output_dir,
    )

    logger.info("Dashboard flow complete")
    return {
        "figures": figure_summary,
        "report_path": report_path,
        "metadata_path": metadata_path,
    }


# ---------------------------------------------------------------------------
# Everything Dashboard — section collection tasks (Tasks 4.2/4.3)
# ---------------------------------------------------------------------------


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


def generate_everything_report(
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


def export_everything_metadata(
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


@flow(name="minivess-everything-dashboard")
def run_everything_dashboard_flow(
    output_dir: Path,
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
    report_path = generate_everything_report(dashboard=dashboard, output_dir=output_dir)
    metadata_path = export_everything_metadata(
        dashboard=dashboard, output_dir=output_dir
    )

    logger.info("Everything dashboard flow complete")
    return {
        "dashboard": dashboard,
        "report_path": report_path,
        "metadata_path": metadata_path,
    }
