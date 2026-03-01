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
