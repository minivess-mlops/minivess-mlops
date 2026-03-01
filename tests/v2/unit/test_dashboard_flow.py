"""Tests for the Dashboard & Reporting Prefect Flow (Flow 5).

RED phase: Tests written before implementation.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import matplotlib

from minivess.pipeline.comparison import (
    ComparisonTable,
    LossResult,
    MetricSummary,
)

if TYPE_CHECKING:
    from pathlib import Path

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _make_metric_summary(mean: float) -> MetricSummary:
    import numpy as np

    rng = np.random.default_rng(42)
    per_fold = [mean + rng.normal(0, 0.02) for _ in range(3)]
    return MetricSummary(
        mean=mean,
        std=0.02,
        ci_lower=mean - 0.04,
        ci_upper=mean + 0.04,
        per_fold=per_fold,
    )


def _make_comparison_table() -> ComparisonTable:
    losses = []
    metric_names = ["dsc", "centreline_dsc"]
    base = {
        "dice_ce": {"dsc": 0.82, "centreline_dsc": 0.83},
        "cbdice": {"dsc": 0.78, "centreline_dsc": 0.80},
    }
    for name, vals in base.items():
        metrics = {m: _make_metric_summary(vals[m]) for m in metric_names}
        losses.append(LossResult(loss_name=name, num_folds=3, metrics=metrics))
    return ComparisonTable(losses=losses, metric_names=metric_names)


# ---------------------------------------------------------------------------
# TestGenerateFiguresTask
# ---------------------------------------------------------------------------


class TestGenerateFiguresTask:
    """Tests for the generate-figures dashboard task."""

    def test_returns_summary_dict(self, tmp_path: Path) -> None:
        """generate_figures_task returns a dict with succeeded/failed."""
        import matplotlib.pyplot as plt

        from minivess.orchestration.flows.dashboard_flow import generate_figures_task

        table = _make_comparison_table()
        result = generate_figures_task(
            comparison_table=table,
            output_dir=tmp_path,
        )
        assert isinstance(result, dict)
        assert "succeeded" in result
        assert "failed" in result
        plt.close("all")

    def test_creates_figure_files(self, tmp_path: Path) -> None:
        """Task creates PNG files in the output directory."""
        import matplotlib.pyplot as plt

        from minivess.orchestration.flows.dashboard_flow import generate_figures_task

        table = _make_comparison_table()
        result = generate_figures_task(
            comparison_table=table,
            output_dir=tmp_path,
        )
        # At least one figure should succeed
        assert len(result["succeeded"]) > 0
        # Check that PNG files exist
        png_files = list(tmp_path.glob("*.png"))
        assert len(png_files) > 0
        plt.close("all")


# ---------------------------------------------------------------------------
# TestGenerateReportTask
# ---------------------------------------------------------------------------


class TestGenerateReportTask:
    """Tests for the markdown report generation task."""

    def test_returns_report_path(self, tmp_path: Path) -> None:
        """generate_report_task returns a Path to the report file."""
        from minivess.orchestration.flows.dashboard_flow import generate_report_task

        table = _make_comparison_table()
        figure_summary = {"succeeded": ["loss_comparison"], "failed": []}
        report_path = generate_report_task(
            comparison_table=table,
            figure_summary=figure_summary,
            output_dir=tmp_path,
        )
        assert report_path is not None
        assert report_path.exists()

    def test_report_is_markdown(self, tmp_path: Path) -> None:
        """Report file has .md extension and contains markdown."""
        from minivess.orchestration.flows.dashboard_flow import generate_report_task

        table = _make_comparison_table()
        figure_summary = {"succeeded": ["loss_comparison"], "failed": []}
        report_path = generate_report_task(
            comparison_table=table,
            figure_summary=figure_summary,
            output_dir=tmp_path,
        )
        assert report_path.suffix == ".md"
        content = report_path.read_text(encoding="utf-8")
        assert "# " in content  # Has headers


# ---------------------------------------------------------------------------
# TestExportMetadataTask
# ---------------------------------------------------------------------------


class TestExportMetadataTask:
    """Tests for the metadata export task."""

    def test_exports_json(self, tmp_path: Path) -> None:
        """export_metadata_task writes a JSON file."""
        from minivess.orchestration.flows.dashboard_flow import export_metadata_task

        table = _make_comparison_table()
        figure_summary = {"succeeded": ["loss_comparison"], "failed": []}
        json_path = export_metadata_task(
            comparison_table=table,
            figure_summary=figure_summary,
            output_dir=tmp_path,
        )
        assert json_path.exists()
        assert json_path.suffix == ".json"

    def test_json_has_required_fields(self, tmp_path: Path) -> None:
        """Exported JSON contains summary fields."""
        from minivess.orchestration.flows.dashboard_flow import export_metadata_task

        table = _make_comparison_table()
        figure_summary = {"succeeded": ["loss_comparison"], "failed": []}
        json_path = export_metadata_task(
            comparison_table=table,
            figure_summary=figure_summary,
            output_dir=tmp_path,
        )
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "generated_at" in data
        assert "figures" in data
        assert "losses" in data


# ---------------------------------------------------------------------------
# TestRunDashboardFlow
# ---------------------------------------------------------------------------


class TestRunDashboardFlow:
    """Tests for the dashboard flow orchestrator."""

    def test_returns_result_dict(self, tmp_path: Path) -> None:
        """run_dashboard_flow returns a summary dict."""
        import matplotlib.pyplot as plt

        from minivess.orchestration.flows.dashboard_flow import run_dashboard_flow

        table = _make_comparison_table()
        result = run_dashboard_flow(
            comparison_table=table,
            output_dir=tmp_path,
        )
        assert isinstance(result, dict)
        assert "figures" in result
        assert "report_path" in result
        assert "metadata_path" in result
        plt.close("all")

    def test_is_idempotent(self, tmp_path: Path) -> None:
        """Running twice produces consistent results."""
        import matplotlib.pyplot as plt

        from minivess.orchestration.flows.dashboard_flow import run_dashboard_flow

        table = _make_comparison_table()
        result1 = run_dashboard_flow(comparison_table=table, output_dir=tmp_path)
        result2 = run_dashboard_flow(comparison_table=table, output_dir=tmp_path)
        assert result1["figures"]["succeeded"] == result2["figures"]["succeeded"]
        plt.close("all")
