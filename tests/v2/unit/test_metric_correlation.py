"""Tests for metric correlation heatmap visualization (F1.5).

RED phase: Tests written before implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

from minivess.pipeline.comparison import (
    ComparisonTable,
    LossResult,
    MetricSummary,
)

if TYPE_CHECKING:
    from pathlib import Path

# Use non-interactive backend for tests
matplotlib.use("Agg")


def _make_metric_summary(
    mean: float, std: float = 0.02, n_folds: int = 3
) -> MetricSummary:
    import numpy as np

    rng = np.random.default_rng(42)
    per_fold = [mean + rng.normal(0, std) for _ in range(n_folds)]
    return MetricSummary(
        mean=mean,
        std=std,
        ci_lower=mean - 0.04,
        ci_upper=mean + 0.04,
        per_fold=per_fold,
    )


def _make_comparison_table() -> ComparisonTable:
    losses = []
    metric_names = ["dsc", "centreline_dsc", "measured_masd"]
    base_values = {
        "dice_ce": {"dsc": 0.82, "centreline_dsc": 0.83, "measured_masd": 0.15},
        "cbdice": {"dsc": 0.78, "centreline_dsc": 0.80, "measured_masd": 0.18},
        "dice_ce_cldice": {"dsc": 0.80, "centreline_dsc": 0.88, "measured_masd": 0.12},
        "cbdice_cldice": {"dsc": 0.79, "centreline_dsc": 0.91, "measured_masd": 0.13},
    }
    for loss_name, values in base_values.items():
        metrics = {m: _make_metric_summary(values[m]) for m in metric_names}
        losses.append(LossResult(loss_name=loss_name, num_folds=3, metrics=metrics))
    return ComparisonTable(losses=losses, metric_names=metric_names)


class TestPlotMetricCorrelation:
    """Tests for the metric correlation heatmap (F1.5)."""

    def test_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.metric_correlation import plot_metric_correlation

        table = _make_comparison_table()
        fig = plot_metric_correlation(table)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_heatmap_is_square(self) -> None:
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.metric_correlation import plot_metric_correlation

        table = _make_comparison_table()
        fig = plot_metric_correlation(table)
        ax = fig.get_axes()[0]
        # Square heatmap: #xticks == #yticks == #metrics
        n_metrics = len(table.metric_names)
        assert len(ax.get_xticklabels()) == n_metrics
        assert len(ax.get_yticklabels()) == n_metrics
        plt.close(fig)

    def test_saves_to_disk(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.figure_export import save_figure
        from minivess.pipeline.viz.metric_correlation import plot_metric_correlation

        table = _make_comparison_table()
        fig = plot_metric_correlation(table)
        path = save_figure(fig, "f1_5_correlation", output_dir=tmp_path)
        assert path is not None
        assert (tmp_path / "f1_5_correlation.png").is_file()
        plt.close(fig)
