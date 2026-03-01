"""Tests for factorial design visualizations (F2.1–F2.5).

RED phase: Tests written before implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import pandas as pd

from minivess.pipeline.comparison import (
    ComparisonTable,
    LossResult,
    MetricSummary,
    PairwiseComparison,
)

if TYPE_CHECKING:
    from pathlib import Path

matplotlib.use("Agg")


def _make_metric_summary(mean: float, std: float = 0.02) -> MetricSummary:
    import numpy as np

    rng = np.random.default_rng(42)
    per_fold = [mean + rng.normal(0, std) for _ in range(3)]
    return MetricSummary(
        mean=mean,
        std=std,
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
        "dice_ce_cldice": {"dsc": 0.80, "centreline_dsc": 0.88},
        "cbdice_cldice": {"dsc": 0.79, "centreline_dsc": 0.91},
    }
    for name, vals in base.items():
        metrics = {m: _make_metric_summary(vals[m]) for m in metric_names}
        losses.append(LossResult(loss_name=name, num_folds=3, metrics=metrics))
    return ComparisonTable(losses=losses, metric_names=metric_names)


def _make_config_df() -> pd.DataFrame:
    """Synthetic config DataFrame for parallel coordinates / spec curve."""
    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    losses = ["dice_ce", "cbdice", "dice_ce_cldice", "cbdice_cldice"]
    for loss in losses:
        for fold in range(3):
            rows.append(
                {
                    "loss_function": loss,
                    "fold_id": fold,
                    "architecture": "DynUNet",
                    "primary_metric": 0.75 + rng.uniform(0, 0.15),
                    "ci_lower": 0.70 + rng.uniform(0, 0.10),
                    "ci_upper": 0.85 + rng.uniform(0, 0.10),
                }
            )
    return pd.DataFrame(rows)


def _make_pairwise() -> list[PairwiseComparison]:
    return [
        PairwiseComparison(
            loss_a="dice_ce",
            loss_b="cbdice",
            metric="dsc",
            p_value=0.03,
            adjusted_p_value=0.09,
            is_significant=False,
            effect_size=0.5,
            direction="A > B",
        ),
    ]


class TestPlotSpecificationCurve:
    """Tests for the specification curve plot (F2.2)."""

    def test_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.factorial_analysis import plot_specification_curve

        df = _make_config_df()
        fig = plot_specification_curve(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_two_panels(self) -> None:
        """Spec curve has top scatter + bottom binary matrix."""
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.factorial_analysis import plot_specification_curve

        df = _make_config_df()
        fig = plot_specification_curve(df)
        assert len(fig.get_axes()) >= 2
        plt.close(fig)

    def test_saves_to_disk(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.factorial_analysis import plot_specification_curve
        from minivess.pipeline.viz.figure_export import save_figure

        df = _make_config_df()
        fig = plot_specification_curve(df)
        path = save_figure(fig, "f2_2_spec_curve", output_dir=tmp_path)
        assert path is not None
        plt.close(fig)


class TestPlotParallelCoordinates:
    """Tests for the parallel coordinates plot (F2.1)."""

    def test_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.factorial_analysis import plot_parallel_coordinates

        df = _make_config_df()
        fig = plot_parallel_coordinates(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotSensitivityHeatmap:
    """Tests for the sensitivity heatmap (F2.3)."""

    def test_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.factorial_analysis import plot_sensitivity_heatmap

        table = _make_comparison_table()
        fig = plot_sensitivity_heatmap(table)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_heatmap(self) -> None:
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.factorial_analysis import plot_sensitivity_heatmap

        table = _make_comparison_table()
        fig = plot_sensitivity_heatmap(table)
        ax = fig.get_axes()[0]
        assert len(ax.collections) > 0 or len(ax.images) > 0
        plt.close(fig)
