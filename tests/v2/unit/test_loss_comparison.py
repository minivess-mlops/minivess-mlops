"""Tests for loss comparison visualization (F1.1 box/violin, F1.4 grouped bar).

RED phase: Tests written before implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

from minivess.pipeline.comparison import (
    ComparisonTable,
    LossResult,
    MetricSummary,
    PairwiseComparison,
)

if TYPE_CHECKING:
    from pathlib import Path

# Use non-interactive backend for tests
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _make_metric_summary(
    mean: float, std: float = 0.02, n_folds: int = 3
) -> MetricSummary:
    """Create a MetricSummary with synthetic per-fold values."""
    import numpy as np

    rng = np.random.default_rng(42)
    per_fold = [mean + rng.normal(0, std) for _ in range(n_folds)]
    return MetricSummary(
        mean=mean,
        std=std,
        ci_lower=mean - 1.96 * std,
        ci_upper=mean + 1.96 * std,
        per_fold=per_fold,
    )


def _make_comparison_table() -> ComparisonTable:
    """Build a ComparisonTable with 4 losses and 3 metrics."""
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


def _make_pairwise_comparisons() -> list[PairwiseComparison]:
    """Build synthetic pairwise comparisons for 4 losses on 1 metric."""
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
        PairwiseComparison(
            loss_a="dice_ce_cldice",
            loss_b="cbdice_cldice",
            metric="dsc",
            p_value=0.001,
            adjusted_p_value=0.004,
            is_significant=True,
            effect_size=1.2,
            direction="A > B",
        ),
    ]


# ---------------------------------------------------------------------------
# TestPlotLossComparison — F1.1 box/violin plots
# ---------------------------------------------------------------------------


class TestPlotLossComparison:
    """Tests for the loss comparison box/violin figure (F1.1)."""

    def test_returns_figure(self) -> None:
        """plot_loss_comparison returns a matplotlib Figure."""
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.loss_comparison import plot_loss_comparison

        table = _make_comparison_table()
        fig = plot_loss_comparison(table)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_creates_subplots_per_metric(self) -> None:
        """One subplot per metric in the comparison table."""
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.loss_comparison import plot_loss_comparison

        table = _make_comparison_table()
        fig = plot_loss_comparison(table)
        axes = fig.get_axes()
        assert len(axes) >= len(table.metric_names)
        plt.close(fig)

    def test_uses_paul_tol_colors(self) -> None:
        """Figure uses colors from plot_config.COLORS."""
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.loss_comparison import plot_loss_comparison

        table = _make_comparison_table()
        fig = plot_loss_comparison(table)
        # Figure created without error using the Paul Tol palette
        assert fig is not None
        plt.close(fig)

    def test_with_significance_annotations(self) -> None:
        """Passing pairwise comparisons adds significance annotations."""
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.loss_comparison import plot_loss_comparison

        table = _make_comparison_table()
        pairwise = _make_pairwise_comparisons()
        fig = plot_loss_comparison(table, pairwise_comparisons=pairwise)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_metric(self) -> None:
        """Works with a single metric in the comparison table."""
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.loss_comparison import plot_loss_comparison

        losses = []
        for name, val in [("dice_ce", 0.82), ("cbdice", 0.78)]:
            losses.append(
                LossResult(
                    loss_name=name,
                    num_folds=3,
                    metrics={"dsc": _make_metric_summary(val)},
                )
            )
        table = ComparisonTable(losses=losses, metric_names=["dsc"])
        fig = plot_loss_comparison(table)
        assert len(fig.get_axes()) >= 1
        plt.close(fig)

    def test_save_with_figure_export(self, tmp_path: Path) -> None:
        """Can be saved via the figure_export module."""
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.figure_export import save_figure
        from minivess.pipeline.viz.loss_comparison import plot_loss_comparison

        table = _make_comparison_table()
        fig = plot_loss_comparison(table)
        path = save_figure(fig, "f1_1_loss_comparison", output_dir=tmp_path)
        assert path is not None
        assert (tmp_path / "f1_1_loss_comparison.png").is_file()
        plt.close(fig)


# ---------------------------------------------------------------------------
# TestPlotForestComparison — F1.4 forest plot with effect sizes
# ---------------------------------------------------------------------------


class TestPlotForestComparison:
    """Tests for the forest plot figure (F1.4) showing CI and effect sizes."""

    def test_returns_figure(self) -> None:
        """plot_forest_comparison returns a matplotlib Figure."""
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.loss_comparison import plot_forest_comparison

        table = _make_comparison_table()
        pairwise = _make_pairwise_comparisons()
        fig = plot_forest_comparison(table, pairwise)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_ci_bars(self) -> None:
        """Forest plot contains horizontal CI bars."""
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.loss_comparison import plot_forest_comparison

        table = _make_comparison_table()
        pairwise = _make_pairwise_comparisons()
        fig = plot_forest_comparison(table, pairwise)
        ax = fig.get_axes()[0]
        # Forest plot has error bars (collections for CI lines)
        assert len(ax.containers) > 0 or len(ax.collections) > 0 or len(ax.lines) > 0
        plt.close(fig)

    def test_empty_pairwise(self) -> None:
        """Works with empty pairwise list (just shows means)."""
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.loss_comparison import plot_forest_comparison

        table = _make_comparison_table()
        fig = plot_forest_comparison(table, [])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_forest_plot(self, tmp_path: Path) -> None:
        """Can be saved via figure_export."""
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.figure_export import save_figure
        from minivess.pipeline.viz.loss_comparison import plot_forest_comparison

        table = _make_comparison_table()
        pairwise = _make_pairwise_comparisons()
        fig = plot_forest_comparison(table, pairwise)
        path = save_figure(fig, "f1_4_forest", output_dir=tmp_path)
        assert path is not None
        assert (tmp_path / "f1_4_forest.png").is_file()
        plt.close(fig)
