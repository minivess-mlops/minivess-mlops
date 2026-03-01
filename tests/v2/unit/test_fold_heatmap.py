"""Tests for fold heatmap visualization (F1.2).

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


# ---------------------------------------------------------------------------
# TestPlotFoldHeatmap — F1.2
# ---------------------------------------------------------------------------


class TestPlotFoldHeatmap:
    """Tests for the fold × metric heatmap (F1.2)."""

    def test_returns_figure(self) -> None:
        """plot_fold_heatmap returns a matplotlib Figure."""
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.fold_heatmap import plot_fold_heatmap

        table = _make_comparison_table()
        fig = plot_fold_heatmap(table)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_heatmap_axes(self) -> None:
        """Figure contains at least one axes with a color mesh."""
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.fold_heatmap import plot_fold_heatmap

        table = _make_comparison_table()
        fig = plot_fold_heatmap(table)
        ax = fig.get_axes()[0]
        # seaborn heatmap creates QuadMesh in collections
        assert len(ax.collections) > 0 or len(ax.images) > 0
        plt.close(fig)

    def test_rows_match_loss_fold_count(self) -> None:
        """Heatmap has rows = n_losses × n_folds."""
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.fold_heatmap import plot_fold_heatmap

        table = _make_comparison_table()
        fig = plot_fold_heatmap(table)
        ax = fig.get_axes()[0]
        ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
        expected_rows = sum(lr.num_folds for lr in table.losses)
        assert len(ytick_labels) == expected_rows
        plt.close(fig)

    def test_saves_to_disk(self, tmp_path: Path) -> None:
        """Can be saved via figure_export."""
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.figure_export import save_figure
        from minivess.pipeline.viz.fold_heatmap import plot_fold_heatmap

        table = _make_comparison_table()
        fig = plot_fold_heatmap(table)
        path = save_figure(fig, "f1_2_heatmap", output_dir=tmp_path)
        assert path is not None
        assert (tmp_path / "f1_2_heatmap.png").is_file()
        plt.close(fig)
