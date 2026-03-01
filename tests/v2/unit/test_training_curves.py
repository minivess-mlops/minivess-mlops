"""Tests for training curves visualization (F1.3).

RED phase: Tests written before implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

# Use non-interactive backend for tests
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _make_training_history() -> pd.DataFrame:
    """Build a synthetic training history DataFrame.

    Columns: epoch, loss_function, metric_name, value, fold_id
    Simulates 4 losses × 3 folds × 10 epochs × 2 metrics.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    losses = ["dice_ce", "cbdice", "dice_ce_cldice", "cbdice_cldice"]
    metrics = ["val_loss", "val_dice"]

    for loss in losses:
        for fold in range(3):
            for epoch in range(1, 11):
                for metric in metrics:
                    base = (
                        0.5 + epoch * 0.04 if "dice" in metric else 0.8 - epoch * 0.03
                    )
                    val = base + rng.normal(0, 0.02)
                    rows.append(
                        {
                            "epoch": epoch,
                            "loss_function": loss,
                            "metric_name": metric,
                            "value": val,
                            "fold_id": fold,
                        }
                    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TestPlotTrainingCurves — F1.3
# ---------------------------------------------------------------------------


class TestPlotTrainingCurves:
    """Tests for the training curves line plot (F1.3)."""

    def test_returns_figure(self) -> None:
        """plot_training_curves returns a matplotlib Figure."""
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.training_curves import plot_training_curves

        df = _make_training_history()
        fig = plot_training_curves(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_one_subplot_per_metric(self) -> None:
        """Creates one subplot per unique metric_name."""
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.training_curves import plot_training_curves

        df = _make_training_history()
        fig = plot_training_curves(df)
        n_metrics = df["metric_name"].nunique()
        visible_axes = [ax for ax in fig.get_axes() if ax.get_visible()]
        assert len(visible_axes) >= n_metrics
        plt.close(fig)

    def test_lines_per_loss(self) -> None:
        """Each loss function gets its own line in each subplot."""
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.training_curves import plot_training_curves

        df = _make_training_history()
        fig = plot_training_curves(df)
        ax = fig.get_axes()[0]
        # At least one line per loss function
        assert len(ax.lines) >= df["loss_function"].nunique()
        plt.close(fig)

    def test_saves_to_disk(self, tmp_path: Path) -> None:
        """Can be saved via figure_export."""
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.figure_export import save_figure
        from minivess.pipeline.viz.training_curves import plot_training_curves

        df = _make_training_history()
        fig = plot_training_curves(df)
        path = save_figure(fig, "f1_3_curves", output_dir=tmp_path)
        assert path is not None
        assert (tmp_path / "f1_3_curves.png").is_file()
        plt.close(fig)

    def test_single_metric(self) -> None:
        """Works when DataFrame has only one metric."""
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.training_curves import plot_training_curves

        df = _make_training_history()
        df = df[df["metric_name"] == "val_loss"]
        fig = plot_training_curves(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
