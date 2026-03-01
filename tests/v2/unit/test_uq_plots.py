"""Tests for UQ visualizations (F3.1–F3.5).

RED phase: Tests written before implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

matplotlib.use("Agg")


def _make_uq_df() -> pd.DataFrame:
    """Synthetic UQ decomposition DataFrame."""
    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    strategies = [
        "per_loss_single_best",
        "all_loss_single_best",
        "per_loss_all_best",
        "all_loss_all_best",
    ]
    for strategy in strategies:
        total = rng.uniform(0.05, 0.3)
        aleatoric = total * rng.uniform(0.3, 0.7)
        epistemic = total - aleatoric
        rows.append(
            {
                "strategy": strategy,
                "total_uncertainty": total,
                "aleatoric": aleatoric,
                "epistemic": epistemic,
            }
        )
    return pd.DataFrame(rows)


def _make_calibration_df() -> pd.DataFrame:
    """Synthetic calibration data: predicted probability bins vs observed frequency."""
    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    losses = ["dice_ce", "cbdice", "dice_ce_cldice", "cbdice_cldice"]
    bins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for loss in losses:
        for b in bins:
            rows.append(
                {
                    "loss_function": loss,
                    "predicted_prob": b,
                    "observed_freq": b + rng.normal(0, 0.05),
                }
            )
    return pd.DataFrame(rows)


class TestPlotUqDecomposition:
    """Tests for UQ decomposition stacked bars (F3.1)."""

    def test_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.uq_plots import plot_uq_decomposition

        df = _make_uq_df()
        fig = plot_uq_decomposition(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_bars(self) -> None:
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.uq_plots import plot_uq_decomposition

        df = _make_uq_df()
        fig = plot_uq_decomposition(df)
        ax = fig.get_axes()[0]
        assert len(ax.containers) > 0 or len(ax.patches) > 0
        plt.close(fig)

    def test_saves_to_disk(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.figure_export import save_figure
        from minivess.pipeline.viz.uq_plots import plot_uq_decomposition

        df = _make_uq_df()
        fig = plot_uq_decomposition(df)
        path = save_figure(fig, "f3_1_uq", output_dir=tmp_path)
        assert path is not None
        plt.close(fig)


class TestPlotCalibrationCurve:
    """Tests for calibration curve (F3.2)."""

    def test_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.uq_plots import plot_calibration_curve

        df = _make_calibration_df()
        fig = plot_calibration_curve(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_diagonal_reference(self) -> None:
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.uq_plots import plot_calibration_curve

        df = _make_calibration_df()
        fig = plot_calibration_curve(df)
        ax = fig.get_axes()[0]
        # At least diagonal + one line per loss
        assert len(ax.lines) >= 2
        plt.close(fig)
