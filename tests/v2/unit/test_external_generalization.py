"""Tests for external generalization visualizations (F5.1–F5.3).

RED phase: Tests written before implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

matplotlib.use("Agg")


def _make_external_df() -> pd.DataFrame:
    """Synthetic external evaluation DataFrame."""
    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    datasets = ["deepvess", "tubenet_2pm"]
    losses = ["dice_ce", "cbdice", "dice_ce_cldice", "cbdice_cldice"]
    metrics = ["dsc", "centreline_dsc", "measured_masd"]

    for dataset in datasets:
        for loss in losses:
            for metric in metrics:
                rows.append(
                    {
                        "dataset": dataset,
                        "loss_function": loss,
                        "metric_name": metric,
                        "value": rng.uniform(0.4, 0.9),
                        "reference_cv_mean": rng.uniform(0.75, 0.92),
                    }
                )
    return pd.DataFrame(rows)


def _make_scatter_df() -> pd.DataFrame:
    """Synthetic per-volume scatter data."""
    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    losses = ["dice_ce", "cbdice"]
    for loss in losses:
        for vol_id in range(10):
            rows.append(
                {
                    "volume_id": f"vol_{vol_id:03d}",
                    "loss_function": loss,
                    "dsc": rng.uniform(0.5, 0.95),
                    "centreline_dsc": rng.uniform(0.4, 0.9),
                }
            )
    return pd.DataFrame(rows)


class TestPlotDomainGap:
    """Tests for the domain gap degradation plot (F5.1)."""

    def test_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.external_generalization import plot_domain_gap

        df = _make_external_df()
        fig = plot_domain_gap(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_bars(self) -> None:
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.external_generalization import plot_domain_gap

        df = _make_external_df()
        fig = plot_domain_gap(df)
        ax = fig.get_axes()[0]
        assert len(ax.containers) > 0 or len(ax.patches) > 0
        plt.close(fig)

    def test_saves_to_disk(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.external_generalization import plot_domain_gap
        from minivess.pipeline.viz.figure_export import save_figure

        df = _make_external_df()
        fig = plot_domain_gap(df)
        path = save_figure(fig, "f5_1_domain_gap", output_dir=tmp_path)
        assert path is not None
        plt.close(fig)


class TestPlotPerVolumeScatter:
    """Tests for per-volume scatter (F5.2)."""

    def test_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.external_generalization import (
            plot_per_volume_scatter,
        )

        df = _make_scatter_df()
        fig = plot_per_volume_scatter(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_scatter_points(self) -> None:
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.external_generalization import (
            plot_per_volume_scatter,
        )

        df = _make_scatter_df()
        fig = plot_per_volume_scatter(df)
        ax = fig.get_axes()[0]
        assert len(ax.collections) > 0
        plt.close(fig)
