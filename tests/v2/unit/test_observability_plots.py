"""Tests for MLOps/observability visualizations (F4.1–F4.5).

RED phase: Tests written before implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

matplotlib.use("Agg")


def _make_drift_df() -> pd.DataFrame:
    """Synthetic metric drift DataFrame."""
    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    for i in range(20):
        rows.append(
            {
                "run_index": i,
                "run_timestamp": f"2026-02-{i + 1:02d}",
                "primary_metric": 0.85 + rng.normal(0, 0.03),
                "loss_function": [
                    "dice_ce",
                    "cbdice",
                    "dice_ce_cldice",
                    "cbdice_cldice",
                ][i % 4],
            }
        )
    return pd.DataFrame(rows)


def _make_resource_df() -> pd.DataFrame:
    """Synthetic GPU/memory utilization DataFrame."""
    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    losses = ["dice_ce", "cbdice"]
    for loss in losses:
        for epoch in range(1, 21):
            rows.append(
                {
                    "epoch": epoch,
                    "loss_function": loss,
                    "gpu_util_pct": 70 + rng.normal(0, 10),
                    "gpu_mem_mb": 4000 + rng.normal(0, 500),
                    "cpu_util_pct": 40 + rng.normal(0, 15),
                    "ram_mb": 6000 + rng.normal(0, 800),
                }
            )
    return pd.DataFrame(rows)


class TestPlotMetricDrift:
    """Tests for metric drift over time (F4.1)."""

    def test_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.observability_plots import plot_metric_drift

        df = _make_drift_df()
        fig = plot_metric_drift(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_scatter_points(self) -> None:
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.observability_plots import plot_metric_drift

        df = _make_drift_df()
        fig = plot_metric_drift(df)
        ax = fig.get_axes()[0]
        assert len(ax.collections) > 0 or len(ax.lines) > 0
        plt.close(fig)

    def test_saves_to_disk(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.figure_export import save_figure
        from minivess.pipeline.viz.observability_plots import plot_metric_drift

        df = _make_drift_df()
        fig = plot_metric_drift(df)
        path = save_figure(fig, "f4_1_drift", output_dir=tmp_path)
        assert path is not None
        plt.close(fig)


class TestPlotResourceUtilization:
    """Tests for GPU/memory utilization (F4.3)."""

    def test_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.observability_plots import plot_resource_utilization

        df = _make_resource_df()
        fig = plot_resource_utilization(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_multiple_panels(self) -> None:
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.observability_plots import plot_resource_utilization

        df = _make_resource_df()
        fig = plot_resource_utilization(df)
        # Should have subplot panels for different resource types
        assert len(fig.get_axes()) >= 2
        plt.close(fig)
