"""Uncertainty quantification visualizations (F3.1–F3.5).

F3.1: UQ decomposition — stacked bars (aleatoric + epistemic).
F3.2: Calibration curve — predicted vs observed with diagonal reference.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from minivess.pipeline.viz.figure_dimensions import get_figsize
from minivess.pipeline.viz.plot_config import COLORS, LOSS_LABELS, setup_style

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def plot_uq_decomposition(
    uq_df: pd.DataFrame,
) -> Figure:
    """Create stacked bar chart of UQ decomposition (F3.1).

    Parameters
    ----------
    uq_df:
        DataFrame with columns: ``strategy``, ``total_uncertainty``,
        ``aleatoric``, ``epistemic``.

    Returns
    -------
    matplotlib Figure.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=get_figsize("single"))

    strategies = uq_df["strategy"].values
    x = np.arange(len(strategies))

    ax.bar(
        x,
        uq_df["aleatoric"],
        label="Aleatoric",
        color="#88CCEE",
        width=0.6,
    )
    ax.bar(
        x,
        uq_df["epistemic"],
        bottom=uq_df["aleatoric"],
        label="Epistemic",
        color="#CC6677",
        width=0.6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [s.replace("_", " ").title() for s in strategies],
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("Uncertainty")
    ax.set_title(
        "UQ Decomposition by Ensemble Strategy", fontsize=14, fontweight="bold"
    )
    ax.legend()

    return fig


def plot_calibration_curve(
    calibration_df: pd.DataFrame,
) -> Figure:
    """Create calibration curve with diagonal reference (F3.2).

    Parameters
    ----------
    calibration_df:
        DataFrame with columns: ``loss_function``, ``predicted_prob``,
        ``observed_freq``.

    Returns
    -------
    matplotlib Figure.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=get_figsize("single"))

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfect")

    for loss_name in sorted(calibration_df["loss_function"].unique()):
        loss_df = calibration_df[
            calibration_df["loss_function"] == loss_name
        ].sort_values("predicted_prob")
        color = COLORS.get(loss_name, "#999999")
        label = LOSS_LABELS.get(loss_name, loss_name)
        ax.plot(
            loss_df["predicted_prob"],
            loss_df["observed_freq"],
            "-o",
            color=color,
            label=label,
            markersize=4,
        )

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Calibration Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return fig
