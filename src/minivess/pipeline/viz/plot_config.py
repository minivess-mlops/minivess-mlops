"""Centralized plot styling with Paul Tol colorblind-safe palette.

All visualization modules import colors and labels from here.
No hardcoded colors or figure sizes anywhere else.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paul Tol colorblind-safe palette
# ---------------------------------------------------------------------------

COLORS: dict[str, str] = {
    "dice_ce": "#332288",
    "cbdice": "#88CCEE",
    "dice_ce_cldice": "#44AA99",
    "cbdice_cldice": "#117733",
    "champion": "#DDCC77",
    "reference": "#CC6677",
}

# ---------------------------------------------------------------------------
# Loss function display labels
# ---------------------------------------------------------------------------

LOSS_LABELS: dict[str, str] = {
    "dice_ce": "Dice + CE",
    "cbdice": "cbDice",
    "dice_ce_cldice": "Dice + CE + clDice",
    "cbdice_cldice": "cbDice + clDice",
}


# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------


def setup_style(context: str = "paper") -> None:
    """Apply consistent styling across all figures.

    Parameters
    ----------
    context:
        Seaborn context: ``"paper"``, ``"talk"``, ``"poster"``, ``"notebook"``.
    """
    sns.set_theme(
        context=context,
        style="whitegrid",
        font_scale=1.2,
        rc={
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )
    plt.rcParams["figure.constrained_layout.use"] = True
