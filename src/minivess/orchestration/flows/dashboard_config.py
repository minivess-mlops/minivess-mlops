"""Progressive disclosure dashboard configuration.

Defines three dashboard levels for different personas:
- Level 1 (PI): Executive summary — key numbers, pass/fail
- Level 2 (Colleague): Experiment comparison, training curves
- Level 3 (Researcher): Per-fold details, UQ, Grafana links
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class DashboardLevel(enum.IntEnum):
    """Dashboard disclosure level."""

    PI = 1
    COLLEAGUE = 2
    RESEARCHER = 3


# Sections available at each level (cumulative)
_LEVEL_SECTIONS: dict[str, list[str]] = {
    "PI": [
        "executive_summary",
        "champion_models",
        "pass_fail_status",
    ],
    "COLLEAGUE": [
        "executive_summary",
        "champion_models",
        "pass_fail_status",
        "experiment_comparison",
        "training_curves",
        "loss_comparison",
    ],
    "RESEARCHER": [
        "executive_summary",
        "champion_models",
        "pass_fail_status",
        "experiment_comparison",
        "training_curves",
        "loss_comparison",
        "per_fold_details",
        "uncertainty_quantification",
        "calibration_plots",
        "topology_metrics",
        "grafana_links",
        "mlflow_links",
    ],
}


@dataclass
class DashboardConfig:
    """Dashboard generation configuration.

    Attributes
    ----------
    level:
        Disclosure level: ``"PI"``, ``"COLLEAGUE"``, or ``"RESEARCHER"``.
    output_formats:
        List of output formats (e.g. ``["html", "pdf", "md"]``).
    """

    level: str = "COLLEAGUE"
    output_formats: list[str] = field(default_factory=lambda: ["md", "html"])


def get_sections_for_level(level: str) -> list[str]:
    """Get dashboard sections for a given disclosure level.

    Parameters
    ----------
    level:
        One of ``"PI"``, ``"COLLEAGUE"``, ``"RESEARCHER"``.

    Returns
    -------
    List of section names to include in the dashboard.

    Raises
    ------
    ValueError
        If level is not recognized.
    """
    level_upper = level.upper()
    if level_upper not in _LEVEL_SECTIONS:
        msg = f"Unknown dashboard level {level!r}. Use PI, COLLEAGUE, or RESEARCHER."
        raise ValueError(msg)
    return list(_LEVEL_SECTIONS[level_upper])
