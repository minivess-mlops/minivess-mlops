from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class WeightWatcherReport:
    """Summary of WeightWatcher spectral analysis."""

    alpha_weighted: float
    log_norm: float
    num_layers: int
    details: dict[str, Any]
    passed_gate: bool


def analyze_model(
    model: nn.Module,
    *,
    alpha_threshold: float = 5.0,
) -> WeightWatcherReport:
    """Run WeightWatcher analysis on a PyTorch model.

    Returns spectral quality metrics. Models with alpha_weighted > threshold
    may have poor generalization (overfit or undertrained).

    Args:
        model: PyTorch model to analyze.
        alpha_threshold: Maximum acceptable alpha_weighted for deployment gate.
    """
    import weightwatcher as ww

    watcher = ww.WeightWatcher(model=model)
    details = watcher.analyze()
    summary = watcher.get_summary(details)

    alpha_weighted = summary.get("alpha_weighted", float("inf"))
    log_norm = summary.get("log_norm", 0.0)
    num_layers = summary.get("num_layers", 0)

    passed = alpha_weighted <= alpha_threshold
    if not passed:
        logger.warning(
            "WeightWatcher gate FAILED: alpha_weighted=%.2f > threshold=%.2f",
            alpha_weighted,
            alpha_threshold,
        )
    else:
        logger.info(
            "WeightWatcher gate passed: alpha_weighted=%.2f <= %.2f",
            alpha_weighted,
            alpha_threshold,
        )

    return WeightWatcherReport(
        alpha_weighted=alpha_weighted,
        log_norm=log_norm,
        num_layers=num_layers,
        details=summary,
        passed_gate=passed,
    )
