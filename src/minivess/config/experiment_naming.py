"""Experiment naming enforcement for debug vs production experiments.

Ensures debug-length experiments (<=20 epochs) use the ``_debug`` suffix
in their experiment name. This prevents accidental pollution of production
MLflow experiments with debug runs.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: Maximum epoch count before the ``_debug`` suffix is required.
DEBUG_EPOCH_THRESHOLD = 20


def is_debug_experiment(name: str) -> bool:
    """Check whether an experiment name indicates a debug experiment.

    Parameters
    ----------
    name:
        Experiment name string.

    Returns
    -------
    True if the name ends with ``_debug``.
    """
    return name.endswith("_debug")


def validate_debug_experiment_name(name: str, *, max_epochs: int) -> str:
    """Validate experiment name against epoch count.

    Experiments with ``max_epochs <= DEBUG_EPOCH_THRESHOLD`` must use the
    ``_debug`` suffix to avoid polluting production MLflow experiments.

    Parameters
    ----------
    name:
        Experiment name to validate.
    max_epochs:
        Maximum number of training epochs.

    Returns
    -------
    The validated name (unchanged).

    Raises
    ------
    ValueError
        If the name is empty, or if ``max_epochs`` is at or below the
        debug threshold without the ``_debug`` suffix.
    """
    if not name:
        msg = "Experiment name must not be empty"
        raise ValueError(msg)

    if max_epochs <= DEBUG_EPOCH_THRESHOLD and not is_debug_experiment(name):
        msg = (
            f"Experiment '{name}' has max_epochs={max_epochs} "
            f"(<= {DEBUG_EPOCH_THRESHOLD}) but does not end with '_debug'. "
            f"Use '{name}_debug' to avoid polluting production MLflow experiments."
        )
        raise ValueError(msg)

    return name
