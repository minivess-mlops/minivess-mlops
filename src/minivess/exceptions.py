"""Custom exception hierarchy for the MinIVess platform.

All domain-specific exceptions inherit from ``MinivessError``, enabling
callers to catch broad categories or specific error types.

Example::

    from minivess.exceptions import CheckpointError

    try:
        model.load_checkpoint(path)
    except CheckpointError as exc:
        logger.error("Failed to load checkpoint: %s", exc)
"""

from __future__ import annotations


class MinivessError(Exception):
    """Base exception for all MinIVess domain errors."""


class CheckpointError(MinivessError):
    """Raised for checkpoint save/load failures."""


class ConfigError(MinivessError):
    """Raised for invalid configuration values."""


class PipelineError(MinivessError):
    """Raised for training/inference pipeline failures."""


class ServingError(MinivessError):
    """Raised for model serving and deployment errors."""


class DataValidationError(MinivessError):
    """Raised for data validation and schema violations."""
