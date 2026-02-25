from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minivess.config.models import DataConfig, TrainingConfig

logger = logging.getLogger(__name__)

# Debug override values — designed for fast smoke testing
_DEBUG_OVERRIDES = {
    "max_epochs": 1,
    "warmup_epochs": 0,
    "early_stopping_patience": 1,
    "num_folds": 2,
    "num_workers": 0,
}

_DEBUG_DATA_OVERRIDES = {
    "num_workers": 0,
}

# Additional non-config values exposed as module constants
DEBUG_MAX_VOLUMES = 10
DEBUG_BOOTSTRAP_ITERATIONS = 10
DEBUG_CACHE_RATE = 0.0


def apply_debug_overrides(
    training_config: TrainingConfig,
    data_config: DataConfig | None = None,
) -> None:
    """Apply debug mode overrides for fast smoke testing.

    Mutates configs in-place to enable rapid iteration:
    - 1 epoch, no warmup, minimal patience
    - 2 folds (minimum for CV)
    - 0 workers (avoids multiprocessing issues)

    Parameters
    ----------
    training_config:
        Training configuration to override.
    data_config:
        Optional data configuration to override (num_workers=0).
    """
    logger.info("Debug mode: applying overrides for fast smoke testing")

    training_config.max_epochs = _DEBUG_OVERRIDES["max_epochs"]
    training_config.warmup_epochs = _DEBUG_OVERRIDES["warmup_epochs"]
    training_config.early_stopping_patience = _DEBUG_OVERRIDES["early_stopping_patience"]
    training_config.num_folds = _DEBUG_OVERRIDES["num_folds"]

    if data_config is not None:
        data_config.num_workers = _DEBUG_DATA_OVERRIDES["num_workers"]

    logger.info(
        "Debug overrides applied: max_epochs=%d, num_folds=%d, num_workers=%d",
        training_config.max_epochs,
        training_config.num_folds,
        _DEBUG_DATA_OVERRIDES["num_workers"],
    )
