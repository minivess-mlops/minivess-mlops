"""Quasi-E2E debug data configuration for mechanics testing.

Provides a Pydantic config for running all model×loss combinations with
minimal resources: 1 epoch, 2+2 volumes, 0 workers, subset external datasets.
Focus is on verifying training mechanics, NOT model performance.

Usage:
    from minivess.testing.debug_data_config import load_quasi_e2e_config
    config = load_quasi_e2e_config()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
_DEFAULT_QUASI_E2E_YAML: Path = (
    _REPO_ROOT / "configs" / "experiment" / "test_practical_combos.yaml"
)


class ExternalTestConfig(BaseModel):
    """Configuration for one external test dataset in debug mode."""

    enabled: bool = True
    max_volumes: int = Field(default=2, ge=1, description="Max volumes to use")


class QuasiE2EConfig(BaseModel):
    """Configuration for quasi-E2E debug experiments.

    All defaults are designed for fast mechanics testing:
    - 1 epoch, 1 fold, 0 warmup
    - 2 train + 2 val volumes (disjoint)
    - 0 workers (no multiprocessing overhead)
    - Subset external test datasets
    """

    # Training overrides
    max_epochs: int = Field(default=1, ge=1, description="Max training epochs")
    num_folds: int = Field(default=1, ge=1, description="Number of CV folds")
    batch_size: int = Field(default=2, ge=1, description="Batch size")
    warmup_epochs: int = Field(default=0, ge=0, description="Warmup epochs")
    num_workers: int = Field(default=0, ge=0, description="DataLoader workers")
    mixed_precision: bool = Field(default=True, description="Enable AMP")
    seed: int = Field(default=42, description="Random seed")

    # Volume selection
    n_train_volumes: int = Field(
        default=2, ge=1, description="Number of training volumes"
    )
    n_val_volumes: int = Field(
        default=2, ge=1, description="Number of validation volumes"
    )
    train_volume_ids: list[str] = Field(
        default_factory=lambda: ["mv01", "mv03"],
        description="Explicit training volume IDs",
    )
    val_volume_ids: list[str] = Field(
        default_factory=lambda: ["mv05", "mv07"],
        description="Explicit validation volume IDs",
    )

    # External test datasets
    # TubeNet excluded — olfactory bulb, different organ, only 1 2PM volume. See CLAUDE.md.
    external_test_datasets: dict[str, ExternalTestConfig] = Field(
        default_factory=lambda: {
            "vesselnn": ExternalTestConfig(enabled=True, max_volumes=2),
            "deepvess": ExternalTestConfig(enabled=False, max_volumes=1),
        },
        description="External test dataset configs",
    )

    # Data config overrides
    cache_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="CacheDataset rate"
    )

    # Post-training
    skip_post_training: bool = Field(
        default=True, description="Skip post-training flow in debug"
    )


def load_quasi_e2e_config(
    yaml_path: Path | None = None,
) -> QuasiE2EConfig:
    """Load quasi-E2E config from YAML file.

    Parameters
    ----------
    yaml_path:
        Path to YAML config. Defaults to
        ``configs/experiment/test_practical_combos.yaml``.

    Returns
    -------
    Validated QuasiE2EConfig instance.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    """
    resolved = yaml_path if yaml_path is not None else _DEFAULT_QUASI_E2E_YAML
    if not resolved.exists():
        msg = f"Quasi-E2E config not found: {resolved}"
        raise FileNotFoundError(msg)
    with resolved.open(encoding="utf-8") as fh:
        data: Any = yaml.safe_load(fh)
    if data is None:
        data = {}
    return QuasiE2EConfig(**data)
