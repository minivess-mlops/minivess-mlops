from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from torchio.transforms import (
    Compose,
    RandomBiasField,
    RandomGamma,
    RandomNoise,
)

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "augmentation" / "default.yaml"
)


@dataclass
class AugmentationConfig:
    """Config-driven augmentation parameters (Rule #29).

    All parameters come from ``configs/augmentation/default.yaml``
    via Hydra config groups. No hardcoded defaults in Python.
    """

    noise_std: float
    gamma_log_range: list[float] = field(default_factory=lambda: [-0.3, 0.3])
    bias_field_coefficients: float = 0.5
    probability: float = 0.2

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> AugmentationConfig:
        """Load augmentation config from YAML file."""
        config_path = path or _DEFAULT_CONFIG_PATH
        raw: dict[str, Any] = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        )
        return cls(
            noise_std=float(raw["noise_std"]),
            gamma_log_range=[float(v) for v in raw["gamma_log_range"]],
            bias_field_coefficients=float(raw["bias_field_coefficients"]),
            probability=float(raw["probability"]),
        )


def build_intensity_augmentation(
    config: AugmentationConfig | None = None,
) -> Compose:
    """Build TorchIO intensity augmentation pipeline.

    Parameters
    ----------
    config:
        Augmentation parameters. Loaded from
        ``configs/augmentation/default.yaml`` when ``None``.
    """
    if config is None:
        config = AugmentationConfig.from_yaml()
    return Compose(
        [
            RandomNoise(std=config.noise_std, p=config.probability),
            RandomGamma(
                log_gamma=tuple(config.gamma_log_range), p=config.probability
            ),
            RandomBiasField(
                coefficients=config.bias_field_coefficients, p=config.probability
            ),
        ]
    )
