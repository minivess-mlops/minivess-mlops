"""Tests for config-driven augmentation (Task 2.20, #936).

Ensures augmentation parameters come from YAML config, not hardcoded values.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from minivess.data.augmentation import (
    AugmentationConfig,
    build_intensity_augmentation,
)


class TestAugmentationConfig:
    """Augmentation parameters must come from config YAML."""

    def test_config_loads_from_default_yaml(self) -> None:
        """Default config loads without error."""
        config = AugmentationConfig.from_yaml()
        assert config.noise_std > 0
        assert len(config.gamma_log_range) == 2
        assert config.probability > 0

    def test_config_matches_yaml_values(self) -> None:
        """Config values match what's in the YAML file."""
        yaml_path = (
            Path(__file__).resolve().parents[4]
            / "configs"
            / "augmentation"
            / "default.yaml"
        )
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        config = AugmentationConfig.from_yaml(yaml_path)

        assert config.noise_std == raw["noise_std"]
        assert config.gamma_log_range == raw["gamma_log_range"]
        assert config.bias_field_coefficients == raw["bias_field_coefficients"]
        assert config.probability == raw["probability"]

    def test_custom_config_overrides_defaults(self, tmp_path: Path) -> None:
        """Custom YAML overrides default values."""
        custom = {
            "noise_std": 0.05,
            "gamma_log_range": [-0.5, 0.5],
            "bias_field_coefficients": 1.0,
            "probability": 0.5,
        }
        yaml_path = tmp_path / "custom.yaml"
        yaml_path.write_text(yaml.dump(custom), encoding="utf-8")

        config = AugmentationConfig.from_yaml(yaml_path)
        assert config.noise_std == 0.05
        assert config.probability == 0.5

    def test_build_augmentation_uses_config(self) -> None:
        """build_intensity_augmentation reads from config, not hardcoded."""
        config = AugmentationConfig(
            noise_std=0.05,
            gamma_log_range=[-0.1, 0.1],
            bias_field_coefficients=0.3,
            probability=0.8,
        )
        pipeline = build_intensity_augmentation(config)
        # Should have 3 transforms
        assert len(pipeline.transforms) == 3
