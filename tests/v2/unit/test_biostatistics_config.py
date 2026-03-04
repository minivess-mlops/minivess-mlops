"""Unit tests for BiostatisticsConfig (Task 1.1).

Tests the Pydantic config model for the biostatistics flow.
"""

from __future__ import annotations

import pytest


class TestBiostatisticsConfigDefaults:
    """Default config is valid and has expected values."""

    def test_default_config_valid(self) -> None:
        """Default constructor produces a valid config."""
        from minivess.config.biostatistics_config import BiostatisticsConfig

        config = BiostatisticsConfig()
        assert config.alpha == 0.05
        assert config.n_bootstrap == 10_000
        assert config.seed == 42
        assert config.primary_metric in config.metrics

    def test_default_metrics_include_cldice(self) -> None:
        """Default metrics list includes clDice as primary."""
        from minivess.config.biostatistics_config import BiostatisticsConfig

        config = BiostatisticsConfig()
        assert "cldice" in config.metrics
        assert config.primary_metric == "cldice"

    def test_default_has_8_metrics(self) -> None:
        """Default config has 8 metrics per plan."""
        from minivess.config.biostatistics_config import BiostatisticsConfig

        config = BiostatisticsConfig()
        assert len(config.metrics) == 8


class TestBiostatisticsConfigCustom:
    """Custom config values are accepted."""

    def test_custom_config(self) -> None:
        """Custom values override defaults."""
        from minivess.config.biostatistics_config import BiostatisticsConfig

        config = BiostatisticsConfig(
            experiment_names=["exp_a", "exp_b"],
            metrics=["dsc", "hd95"],
            primary_metric="dsc",
            alpha=0.01,
            n_bootstrap=5000,
            seed=123,
        )
        assert config.experiment_names == ["exp_a", "exp_b"]
        assert config.alpha == 0.01
        assert config.n_bootstrap == 5000
        assert config.seed == 123
        assert config.primary_metric == "dsc"

    def test_rope_values_custom(self) -> None:
        """Custom ROPE values are accepted."""
        from minivess.config.biostatistics_config import BiostatisticsConfig

        config = BiostatisticsConfig(
            metrics=["dsc", "cldice"],
            primary_metric="cldice",
            rope_values={"dsc": 0.02, "cldice": 0.015},
        )
        assert config.rope_values["dsc"] == 0.02
        assert config.rope_values["cldice"] == 0.015


class TestBiostatisticsConfigValidation:
    """Validation catches invalid configs."""

    def test_primary_metric_must_be_in_metrics_list(self) -> None:
        """primary_metric not in metrics raises ValidationError."""
        from pydantic import ValidationError

        from minivess.config.biostatistics_config import BiostatisticsConfig

        with pytest.raises(ValidationError, match="primary_metric"):
            BiostatisticsConfig(
                metrics=["dsc", "hd95"],
                primary_metric="cldice",
            )

    def test_alpha_too_low(self) -> None:
        """alpha <= 0 raises ValidationError."""
        from pydantic import ValidationError

        from minivess.config.biostatistics_config import BiostatisticsConfig

        with pytest.raises(ValidationError):
            BiostatisticsConfig(alpha=0.0)

    def test_alpha_too_high(self) -> None:
        """alpha >= 1 raises ValidationError."""
        from pydantic import ValidationError

        from minivess.config.biostatistics_config import BiostatisticsConfig

        with pytest.raises(ValidationError):
            BiostatisticsConfig(alpha=1.0)

    def test_n_bootstrap_too_low(self) -> None:
        """n_bootstrap < 100 raises ValidationError."""
        from pydantic import ValidationError

        from minivess.config.biostatistics_config import BiostatisticsConfig

        with pytest.raises(ValidationError):
            BiostatisticsConfig(n_bootstrap=50)

    def test_rope_values_positive(self) -> None:
        """ROPE values must be positive."""
        from pydantic import ValidationError

        from minivess.config.biostatistics_config import BiostatisticsConfig

        with pytest.raises(ValidationError, match="rope"):
            BiostatisticsConfig(
                metrics=["dsc"],
                primary_metric="dsc",
                rope_values={"dsc": -0.01},
            )


class TestBiostatisticsConfigYaml:
    """YAML config loading (Task 1.3)."""

    def test_default_yaml_loads(self) -> None:
        """configs/biostatistics/default.yaml loads into BiostatisticsConfig."""
        from pathlib import Path

        import yaml

        from minivess.config.biostatistics_config import BiostatisticsConfig

        yaml_path = Path("configs/biostatistics/default.yaml")
        assert yaml_path.exists(), f"YAML config not found: {yaml_path}"
        with yaml_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        config = BiostatisticsConfig(**data)
        assert config.primary_metric == "cldice"
        assert config.alpha == 0.05

    def test_default_yaml_has_all_8_metrics(self) -> None:
        """YAML config has all 8 default metrics."""
        from pathlib import Path

        import yaml

        from minivess.config.biostatistics_config import BiostatisticsConfig

        yaml_path = Path("configs/biostatistics/default.yaml")
        with yaml_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        config = BiostatisticsConfig(**data)
        assert len(config.metrics) == 8
        assert "cldice" in config.metrics
        assert "dsc" in config.metrics


class TestBiostatisticsConfigSerialization:
    """Config round-trips through JSON."""

    def test_serialization_round_trip(self) -> None:
        """Config can be serialized to JSON and back."""
        from minivess.config.biostatistics_config import BiostatisticsConfig

        original = BiostatisticsConfig(
            experiment_names=["exp_a"],
            metrics=["dsc", "cldice"],
            primary_metric="cldice",
        )
        json_str = original.model_dump_json()
        restored = BiostatisticsConfig.model_validate_json(json_str)
        assert restored.experiment_names == original.experiment_names
        assert restored.metrics == original.metrics
        assert restored.primary_metric == original.primary_metric
        assert restored.alpha == original.alpha
