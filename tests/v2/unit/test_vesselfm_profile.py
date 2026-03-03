"""Tests for VesselFM model profile and normalization config (#290)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


class TestVesselFMProfile:
    """Test VesselFM model profile YAML."""

    def test_vesselfm_profile_exists(self) -> None:
        path = CONFIGS_DIR / "model_profiles" / "vesselfm.yaml"
        assert path.exists(), "configs/model_profiles/vesselfm.yaml missing"

    def test_vesselfm_profile_loads(self) -> None:
        path = CONFIGS_DIR / "model_profiles" / "vesselfm.yaml"
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict)

    def test_vesselfm_profile_divisor(self) -> None:
        path = CONFIGS_DIR / "model_profiles" / "vesselfm.yaml"
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg.get("divisor") == 32

    def test_vesselfm_profile_has_max_batch_size(self) -> None:
        path = CONFIGS_DIR / "model_profiles" / "vesselfm.yaml"
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "max_batch_size" in cfg
        assert "gpu_low" in cfg["max_batch_size"]

    def test_vesselfm_profile_loads_via_api(self) -> None:
        """ModelProfile API should be able to load vesselfm profile."""
        from minivess.config.model_profiles import load_model_profile

        profile = load_model_profile("vesselfm")
        assert profile.name == "vesselfm"
        assert profile.divisor == 32


class TestDataConfigNormalization:
    """Test normalization fields on DataConfig."""

    def test_dataconfig_normalization_default(self) -> None:
        from minivess.config.models import DataConfig

        cfg = DataConfig(dataset_name="test")
        assert cfg.normalization == "zscore"

    def test_dataconfig_normalization_percentile(self) -> None:
        from minivess.config.models import DataConfig

        cfg = DataConfig(
            dataset_name="test",
            normalization="percentile",
            percentile_lower=1.0,
            percentile_upper=99.0,
        )
        assert cfg.normalization == "percentile"
        assert cfg.percentile_lower == 1.0
        assert cfg.percentile_upper == 99.0

    def test_percentile_lower_less_than_upper(self) -> None:
        """Validation: lower must be < upper."""
        from minivess.config.models import DataConfig

        with pytest.raises(
            ValueError, match="percentile_lower.*less than.*percentile_upper"
        ):
            DataConfig(
                dataset_name="test",
                normalization="percentile",
                percentile_lower=99.0,
                percentile_upper=1.0,
            )

    def test_normalization_validates_choices(self) -> None:
        """Only 'zscore' and 'percentile' are allowed."""
        from minivess.config.models import DataConfig

        with pytest.raises(ValueError):
            DataConfig(dataset_name="test", normalization="invalid")
