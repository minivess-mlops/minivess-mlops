"""Tests for PostTrainingConfig Pydantic model.

Phase 0 of post-training plugin architecture (#314).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


class TestPostTrainingConfigDefaults:
    """PostTrainingConfig should construct with sensible defaults."""

    def test_default_construction(self) -> None:
        from minivess.config.post_training_config import PostTrainingConfig

        cfg = PostTrainingConfig()
        assert cfg.mlflow_experiment == "minivess_post_training"

    def test_swa_defaults(self) -> None:
        from minivess.config.post_training_config import PostTrainingConfig

        cfg = PostTrainingConfig()
        assert cfg.swa.enabled is True
        assert cfg.swa.per_loss is True
        assert cfg.swa.cross_loss is False

    def test_multi_swa_defaults(self) -> None:
        from minivess.config.post_training_config import PostTrainingConfig

        cfg = PostTrainingConfig()
        assert cfg.multi_swa.enabled is False
        assert cfg.multi_swa.n_models == 3
        assert cfg.multi_swa.subsample_fraction == pytest.approx(0.7)
        assert cfg.multi_swa.seed == 42

    def test_model_merging_defaults(self) -> None:
        from minivess.config.post_training_config import (
            MergeMethod,
            PostTrainingConfig,
        )

        cfg = PostTrainingConfig()
        assert cfg.model_merging.enabled is True
        assert cfg.model_merging.method == MergeMethod.SLERP
        assert cfg.model_merging.t == pytest.approx(0.5)

    def test_calibration_defaults(self) -> None:
        from minivess.config.post_training_config import (
            CalibrationMethod,
            PostTrainingConfig,
        )

        cfg = PostTrainingConfig()
        assert cfg.calibration.enabled is True
        assert CalibrationMethod.GLOBAL_TEMPERATURE in cfg.calibration.methods
        assert cfg.calibration.n_bins == 15

    def test_crc_conformal_defaults(self) -> None:
        from minivess.config.post_training_config import PostTrainingConfig

        cfg = PostTrainingConfig()
        assert cfg.crc_conformal.enabled is True
        assert cfg.crc_conformal.alpha == pytest.approx(0.1)

    def test_conseco_defaults(self) -> None:
        from minivess.config.post_training_config import (
            PostTrainingConfig,
            ShrinkMethod,
        )

        cfg = PostTrainingConfig()
        assert cfg.conseco_fp_control.enabled is False
        assert cfg.conseco_fp_control.tolerance == pytest.approx(0.05)
        assert cfg.conseco_fp_control.shrink_method == ShrinkMethod.EROSION


class TestPostTrainingConfigValidation:
    """Validation should reject invalid parameter ranges."""

    def test_merge_t_out_of_range(self) -> None:
        from minivess.config.post_training_config import ModelMergingPluginConfig

        with pytest.raises(ValidationError):
            ModelMergingPluginConfig(t=1.5)

    def test_merge_t_negative(self) -> None:
        from minivess.config.post_training_config import ModelMergingPluginConfig

        with pytest.raises(ValidationError):
            ModelMergingPluginConfig(t=-0.1)

    def test_alpha_out_of_range(self) -> None:
        from minivess.config.post_training_config import CRCConformalPluginConfig

        with pytest.raises(ValidationError):
            CRCConformalPluginConfig(alpha=1.5)

    def test_alpha_zero_rejected(self) -> None:
        from minivess.config.post_training_config import CRCConformalPluginConfig

        with pytest.raises(ValidationError):
            CRCConformalPluginConfig(alpha=0.0)

    def test_n_models_minimum_2(self) -> None:
        from minivess.config.post_training_config import MultiSWAPluginConfig

        with pytest.raises(ValidationError):
            MultiSWAPluginConfig(n_models=1)

    def test_subsample_fraction_bounds(self) -> None:
        from minivess.config.post_training_config import MultiSWAPluginConfig

        with pytest.raises(ValidationError):
            MultiSWAPluginConfig(subsample_fraction=0.0)
        with pytest.raises(ValidationError):
            MultiSWAPluginConfig(subsample_fraction=1.5)

    def test_tolerance_bounds(self) -> None:
        from minivess.config.post_training_config import ConSeCoPluginConfig

        with pytest.raises(ValidationError):
            ConSeCoPluginConfig(tolerance=0.0)
        with pytest.raises(ValidationError):
            ConSeCoPluginConfig(tolerance=1.0)

    def test_n_bins_positive(self) -> None:
        from minivess.config.post_training_config import CalibrationPluginConfig

        with pytest.raises(ValidationError):
            CalibrationPluginConfig(n_bins=0)


class TestPostTrainingConfigYAML:
    """Config should load from YAML."""

    def test_yaml_loading(self, tmp_path: pytest.TempPathFactory) -> None:
        from pathlib import Path

        import yaml

        from minivess.config.post_training_config import PostTrainingConfig

        yaml_content = {
            "mlflow_experiment": "test_experiment",
            "swa": {"enabled": False},
            "multi_swa": {"enabled": True, "n_models": 5},
        }
        yaml_path = Path(tmp_path) / "test_config.yaml"
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f)

        with yaml_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)

        cfg = PostTrainingConfig(**data)
        assert cfg.mlflow_experiment == "test_experiment"
        assert cfg.swa.enabled is False
        assert cfg.multi_swa.enabled is True
        assert cfg.multi_swa.n_models == 5

    def test_all_plugins_disabled(self) -> None:
        from minivess.config.post_training_config import PostTrainingConfig

        cfg = PostTrainingConfig(
            swa={"enabled": False},  # type: ignore[arg-type]
            multi_swa={"enabled": False},  # type: ignore[arg-type]
            model_merging={"enabled": False},  # type: ignore[arg-type]
            calibration={"enabled": False},  # type: ignore[arg-type]
            crc_conformal={"enabled": False},  # type: ignore[arg-type]
            conseco_fp_control={"enabled": False},  # type: ignore[arg-type]
        )
        assert cfg.swa.enabled is False
        assert cfg.multi_swa.enabled is False
        assert cfg.model_merging.enabled is False
        assert cfg.calibration.enabled is False
        assert cfg.crc_conformal.enabled is False
        assert cfg.conseco_fp_control.enabled is False

    def test_enabled_plugin_names(self) -> None:
        from minivess.config.post_training_config import PostTrainingConfig

        cfg = PostTrainingConfig()
        names = cfg.enabled_plugin_names()
        assert "swa" in names
        assert "model_merging" in names
        assert "calibration" in names
        assert "crc_conformal" in names
        # Disabled by default
        assert "multi_swa" not in names
        assert "conseco_fp_control" not in names
