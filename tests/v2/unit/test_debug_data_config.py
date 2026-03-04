"""Tests for quasi-E2E debug data configuration (Phase 3, #335).

Verifies that QuasiE2EConfig provides correct debug overrides:
1 epoch, 2+2 volumes, subset external datasets, correct defaults.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from minivess.testing.debug_data_config import (
    QuasiE2EConfig,
    load_quasi_e2e_config,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestQuasiE2EConfigDefaults:
    """Default config values are correct for debug mode."""

    def test_max_epochs_is_one(self) -> None:
        config = QuasiE2EConfig()
        assert config.max_epochs == 1

    def test_num_folds_is_one(self) -> None:
        config = QuasiE2EConfig()
        assert config.num_folds == 1

    def test_batch_size_is_two(self) -> None:
        config = QuasiE2EConfig()
        assert config.batch_size == 2

    def test_n_train_volumes(self) -> None:
        config = QuasiE2EConfig()
        assert config.n_train_volumes == 2

    def test_n_val_volumes(self) -> None:
        config = QuasiE2EConfig()
        assert config.n_val_volumes == 2

    def test_warmup_epochs_is_zero(self) -> None:
        config = QuasiE2EConfig()
        assert config.warmup_epochs == 0

    def test_num_workers_is_zero(self) -> None:
        config = QuasiE2EConfig()
        assert config.num_workers == 0

    def test_seed_is_42(self) -> None:
        config = QuasiE2EConfig()
        assert config.seed == 42

    def test_mixed_precision_enabled(self) -> None:
        config = QuasiE2EConfig()
        assert config.mixed_precision is True


class TestQuasiE2EConfigValidation:
    """Config validation catches invalid values."""

    def test_max_epochs_positive(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            QuasiE2EConfig(max_epochs=0)

    def test_n_train_volumes_positive(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            QuasiE2EConfig(n_train_volumes=0)

    def test_n_val_volumes_positive(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            QuasiE2EConfig(n_val_volumes=0)

    def test_batch_size_positive(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            QuasiE2EConfig(batch_size=0)


class TestExternalTestConfig:
    """External test dataset config has correct defaults."""

    def test_tubenet_enabled_by_default(self) -> None:
        config = QuasiE2EConfig()
        assert "tubenet_2pm" in config.external_test_datasets
        tubenet = config.external_test_datasets["tubenet_2pm"]
        assert tubenet.enabled is True

    def test_vesselnn_enabled_by_default(self) -> None:
        config = QuasiE2EConfig()
        assert "vesselnn" in config.external_test_datasets
        vesselnn = config.external_test_datasets["vesselnn"]
        assert vesselnn.enabled is True

    def test_tubenet_max_volumes_is_one(self) -> None:
        """tubenet_2pm has only 1 volume, so max is 1."""
        config = QuasiE2EConfig()
        tubenet = config.external_test_datasets["tubenet_2pm"]
        assert tubenet.max_volumes == 1

    def test_vesselnn_subset(self) -> None:
        """vesselnn has 12 volumes, we use 2 in debug mode."""
        config = QuasiE2EConfig()
        vesselnn = config.external_test_datasets["vesselnn"]
        assert vesselnn.max_volumes == 2

    def test_deepvess_disabled_by_default(self) -> None:
        config = QuasiE2EConfig()
        deepvess = config.external_test_datasets["deepvess"]
        assert deepvess.enabled is False


class TestQuasiE2EConfigTrainVolumes:
    """Volume selection produces correct counts."""

    def test_default_train_volume_ids(self) -> None:
        config = QuasiE2EConfig()
        assert config.train_volume_ids == ["mv01", "mv03"]

    def test_default_val_volume_ids(self) -> None:
        config = QuasiE2EConfig()
        assert config.val_volume_ids == ["mv05", "mv07"]

    def test_train_and_val_disjoint(self) -> None:
        config = QuasiE2EConfig()
        assert not set(config.train_volume_ids) & set(config.val_volume_ids)

    def test_total_volumes_is_four(self) -> None:
        config = QuasiE2EConfig()
        total = len(config.train_volume_ids) + len(config.val_volume_ids)
        assert total == 4


class TestLoadQuasiE2EConfig:
    """load_quasi_e2e_config loads from YAML file."""

    def test_loads_default_config(self) -> None:
        config = load_quasi_e2e_config()
        assert isinstance(config, QuasiE2EConfig)
        assert config.max_epochs == 1

    def test_loads_from_custom_path(self, tmp_path: Path) -> None:
        import yaml

        data = {
            "max_epochs": 3,
            "n_train_volumes": 4,
            "n_val_volumes": 4,
        }
        path = tmp_path / "custom.yaml"
        with path.open("w", encoding="utf-8") as fh:
            yaml.dump(data, fh)

        config = load_quasi_e2e_config(path)
        assert config.max_epochs == 3
        assert config.n_train_volumes == 4

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_quasi_e2e_config(tmp_path / "nonexistent.yaml")
