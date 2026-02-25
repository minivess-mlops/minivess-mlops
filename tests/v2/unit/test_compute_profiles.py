from __future__ import annotations

import pytest

from minivess.config.compute_profiles import (
    ComputeProfile,
    apply_profile,
    get_compute_profile,
    list_profiles,
)
from minivess.config.models import DataConfig, TrainingConfig


class TestGetComputeProfile:
    def test_all_profiles_instantiate(self) -> None:
        for name in list_profiles():
            profile = get_compute_profile(name)
            assert isinstance(profile, ComputeProfile)
            assert profile.name == name

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown compute profile"):
            get_compute_profile("nonexistent_profile")

    @pytest.mark.parametrize(
        "name,expected_batch",
        [
            ("cpu", 1),
            ("gpu_low", 2),
            ("gpu_high", 4),
            ("dgx_spark", 8),
            ("cloud_single", 8),
            ("cloud_multi", 32),
        ],
    )
    def test_batch_sizes(self, name: str, expected_batch: int) -> None:
        profile = get_compute_profile(name)
        assert profile.batch_size == expected_batch

    def test_cpu_no_amp(self) -> None:
        profile = get_compute_profile("cpu")
        assert profile.mixed_precision is False

    def test_gpu_profiles_have_amp(self) -> None:
        for name in ["gpu_low", "gpu_high", "dgx_spark", "cloud_single", "cloud_multi"]:
            profile = get_compute_profile(name)
            assert profile.mixed_precision is True

    def test_list_profiles_returns_all_six(self) -> None:
        profiles = list_profiles()
        assert len(profiles) == 6


class TestApplyProfile:
    def test_apply_modifies_configs(self) -> None:
        profile = get_compute_profile("gpu_low")
        data_config = DataConfig(dataset_name="test")
        training_config = TrainingConfig()

        apply_profile(profile, data_config, training_config)

        assert data_config.patch_size == (96, 96, 24)
        assert data_config.num_workers == 4
        assert training_config.batch_size == 2
        assert training_config.mixed_precision is True

    def test_apply_cpu_profile(self) -> None:
        profile = get_compute_profile("cpu")
        data_config = DataConfig(dataset_name="test")
        training_config = TrainingConfig()

        apply_profile(profile, data_config, training_config)

        assert data_config.patch_size == (64, 64, 16)
        assert training_config.batch_size == 1
        assert training_config.mixed_precision is False
