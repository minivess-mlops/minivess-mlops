from __future__ import annotations

from minivess.config.debug import (
    DEBUG_BOOTSTRAP_ITERATIONS,
    DEBUG_CACHE_RATE,
    DEBUG_MAX_VOLUMES,
    apply_debug_overrides,
)
from minivess.config.models import DataConfig, TrainingConfig


class TestApplyDebugOverrides:
    def test_training_overrides_applied(self) -> None:
        config = TrainingConfig()
        assert config.max_epochs == 100  # default

        apply_debug_overrides(config)

        assert config.max_epochs == 1
        assert config.warmup_epochs == 0
        assert config.early_stopping_patience == 1
        assert config.num_folds == 2

    def test_data_config_overrides(self) -> None:
        training_config = TrainingConfig()
        data_config = DataConfig(dataset_name="test")
        assert data_config.num_workers == 4  # default

        apply_debug_overrides(training_config, data_config)

        assert data_config.num_workers == 0

    def test_debug_off_by_default(self) -> None:
        config = TrainingConfig()
        # Default values should NOT be debug values
        assert config.max_epochs != 1
        assert config.num_folds != 2

    def test_debug_constants(self) -> None:
        assert DEBUG_MAX_VOLUMES == 10
        assert DEBUG_BOOTSTRAP_ITERATIONS == 10
        assert DEBUG_CACHE_RATE == 0.0

    def test_debug_with_compute_profile_interaction(self) -> None:
        """Debug should override compute profile settings where applicable."""
        from minivess.config.compute_profiles import apply_profile, get_compute_profile

        profile = get_compute_profile("gpu_high")
        data_config = DataConfig(dataset_name="test")
        training_config = TrainingConfig()

        # First apply compute profile
        apply_profile(profile, data_config, training_config)
        assert training_config.batch_size == 4
        assert data_config.num_workers == 8

        # Then apply debug overrides (should override some settings)
        apply_debug_overrides(training_config, data_config)
        assert training_config.max_epochs == 1
        assert training_config.num_folds == 2
        assert data_config.num_workers == 0
        # batch_size is NOT overridden by debug (kept from profile)
        assert training_config.batch_size == 4
