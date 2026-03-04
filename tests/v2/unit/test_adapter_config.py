"""Tests for configurable adapter hyperparameters (R5.15).

Verifies that hardcoded architecture values in adapters can be overridden
via ``ModelConfig.architecture_params`` while preserving default behavior.
"""

from __future__ import annotations

import pytest

from minivess.config.models import ModelConfig, ModelFamily

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def dynunet_config() -> ModelConfig:
    return ModelConfig(
        family=ModelFamily.MONAI_DYNUNET,
        name="dynunet-test",
        in_channels=1,
        out_channels=2,
    )


# ---------------------------------------------------------------------------
# Test: ModelConfig has architecture_params field
# ---------------------------------------------------------------------------


class TestModelConfigArchitectureParams:
    """ModelConfig.architecture_params is available and defaults to empty."""

    def test_architecture_params_defaults_to_empty_dict(self) -> None:
        config = ModelConfig(
            family=ModelFamily.MONAI_DYNUNET,
            name="test",
        )
        assert config.architecture_params == {}

    def test_architecture_params_accepts_custom_values(self) -> None:
        config = ModelConfig(
            family=ModelFamily.MONAI_DYNUNET,
            name="test",
            architecture_params={"filters": [16, 32, 64, 128]},
        )
        assert config.architecture_params["filters"] == [16, 32, 64, 128]


# ---------------------------------------------------------------------------
# Test: DynUNet adapter respects architecture_params
# ---------------------------------------------------------------------------


class TestDynUNetConfigurable:
    """DynUNetAdapter uses architecture_params for filter override."""

    def test_default_filters(self, dynunet_config: ModelConfig) -> None:
        from minivess.adapters.dynunet import DynUNetAdapter

        adapter = DynUNetAdapter(dynunet_config)
        cfg = adapter.get_config()
        assert cfg.extras["filters"] == [32, 64, 128, 256]

    def test_custom_filters_via_architecture_params(self) -> None:
        from minivess.adapters.dynunet import DynUNetAdapter

        config = ModelConfig(
            family=ModelFamily.MONAI_DYNUNET,
            name="dynunet-custom",
            in_channels=1,
            out_channels=2,
            architecture_params={"filters": [16, 32, 64, 128]},
        )
        adapter = DynUNetAdapter(config)
        cfg = adapter.get_config()
        assert cfg.extras["filters"] == [16, 32, 64, 128]
