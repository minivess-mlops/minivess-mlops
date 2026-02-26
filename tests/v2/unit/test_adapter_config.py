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
def segresnet_config() -> ModelConfig:
    return ModelConfig(
        family=ModelFamily.MONAI_SEGRESNET,
        name="segresnet-test",
        in_channels=1,
        out_channels=2,
    )


@pytest.fixture()
def swinunetr_config() -> ModelConfig:
    return ModelConfig(
        family=ModelFamily.MONAI_SWINUNETR,
        name="swinunetr-test",
        in_channels=1,
        out_channels=2,
    )


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
            family=ModelFamily.MONAI_SEGRESNET,
            name="test",
        )
        assert config.architecture_params == {}

    def test_architecture_params_accepts_custom_values(self) -> None:
        config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="test",
            architecture_params={"init_filters": 64, "blocks_down": (2, 3, 3, 4)},
        )
        assert config.architecture_params["init_filters"] == 64
        assert config.architecture_params["blocks_down"] == (2, 3, 3, 4)


# ---------------------------------------------------------------------------
# Test: SegResNet adapter respects architecture_params
# ---------------------------------------------------------------------------


class TestSegResNetConfigurable:
    """SegResNetAdapter uses architecture_params for overrides, defaults otherwise."""

    def test_default_init_filters(self, segresnet_config: ModelConfig) -> None:
        from minivess.adapters.segresnet import SegResNetAdapter

        adapter = SegResNetAdapter(segresnet_config)
        cfg = adapter.get_config()
        assert cfg.extras["init_filters"] == 32

    def test_custom_init_filters_via_architecture_params(self) -> None:
        from minivess.adapters.segresnet import SegResNetAdapter

        config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="segresnet-custom",
            in_channels=1,
            out_channels=2,
            architecture_params={"init_filters": 64},
        )
        adapter = SegResNetAdapter(config)
        cfg = adapter.get_config()
        assert cfg.extras["init_filters"] == 64

    def test_custom_blocks_via_architecture_params(self) -> None:
        from minivess.adapters.segresnet import SegResNetAdapter

        config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="segresnet-custom",
            in_channels=1,
            out_channels=2,
            architecture_params={
                "blocks_down": (2, 3, 3, 4),
                "blocks_up": (2, 2, 2),
            },
        )
        adapter = SegResNetAdapter(config)
        cfg = adapter.get_config()
        assert cfg.extras["blocks_down"] == (2, 3, 3, 4)
        assert cfg.extras["blocks_up"] == (2, 2, 2)


# ---------------------------------------------------------------------------
# Test: SwinUNETR adapter respects architecture_params
# ---------------------------------------------------------------------------


class TestSwinUNETRConfigurable:
    """SwinUNETRAdapter uses architecture_params for overrides."""

    def test_default_feature_size(self, swinunetr_config: ModelConfig) -> None:
        from minivess.adapters.swinunetr import SwinUNETRAdapter

        adapter = SwinUNETRAdapter(swinunetr_config)
        cfg = adapter.get_config()
        assert cfg.extras["feature_size"] == 48

    def test_custom_feature_size_via_architecture_params(self) -> None:
        from minivess.adapters.swinunetr import SwinUNETRAdapter

        config = ModelConfig(
            family=ModelFamily.MONAI_SWINUNETR,
            name="swinunetr-custom",
            in_channels=1,
            out_channels=2,
            architecture_params={"feature_size": 24},
        )
        adapter = SwinUNETRAdapter(config)
        cfg = adapter.get_config()
        assert cfg.extras["feature_size"] == 24

    def test_custom_depths_num_heads_via_architecture_params(self) -> None:
        from minivess.adapters.swinunetr import SwinUNETRAdapter

        config = ModelConfig(
            family=ModelFamily.MONAI_SWINUNETR,
            name="swinunetr-custom",
            in_channels=1,
            out_channels=2,
            architecture_params={
                "depths": (2, 2, 6, 2),
                "num_heads": (3, 6, 12, 24),
            },
        )
        adapter = SwinUNETRAdapter(config)
        cfg = adapter.get_config()
        assert cfg.extras["depths"] == (2, 2, 6, 2)
        assert cfg.extras["num_heads"] == (3, 6, 12, 24)


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
