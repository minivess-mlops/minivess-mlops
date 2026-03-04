"""Tests for model builder factory (SAM-10).

Validates build_adapter() dispatches correctly for all model families.
"""

from __future__ import annotations

import pytest

from minivess.config.models import ModelConfig, ModelFamily


class TestBuildAdapter:
    """build_adapter() dispatches on ModelFamily enum."""

    def test_build_dynunet(self) -> None:
        from minivess.adapters.model_builder import build_adapter

        config = ModelConfig(
            family=ModelFamily.MONAI_DYNUNET,
            name="dynunet-test",
            in_channels=1,
            out_channels=2,
        )
        adapter = build_adapter(config)
        cfg = adapter.get_config()
        assert cfg.family == "dynunet"

    def test_build_sam3_vanilla(self) -> None:
        from minivess.adapters.model_builder import build_adapter

        config = ModelConfig(
            family=ModelFamily.SAM3_VANILLA,
            name="vanilla-test",
            in_channels=1,
            out_channels=2,
        )
        adapter = build_adapter(config, use_stub=True)
        cfg = adapter.get_config()
        assert cfg.family == "sam3_vanilla"

    def test_build_sam3_topolora(self) -> None:
        from minivess.adapters.model_builder import build_adapter

        config = ModelConfig(
            family=ModelFamily.SAM3_TOPOLORA,
            name="topolora-test",
            in_channels=1,
            out_channels=2,
            lora_rank=2,
        )
        adapter = build_adapter(config, use_stub=True)
        cfg = adapter.get_config()
        assert cfg.family == "sam3_topolora"

    def test_build_sam3_hybrid(self) -> None:
        from minivess.adapters.model_builder import build_adapter

        config = ModelConfig(
            family=ModelFamily.SAM3_HYBRID,
            name="hybrid-test",
            in_channels=1,
            out_channels=2,
            architecture_params={"filters": [16, 32, 64]},
        )
        adapter = build_adapter(config, use_stub=True)
        cfg = adapter.get_config()
        assert cfg.family == "sam3_hybrid"

    def test_build_unknown_raises(self) -> None:
        from minivess.adapters.model_builder import build_adapter

        config = ModelConfig(
            family=ModelFamily.CUSTOM,
            name="unknown-test",
        )
        with pytest.raises(ValueError, match="Unsupported model family"):
            build_adapter(config)
