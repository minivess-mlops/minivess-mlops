"""Tests for SwinUNETR adapter (T-02.5a).

Closes: #474 (MONAI ecosystem audit)
"""

from __future__ import annotations

import pytest
import torch

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput
from minivess.config.models import ModelConfig, ModelFamily

pytestmark = pytest.mark.model_loading


def _make_config(**kwargs: object) -> ModelConfig:
    return ModelConfig(
        family=ModelFamily.MONAI_SWINUNETR,
        name="test-swinunetr",
        in_channels=1,
        out_channels=2,
        **kwargs,
    )


class TestSwinUNETRAdapter:
    """SwinUNETR adapter must satisfy the ModelAdapter ABC."""

    def test_instantiation(self) -> None:
        """SwinUNETRAdapter constructs with default config."""
        from minivess.adapters.swinunetr import SwinUNETRAdapter

        adapter = SwinUNETRAdapter(_make_config())
        assert isinstance(adapter, ModelAdapter)

    def test_forward_shape(self) -> None:
        """Output shape must match input spatial dims."""
        from minivess.adapters.swinunetr import SwinUNETRAdapter

        # img_size must match the patch size used in __init__
        config = _make_config(architecture_params={"img_size": (64, 64, 32)})
        adapter = SwinUNETRAdapter(config)
        x = torch.randn(1, 1, 64, 64, 32)
        output = adapter(x)
        assert isinstance(output, SegmentationOutput)
        assert output.prediction.shape == (1, 2, 64, 64, 32)
        assert output.logits.shape == (1, 2, 64, 64, 32)

    def test_get_config(self) -> None:
        """get_config() returns AdapterConfigInfo."""
        from minivess.adapters.swinunetr import SwinUNETRAdapter

        adapter = SwinUNETRAdapter(_make_config())
        cfg = adapter.get_config()
        assert isinstance(cfg, AdapterConfigInfo)
        assert cfg.family == "swinunetr"
        assert "feature_size" in cfg.extras

    def test_feature_size_configurable(self) -> None:
        """feature_size architecture_param is respected."""
        from minivess.adapters.swinunetr import SwinUNETRAdapter

        config = _make_config(architecture_params={"feature_size": 24})
        adapter = SwinUNETRAdapter(config)
        cfg = adapter.get_config()
        assert cfg.extras["feature_size"] == 24

    def test_registered_in_factory(self) -> None:
        """build_adapter with MONAI_SWINUNETR config returns SwinUNETRAdapter."""
        from minivess.adapters.model_builder import build_adapter
        from minivess.adapters.swinunetr import SwinUNETRAdapter

        adapter = build_adapter(_make_config())
        assert isinstance(adapter, SwinUNETRAdapter)
