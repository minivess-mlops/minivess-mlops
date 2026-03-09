"""Tests for SegResNet adapter (T-02.4a).

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
        family=ModelFamily.MONAI_SEGRESNET,
        name="test-segresnet",
        in_channels=1,
        out_channels=2,
        **kwargs,
    )


class TestSegResNetAdapter:
    """SegResNet adapter must satisfy the ModelAdapter ABC."""

    def test_instantiation_default(self) -> None:
        """SegResNetAdapter constructs with default config."""
        from minivess.adapters.segresnet import SegResNetAdapter

        adapter = SegResNetAdapter(_make_config())
        assert isinstance(adapter, ModelAdapter)

    def test_instantiation_custom_filters(self) -> None:
        """init_filters parameter is respected."""
        from minivess.adapters.segresnet import SegResNetAdapter

        config = _make_config(architecture_params={"init_filters": 32})
        adapter = SegResNetAdapter(config)
        assert isinstance(adapter, ModelAdapter)

    def test_forward_shape(self) -> None:
        """Input (1,1,64,64,32) → SegmentationOutput with matching shape."""
        from minivess.adapters.segresnet import SegResNetAdapter

        adapter = SegResNetAdapter(_make_config())
        x = torch.randn(1, 1, 64, 64, 32)
        output = adapter(x)
        assert isinstance(output, SegmentationOutput)
        assert output.prediction.shape == (1, 2, 64, 64, 32)
        assert output.logits.shape == (1, 2, 64, 64, 32)

    def test_prediction_sums_to_one(self) -> None:
        """Softmax prediction sums to 1 across class dim."""
        from minivess.adapters.segresnet import SegResNetAdapter

        adapter = SegResNetAdapter(_make_config())
        x = torch.randn(1, 1, 32, 32, 16)
        output = adapter(x)
        sums = output.prediction.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_get_config_returns_adapter_config_info(self) -> None:
        """get_config() returns AdapterConfigInfo with expected fields."""
        from minivess.adapters.segresnet import SegResNetAdapter

        adapter = SegResNetAdapter(_make_config())
        cfg = adapter.get_config()
        assert isinstance(cfg, AdapterConfigInfo)
        assert cfg.family == "segresnet"
        assert cfg.in_channels == 1
        assert cfg.out_channels == 2
        assert "init_filters" in cfg.extras

    def test_trainable_params_positive(self) -> None:
        """trainable_parameters() returns a positive integer."""
        from minivess.adapters.segresnet import SegResNetAdapter

        adapter = SegResNetAdapter(_make_config())
        count = adapter.trainable_parameters()
        assert isinstance(count, int)
        assert count > 0

    def test_registered_in_factory(self) -> None:
        """build_adapter with MONAI_SEGRESNET config returns SegResNetAdapter."""
        from minivess.adapters.model_builder import build_adapter
        from minivess.adapters.segresnet import SegResNetAdapter

        adapter = build_adapter(_make_config())
        assert isinstance(adapter, SegResNetAdapter)
