"""Tests for AttentionUnet adapter (T-02.7a).

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
        family=ModelFamily.MONAI_ATTENTIONUNET,
        name="test-attentionunet",
        in_channels=1,
        out_channels=2,
        **kwargs,
    )


class TestAttentionUnetAdapter:
    """AttentionUnet adapter must satisfy the ModelAdapter ABC."""

    def test_instantiation(self) -> None:
        """AttentionUnetAdapter constructs with default config."""
        from minivess.adapters.attentionunet import AttentionUnetAdapter

        adapter = AttentionUnetAdapter(_make_config())
        assert isinstance(adapter, ModelAdapter)

    def test_forward_shape(self) -> None:
        """Output shape must match input spatial dims."""
        from minivess.adapters.attentionunet import AttentionUnetAdapter

        adapter = AttentionUnetAdapter(_make_config())
        x = torch.randn(1, 1, 64, 64, 32)
        output = adapter(x)
        assert isinstance(output, SegmentationOutput)
        assert output.prediction.shape == (1, 2, 64, 64, 32)
        assert output.logits.shape == (1, 2, 64, 64, 32)

    def test_get_config(self) -> None:
        """get_config() returns AdapterConfigInfo."""
        from minivess.adapters.attentionunet import AttentionUnetAdapter

        adapter = AttentionUnetAdapter(_make_config())
        cfg = adapter.get_config()
        assert isinstance(cfg, AdapterConfigInfo)
        assert cfg.family == "attentionunet"
        assert "channels" in cfg.extras

    def test_registered_in_factory(self) -> None:
        """build_adapter with MONAI_ATTENTIONUNET config returns AttentionUnetAdapter."""
        from minivess.adapters.attentionunet import AttentionUnetAdapter
        from minivess.adapters.model_builder import build_adapter

        adapter = build_adapter(_make_config())
        assert isinstance(adapter, AttentionUnetAdapter)
