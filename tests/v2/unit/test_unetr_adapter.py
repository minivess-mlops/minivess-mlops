"""Tests for UNETR adapter (T-02.6a).

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
        family=ModelFamily.MONAI_UNETR,
        name="test-unetr",
        in_channels=1,
        out_channels=2,
        **kwargs,
    )


class TestUNETRAdapter:
    """UNETR adapter must satisfy the ModelAdapter ABC."""

    def test_instantiation(self) -> None:
        """UNETRAdapter constructs with default config."""
        from minivess.adapters.unetr import UNETRAdapter

        adapter = UNETRAdapter(_make_config())
        assert isinstance(adapter, ModelAdapter)

    def test_forward_shape(self) -> None:
        """Output shape must match input spatial dims."""
        from minivess.adapters.unetr import UNETRAdapter

        config = _make_config(architecture_params={"img_size": (64, 64, 32)})
        adapter = UNETRAdapter(config)
        x = torch.randn(1, 1, 64, 64, 32)
        output = adapter(x)
        assert isinstance(output, SegmentationOutput)
        assert output.prediction.shape == (1, 2, 64, 64, 32)
        assert output.logits.shape == (1, 2, 64, 64, 32)

    def test_get_config(self) -> None:
        """get_config() returns AdapterConfigInfo."""
        from minivess.adapters.unetr import UNETRAdapter

        adapter = UNETRAdapter(_make_config())
        cfg = adapter.get_config()
        assert isinstance(cfg, AdapterConfigInfo)
        assert cfg.family == "unetr"
        assert "hidden_size" in cfg.extras

    def test_registered_in_factory(self) -> None:
        """build_adapter with MONAI_UNETR config returns UNETRAdapter."""
        from minivess.adapters.model_builder import build_adapter
        from minivess.adapters.unetr import UNETRAdapter

        adapter = build_adapter(_make_config())
        assert isinstance(adapter, UNETRAdapter)
