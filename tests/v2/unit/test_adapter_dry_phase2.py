"""Tests for Adapter DRY Phase 2 — R5.3, R5.5, R5.28 (Issue #52).

R5.3: _build_output() helper on ModelAdapter base class
R5.5: _build_config() helper on ModelAdapter base class
R5.28: Standardize comma.py parameter names
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor, nn

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput
from minivess.config.models import ModelConfig, ModelFamily

# ---------------------------------------------------------------------------
# Minimal concrete adapter for testing base helpers
# ---------------------------------------------------------------------------


class _StubAdapter(ModelAdapter):
    """Minimal adapter for testing _build_output / _build_config helpers."""

    def __init__(self) -> None:
        super().__init__()
        self.config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="stub",
            in_channels=1,
            out_channels=2,
        )
        self.net = nn.Conv3d(1, 2, kernel_size=1)

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        logits = self.net(images)
        return self._build_output(logits, "stub")

    def get_config(self) -> AdapterConfigInfo:
        return self._build_config(custom_key="custom_val")


# =========================================================================
# R5.3: _build_output() helper
# =========================================================================


class TestBuildOutput:
    """Test _build_output() helper on ModelAdapter base class."""

    def test_build_output_returns_segmentation_output(self) -> None:
        """_build_output should return a SegmentationOutput instance."""
        adapter = _StubAdapter()
        logits = torch.randn(1, 2, 4, 4, 4)
        result = adapter._build_output(logits, "test_arch")
        assert isinstance(result, SegmentationOutput)

    def test_build_output_prediction_is_softmax(self) -> None:
        """_build_output prediction should be softmax of logits."""
        adapter = _StubAdapter()
        logits = torch.randn(1, 2, 4, 4, 4)
        result = adapter._build_output(logits, "test_arch")
        # Softmax probabilities sum to 1 along dim=1
        sums = result.prediction.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_build_output_logits_unchanged(self) -> None:
        """_build_output should pass logits through unchanged."""
        adapter = _StubAdapter()
        logits = torch.randn(1, 2, 4, 4, 4)
        result = adapter._build_output(logits, "test_arch")
        assert torch.equal(result.logits, logits)

    def test_build_output_metadata_architecture(self) -> None:
        """_build_output should set architecture in metadata."""
        adapter = _StubAdapter()
        logits = torch.randn(1, 2, 4, 4, 4)
        result = adapter._build_output(logits, "my_arch")
        assert result.metadata["architecture"] == "my_arch"

    def test_build_output_used_in_forward(self) -> None:
        """_StubAdapter.forward should use _build_output correctly."""
        adapter = _StubAdapter()
        x = torch.randn(1, 1, 4, 4, 4)
        output = adapter(x)
        assert isinstance(output, SegmentationOutput)
        assert output.metadata["architecture"] == "stub"


# =========================================================================
# R5.5: _build_config() helper
# =========================================================================


class TestBuildConfig:
    """Test _build_config() helper on ModelAdapter base class."""

    def test_build_config_returns_adapter_config_info(self) -> None:
        """_build_config should return an AdapterConfigInfo instance."""
        adapter = _StubAdapter()
        cfg = adapter._build_config()
        assert isinstance(cfg, AdapterConfigInfo)

    def test_build_config_auto_populates_from_self_config(self) -> None:
        """_build_config should auto-populate family, name, channels."""
        adapter = _StubAdapter()
        cfg = adapter._build_config()
        assert cfg.family == "segresnet"
        assert cfg.name == "stub"
        assert cfg.in_channels == 1
        assert cfg.out_channels == 2

    def test_build_config_includes_trainable_params(self) -> None:
        """_build_config should include trainable_params count."""
        adapter = _StubAdapter()
        cfg = adapter._build_config()
        assert cfg.trainable_params is not None
        assert cfg.trainable_params > 0

    def test_build_config_extras_forwarded(self) -> None:
        """_build_config(**extras) should put extras in the extras dict."""
        adapter = _StubAdapter()
        cfg = adapter._build_config(custom_key="custom_val", another=42)
        assert cfg.extras["custom_key"] == "custom_val"
        assert cfg.extras["another"] == 42

    def test_build_config_used_in_get_config(self) -> None:
        """_StubAdapter.get_config should use _build_config correctly."""
        adapter = _StubAdapter()
        cfg = adapter.get_config()
        assert cfg.family == "segresnet"
        assert cfg.extras["custom_key"] == "custom_val"


# =========================================================================
# R5.3/R5.5: Verify real adapters can use helpers
# =========================================================================


class TestAdaptersUseBuildHelpers:
    """Verify that real adapters use _build_output and _build_config."""

    @pytest.fixture
    def segresnet_adapter(self) -> ModelAdapter:
        config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="test-segresnet",
            in_channels=1,
            out_channels=2,
        )
        from minivess.adapters.segresnet import SegResNetAdapter

        return SegResNetAdapter(config)

    @pytest.fixture
    def swinunetr_adapter(self) -> ModelAdapter:
        config = ModelConfig(
            family=ModelFamily.MONAI_SWINUNETR,
            name="test-swinunetr",
            in_channels=1,
            out_channels=2,
        )
        from minivess.adapters.swinunetr import SwinUNETRAdapter

        return SwinUNETRAdapter(config, feature_size=24)

    def test_segresnet_forward_still_works(
        self, segresnet_adapter: ModelAdapter
    ) -> None:
        """SegResNetAdapter should still produce correct output after refactor."""
        x = torch.randn(1, 1, 32, 32, 16)
        output = segresnet_adapter(x)
        assert isinstance(output, SegmentationOutput)
        assert output.prediction.shape == (1, 2, 32, 32, 16)
        assert output.metadata["architecture"] == "segresnet"
        sums = output.prediction.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_segresnet_get_config_still_works(
        self, segresnet_adapter: ModelAdapter
    ) -> None:
        """SegResNetAdapter.get_config should still produce correct output."""
        cfg = segresnet_adapter.get_config()
        assert cfg.family == "segresnet"
        assert cfg.in_channels == 1
        assert cfg.out_channels == 2
        assert cfg.trainable_params is not None
        assert cfg.extras["init_filters"] == 32

    def test_swinunetr_forward_still_works(
        self, swinunetr_adapter: ModelAdapter
    ) -> None:
        """SwinUNETRAdapter should still produce correct output after refactor."""
        x = torch.randn(1, 1, 64, 64, 32)
        output = swinunetr_adapter(x)
        assert isinstance(output, SegmentationOutput)
        assert output.prediction.shape == (1, 2, 64, 64, 32)
        assert output.metadata["architecture"] == "swinunetr"

    def test_swinunetr_get_config_still_works(
        self, swinunetr_adapter: ModelAdapter
    ) -> None:
        """SwinUNETRAdapter.get_config should still produce correct output."""
        cfg = swinunetr_adapter.get_config()
        assert cfg.family == "swinunetr"
        assert cfg.extras["feature_size"] == 24


# =========================================================================
# R5.28: Standardize comma.py parameter names
# =========================================================================


class TestCommaParameterStandardization:
    """Test that comma.py internal blocks use standardized parameter names."""

    def test_encoder_block_uses_full_names(self) -> None:
        """_CommaEncoderBlock should accept in_channels, out_channels."""
        import inspect

        from minivess.adapters.comma import _CommaEncoderBlock

        sig = inspect.signature(_CommaEncoderBlock.__init__)
        params = list(sig.parameters.keys())
        assert "in_channels" in params, f"Expected 'in_channels', got {params}"
        assert "out_channels" in params, f"Expected 'out_channels', got {params}"

    def test_decoder_block_uses_full_names(self) -> None:
        """_CommaDecoderBlock should accept in_channels, skip_channels, out_channels."""
        import inspect

        from minivess.adapters.comma import _CommaDecoderBlock

        sig = inspect.signature(_CommaDecoderBlock.__init__)
        params = list(sig.parameters.keys())
        assert "in_channels" in params, f"Expected 'in_channels', got {params}"
        assert "out_channels" in params, f"Expected 'out_channels', got {params}"
        assert "skip_channels" in params, f"Expected 'skip_channels', got {params}"

    def test_encoder_block_functional(self) -> None:
        """_CommaEncoderBlock should still work with standardized names."""
        from minivess.adapters.comma import _CommaEncoderBlock

        block = _CommaEncoderBlock(in_channels=1, out_channels=16, d_state=8)
        x = torch.randn(1, 1, 8, 8, 8)
        down, skip = block(x)
        assert skip.shape[1] == 16  # out_channels
        assert down.shape[1] == 16

    def test_decoder_block_functional(self) -> None:
        """_CommaDecoderBlock should still work with standardized names."""
        from minivess.adapters.comma import _CommaDecoderBlock

        block = _CommaDecoderBlock(in_channels=32, skip_channels=16, out_channels=16)
        x = torch.randn(1, 32, 4, 4, 4)
        skip = torch.randn(1, 16, 8, 8, 8)
        out = block(x, skip)
        assert out.shape[1] == 16  # out_channels

    def test_comma_adapter_still_functional(self) -> None:
        """CommaAdapter should still work end-to-end after renaming."""
        from minivess.adapters.comma import CommaAdapter

        config = ModelConfig(
            family=ModelFamily.COMMA_MAMBA,
            name="test-comma",
            in_channels=1,
            out_channels=2,
        )
        adapter = CommaAdapter(config)
        x = torch.randn(1, 1, 32, 32, 16)
        output = adapter(x)
        assert isinstance(output, SegmentationOutput)
        assert output.prediction.shape == (1, 2, 32, 32, 16)
        assert output.metadata["architecture"] == "comma_mamba"
