"""Tests for COMMA/Mamba architecture adapter (Issue #9)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from minivess.adapters.base import ModelAdapter, SegmentationOutput
from minivess.config.models import ModelConfig, ModelFamily

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# T1: CommaAdapter — ModelAdapter ABC compliance
# ---------------------------------------------------------------------------


class TestCommaAdapter:
    """Test CommaAdapter implements ModelAdapter correctly."""

    @pytest.fixture
    def config(self) -> ModelConfig:
        return ModelConfig(
            family=ModelFamily.COMMA_MAMBA,
            name="test-comma",
            in_channels=1,
            out_channels=2,
        )

    @pytest.fixture
    def adapter(self, config: ModelConfig) -> ModelAdapter:
        from minivess.adapters.comma import CommaAdapter

        return CommaAdapter(config)

    def test_is_model_adapter(self, adapter: ModelAdapter) -> None:
        """CommaAdapter must be a ModelAdapter instance."""
        assert isinstance(adapter, ModelAdapter)

    def test_forward_shape(self, adapter: ModelAdapter) -> None:
        """Forward should return correct (B, C, D, H, W) shapes."""
        x = torch.randn(1, 1, 32, 32, 16)
        output = adapter(x)
        assert isinstance(output, SegmentationOutput)
        assert output.prediction.shape == (1, 2, 32, 32, 16)
        assert output.logits.shape == (1, 2, 32, 32, 16)

    def test_prediction_is_probability(self, adapter: ModelAdapter) -> None:
        """Prediction should be softmax probabilities summing to ~1."""
        x = torch.randn(1, 1, 32, 32, 16)
        output = adapter(x)
        sums = output.prediction.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_prediction_non_negative(self, adapter: ModelAdapter) -> None:
        """Softmax outputs must be non-negative."""
        x = torch.randn(1, 1, 32, 32, 16)
        output = adapter(x)
        assert (output.prediction >= 0.0).all()

    def test_batch_size_preserved(self, adapter: ModelAdapter) -> None:
        """Output batch size should match input batch size."""
        x = torch.randn(2, 1, 32, 32, 16)
        output = adapter(x)
        assert output.prediction.shape[0] == 2
        assert output.logits.shape[0] == 2

    def test_metadata_contains_architecture(self, adapter: ModelAdapter) -> None:
        """Metadata should identify the architecture."""
        x = torch.randn(1, 1, 32, 32, 16)
        output = adapter(x)
        assert output.metadata["architecture"] == "comma_mamba"


# ---------------------------------------------------------------------------
# T2: MambaBlock — standalone SSM block
# ---------------------------------------------------------------------------


class TestMambaBlock:
    """Test the pure-PyTorch Mamba-style SSM block."""

    def test_forward_shape(self) -> None:
        """MambaBlock should preserve (B, L, D) sequence shape."""
        from minivess.adapters.comma import MambaBlock

        block = MambaBlock(d_model=32, d_state=16, d_conv=4, expand=2)
        x = torch.randn(2, 64, 32)  # (B, L, D)
        y = block(x)
        assert y.shape == (2, 64, 32)

    def test_different_sequence_lengths(self) -> None:
        """MambaBlock should handle variable-length sequences."""
        from minivess.adapters.comma import MambaBlock

        block = MambaBlock(d_model=16, d_state=8, d_conv=4, expand=2)
        for seq_len in [32, 128, 256]:
            x = torch.randn(1, seq_len, 16)
            y = block(x)
            assert y.shape == (1, seq_len, 16)

    def test_has_trainable_parameters(self) -> None:
        """MambaBlock should have learnable parameters."""
        from minivess.adapters.comma import MambaBlock

        block = MambaBlock(d_model=32, d_state=16, d_conv=4, expand=2)
        n_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
        assert n_params > 0


# ---------------------------------------------------------------------------
# T3: CoordinateEmbedding — 3D coordinate-aware positional embedding
# ---------------------------------------------------------------------------


class TestCoordinateEmbedding:
    """Test coordinate-aware 3D positional embedding."""

    def test_output_shape(self) -> None:
        """Embedding should preserve spatial shape and add channels."""
        from minivess.adapters.comma import CoordinateEmbedding

        embed = CoordinateEmbedding(in_channels=32)
        x = torch.randn(1, 32, 16, 16, 8)
        y = embed(x)
        # Output channels = in_channels + 3 coordinate channels processed back
        assert y.shape[0] == 1
        assert y.shape[2:] == (16, 16, 8)

    def test_different_spatial_sizes(self) -> None:
        """Embedding should work with varying spatial dimensions."""
        from minivess.adapters.comma import CoordinateEmbedding

        embed = CoordinateEmbedding(in_channels=16)
        for d, h, w in [(8, 8, 4), (16, 16, 8), (32, 32, 16)]:
            x = torch.randn(1, 16, d, h, w)
            y = embed(x)
            assert y.shape[2:] == (d, h, w)


# ---------------------------------------------------------------------------
# T4: Config and checkpoint management
# ---------------------------------------------------------------------------


class TestCommaConfig:
    """Test CommaAdapter configuration and checkpoint management."""

    @pytest.fixture
    def config(self) -> ModelConfig:
        return ModelConfig(
            family=ModelFamily.COMMA_MAMBA,
            name="test-comma",
            in_channels=1,
            out_channels=2,
        )

    @pytest.fixture
    def adapter(self, config: ModelConfig) -> ModelAdapter:
        from minivess.adapters.comma import CommaAdapter

        return CommaAdapter(config)

    def test_get_config(self, adapter: ModelAdapter) -> None:
        """Config dict should contain architecture details."""
        cfg = adapter.get_config()
        assert cfg["family"] == "comma_mamba"
        assert cfg["in_channels"] == 1
        assert cfg["out_channels"] == 2
        assert "trainable_params" in cfg
        assert "d_state" in cfg

    def test_trainable_parameters(self, adapter: ModelAdapter) -> None:
        """Should report positive trainable parameter count."""
        count = adapter.trainable_parameters()
        assert count > 0
        assert isinstance(count, int)

    def test_save_load_checkpoint(
        self, adapter: ModelAdapter, config: ModelConfig, tmp_path: Path
    ) -> None:
        """Checkpoint save/load should preserve model weights."""
        from minivess.adapters.comma import CommaAdapter

        ckpt_path = tmp_path / "comma.pth"
        adapter.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

        new_adapter = CommaAdapter(config)
        new_adapter.load_checkpoint(ckpt_path)

        adapter.eval()
        new_adapter.eval()
        x = torch.randn(1, 1, 32, 32, 16)
        with torch.no_grad():
            orig = adapter(x)
            loaded = new_adapter(x)
        assert torch.allclose(orig.logits, loaded.logits, atol=1e-6)

    def test_save_creates_parent_dirs(
        self, adapter: ModelAdapter, tmp_path: Path
    ) -> None:
        """Save should create intermediate directories."""
        ckpt_path = tmp_path / "nested" / "dir" / "comma.pth"
        adapter.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

    def test_model_family_enum(self) -> None:
        """COMMA_MAMBA should be in ModelFamily enum."""
        assert hasattr(ModelFamily, "COMMA_MAMBA")
        assert ModelFamily.COMMA_MAMBA.value == "comma_mamba"
