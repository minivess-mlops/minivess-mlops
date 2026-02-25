from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from minivess.adapters.base import ModelAdapter, SegmentationOutput
from minivess.adapters.segresnet import SegResNetAdapter
from minivess.adapters.swinunetr import SwinUNETRAdapter
from minivess.config.models import ModelConfig, ModelFamily

if TYPE_CHECKING:
    from pathlib import Path


class TestSegmentationOutput:
    """Test SegmentationOutput dataclass."""

    def test_creation(self) -> None:
        out = SegmentationOutput(
            prediction=torch.randn(1, 2, 8, 8, 4),
            logits=torch.randn(1, 2, 8, 8, 4),
        )
        assert out.prediction.shape == (1, 2, 8, 8, 4)
        assert out.metadata == {}

    def test_with_metadata(self) -> None:
        out = SegmentationOutput(
            prediction=torch.randn(1, 2, 8, 8, 4),
            logits=torch.randn(1, 2, 8, 8, 4),
            metadata={"architecture": "test"},
        )
        assert out.metadata["architecture"] == "test"

    def test_logits_shape_matches_prediction(self) -> None:
        shape = (2, 3, 16, 16, 8)
        out = SegmentationOutput(
            prediction=torch.randn(*shape),
            logits=torch.randn(*shape),
        )
        assert out.logits.shape == out.prediction.shape

    def test_empty_metadata_is_independent_per_instance(self) -> None:
        """Default metadata dicts should not be shared across instances."""
        out1 = SegmentationOutput(
            prediction=torch.zeros(1, 1, 1, 1, 1),
            logits=torch.zeros(1, 1, 1, 1, 1),
        )
        out2 = SegmentationOutput(
            prediction=torch.zeros(1, 1, 1, 1, 1),
            logits=torch.zeros(1, 1, 1, 1, 1),
        )
        out1.metadata["key"] = "value"
        assert "key" not in out2.metadata


class TestSegResNetAdapter:
    """Test SegResNet adapter."""

    @pytest.fixture
    def config(self) -> ModelConfig:
        return ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="test-segresnet",
            in_channels=1,
            out_channels=2,
        )

    @pytest.fixture
    def adapter(self, config: ModelConfig) -> SegResNetAdapter:
        return SegResNetAdapter(config)

    def test_is_model_adapter(self, adapter: SegResNetAdapter) -> None:
        assert isinstance(adapter, ModelAdapter)

    def test_forward_shape(self, adapter: SegResNetAdapter) -> None:
        x = torch.randn(1, 1, 32, 32, 16)
        output = adapter(x)
        assert isinstance(output, SegmentationOutput)
        assert output.prediction.shape == (1, 2, 32, 32, 16)
        assert output.logits.shape == (1, 2, 32, 32, 16)

    def test_prediction_is_probability(self, adapter: SegResNetAdapter) -> None:
        x = torch.randn(1, 1, 32, 32, 16)
        output = adapter(x)
        # Probabilities should sum to ~1 along channel dim
        sums = output.prediction.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_prediction_non_negative(self, adapter: SegResNetAdapter) -> None:
        x = torch.randn(1, 1, 32, 32, 16)
        output = adapter(x)
        assert (output.prediction >= 0.0).all()

    def test_get_config(self, adapter: SegResNetAdapter) -> None:
        cfg = adapter.get_config()
        assert cfg.family == "segresnet"
        assert cfg.in_channels == 1
        assert cfg.out_channels == 2
        assert cfg.trainable_params is not None
        assert cfg.extras["init_filters"] == 32

    def test_trainable_parameters(self, adapter: SegResNetAdapter) -> None:
        count = adapter.trainable_parameters()
        assert count > 0
        assert isinstance(count, int)

    def test_save_load_checkpoint(
        self, adapter: SegResNetAdapter, tmp_path: Path
    ) -> None:
        ckpt_path = tmp_path / "model.pth"
        adapter.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

        # Load into a new adapter with same architecture
        new_config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="loaded",
            in_channels=1,
            out_channels=2,
        )
        new_adapter = SegResNetAdapter(new_config)
        new_adapter.load_checkpoint(ckpt_path)

        # Verify weights match after loading (eval mode to disable dropout)
        adapter.eval()
        new_adapter.eval()
        x = torch.randn(1, 1, 32, 32, 16)
        with torch.no_grad():
            original_output = adapter(x)
            loaded_output = new_adapter(x)
        assert torch.allclose(original_output.logits, loaded_output.logits, atol=1e-6)

    def test_save_creates_parent_dirs(
        self, adapter: SegResNetAdapter, tmp_path: Path
    ) -> None:
        ckpt_path = tmp_path / "nested" / "dir" / "model.pth"
        adapter.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

    def test_metadata(self, adapter: SegResNetAdapter) -> None:
        x = torch.randn(1, 1, 32, 32, 16)
        output = adapter(x)
        assert output.metadata["architecture"] == "segresnet"

    def test_batch_size_preserved(self, adapter: SegResNetAdapter) -> None:
        """Output batch size should match input batch size."""
        x = torch.randn(2, 1, 32, 32, 16)
        output = adapter(x)
        assert output.prediction.shape[0] == 2
        assert output.logits.shape[0] == 2


class TestSwinUNETRAdapter:
    """Test SwinUNETR adapter."""

    @pytest.fixture
    def config(self) -> ModelConfig:
        return ModelConfig(
            family=ModelFamily.MONAI_SWINUNETR,
            name="test-swinunetr",
            in_channels=1,
            out_channels=2,
        )

    @pytest.fixture
    def adapter(self, config: ModelConfig) -> SwinUNETRAdapter:
        return SwinUNETRAdapter(config, feature_size=24)

    def test_is_model_adapter(self, adapter: SwinUNETRAdapter) -> None:
        assert isinstance(adapter, ModelAdapter)

    def test_forward_shape(self, adapter: SwinUNETRAdapter) -> None:
        x = torch.randn(1, 1, 64, 64, 32)
        output = adapter(x)
        assert isinstance(output, SegmentationOutput)
        assert output.prediction.shape == (1, 2, 64, 64, 32)

    def test_prediction_is_probability(self, adapter: SwinUNETRAdapter) -> None:
        x = torch.randn(1, 1, 64, 64, 32)
        output = adapter(x)
        sums = output.prediction.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_get_config(self, adapter: SwinUNETRAdapter) -> None:
        cfg = adapter.get_config()
        assert cfg.family == "swinunetr"
        assert cfg.extras["feature_size"] == 24

    def test_trainable_parameters(self, adapter: SwinUNETRAdapter) -> None:
        count = adapter.trainable_parameters()
        assert count > 0
        assert isinstance(count, int)

    def test_metadata(self, adapter: SwinUNETRAdapter) -> None:
        x = torch.randn(1, 1, 64, 64, 32)
        output = adapter(x)
        assert output.metadata["architecture"] == "swinunetr"

    def test_save_load_checkpoint(
        self, adapter: SwinUNETRAdapter, tmp_path: Path
    ) -> None:
        ckpt_path = tmp_path / "swin.pth"
        adapter.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

        new_config = ModelConfig(
            family=ModelFamily.MONAI_SWINUNETR,
            name="loaded",
            in_channels=1,
            out_channels=2,
        )
        new_adapter = SwinUNETRAdapter(new_config, feature_size=24)
        new_adapter.load_checkpoint(ckpt_path)
