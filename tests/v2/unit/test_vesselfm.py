"""Tests for vesselFM foundation model adapter (Issue #3)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import torch

from minivess.adapters.base import ModelAdapter, SegmentationOutput
from minivess.config.models import ModelConfig, ModelFamily

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# T1: VesselFMAdapter — ModelAdapter ABC compliance
# ---------------------------------------------------------------------------


class TestVesselFMAdapter:
    """Test vesselFM adapter implementation."""

    @pytest.fixture
    def config(self) -> ModelConfig:
        return ModelConfig(
            family=ModelFamily.VESSEL_FM,
            name="test-vesselfm",
            in_channels=1,
            out_channels=2,
        )

    @pytest.fixture
    def adapter(self, config: ModelConfig) -> ModelAdapter:
        from minivess.adapters.vesselfm import VesselFMAdapter

        return VesselFMAdapter(config, pretrained=False)

    def test_is_model_adapter(self, adapter: ModelAdapter) -> None:
        """VesselFMAdapter must be a ModelAdapter instance."""
        assert isinstance(adapter, ModelAdapter)

    def test_forward_shape(self, adapter: ModelAdapter) -> None:
        """Forward should return correct (B, C, D, H, W) shapes."""
        x = torch.randn(1, 1, 64, 64, 64)
        output = adapter(x)
        assert isinstance(output, SegmentationOutput)
        assert output.prediction.shape == (1, 2, 64, 64, 64)
        assert output.logits.shape == (1, 2, 64, 64, 64)

    def test_prediction_is_probability(self, adapter: ModelAdapter) -> None:
        """Prediction should be softmax probabilities summing to ~1."""
        x = torch.randn(1, 1, 64, 64, 64)
        output = adapter(x)
        sums = output.prediction.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_get_config(self, adapter: ModelAdapter) -> None:
        """Config should identify vesselFM architecture."""
        cfg = adapter.get_config()
        assert cfg["family"] == "vesselfm"
        assert cfg["architecture"] == "vesselfm"
        assert "trainable_params" in cfg

    def test_trainable_parameters(self, adapter: ModelAdapter) -> None:
        """Should report positive trainable parameter count."""
        count = adapter.trainable_parameters()
        assert count > 0
        assert isinstance(count, int)

    def test_save_load_checkpoint(
        self, adapter: ModelAdapter, config: ModelConfig, tmp_path: Path
    ) -> None:
        """Checkpoint save/load should preserve weights."""
        from minivess.adapters.vesselfm import VesselFMAdapter

        ckpt_path = tmp_path / "vesselfm.pth"
        adapter.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

        new_adapter = VesselFMAdapter(config, pretrained=False)
        new_adapter.load_checkpoint(ckpt_path)

        adapter.eval()
        new_adapter.eval()
        x = torch.randn(1, 1, 64, 64, 64)
        with torch.no_grad():
            orig = adapter(x)
            loaded = new_adapter(x)
        assert torch.allclose(orig.logits, loaded.logits, atol=1e-6)

    def test_metadata(self, adapter: ModelAdapter) -> None:
        """Metadata should identify the architecture."""
        x = torch.randn(1, 1, 64, 64, 64)
        output = adapter(x)
        assert output.metadata["architecture"] == "vesselfm"


# ---------------------------------------------------------------------------
# T2: VesselFM config and enum
# ---------------------------------------------------------------------------


class TestVesselFMConfig:
    """Test vesselFM configuration."""

    def test_model_family_enum(self) -> None:
        """VESSEL_FM should be in ModelFamily enum."""
        assert hasattr(ModelFamily, "VESSEL_FM")
        assert ModelFamily.VESSEL_FM.value == "vesselfm"

    def test_default_filters(self) -> None:
        """vesselFM uses [32, 64, 128, 256, 320, 320] filters."""
        from minivess.adapters.vesselfm import VESSELFM_FILTERS

        assert VESSELFM_FILTERS == [32, 64, 128, 256, 320, 320]

    @patch("minivess.adapters.vesselfm.hf_hub_download")
    def test_pretrained_downloads_from_hf(
        self, mock_download: MagicMock, tmp_path: Path
    ) -> None:
        """pretrained=True should attempt HuggingFace download."""
        from minivess.adapters.vesselfm import VesselFMAdapter

        # Mock the download to return a fake path
        fake_ckpt = tmp_path / "fake.pt"
        # Create a valid state dict for the model
        config = ModelConfig(
            family=ModelFamily.VESSEL_FM,
            name="test",
            in_channels=1,
            out_channels=2,
        )
        temp_adapter = VesselFMAdapter(config, pretrained=False)
        torch.save(temp_adapter.net.state_dict(), fake_ckpt)

        mock_download.return_value = str(fake_ckpt)
        adapter = VesselFMAdapter(config, pretrained=True)
        mock_download.assert_called_once()
        assert adapter is not None
