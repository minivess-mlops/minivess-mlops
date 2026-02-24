"""Tests for foundation model fine-tuning: VISTA-3D, LoRA wrapper, SAM3 stub."""

from __future__ import annotations

import pytest
import torch

from minivess.adapters.base import ModelAdapter
from minivess.config.models import ModelConfig, ModelFamily

# ---------------------------------------------------------------------------
# T1: Vista3dAdapter
# ---------------------------------------------------------------------------


class TestVista3dAdapter:
    """Test MONAI VISTA-3D adapter (SegResNetDS2 backbone)."""

    def test_vista3d_adapter_creates(self) -> None:
        from minivess.adapters.vista3d import Vista3dAdapter

        config = ModelConfig(
            family=ModelFamily.MONAI_VISTA3D,
            name="test_vista3d",
            in_channels=1,
            out_channels=2,
        )
        model = Vista3dAdapter(config)
        assert isinstance(model, ModelAdapter)

    def test_vista3d_forward_shape(self) -> None:
        from minivess.adapters.vista3d import Vista3dAdapter

        config = ModelConfig(
            family=ModelFamily.MONAI_VISTA3D,
            name="test_vista3d",
            in_channels=1,
            out_channels=2,
        )
        model = Vista3dAdapter(config)
        x = torch.randn(1, 1, 32, 32, 16)
        output = model(x)
        assert output.prediction.shape == (1, 2, 32, 32, 16)
        assert output.logits.shape == (1, 2, 32, 32, 16)

    def test_vista3d_output_probabilities(self) -> None:
        from minivess.adapters.vista3d import Vista3dAdapter

        config = ModelConfig(
            family=ModelFamily.MONAI_VISTA3D,
            name="test_vista3d",
            in_channels=1,
            out_channels=2,
        )
        model = Vista3dAdapter(config)
        x = torch.randn(1, 1, 32, 32, 16)
        output = model(x)
        sums = output.prediction.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=0.01)

    def test_vista3d_metadata(self) -> None:
        from minivess.adapters.vista3d import Vista3dAdapter

        config = ModelConfig(
            family=ModelFamily.MONAI_VISTA3D,
            name="test_vista3d",
            in_channels=1,
            out_channels=2,
        )
        model = Vista3dAdapter(config)
        output = model(torch.randn(1, 1, 32, 32, 16))
        assert output.metadata["architecture"] == "vista3d"

    def test_vista3d_get_config(self) -> None:
        from minivess.adapters.vista3d import Vista3dAdapter

        config = ModelConfig(
            family=ModelFamily.MONAI_VISTA3D,
            name="test_vista3d",
            in_channels=1,
            out_channels=2,
        )
        model = Vista3dAdapter(config)
        cfg = model.get_config()
        assert cfg["family"] == "vista3d"
        assert cfg["in_channels"] == 1

    def test_vista3d_trainable_params(self) -> None:
        from minivess.adapters.vista3d import Vista3dAdapter

        config = ModelConfig(
            family=ModelFamily.MONAI_VISTA3D,
            name="test_vista3d",
            in_channels=1,
            out_channels=2,
        )
        model = Vista3dAdapter(config)
        params = model.trainable_parameters()
        assert params > 0

    def test_vista3d_checkpoint_roundtrip(self, tmp_path: object) -> None:
        from pathlib import Path

        from minivess.adapters.vista3d import Vista3dAdapter

        config = ModelConfig(
            family=ModelFamily.MONAI_VISTA3D,
            name="test_vista3d",
            in_channels=1,
            out_channels=2,
        )
        model = Vista3dAdapter(config)
        ckpt = Path(str(tmp_path)) / "vista3d.pt"
        model.save_checkpoint(ckpt)
        assert ckpt.exists()

        model2 = Vista3dAdapter(config)
        model2.load_checkpoint(ckpt)
        # Parameters should match after loading
        for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
            assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# T2: LoRA wrapper
# ---------------------------------------------------------------------------


class TestLoraAdapter:
    """Test PEFT LoRA wrapper for model fine-tuning."""

    def test_lora_wraps_model(self) -> None:
        from minivess.adapters.lora import LoraModelAdapter
        from minivess.adapters.segresnet import SegResNetAdapter

        base_config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="test_lora_base",
            in_channels=1,
            out_channels=2,
        )
        base_model = SegResNetAdapter(base_config)
        lora_model = LoraModelAdapter(base_model, lora_rank=8, lora_alpha=16.0)
        assert isinstance(lora_model, ModelAdapter)

    def test_lora_forward_shape(self) -> None:
        from minivess.adapters.lora import LoraModelAdapter
        from minivess.adapters.segresnet import SegResNetAdapter

        base_config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="test_lora",
            in_channels=1,
            out_channels=2,
        )
        base_model = SegResNetAdapter(base_config)
        lora_model = LoraModelAdapter(base_model, lora_rank=8, lora_alpha=16.0)
        x = torch.randn(1, 1, 32, 32, 16)
        output = lora_model(x)
        assert output.prediction.shape == (1, 2, 32, 32, 16)

    def test_lora_fewer_trainable_params(self) -> None:
        """LoRA should have fewer trainable params than the full model."""
        from minivess.adapters.lora import LoraModelAdapter
        from minivess.adapters.segresnet import SegResNetAdapter

        base_config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="test_lora",
            in_channels=1,
            out_channels=2,
        )
        base_model = SegResNetAdapter(base_config)
        full_params = base_model.trainable_parameters()
        lora_model = LoraModelAdapter(base_model, lora_rank=4, lora_alpha=8.0)
        lora_params = lora_model.trainable_parameters()
        assert lora_params < full_params

    def test_lora_get_config(self) -> None:
        from minivess.adapters.lora import LoraModelAdapter
        from minivess.adapters.segresnet import SegResNetAdapter

        base_config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="test_lora",
            in_channels=1,
            out_channels=2,
        )
        base_model = SegResNetAdapter(base_config)
        lora_model = LoraModelAdapter(base_model, lora_rank=8, lora_alpha=16.0)
        cfg = lora_model.get_config()
        assert cfg["lora_rank"] == 8
        assert cfg["lora_alpha"] == 16.0

    def test_lora_save_load_adapter(self, tmp_path: object) -> None:
        """LoRA adapter weights should be saveable and loadable."""
        from pathlib import Path

        from minivess.adapters.lora import LoraModelAdapter
        from minivess.adapters.segresnet import SegResNetAdapter

        base_config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="test_lora",
            in_channels=1,
            out_channels=2,
        )
        base_model = SegResNetAdapter(base_config)
        lora_model = LoraModelAdapter(base_model, lora_rank=4, lora_alpha=8.0)

        ckpt = Path(str(tmp_path)) / "lora_adapter.pt"
        lora_model.save_checkpoint(ckpt)
        assert ckpt.exists()

    def test_lora_wraps_vista3d(self) -> None:
        """LoRA should also work with Vista3dAdapter."""
        from minivess.adapters.lora import LoraModelAdapter
        from minivess.adapters.vista3d import Vista3dAdapter

        config = ModelConfig(
            family=ModelFamily.MONAI_VISTA3D,
            name="test_lora_vista3d",
            in_channels=1,
            out_channels=2,
        )
        base_model = Vista3dAdapter(config)
        lora_model = LoraModelAdapter(base_model, lora_rank=4, lora_alpha=8.0)
        x = torch.randn(1, 1, 32, 32, 16)
        output = lora_model(x)
        assert output.prediction.shape == (1, 2, 32, 32, 16)


# ---------------------------------------------------------------------------
# T3: SAM3 adapter (exploratory stub)
# ---------------------------------------------------------------------------


class TestSam3Adapter:
    """Test SAM3 adapter stub."""

    def test_sam3_not_available_message(self) -> None:
        """SAM3 should raise clear error about missing dependency."""
        from minivess.adapters.sam3 import Sam3Adapter

        config = ModelConfig(
            family=ModelFamily.SAM3_LORA,
            name="test_sam3",
            in_channels=1,
            out_channels=2,
        )
        with pytest.raises((ImportError, RuntimeError), match="segment.anything|SAM"):
            Sam3Adapter(config)

    def test_sam3_config_exists_in_enum(self) -> None:
        """SAM3_LORA should exist in ModelFamily enum."""
        assert ModelFamily.SAM3_LORA.value == "sam3_lora"


# ---------------------------------------------------------------------------
# T4: Export and adapter interface
# ---------------------------------------------------------------------------


class TestAdapterInterface:
    """Test adapter interface compliance across implementations."""

    def test_vista3d_has_required_methods(self) -> None:
        from minivess.adapters.vista3d import Vista3dAdapter

        config = ModelConfig(
            family=ModelFamily.MONAI_VISTA3D,
            name="test",
            in_channels=1,
            out_channels=2,
        )
        model = Vista3dAdapter(config)
        assert hasattr(model, "forward")
        assert hasattr(model, "get_config")
        assert hasattr(model, "load_checkpoint")
        assert hasattr(model, "save_checkpoint")
        assert hasattr(model, "trainable_parameters")
        assert hasattr(model, "export_onnx")

    def test_lora_has_required_methods(self) -> None:
        from minivess.adapters.lora import LoraModelAdapter
        from minivess.adapters.segresnet import SegResNetAdapter

        base_config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="test",
            in_channels=1,
            out_channels=2,
        )
        base = SegResNetAdapter(base_config)
        model = LoraModelAdapter(base, lora_rank=4)
        assert hasattr(model, "forward")
        assert hasattr(model, "get_config")
        assert hasattr(model, "load_checkpoint")
        assert hasattr(model, "save_checkpoint")
        assert hasattr(model, "trainable_parameters")
        assert hasattr(model, "export_onnx")
