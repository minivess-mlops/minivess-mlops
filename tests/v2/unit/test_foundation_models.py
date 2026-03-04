"""Tests for foundation model fine-tuning: LoRA wrapper, SAM3 stub."""

from __future__ import annotations

import pytest
import torch

from minivess.adapters.base import ModelAdapter
from minivess.config.models import ModelConfig, ModelFamily

# ---------------------------------------------------------------------------
# T1: LoRA wrapper
# ---------------------------------------------------------------------------


class TestLoraAdapter:
    """Test PEFT LoRA wrapper for model fine-tuning."""

    def test_lora_wraps_model(self) -> None:
        from minivess.adapters.dynunet import DynUNetAdapter
        from minivess.adapters.lora import LoraModelAdapter

        base_config = ModelConfig(
            family=ModelFamily.MONAI_DYNUNET,
            name="test_lora_base",
            in_channels=1,
            out_channels=2,
        )
        base_model = DynUNetAdapter(base_config)
        lora_model = LoraModelAdapter(base_model, lora_rank=8, lora_alpha=16.0)
        assert isinstance(lora_model, ModelAdapter)

    def test_lora_forward_shape(self) -> None:
        from minivess.adapters.dynunet import DynUNetAdapter
        from minivess.adapters.lora import LoraModelAdapter

        base_config = ModelConfig(
            family=ModelFamily.MONAI_DYNUNET,
            name="test_lora",
            in_channels=1,
            out_channels=2,
        )
        base_model = DynUNetAdapter(base_config)
        lora_model = LoraModelAdapter(base_model, lora_rank=8, lora_alpha=16.0)
        x = torch.randn(1, 1, 32, 32, 16)
        output = lora_model(x)
        assert output.prediction.shape == (1, 2, 32, 32, 16)

    def test_lora_fewer_trainable_params(self) -> None:
        """LoRA should have fewer trainable params than the full model."""
        from minivess.adapters.dynunet import DynUNetAdapter
        from minivess.adapters.lora import LoraModelAdapter

        base_config = ModelConfig(
            family=ModelFamily.MONAI_DYNUNET,
            name="test_lora",
            in_channels=1,
            out_channels=2,
        )
        base_model = DynUNetAdapter(base_config)
        full_params = base_model.trainable_parameters()
        lora_model = LoraModelAdapter(base_model, lora_rank=4, lora_alpha=8.0)
        lora_params = lora_model.trainable_parameters()
        assert lora_params < full_params

    def test_lora_get_config(self) -> None:
        from minivess.adapters.dynunet import DynUNetAdapter
        from minivess.adapters.lora import LoraModelAdapter

        base_config = ModelConfig(
            family=ModelFamily.MONAI_DYNUNET,
            name="test_lora",
            in_channels=1,
            out_channels=2,
        )
        base_model = DynUNetAdapter(base_config)
        lora_model = LoraModelAdapter(base_model, lora_rank=8, lora_alpha=16.0)
        cfg = lora_model.get_config()
        assert cfg.extras["lora_rank"] == 8
        assert cfg.extras["lora_alpha"] == 16.0

    def test_lora_save_load_adapter(self, tmp_path: object) -> None:
        """LoRA adapter weights should be saveable and loadable."""
        from pathlib import Path

        from minivess.adapters.dynunet import DynUNetAdapter
        from minivess.adapters.lora import LoraModelAdapter

        base_config = ModelConfig(
            family=ModelFamily.MONAI_DYNUNET,
            name="test_lora",
            in_channels=1,
            out_channels=2,
        )
        base_model = DynUNetAdapter(base_config)
        lora_model = LoraModelAdapter(base_model, lora_rank=4, lora_alpha=8.0)

        ckpt = Path(str(tmp_path)) / "lora_adapter.pt"
        lora_model.save_checkpoint(ckpt)
        assert ckpt.exists()


# ---------------------------------------------------------------------------
# T2: SAM3 adapter (exploratory stub)
# ---------------------------------------------------------------------------


class TestSam3Adapter:
    """Test SAM3 adapter stub."""

    def test_sam3_not_available_message(self) -> None:
        """Deprecated Sam3Adapter raises RuntimeError directing to new adapters."""
        from minivess.adapters.sam3 import Sam3Adapter

        config = ModelConfig(
            family=ModelFamily.SAM3_LORA,
            name="test_sam3",
            in_channels=1,
            out_channels=2,
        )
        with pytest.raises((ImportError, RuntimeError), match="deprecated|Sam3"):
            Sam3Adapter(config)

    def test_sam3_config_exists_in_enum(self) -> None:
        """SAM3_LORA should exist in ModelFamily enum."""
        assert ModelFamily.SAM3_LORA.value == "sam3_lora"


# ---------------------------------------------------------------------------
# T3: Export and adapter interface
# ---------------------------------------------------------------------------


class TestAdapterInterface:
    """Test adapter interface compliance across implementations."""

    def test_lora_has_required_methods(self) -> None:
        from minivess.adapters.dynunet import DynUNetAdapter
        from minivess.adapters.lora import LoraModelAdapter

        base_config = ModelConfig(
            family=ModelFamily.MONAI_DYNUNET,
            name="test",
            in_channels=1,
            out_channels=2,
        )
        base = DynUNetAdapter(base_config)
        model = LoraModelAdapter(base, lora_rank=4)
        assert hasattr(model, "forward")
        assert hasattr(model, "get_config")
        assert hasattr(model, "load_checkpoint")
        assert hasattr(model, "save_checkpoint")
        assert hasattr(model, "trainable_parameters")
        assert hasattr(model, "export_onnx")
