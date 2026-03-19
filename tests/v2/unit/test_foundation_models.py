"""Tests for foundation model fine-tuning: LoRA wrapper, SAM3 stub."""

from __future__ import annotations

import pytest
import torch

from minivess.adapters.base import ModelAdapter
from minivess.config.models import ModelConfig, ModelFamily

pytestmark = pytest.mark.model_loading

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

    def test_lora_load_trainer_format_checkpoint(
        self, tmp_path: pytest.FixtureRequest
    ) -> None:
        """LoRA load_checkpoint must actually load weights from trainer-wrapped format.

        The trainer saves: {"model_state_dict": model.state_dict(), "optimizer_state_dict": ...}
        lora.py used strict=False which silently ignored all keys (cosmetic success).
        This test verifies weights are actually loaded, not just "no exception raised".
        """
        from pathlib import Path

        import torch

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

        # Mutate LoRA weights so we can detect if loading restores them
        for name, p in lora_model.named_parameters():
            if "lora_" in name:
                p.data.fill_(3.14)

        # Simulate what save_metric_checkpoint() produces
        ckpt = Path(str(tmp_path)) / "trainer_lora.pt"
        wrapped = {
            "model_state_dict": lora_model.state_dict(),
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "checkpoint_metadata": {"epoch": 10},
            "scaler_state_dict": None,
        }
        torch.save(wrapped, ckpt)

        # Create a fresh adapter (LoRA weights randomly initialized, NOT 3.14)
        base_model2 = DynUNetAdapter(base_config)
        lora_model2 = LoraModelAdapter(base_model2, lora_rank=4, lora_alpha=8.0)
        lora_model2.load_checkpoint(ckpt)

        # Verify LoRA weights were actually loaded (should be 3.14, not random)
        for name, p in lora_model2.named_parameters():
            if "lora_" in name:
                assert torch.allclose(p.data, torch.full_like(p.data, 3.14)), (
                    f"LoRA weight {name!r} was not loaded from trainer checkpoint"
                )


# ---------------------------------------------------------------------------
# T2: SAM3 adapter (exploratory stub)
# ---------------------------------------------------------------------------


class TestSam3Adapter:
    """Test SAM3 adapter stub."""

    def test_sam3_not_available_message(self) -> None:
        """Deprecated Sam3Adapter raises RuntimeError directing to new adapters."""
        from minivess.adapters.sam3 import Sam3Adapter

        config = ModelConfig(
            family=ModelFamily.SAM3_TOPOLORA,
            name="test_sam3",
            in_channels=1,
            out_channels=2,
        )
        with pytest.raises((ImportError, RuntimeError), match="deprecated|Sam3"):
            Sam3Adapter(config)


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
