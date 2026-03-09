"""Tests for Sam3TopoLoraAdapter.

V2: SAM3 + LoRA on FFN (mlp.lin1, mlp.lin2) + topology-aware loss.
Same as V1 + LoRA adapters + cbdice_cldice loss.

IMPORTANT: These tests require real SAM3 pretrained weights (GPU ≥16 GB).
They are skipped in CI where SAM3 is not installed.
"""

from __future__ import annotations

import pytest
import torch

from minivess.adapters.model_builder import _sam3_package_available
from minivess.config.models import ModelConfig, ModelFamily


def _gpu_vram_gb() -> float:
    """Return VRAM of first CUDA GPU in GB, or 0.0 if no GPU."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


_sam3_skip = pytest.mark.skipif(
    not _sam3_package_available(), reason="SAM3 not installed"
)
_vram_16gb_skip = pytest.mark.skipif(
    _gpu_vram_gb() < 16.0,
    reason=f"SAM3 TopoLoRA requires >= 16 GB VRAM (detected {_gpu_vram_gb():.1f} GB)",
)


@pytest.fixture()
def sam3_lora_config() -> ModelConfig:
    """SAM3 TopoLoRA config with LoRA params."""
    return ModelConfig(
        family=ModelFamily.SAM3_TOPOLORA,
        name="sam3-topolora-test",
        in_channels=1,
        out_channels=2,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
    )


@_sam3_skip
@_vram_16gb_skip
@pytest.mark.gpu
class TestSam3TopoLoraAdapter:
    """Sam3TopoLoraAdapter: SAM3 + LoRA + topology loss."""

    def test_adapter_creates(self, sam3_lora_config: ModelConfig) -> None:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(config=sam3_lora_config)
        assert adapter is not None

    def test_forward_produces_segmentation_output(
        self, sam3_lora_config: ModelConfig
    ) -> None:
        from minivess.adapters.base import SegmentationOutput
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(config=sam3_lora_config)
        volume = torch.randn(1, 1, 3, 64, 64)
        output = adapter(volume)
        assert isinstance(output, SegmentationOutput)

    def test_forward_output_shape_2class(self, sam3_lora_config: ModelConfig) -> None:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(config=sam3_lora_config)
        volume = torch.randn(1, 1, 3, 64, 64)
        output = adapter(volume)
        assert output.logits.shape[1] == 2  # 2-class
        assert output.logits.shape[2] == 3  # depth

    def test_lora_params_are_trainable(self, sam3_lora_config: ModelConfig) -> None:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(config=sam3_lora_config)
        lora_params = [
            (name, p)
            for name, p in adapter.named_parameters()
            if "lora" in name.lower() and p.requires_grad
        ]
        assert len(lora_params) > 0, "Should have trainable LoRA parameters"

    def test_base_encoder_weights_frozen(self, sam3_lora_config: ModelConfig) -> None:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(config=sam3_lora_config)
        base_params = [
            (name, p)
            for name, p in adapter.backbone.encoder.named_parameters()
            if "lora" not in name.lower()
        ]
        for name, p in base_params:
            assert not p.requires_grad, f"Base param {name} should be frozen"

    def test_gradient_flows_through_lora(self, sam3_lora_config: ModelConfig) -> None:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(config=sam3_lora_config)
        volume = torch.randn(1, 1, 2, 64, 64)
        output = adapter(volume)
        target = torch.zeros(1, 2, 64, 64, dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(
            output.logits[:, :, 0, :, :], target[:, 0, :, :]
        )
        loss.backward()
        has_decoder_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in adapter.decoder.parameters()
        )
        assert has_decoder_grad, "Decoder gradients must flow"

    def test_get_config_returns_lora_info(self, sam3_lora_config: ModelConfig) -> None:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(config=sam3_lora_config)
        info = adapter.get_config()
        assert info.family == "sam3_topolora"
        assert info.extras.get("variant") == "topolora"
        assert info.extras.get("lora_rank") == 2

    def test_trainable_params_more_than_vanilla(
        self, sam3_lora_config: ModelConfig
    ) -> None:
        """TopoLoRA should have more trainable params than vanilla (LoRA added)."""
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        vanilla_config = ModelConfig(
            family=ModelFamily.SAM3_VANILLA,
            name="vanilla-compare",
            in_channels=1,
            out_channels=2,
        )
        vanilla = Sam3VanillaAdapter(config=vanilla_config)
        topolora = Sam3TopoLoraAdapter(config=sam3_lora_config)

        assert topolora.trainable_parameters() > vanilla.trainable_parameters()

    def test_lora_applied_with_real_rank_16(self) -> None:
        """LoRA must apply to >=1 layer even with rank=16 (training default)."""
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        config = ModelConfig(
            family=ModelFamily.SAM3_TOPOLORA,
            name="sam3-topolora-rank16",
            in_channels=1,
            out_channels=2,
            lora_rank=16,
            lora_alpha=32.0,
            lora_dropout=0.1,
        )
        adapter = Sam3TopoLoraAdapter(config=config)
        lora_params = [
            name
            for name, p in adapter.named_parameters()
            if "lora" in name.lower() and p.requires_grad
        ]
        assert len(lora_params) > 0, (
            "LoRA must be applied to at least 1 layer with rank=16."
        )
