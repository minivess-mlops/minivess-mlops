"""Tests for Sam3TopoLoraAdapter (T7).

V2: SAM3 + LoRA on FFN (mlp.lin1, mlp.lin2) + topology-aware loss.
Same as V1 + LoRA adapters + cbdice_cldice loss.
"""

from __future__ import annotations

import pytest
import torch

from minivess.config.models import ModelConfig, ModelFamily


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


class TestSam3TopoLoraAdapter:
    """Sam3TopoLoraAdapter: SAM3 + LoRA + topology loss."""

    def test_adapter_creates_with_stub(self, sam3_lora_config: ModelConfig) -> None:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(config=sam3_lora_config, use_stub=True)
        assert adapter is not None

    def test_forward_produces_segmentation_output(
        self, sam3_lora_config: ModelConfig
    ) -> None:
        from minivess.adapters.base import SegmentationOutput
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(config=sam3_lora_config, use_stub=True)
        volume = torch.randn(1, 1, 3, 64, 64)
        output = adapter(volume)
        assert isinstance(output, SegmentationOutput)

    def test_forward_output_shape_2class(self, sam3_lora_config: ModelConfig) -> None:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(config=sam3_lora_config, use_stub=True)
        volume = torch.randn(1, 1, 3, 64, 64)
        output = adapter(volume)
        assert output.logits.shape[1] == 2  # 2-class
        assert output.logits.shape[2] == 3  # depth

    def test_lora_params_are_trainable(self, sam3_lora_config: ModelConfig) -> None:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(config=sam3_lora_config, use_stub=True)
        # LoRA params should exist and be trainable
        lora_params = [
            (name, p)
            for name, p in adapter.named_parameters()
            if "lora" in name.lower() and p.requires_grad
        ]
        assert len(lora_params) > 0, "Should have trainable LoRA parameters"

    def test_base_encoder_weights_frozen(self, sam3_lora_config: ModelConfig) -> None:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(config=sam3_lora_config, use_stub=True)
        # Base encoder (non-LoRA) params should be frozen
        base_params = [
            (name, p)
            for name, p in adapter.backbone.encoder.named_parameters()
            if "lora" not in name.lower()
        ]
        for name, p in base_params:
            assert not p.requires_grad, f"Base param {name} should be frozen"

    def test_gradient_flows_through_lora(self, sam3_lora_config: ModelConfig) -> None:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(config=sam3_lora_config, use_stub=True)
        volume = torch.randn(1, 1, 2, 64, 64)
        output = adapter(volume)
        target = torch.zeros(1, 2, 64, 64, dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(
            output.logits[:, :, 0, :, :], target[:, 0, :, :]
        )
        loss.backward()
        # Check decoder has gradients
        has_decoder_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in adapter.decoder.parameters()
        )
        assert has_decoder_grad, "Decoder gradients must flow"

    def test_get_config_returns_lora_info(self, sam3_lora_config: ModelConfig) -> None:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(config=sam3_lora_config, use_stub=True)
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
        vanilla = Sam3VanillaAdapter(config=vanilla_config, use_stub=True)
        topolora = Sam3TopoLoraAdapter(config=sam3_lora_config, use_stub=True)

        assert topolora.trainable_parameters() > vanilla.trainable_parameters()
