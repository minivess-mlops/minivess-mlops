"""Tests for Sam3VanillaAdapter.

V1: Frozen SAM3 ViT-32L encoder + trainable decoder.
Slice-by-slice 2D inference, null prompts, dice_ce loss.

IMPORTANT: These tests require real SAM3 pretrained weights (GPU ≥16 GB).
They are skipped in CI where SAM3 is not installed.
"""

from __future__ import annotations

import pytest
import torch

from minivess.adapters.model_builder import _sam3_package_available
from minivess.config.models import ModelConfig, ModelFamily

_sam3_skip = pytest.mark.skipif(
    not _sam3_package_available(), reason="SAM3 not installed"
)


@pytest.fixture()
def sam3_config() -> ModelConfig:
    """Minimal SAM3 vanilla config."""
    return ModelConfig(
        family=ModelFamily.SAM3_VANILLA,
        name="sam3-vanilla-test",
        in_channels=1,
        out_channels=2,
    )


@_sam3_skip
@pytest.mark.slow
class TestSam3VanillaAdapter:
    """Sam3VanillaAdapter: frozen encoder + trainable decoder."""

    def test_adapter_creates(self, sam3_config: ModelConfig) -> None:
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        adapter = Sam3VanillaAdapter(config=sam3_config)
        assert adapter is not None

    def test_forward_produces_segmentation_output(
        self, sam3_config: ModelConfig
    ) -> None:
        from minivess.adapters.base import SegmentationOutput
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        adapter = Sam3VanillaAdapter(config=sam3_config)
        volume = torch.randn(1, 1, 3, 64, 64)
        output = adapter(volume)
        assert isinstance(output, SegmentationOutput)

    def test_forward_output_shape_2class(self, sam3_config: ModelConfig) -> None:
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        adapter = Sam3VanillaAdapter(config=sam3_config)
        volume = torch.randn(1, 1, 3, 64, 64)
        output = adapter(volume)
        assert output.logits.shape[0] == 1  # batch
        assert output.logits.shape[1] == 2  # classes
        assert output.logits.shape[2] == 3  # depth

    def test_encoder_frozen_decoder_trainable(self, sam3_config: ModelConfig) -> None:
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        adapter = Sam3VanillaAdapter(config=sam3_config)
        for p in adapter.backbone.encoder.parameters():
            assert not p.requires_grad, "Encoder must be frozen"
        trainable = [p for p in adapter.decoder.parameters() if p.requires_grad]
        assert len(trainable) > 0, "Decoder must have trainable params"

    def test_gradient_flows_through_decoder(self, sam3_config: ModelConfig) -> None:
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        adapter = Sam3VanillaAdapter(config=sam3_config)
        volume = torch.randn(1, 1, 2, 64, 64)
        output = adapter(volume)
        target = torch.zeros(1, 2, 64, 64, dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(
            output.logits[:, :, 0, :, :], target[:, 0, :, :]
        )
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in adapter.decoder.parameters()
        )
        assert has_grad, "Gradients must flow through decoder"

    def test_get_config_returns_info(self, sam3_config: ModelConfig) -> None:
        from minivess.adapters.base import AdapterConfigInfo
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        adapter = Sam3VanillaAdapter(config=sam3_config)
        info = adapter.get_config()
        assert isinstance(info, AdapterConfigInfo)
        assert info.family == "sam3_vanilla"

    def test_trainable_parameters_count(self, sam3_config: ModelConfig) -> None:
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        adapter = Sam3VanillaAdapter(config=sam3_config)
        trainable = adapter.trainable_parameters()
        assert trainable > 0, "Should have trainable parameters (decoder)"
        total = sum(p.numel() for p in adapter.parameters())
        assert trainable < total
