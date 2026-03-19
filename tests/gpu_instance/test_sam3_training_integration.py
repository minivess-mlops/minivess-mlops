"""Integration smoke tests for SAM3 adapter training loop.

Tests that each SAM3 adapter can:
1. Complete one forward + backward step
2. Have gradients flow to trainable parameters
3. Work with AMP (mixed precision)

IMPORTANT: These tests require real SAM3 pretrained weights (GPU ≥16 GB).
They are skipped in CI where SAM3 is not installed.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from minivess.adapters.base import SegmentationOutput
from minivess.adapters.model_builder import _sam3_package_available
from minivess.config.models import ModelConfig, ModelFamily


def _insufficient_vram() -> bool:
    """Check if GPU VRAM is too small for SAM3 (needs >= 16 GB)."""
    if not torch.cuda.is_available():
        return True
    return torch.cuda.get_device_properties(0).total_memory < 16 * 1024**3


_sam3_skip = pytest.mark.skipif(
    not _sam3_package_available(), reason="SAM3 not installed"
)
_vram_skip = pytest.mark.skipif(
    _insufficient_vram(), reason="SAM3 requires >= 16 GB VRAM"
)


@pytest.fixture()
def sam3_vanilla_config() -> ModelConfig:
    """Minimal SAM3 vanilla config."""
    return ModelConfig(
        family=ModelFamily.SAM3_VANILLA,
        name="sam3-vanilla-integration",
        in_channels=1,
        out_channels=2,
    )


@pytest.fixture()
def sam3_topolora_config() -> ModelConfig:
    """Minimal SAM3 topolora config."""
    return ModelConfig(
        family=ModelFamily.SAM3_TOPOLORA,
        name="sam3-topolora-integration",
        in_channels=1,
        out_channels=2,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
    )


@pytest.fixture()
def sam3_hybrid_config() -> ModelConfig:
    """Minimal SAM3 hybrid config."""
    return ModelConfig(
        family=ModelFamily.SAM3_HYBRID,
        name="sam3-hybrid-integration",
        in_channels=1,
        out_channels=2,
        architecture_params={"filters": [16, 32, 64], "fusion_gate_init": 0.0},
    )


@_sam3_skip
@_vram_skip
class TestSam3VanillaTraining:
    """Sam3VanillaAdapter: one training step integration."""

    def test_one_training_step(self, sam3_vanilla_config: ModelConfig) -> None:
        """Forward + backward + optimizer step completes without error."""
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        adapter = Sam3VanillaAdapter(config=sam3_vanilla_config)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, adapter.parameters()), lr=1e-3
        )

        volume = torch.randn(1, 1, 2, 64, 64)
        output = adapter(volume)
        assert isinstance(output, SegmentationOutput)

        target = torch.zeros(1, 64, 64, dtype=torch.long)
        loss = F.cross_entropy(output.logits[:, :, 0, :, :], target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def test_gradient_flow(self, sam3_vanilla_config: ModelConfig) -> None:
        """Gradients reach decoder parameters."""
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        adapter = Sam3VanillaAdapter(config=sam3_vanilla_config)
        volume = torch.randn(1, 1, 2, 64, 64)
        output = adapter(volume)
        target = torch.zeros(1, 64, 64, dtype=torch.long)
        loss = F.cross_entropy(output.logits[:, :, 0, :, :], target)
        loss.backward()

        decoder_grads = [
            p
            for p in adapter.decoder.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        ]
        assert len(decoder_grads) > 0, "Gradients must flow through decoder"

    def test_amp_forward(self, sam3_vanilla_config: ModelConfig) -> None:
        """Forward pass works under AMP autocast."""
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        adapter = Sam3VanillaAdapter(config=sam3_vanilla_config)
        volume = torch.randn(1, 1, 2, 64, 64)

        with torch.amp.autocast("cpu"):
            output = adapter(volume)
        assert output.logits.shape == (1, 2, 2, 64, 64)


@_sam3_skip
@_vram_skip
class TestSam3TopoLoraTraining:
    """Sam3TopoLoraAdapter: one training step integration."""

    def test_one_training_step(self, sam3_topolora_config: ModelConfig) -> None:
        """Forward + backward + optimizer step completes without error."""
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(config=sam3_topolora_config)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, adapter.parameters()), lr=1e-3
        )

        volume = torch.randn(1, 1, 2, 64, 64)
        output = adapter(volume)
        assert isinstance(output, SegmentationOutput)

        target = torch.zeros(1, 64, 64, dtype=torch.long)
        loss = F.cross_entropy(output.logits[:, :, 0, :, :], target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def test_lora_params_get_gradients(self, sam3_topolora_config: ModelConfig) -> None:
        """LoRA A/B matrices receive gradients."""
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(config=sam3_topolora_config)
        volume = torch.randn(1, 1, 2, 64, 64)
        output = adapter(volume)
        target = torch.zeros(1, 64, 64, dtype=torch.long)
        loss = F.cross_entropy(output.logits[:, :, 0, :, :], target)
        loss.backward()

        lora_grads = [
            name
            for name, p in adapter.named_parameters()
            if "lora_" in name and p.grad is not None and p.grad.abs().sum() > 0
        ]
        assert len(lora_grads) > 0, "LoRA parameters must receive gradients"

    def test_amp_forward(self, sam3_topolora_config: ModelConfig) -> None:
        """Forward pass works under AMP autocast."""
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(config=sam3_topolora_config)
        volume = torch.randn(1, 1, 2, 64, 64)

        with torch.amp.autocast("cpu"):
            output = adapter(volume)
        assert output.logits.shape == (1, 2, 2, 64, 64)


@_sam3_skip
@_vram_skip
class TestSam3HybridTraining:
    """Sam3HybridAdapter: one training step integration."""

    def test_one_training_step(self, sam3_hybrid_config: ModelConfig) -> None:
        """Forward + backward + optimizer step completes without error."""
        from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

        adapter = Sam3HybridAdapter(config=sam3_hybrid_config)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, adapter.parameters()), lr=1e-3
        )

        volume = torch.randn(1, 1, 4, 64, 64)
        output = adapter(volume)
        assert isinstance(output, SegmentationOutput)

        target = torch.zeros(1, 64, 64, dtype=torch.long)
        loss = F.cross_entropy(output.logits[:, :, 0, :, :], target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def test_dynunet_gets_gradients(self, sam3_hybrid_config: ModelConfig) -> None:
        """DynUNet encoder/decoder receives gradients."""
        from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

        adapter = Sam3HybridAdapter(config=sam3_hybrid_config)
        volume = torch.randn(1, 1, 4, 64, 64)
        output = adapter(volume)
        target = torch.zeros(1, 64, 64, dtype=torch.long)
        loss = F.cross_entropy(output.logits[:, :, 0, :, :], target)
        loss.backward()

        dynunet_grads = [
            name
            for name, p in adapter.dynunet.named_parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        ]
        assert len(dynunet_grads) > 0, "DynUNet must receive gradients"

    def test_gate_alpha_gets_gradient(self, sam3_hybrid_config: ModelConfig) -> None:
        """Fusion gate_alpha parameter receives gradient."""
        from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

        adapter = Sam3HybridAdapter(config=sam3_hybrid_config)
        volume = torch.randn(1, 1, 4, 64, 64)
        output = adapter(volume)
        target = torch.zeros(1, 64, 64, dtype=torch.long)
        loss = F.cross_entropy(output.logits[:, :, 0, :, :], target)
        loss.backward()

        assert adapter.fusion.gate_alpha.grad is not None
        assert adapter.fusion.gate_alpha.grad.abs().sum() > 0

    def test_amp_forward(self, sam3_hybrid_config: ModelConfig) -> None:
        """Forward pass works under AMP autocast."""
        from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

        adapter = Sam3HybridAdapter(config=sam3_hybrid_config)
        volume = torch.randn(1, 1, 4, 64, 64)

        with torch.amp.autocast("cpu"):
            output = adapter(volume)
        assert output.logits.shape == (1, 2, 4, 64, 64)


@_sam3_skip
@_vram_skip
class TestBuildAdapterIntegration:
    """build_adapter() → training step for each SAM3 family."""

    def test_build_and_train_vanilla(self, sam3_vanilla_config: ModelConfig) -> None:
        from minivess.adapters.model_builder import build_adapter

        adapter = build_adapter(sam3_vanilla_config)
        volume = torch.randn(1, 1, 2, 64, 64)
        output = adapter(volume)
        target = torch.zeros(1, 64, 64, dtype=torch.long)
        loss = F.cross_entropy(output.logits[:, :, 0, :, :], target)
        loss.backward()
        assert loss.item() > 0

    def test_build_and_train_topolora(self, sam3_topolora_config: ModelConfig) -> None:
        from minivess.adapters.model_builder import build_adapter

        adapter = build_adapter(sam3_topolora_config)
        volume = torch.randn(1, 1, 2, 64, 64)
        output = adapter(volume)
        target = torch.zeros(1, 64, 64, dtype=torch.long)
        loss = F.cross_entropy(output.logits[:, :, 0, :, :], target)
        loss.backward()
        assert loss.item() > 0

    def test_build_and_train_hybrid(self, sam3_hybrid_config: ModelConfig) -> None:
        from minivess.adapters.model_builder import build_adapter

        adapter = build_adapter(sam3_hybrid_config)
        volume = torch.randn(1, 1, 4, 64, 64)
        output = adapter(volume)
        target = torch.zeros(1, 64, 64, dtype=torch.long)
        loss = F.cross_entropy(output.logits[:, :, 0, :, :], target)
        loss.backward()
        assert loss.item() > 0
