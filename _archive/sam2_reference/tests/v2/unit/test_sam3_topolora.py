"""Tests for Sam3TopoLoraAdapter (SAM-07).

SAM2 encoder with PEFT LoRA on attention layers + topology-aware loss.
"""

from __future__ import annotations

import pytest
import torch

from minivess.config.models import ModelConfig, ModelFamily


@pytest.fixture()
def topolora_config() -> ModelConfig:
    return ModelConfig(
        family=ModelFamily.SAM3_TOPOLORA,
        name="sam3-topolora-test",
        in_channels=1,
        out_channels=2,
        lora_rank=4,
        lora_alpha=8.0,
        lora_dropout=0.0,
    )


class TestSam3TopoLoraInit:
    """Sam3TopoLoraAdapter initialization and LoRA application."""

    def test_creates_with_config(self, topolora_config: ModelConfig) -> None:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(topolora_config)
        assert adapter is not None

    def test_lora_params_trainable(self, topolora_config: ModelConfig) -> None:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(topolora_config)
        n_trainable = adapter.trainable_parameters()
        assert n_trainable > 0

    def test_more_trainable_than_vanilla(self, topolora_config: ModelConfig) -> None:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        vanilla_config = ModelConfig(
            family=ModelFamily.SAM3_VANILLA,
            name="vanilla-compare",
            in_channels=1,
            out_channels=2,
        )
        vanilla = Sam3VanillaAdapter(vanilla_config)
        topolora = Sam3TopoLoraAdapter(topolora_config)

        # TopoLoRA should have more trainable params (LoRA + decoder > decoder only)
        assert topolora.trainable_parameters() > vanilla.trainable_parameters()

    def test_get_config_has_lora_info(self, topolora_config: ModelConfig) -> None:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(topolora_config)
        cfg = adapter.get_config()
        assert cfg.family == "sam3_topolora"
        assert "lora_rank" in cfg.extras
        assert cfg.extras["lora_rank"] == 4


class TestSam3TopoLoraForward:
    """Sam3TopoLoraAdapter forward pass."""

    def test_forward_output_shape(self, topolora_config: ModelConfig) -> None:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(topolora_config)
        adapter.eval()
        volume = torch.randn(1, 1, 4, 64, 64)
        with torch.no_grad():
            output = adapter(volume)
        assert output.logits.shape == (1, 2, 4, 64, 64)
        assert output.prediction.shape == (1, 2, 4, 64, 64)

    def test_forward_predictions_valid(self, topolora_config: ModelConfig) -> None:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(topolora_config)
        adapter.eval()
        volume = torch.randn(1, 1, 2, 32, 32)
        with torch.no_grad():
            output = adapter(volume)
        assert output.prediction.min() >= 0
        sums = output.prediction.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_gradient_flows_through_lora(self, topolora_config: ModelConfig) -> None:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(topolora_config)
        adapter.train()
        volume = torch.randn(1, 1, 2, 32, 32)
        output = adapter(volume)
        loss = output.logits.sum()
        loss.backward()

        # At least some LoRA params should have gradients
        has_grad = False
        for param in adapter.parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        assert has_grad


class TestSam3TopoLoraCheckpoint:
    """Checkpoint for LoRA + decoder weights."""

    def test_save_load_roundtrip(self, topolora_config: ModelConfig, tmp_path) -> None:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(topolora_config)
        ckpt_path = tmp_path / "topolora_ckpt.pth"
        adapter.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

        adapter2 = Sam3TopoLoraAdapter(topolora_config)
        adapter2.load_checkpoint(ckpt_path)

    def test_checkpoint_excludes_frozen_encoder(
        self, topolora_config: ModelConfig, tmp_path
    ) -> None:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter = Sam3TopoLoraAdapter(topolora_config)
        ckpt_path = tmp_path / "topolora_ckpt.pth"
        adapter.save_checkpoint(ckpt_path)

        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Checkpoint should NOT contain full encoder weights
        total_params = 0
        for v in state.values():
            if isinstance(v, dict):
                total_params += sum(t.numel() for t in v.values())
            elif hasattr(v, "numel"):
                total_params += v.numel()
        full_params = sum(p.numel() for p in adapter.parameters())
        assert total_params < full_params
