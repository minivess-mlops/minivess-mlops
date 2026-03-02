"""Tests for Sam3VanillaAdapter (SAM-06).

Frozen SAM2 encoder + trainable decoder, slice-by-slice inference.
"""

from __future__ import annotations

import pytest
import torch

from minivess.config.models import ModelConfig, ModelFamily


@pytest.fixture()
def vanilla_config() -> ModelConfig:
    return ModelConfig(
        family=ModelFamily.SAM3_VANILLA,
        name="sam3-vanilla-test",
        in_channels=1,
        out_channels=2,
    )


class TestSam3VanillaInit:
    """Sam3VanillaAdapter initialization and configuration."""

    def test_creates_with_config(self, vanilla_config: ModelConfig) -> None:
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        adapter = Sam3VanillaAdapter(vanilla_config)
        assert adapter is not None

    def test_encoder_frozen(self, vanilla_config: ModelConfig) -> None:
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        adapter = Sam3VanillaAdapter(vanilla_config)
        for param in adapter.backbone.parameters():
            assert not param.requires_grad

    def test_decoder_trainable(self, vanilla_config: ModelConfig) -> None:
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        adapter = Sam3VanillaAdapter(vanilla_config)
        trainable = [p for p in adapter.decoder.parameters() if p.requires_grad]
        assert len(trainable) > 0

    def test_get_config_returns_info(self, vanilla_config: ModelConfig) -> None:
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        adapter = Sam3VanillaAdapter(vanilla_config)
        cfg = adapter.get_config()
        assert cfg.family == "sam3_vanilla"
        assert "variant" in cfg.extras


class TestSam3VanillaForward:
    """Sam3VanillaAdapter forward pass produces correct shapes."""

    def test_forward_output_shape(self, vanilla_config: ModelConfig) -> None:
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        adapter = Sam3VanillaAdapter(vanilla_config)
        adapter.eval()
        # (B=1, C=1, D=4, H=64, W=64)
        volume = torch.randn(1, 1, 4, 64, 64)
        with torch.no_grad():
            output = adapter(volume)
        assert output.logits.shape == (1, 2, 4, 64, 64)
        assert output.prediction.shape == (1, 2, 4, 64, 64)

    def test_forward_predictions_sum_to_one(self, vanilla_config: ModelConfig) -> None:
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        adapter = Sam3VanillaAdapter(vanilla_config)
        adapter.eval()
        volume = torch.randn(1, 1, 2, 32, 32)
        with torch.no_grad():
            output = adapter(volume)
        sums = output.prediction.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_forward_metadata_has_architecture(
        self, vanilla_config: ModelConfig
    ) -> None:
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        adapter = Sam3VanillaAdapter(vanilla_config)
        adapter.eval()
        volume = torch.randn(1, 1, 2, 32, 32)
        with torch.no_grad():
            output = adapter(volume)
        assert output.metadata["architecture"] == "sam3_vanilla"


class TestSam3VanillaCheckpoint:
    """Checkpoint save/load for decoder-only weights."""

    def test_save_load_roundtrip(self, vanilla_config: ModelConfig, tmp_path) -> None:
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        adapter = Sam3VanillaAdapter(vanilla_config)
        ckpt_path = tmp_path / "vanilla_ckpt.pth"
        adapter.save_checkpoint(ckpt_path)

        adapter2 = Sam3VanillaAdapter(vanilla_config)
        adapter2.load_checkpoint(ckpt_path)

        # Verify decoder weights match
        for p1, p2 in zip(
            adapter.decoder.parameters(), adapter2.decoder.parameters(), strict=True
        ):
            assert torch.allclose(p1, p2)

    def test_trainable_parameters_count(self, vanilla_config: ModelConfig) -> None:
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        adapter = Sam3VanillaAdapter(vanilla_config)
        n_trainable = adapter.trainable_parameters()
        assert n_trainable > 0
        # Trainable should be much less than total (encoder is frozen)
        n_total = sum(p.numel() for p in adapter.parameters())
        assert n_trainable < n_total
