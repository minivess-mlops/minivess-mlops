"""Error-path tests for ModelAdapter implementations (Code Review R2.1).

Tests that adapters handle failure conditions gracefully:
corrupted checkpoints, missing files, shape mismatches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path
import torch

from minivess.config.models import ModelConfig, ModelFamily


def _segresnet_config() -> ModelConfig:
    return ModelConfig(
        family=ModelFamily.MONAI_SEGRESNET,
        name="test_error",
        in_channels=1,
        out_channels=2,
    )


# ---------------------------------------------------------------------------
# T1: Checkpoint error paths
# ---------------------------------------------------------------------------


class TestCheckpointErrors:
    """Test checkpoint save/load error handling."""

    def test_load_nonexistent_checkpoint(self, tmp_path: Path) -> None:
        """Loading a nonexistent checkpoint should raise an error."""
        from minivess.adapters.segresnet import SegResNetAdapter

        model = SegResNetAdapter(_segresnet_config())
        missing = tmp_path / "does_not_exist.pt"
        with pytest.raises(FileNotFoundError):
            model.load_checkpoint(missing)

    def test_load_corrupted_checkpoint(self, tmp_path: Path) -> None:
        """Loading a corrupted file should raise an error."""
        from minivess.adapters.segresnet import SegResNetAdapter

        model = SegResNetAdapter(_segresnet_config())
        corrupt = tmp_path / "corrupt.pt"
        corrupt.write_bytes(b"not a valid checkpoint file contents")
        with pytest.raises(Exception):  # noqa: B017
            model.load_checkpoint(corrupt)

    def test_load_wrong_architecture(self, tmp_path: Path) -> None:
        """Loading weights from a different architecture should fail."""
        from minivess.adapters.segresnet import SegResNetAdapter

        config1 = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="model_a",
            in_channels=1,
            out_channels=2,
        )
        config2 = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="model_b",
            in_channels=1,
            out_channels=5,  # Different output channels
        )
        model_a = SegResNetAdapter(config1)
        ckpt = tmp_path / "model_a.pt"
        model_a.save_checkpoint(ckpt)

        model_b = SegResNetAdapter(config2)
        with pytest.raises(RuntimeError, match="size mismatch"):
            model_b.load_checkpoint(ckpt)

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_checkpoint should create nested parent directories."""
        from minivess.adapters.segresnet import SegResNetAdapter

        model = SegResNetAdapter(_segresnet_config())
        deep_path = tmp_path / "a" / "b" / "c" / "model.pt"
        model.save_checkpoint(deep_path)
        assert deep_path.exists()

    def test_checkpoint_roundtrip_preserves_weights(self, tmp_path: Path) -> None:
        """Weights should be identical after save/load cycle."""
        from minivess.adapters.segresnet import SegResNetAdapter

        model = SegResNetAdapter(_segresnet_config())
        ckpt = tmp_path / "model.pt"
        model.save_checkpoint(ckpt)

        model2 = SegResNetAdapter(_segresnet_config())
        model2.load_checkpoint(ckpt)

        for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
            assert torch.equal(p1, p2)


# ---------------------------------------------------------------------------
# T2: Forward pass edge cases
# ---------------------------------------------------------------------------


class TestForwardEdgeCases:
    """Test forward pass with unusual inputs."""

    def test_forward_batch_size_one(self) -> None:
        """Forward pass with batch size 1 should work."""
        from minivess.adapters.segresnet import SegResNetAdapter

        model = SegResNetAdapter(_segresnet_config())
        x = torch.randn(1, 1, 16, 16, 16)
        output = model(x)
        assert output.prediction.shape[0] == 1

    def test_forward_output_sums_to_one(self) -> None:
        """Softmax predictions should sum to ~1.0 along class dimension."""
        from minivess.adapters.segresnet import SegResNetAdapter

        model = SegResNetAdapter(_segresnet_config())
        x = torch.randn(1, 1, 16, 16, 16)
        output = model(x)
        sums = output.prediction.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_forward_predictions_in_01(self) -> None:
        """Predictions should be in [0, 1] range after softmax."""
        from minivess.adapters.segresnet import SegResNetAdapter

        model = SegResNetAdapter(_segresnet_config())
        x = torch.randn(1, 1, 16, 16, 16)
        output = model(x)
        assert output.prediction.min() >= 0.0
        assert output.prediction.max() <= 1.0

    def test_forward_logits_have_same_shape(self) -> None:
        """Logits and predictions should have identical shapes."""
        from minivess.adapters.segresnet import SegResNetAdapter

        model = SegResNetAdapter(_segresnet_config())
        x = torch.randn(1, 1, 16, 16, 16)
        output = model(x)
        assert output.logits.shape == output.prediction.shape


# ---------------------------------------------------------------------------
# T3: get_config validation
# ---------------------------------------------------------------------------


class TestGetConfig:
    """Test get_config returns complete information."""

    def test_segresnet_config_has_required_keys(self) -> None:
        """get_config should return all critical configuration keys."""
        from minivess.adapters.segresnet import SegResNetAdapter

        model = SegResNetAdapter(_segresnet_config())
        cfg = model.get_config()
        d = cfg.to_dict()
        required_keys = {
            "family",
            "name",
            "in_channels",
            "out_channels",
            "trainable_params",
        }
        assert required_keys.issubset(d.keys())

    def test_swinunetr_config_has_required_keys(self) -> None:
        """SwinUNETR get_config should include transformer-specific fields."""
        from minivess.adapters.swinunetr import SwinUNETRAdapter

        config = ModelConfig(
            family=ModelFamily.MONAI_SWINUNETR,
            name="test",
            in_channels=1,
            out_channels=2,
        )
        model = SwinUNETRAdapter(config)
        cfg = model.get_config()
        d = cfg.to_dict()
        assert "feature_size" in d
        assert "depths" in d
        assert "num_heads" in d

    def test_trainable_params_positive(self) -> None:
        """All adapters should report positive trainable parameter counts."""
        from minivess.adapters.segresnet import SegResNetAdapter

        model = SegResNetAdapter(_segresnet_config())
        assert model.trainable_parameters() > 0
        assert model.get_config().trainable_params > 0
