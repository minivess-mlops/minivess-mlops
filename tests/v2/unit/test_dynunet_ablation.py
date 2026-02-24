"""Tests for DynUNet width ablation study (Issue #33)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from minivess.adapters.base import ModelAdapter, SegmentationOutput
from minivess.config.models import ModelConfig, ModelFamily

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# T1: DynUNetAdapter — ModelAdapter ABC compliance
# ---------------------------------------------------------------------------


class TestDynUNetAdapter:
    """Test DynUNet adapter implementation."""

    @pytest.fixture
    def config(self) -> ModelConfig:
        return ModelConfig(
            family=ModelFamily.MONAI_DYNUNET,
            name="test-dynunet",
            in_channels=1,
            out_channels=2,
        )

    @pytest.fixture
    def adapter(self, config: ModelConfig) -> ModelAdapter:
        from minivess.adapters.dynunet import DynUNetAdapter

        return DynUNetAdapter(
            config,
            filters=[8, 16, 32, 64],
        )

    def test_is_model_adapter(self, adapter: ModelAdapter) -> None:
        """DynUNetAdapter must be a ModelAdapter instance."""
        assert isinstance(adapter, ModelAdapter)

    def test_forward_shape(self, adapter: ModelAdapter) -> None:
        """Forward should return correct (B, C, D, H, W) shapes."""
        x = torch.randn(1, 1, 32, 32, 16)
        output = adapter(x)
        assert isinstance(output, SegmentationOutput)
        assert output.prediction.shape == (1, 2, 32, 32, 16)
        assert output.logits.shape == (1, 2, 32, 32, 16)

    def test_prediction_is_probability(self, adapter: ModelAdapter) -> None:
        """Prediction should be softmax probabilities summing to ~1."""
        x = torch.randn(1, 1, 32, 32, 16)
        output = adapter(x)
        sums = output.prediction.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_get_config(self, adapter: ModelAdapter) -> None:
        """Config should contain architecture details."""
        cfg = adapter.get_config()
        assert cfg["family"] == "dynunet"
        assert cfg["filters"] == [8, 16, 32, 64]
        assert "trainable_params" in cfg

    def test_trainable_parameters(self, adapter: ModelAdapter) -> None:
        """Should report positive trainable parameter count."""
        count = adapter.trainable_parameters()
        assert count > 0
        assert isinstance(count, int)

    def test_save_load_checkpoint(
        self, adapter: ModelAdapter, config: ModelConfig, tmp_path: Path
    ) -> None:
        """Checkpoint save/load should preserve weights."""
        from minivess.adapters.dynunet import DynUNetAdapter

        ckpt_path = tmp_path / "dynunet.pth"
        adapter.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

        new_adapter = DynUNetAdapter(config, filters=[8, 16, 32, 64])
        new_adapter.load_checkpoint(ckpt_path)

        adapter.eval()
        new_adapter.eval()
        x = torch.randn(1, 1, 32, 32, 16)
        with torch.no_grad():
            orig = adapter(x)
            loaded = new_adapter(x)
        assert torch.allclose(orig.logits, loaded.logits, atol=1e-6)

    def test_metadata(self, adapter: ModelAdapter) -> None:
        """Metadata should identify the architecture."""
        x = torch.randn(1, 1, 32, 32, 16)
        output = adapter(x)
        assert output.metadata["architecture"] == "dynunet"

    def test_model_family_enum(self) -> None:
        """MONAI_DYNUNET should be in ModelFamily enum."""
        assert hasattr(ModelFamily, "MONAI_DYNUNET")
        assert ModelFamily.MONAI_DYNUNET.value == "dynunet"


# ---------------------------------------------------------------------------
# T2: Topology-aware loss functions
# ---------------------------------------------------------------------------


class TestTopologyLoss:
    """Test topology-aware loss integration."""

    def test_cldice_loss_exists(self) -> None:
        """build_loss_function should support 'cldice'."""
        from minivess.pipeline.loss_functions import build_loss_function

        loss = build_loss_function("cldice")
        assert loss is not None

    def test_compound_loss_exists(self) -> None:
        """build_loss_function should support 'dice_ce_cldice'."""
        from minivess.pipeline.loss_functions import build_loss_function

        loss = build_loss_function("dice_ce_cldice")
        assert loss is not None

    def test_compound_loss_callable(self) -> None:
        """Compound loss should be callable with logits and labels."""
        from minivess.pipeline.loss_functions import build_loss_function

        loss_fn = build_loss_function("dice_ce_cldice")
        logits = torch.randn(1, 2, 16, 16, 8)
        labels = torch.zeros(1, 1, 16, 16, 8, dtype=torch.long)
        loss_val = loss_fn(logits, labels)
        assert loss_val.ndim == 0  # scalar
        assert loss_val.item() > 0


# ---------------------------------------------------------------------------
# T3: Ablation config and grid
# ---------------------------------------------------------------------------


class TestAblationConfig:
    """Test ablation grid configuration."""

    def test_width_presets(self) -> None:
        """Should define FULL, HALF, QUARTER width presets."""
        from minivess.pipeline.ablation import DYNUNET_WIDTH_PRESETS

        assert "full" in DYNUNET_WIDTH_PRESETS
        assert "half" in DYNUNET_WIDTH_PRESETS
        assert "quarter" in DYNUNET_WIDTH_PRESETS
        assert DYNUNET_WIDTH_PRESETS["full"] == [32, 64, 128, 256]
        assert DYNUNET_WIDTH_PRESETS["quarter"] == [8, 16, 32, 64]

    def test_build_ablation_grid(self) -> None:
        """Should generate all width × loss combinations."""
        from minivess.pipeline.ablation import build_ablation_grid

        grid = build_ablation_grid(
            widths=["quarter", "half"],
            losses=["dice_ce", "dice_ce_cldice"],
        )
        assert len(grid) == 4  # 2 widths × 2 losses

    def test_grid_entries_have_required_fields(self) -> None:
        """Each grid entry should have width_name, filters, loss_name."""
        from minivess.pipeline.ablation import build_ablation_grid

        grid = build_ablation_grid(
            widths=["quarter"],
            losses=["dice_ce"],
        )
        entry = grid[0]
        assert entry["width_name"] == "quarter"
        assert entry["filters"] == [8, 16, 32, 64]
        assert entry["loss_name"] == "dice_ce"

    def test_full_ablation_grid(self) -> None:
        """Full 3×3 grid should have 9 entries."""
        from minivess.pipeline.ablation import build_ablation_grid

        grid = build_ablation_grid(
            widths=["quarter", "half", "full"],
            losses=["dice_ce", "cldice", "dice_ce_cldice"],
        )
        assert len(grid) == 9

    def test_grid_experiment_names(self) -> None:
        """Grid entries should have unique experiment names."""
        from minivess.pipeline.ablation import build_ablation_grid

        grid = build_ablation_grid(
            widths=["quarter", "half"],
            losses=["dice_ce", "cldice"],
        )
        names = [e["experiment_name"] for e in grid]
        assert len(names) == len(set(names))  # all unique
