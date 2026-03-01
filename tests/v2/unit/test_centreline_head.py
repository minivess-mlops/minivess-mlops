"""Tests for centreline prediction head multi-task adapter (#120).

Covers: CentrelineHeadAdapter wrapping DynUNetAdapter, multi-task output
(segmentation + centreline distance map), configurable weights, disable mode.

Uses composition pattern: CentrelineHeadAdapter(DynUNetAdapter(config)).
Centreline distance map stored in SegmentationOutput.metadata['centreline_map'].

TDD RED phase: tests written before implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput
from minivess.config.models import ModelConfig, ModelFamily

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dynunet_config() -> ModelConfig:
    """Small DynUNet config for testing."""
    return ModelConfig(
        family=ModelFamily.MONAI_DYNUNET,
        name="test-dynunet-for-centreline",
        in_channels=1,
        out_channels=2,
        architecture_params={"filters": [8, 16, 32]},
    )


@pytest.fixture
def base_adapter(dynunet_config: ModelConfig) -> ModelAdapter:
    from minivess.adapters.dynunet import DynUNetAdapter

    return DynUNetAdapter(dynunet_config, filters=[8, 16, 32])


@pytest.fixture
def centreline_adapter(base_adapter: ModelAdapter) -> ModelAdapter:
    from minivess.adapters.centreline_head import CentrelineHeadAdapter

    return CentrelineHeadAdapter(base_adapter)


@pytest.fixture
def small_input() -> torch.Tensor:
    """Small 3D volume: (B=1, C=1, D=16, H=16, W=8)."""
    return torch.randn(1, 1, 16, 16, 8)


# ---------------------------------------------------------------------------
# CentrelineHeadAdapter tests
# ---------------------------------------------------------------------------


class TestCentrelineHeadAdapter:
    """Tests for the multi-task centreline prediction head."""

    def test_centreline_head_is_model_adapter(
        self, centreline_adapter: ModelAdapter
    ) -> None:
        assert isinstance(centreline_adapter, ModelAdapter)

    def test_centreline_head_forward_shape(
        self, centreline_adapter: ModelAdapter, small_input: torch.Tensor
    ) -> None:
        output = centreline_adapter(small_input)
        assert isinstance(output, SegmentationOutput)
        # Segmentation output should preserve original shape
        assert output.prediction.shape == (1, 2, 16, 16, 8)
        assert output.logits.shape == (1, 2, 16, 16, 8)

    def test_centreline_head_produces_both_outputs(
        self, centreline_adapter: ModelAdapter, small_input: torch.Tensor
    ) -> None:
        output = centreline_adapter(small_input)
        # Should have segmentation output
        assert output.prediction is not None
        # Should have centreline map in metadata
        assert "centreline_map" in output.metadata

    def test_centreline_head_metadata_has_centreline_map(
        self, centreline_adapter: ModelAdapter, small_input: torch.Tensor
    ) -> None:
        output = centreline_adapter(small_input)
        cmap = output.metadata["centreline_map"]
        assert isinstance(cmap, torch.Tensor)
        # Centreline map should be (B, 1, D, H, W)
        assert cmap.shape == (1, 1, 16, 16, 8)
        # Distances should be non-negative (ReLU output)
        assert (cmap >= 0).all()

    def test_centreline_head_backward_succeeds(
        self, centreline_adapter: ModelAdapter, small_input: torch.Tensor
    ) -> None:
        output = centreline_adapter(small_input)
        # Multi-task loss: segmentation + centreline regression
        seg_loss = output.logits.sum()
        centreline_loss = output.metadata["centreline_map"].sum()
        total_loss = seg_loss + centreline_loss
        total_loss.backward()
        # Check gradients exist
        has_grads = False
        for param in centreline_adapter.parameters():
            if param.requires_grad and param.grad is not None:
                has_grads = True
                break
        assert has_grads

    def test_centreline_head_weights_configurable(
        self, base_adapter: ModelAdapter
    ) -> None:
        from minivess.adapters.centreline_head import CentrelineHeadAdapter

        adapter = CentrelineHeadAdapter(
            base_adapter, seg_weight=0.7, centreline_weight=0.3
        )
        cfg = adapter.get_config()
        assert cfg.extras["seg_weight"] == 0.7
        assert cfg.extras["centreline_weight"] == 0.3

    def test_centreline_head_disabled_mode(
        self, base_adapter: ModelAdapter, small_input: torch.Tensor
    ) -> None:
        from minivess.adapters.centreline_head import CentrelineHeadAdapter

        # Disabled centreline head should behave like base model
        adapter = CentrelineHeadAdapter(base_adapter, enabled=False)
        output = adapter(small_input)
        assert isinstance(output, SegmentationOutput)
        # Should NOT have centreline map when disabled
        assert output.metadata.get("centreline_map") is None
        assert output.metadata.get("centreline_head_enabled") is False

    def test_centreline_head_get_config(self, centreline_adapter: ModelAdapter) -> None:
        cfg = centreline_adapter.get_config()
        assert isinstance(cfg, AdapterConfigInfo)
        # Base model config should be preserved
        assert cfg.in_channels == 1
        assert cfg.out_channels == 2
        # Centreline-specific params
        assert "centreline_head_enabled" in cfg.extras
        assert cfg.extras["centreline_head_enabled"] is True

    def test_centreline_head_vram_small_patch(
        self, centreline_adapter: ModelAdapter, small_input: torch.Tensor
    ) -> None:
        """Forward + backward should succeed within budget."""
        output = centreline_adapter(small_input)
        loss = output.logits.sum() + output.metadata["centreline_map"].sum()
        loss.backward()
        # Model should not be unreasonably large
        total_params = sum(p.numel() for p in centreline_adapter.parameters())
        assert total_params < 5_000_000

    def test_centreline_head_prediction_is_probability(
        self, centreline_adapter: ModelAdapter, small_input: torch.Tensor
    ) -> None:
        output = centreline_adapter(small_input)
        sums = output.prediction.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        assert (output.prediction >= 0.0).all()

    def test_centreline_head_save_load_checkpoint(
        self,
        centreline_adapter: ModelAdapter,
        base_adapter: ModelAdapter,
        small_input: torch.Tensor,
        tmp_path: Path,
    ) -> None:
        from minivess.adapters.centreline_head import CentrelineHeadAdapter

        ckpt_path = tmp_path / "centreline_model.pth"
        centreline_adapter.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

        new_adapter = CentrelineHeadAdapter(base_adapter)
        new_adapter.load_checkpoint(ckpt_path)

        centreline_adapter.eval()
        new_adapter.eval()
        with torch.no_grad():
            out_orig = centreline_adapter(small_input)
            out_loaded = new_adapter(small_input)
        assert torch.allclose(out_orig.logits, out_loaded.logits, atol=1e-5)

    def test_centreline_head_eval_deterministic(
        self, centreline_adapter: ModelAdapter, small_input: torch.Tensor
    ) -> None:
        centreline_adapter.eval()
        with torch.no_grad():
            out1 = centreline_adapter(small_input)
            out2 = centreline_adapter(small_input)
        assert torch.allclose(
            out1.metadata["centreline_map"],
            out2.metadata["centreline_map"],
            atol=1e-6,
        )


class TestCentrelineDistanceMap:
    """Tests for GT centreline distance map computation."""

    def test_compute_centreline_distance_map(self) -> None:
        import numpy as np

        from minivess.adapters.centreline_head import compute_centreline_distance_map

        # Create a simple tube mask
        mask = np.zeros((16, 16, 16), dtype=bool)
        for z in range(4, 12):
            for y in range(6, 10):
                for x in range(6, 10):
                    if (y - 8) ** 2 + (x - 8) ** 2 <= 4:
                        mask[z, y, x] = True

        dist_map = compute_centreline_distance_map(mask)
        assert dist_map.shape == mask.shape
        # All values should be non-negative
        assert (dist_map >= 0).all()
        # Skeleton voxels should have distance 0
        assert dist_map.min() == 0.0

    def test_compute_centreline_distance_map_empty(self) -> None:
        import numpy as np

        from minivess.adapters.centreline_head import compute_centreline_distance_map

        mask = np.zeros((8, 8, 8), dtype=bool)
        dist_map = compute_centreline_distance_map(mask)
        assert dist_map.shape == mask.shape
        # Empty mask should return all zeros
        assert (dist_map == 0).all()
