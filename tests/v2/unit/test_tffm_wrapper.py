"""Tests for TFFM (Topology Feature Fusion Module) wrapper adapter (#119).

Covers: TFFMBlock3D graph attention module, TFFMWrapper composition pattern,
forward/backward pass, metadata, config, checkpoint save/load.

Architecture: TFFMWrapper(DynUNetAdapter(config)) — wraps any ModelAdapter
via forward hooks on the bottleneck layer. Uses dense graph attention
(no torch_geometric dependency).

Reference: Ahmed et al. (2026). "TFFM: Topology-Aware Feature Fusion Module
via Latent Graph Reasoning for Retinal Vessel Segmentation." WACV 2026.

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
    """Small DynUNet config for testing (tiny filters for speed)."""
    return ModelConfig(
        family=ModelFamily.MONAI_DYNUNET,
        name="test-dynunet-for-tffm",
        in_channels=1,
        out_channels=2,
        architecture_params={"filters": [8, 16, 32]},
    )


@pytest.fixture
def base_adapter(dynunet_config: ModelConfig) -> ModelAdapter:
    from minivess.adapters.dynunet import DynUNetAdapter

    return DynUNetAdapter(dynunet_config, filters=[8, 16, 32])


@pytest.fixture
def tffm_wrapper(base_adapter: ModelAdapter) -> ModelAdapter:
    from minivess.adapters.tffm_wrapper import TFFMWrapper

    return TFFMWrapper(base_adapter, grid_size=4, hidden_dim=8, n_heads=2)


@pytest.fixture
def small_input() -> torch.Tensor:
    """Small 3D volume for testing: (B=1, C=1, D=16, H=16, W=8)."""
    return torch.randn(1, 1, 16, 16, 8)


# ---------------------------------------------------------------------------
# TFFMBlock3D — standalone graph attention module tests
# ---------------------------------------------------------------------------


class TestTFFMBlock3D:
    """Tests for the standalone TFFM graph attention block."""

    def test_tffm_block_preserves_shape(self) -> None:
        from minivess.adapters.graph_modules import TFFMBlock3D

        block = TFFMBlock3D(in_channels=32, grid_size=4, hidden_dim=16, n_heads=2)
        x = torch.randn(1, 32, 8, 8, 4)
        out = block(x)
        assert out.shape == x.shape

    def test_tffm_block_gradient_flow(self) -> None:
        from minivess.adapters.graph_modules import TFFMBlock3D

        block = TFFMBlock3D(in_channels=16, grid_size=4, hidden_dim=8, n_heads=2)
        x = torch.randn(1, 16, 8, 8, 4, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_tffm_block_different_spatial_sizes(self) -> None:
        from minivess.adapters.graph_modules import TFFMBlock3D

        block = TFFMBlock3D(in_channels=16, grid_size=4, hidden_dim=8, n_heads=1)
        # Non-square spatial dimensions
        for shape in [(1, 16, 8, 8, 4), (1, 16, 4, 4, 4), (1, 16, 16, 8, 4)]:
            x = torch.randn(*shape)
            out = block(x)
            assert out.shape == x.shape

    def test_tffm_block_batch_independence(self) -> None:
        """Each sample in the batch should be processed independently."""
        from minivess.adapters.graph_modules import TFFMBlock3D

        block = TFFMBlock3D(in_channels=16, grid_size=4, hidden_dim=8, n_heads=1)
        block.eval()
        x = torch.randn(2, 16, 8, 8, 4)
        out_batch = block(x)
        out_single_0 = block(x[0:1])
        out_single_1 = block(x[1:2])
        assert torch.allclose(out_batch[0:1], out_single_0, atol=1e-5)
        assert torch.allclose(out_batch[1:2], out_single_1, atol=1e-5)


class TestGraphAttentionLayer3D:
    """Tests for the dense graph attention layer."""

    def test_attention_layer_output_shape(self) -> None:
        from minivess.adapters.graph_modules import GraphAttentionLayer

        layer = GraphAttentionLayer(in_features=16, out_features=8, n_heads=2)
        # B=1, N=16 nodes, F=16 features
        x = torch.randn(1, 16, 16)
        adj = torch.ones(1, 16, 16)  # fully connected
        out = layer(x, adj)
        assert out.shape == (1, 16, 8)

    def test_attention_layer_gradient_flow(self) -> None:
        from minivess.adapters.graph_modules import GraphAttentionLayer

        layer = GraphAttentionLayer(in_features=8, out_features=8, n_heads=1)
        x = torch.randn(1, 8, 8, requires_grad=True)
        adj = torch.ones(1, 8, 8)
        out = layer(x, adj)
        out.sum().backward()
        assert x.grad is not None

    def test_attention_layer_masked_adjacency(self) -> None:
        """Disconnected nodes should not attend to each other."""
        from minivess.adapters.graph_modules import GraphAttentionLayer

        layer = GraphAttentionLayer(in_features=8, out_features=8, n_heads=1)
        layer.eval()
        x = torch.randn(1, 4, 8)
        # Block-diagonal: nodes {0,1} and {2,3} are disconnected
        adj = torch.zeros(1, 4, 4)
        adj[0, 0, 0] = adj[0, 0, 1] = adj[0, 1, 0] = adj[0, 1, 1] = 1
        adj[0, 2, 2] = adj[0, 2, 3] = adj[0, 3, 2] = adj[0, 3, 3] = 1
        out_block = layer(x, adj)
        # Fully connected
        adj_full = torch.ones(1, 4, 4)
        out_full = layer(x, adj_full)
        # Outputs should differ (different connectivity)
        assert not torch.allclose(out_block, out_full, atol=1e-5)


# ---------------------------------------------------------------------------
# TFFMWrapper — composition pattern tests
# ---------------------------------------------------------------------------


class TestTFFMWrapper:
    """Tests for the TFFMWrapper adapter composition."""

    def test_tffm_wrapper_is_model_adapter(self, tffm_wrapper: ModelAdapter) -> None:
        assert isinstance(tffm_wrapper, ModelAdapter)

    def test_tffm_wrapper_returns_segmentation_output(
        self, tffm_wrapper: ModelAdapter, small_input: torch.Tensor
    ) -> None:
        output = tffm_wrapper(small_input)
        assert isinstance(output, SegmentationOutput)

    def test_tffm_wrapper_preserves_output_shape(
        self, tffm_wrapper: ModelAdapter, small_input: torch.Tensor
    ) -> None:
        output = tffm_wrapper(small_input)
        assert output.prediction.shape == (1, 2, 16, 16, 8)
        assert output.logits.shape == (1, 2, 16, 16, 8)

    def test_tffm_wrapper_prediction_is_probability(
        self, tffm_wrapper: ModelAdapter, small_input: torch.Tensor
    ) -> None:
        output = tffm_wrapper(small_input)
        # Probabilities should sum to ~1 along channel dim
        sums = output.prediction.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        # All non-negative
        assert (output.prediction >= 0.0).all()

    def test_tffm_wrapper_metadata_contains_tffm_info(
        self, tffm_wrapper: ModelAdapter, small_input: torch.Tensor
    ) -> None:
        output = tffm_wrapper(small_input)
        assert "tffm_applied" in output.metadata
        assert output.metadata["tffm_applied"] is True
        assert "architecture" in output.metadata

    def test_tffm_wrapper_get_config(self, tffm_wrapper: ModelAdapter) -> None:
        cfg = tffm_wrapper.get_config()
        assert isinstance(cfg, AdapterConfigInfo)
        # Base model config should be preserved
        assert cfg.in_channels == 1
        assert cfg.out_channels == 2
        # TFFM-specific params in extras
        assert "tffm_grid_size" in cfg.extras
        assert "tffm_hidden_dim" in cfg.extras
        assert "tffm_n_heads" in cfg.extras

    def test_tffm_wrapper_trainable_parameters(
        self, tffm_wrapper: ModelAdapter, base_adapter: ModelAdapter
    ) -> None:
        # TFFM wrapper should have MORE trainable params than base
        tffm_params = tffm_wrapper.trainable_parameters()
        base_params = base_adapter.trainable_parameters()
        assert tffm_params > base_params

    def test_tffm_wrapper_gradient_flow(
        self, tffm_wrapper: ModelAdapter, small_input: torch.Tensor
    ) -> None:
        """Gradients should flow through TFFM block to all trainable params."""
        output = tffm_wrapper(small_input)
        loss = output.logits.sum()
        loss.backward()
        # Check that TFFM block parameters received gradients
        tffm_has_grads = False
        for name, param in tffm_wrapper.named_parameters():
            if "tffm" in name and param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                tffm_has_grads = True
        assert tffm_has_grads, "No TFFM parameters found with gradients"

    def test_tffm_wrapper_save_load_checkpoint(
        self,
        tffm_wrapper: ModelAdapter,
        base_adapter: ModelAdapter,
        small_input: torch.Tensor,
        tmp_path: Path,
    ) -> None:
        from minivess.adapters.tffm_wrapper import TFFMWrapper

        ckpt_path = tmp_path / "tffm_model.pth"
        tffm_wrapper.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

        # Create a new wrapper and load checkpoint
        new_wrapper = TFFMWrapper(base_adapter, grid_size=4, hidden_dim=8, n_heads=2)
        new_wrapper.load_checkpoint(ckpt_path)

        # Outputs should match
        tffm_wrapper.eval()
        new_wrapper.eval()
        with torch.no_grad():
            out_orig = tffm_wrapper(small_input)
            out_loaded = new_wrapper(small_input)
        assert torch.allclose(out_orig.logits, out_loaded.logits, atol=1e-5)

    def test_tffm_wrapper_vram_budget(
        self, tffm_wrapper: ModelAdapter, small_input: torch.Tensor
    ) -> None:
        """Forward + backward should work within budget (CPU test as proxy)."""
        output = tffm_wrapper(small_input)
        loss = output.logits.sum()
        loss.backward()
        # If we get here without OOM, the test passes
        # Also verify the model is not unreasonably large
        total_params = sum(p.numel() for p in tffm_wrapper.parameters())
        # Should be < 5M params for small config
        assert total_params < 5_000_000

    def test_tffm_wrapper_eval_mode_deterministic(
        self, tffm_wrapper: ModelAdapter, small_input: torch.Tensor
    ) -> None:
        """In eval mode, repeated calls should produce identical outputs."""
        tffm_wrapper.eval()
        with torch.no_grad():
            out1 = tffm_wrapper(small_input)
            out2 = tffm_wrapper(small_input)
        assert torch.allclose(out1.logits, out2.logits, atol=1e-6)
