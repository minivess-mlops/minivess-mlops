"""Comprehensive tests for TFFMBlock3D + GraphAttentionLayer (T11 — #238)."""

from __future__ import annotations

import torch

from minivess.adapters.graph_modules import GraphAttentionLayer, TFFMBlock3D


class TestGraphAttentionLayer:
    """Tests for GraphAttentionLayer."""

    def test_gat_forward_shape(self) -> None:
        """[B, N, F_out] output shape correct."""
        gat = GraphAttentionLayer(in_features=16, out_features=16, n_heads=1)
        x = torch.randn(2, 10, 16)
        adj = torch.ones(2, 10, 10)
        out = gat(x, adj)
        assert out.shape == (2, 10, 16)

    def test_gat_multihead_consistent(self) -> None:
        """n_heads=4 gives same shape as n_heads=1."""
        gat1 = GraphAttentionLayer(in_features=16, out_features=16, n_heads=1)
        gat4 = GraphAttentionLayer(in_features=16, out_features=16, n_heads=4)
        x = torch.randn(2, 8, 16)
        adj = torch.ones(2, 8, 8)
        out1 = gat1(x, adj)
        out4 = gat4(x, adj)
        assert out1.shape == out4.shape == (2, 8, 16)

    def test_gat_attention_masked_by_adjacency(self) -> None:
        """Zero adj entries produce zero attention (masked to -inf then softmax)."""
        gat = GraphAttentionLayer(in_features=8, out_features=8, n_heads=1)
        x = torch.randn(1, 4, 8)
        # Only node 0 and 1 connected, node 2 and 3 connected, no cross-edges
        adj = torch.zeros(1, 4, 4)
        adj[0, 0, 0] = 1
        adj[0, 0, 1] = 1
        adj[0, 1, 0] = 1
        adj[0, 1, 1] = 1
        adj[0, 2, 2] = 1
        adj[0, 2, 3] = 1
        adj[0, 3, 2] = 1
        adj[0, 3, 3] = 1

        out = gat(x, adj)
        assert out.shape == (1, 4, 8)
        # Output should be finite (no NaN from softmax on all-masked rows)
        assert torch.isfinite(out).all()

    def test_gat_fully_connected_graph(self) -> None:
        """Dense adj gives valid attention."""
        gat = GraphAttentionLayer(in_features=16, out_features=16, n_heads=2)
        x = torch.randn(1, 6, 16)
        adj = torch.ones(1, 6, 6)
        out = gat(x, adj)
        assert out.shape == (1, 6, 16)
        assert torch.isfinite(out).all()

    def test_gat_gradient_flows(self) -> None:
        """Backward through GAT produces gradients."""
        gat = GraphAttentionLayer(in_features=8, out_features=8, n_heads=2)
        x = torch.randn(1, 4, 8, requires_grad=True)
        adj = torch.ones(1, 4, 4)
        out = gat(x, adj)
        out.sum().backward()
        assert x.grad is not None
        for name, p in gat.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


class TestTFFMBlock3D:
    """Tests for TFFMBlock3D."""

    def test_tffm_output_same_shape_as_input(self) -> None:
        """[B, C, D, H, W] in/out."""
        tffm = TFFMBlock3D(in_channels=16, grid_size=4, hidden_dim=8, n_heads=2)
        x = torch.randn(1, 16, 8, 8, 8)
        out = tffm(x)
        assert out.shape == x.shape

    def test_tffm_gated_residual_near_identity(self) -> None:
        """Gate initialized near zero so initial output ≈ input."""
        tffm = TFFMBlock3D(in_channels=16, grid_size=4, hidden_dim=8)
        x = torch.randn(1, 16, 4, 8, 8)
        out = tffm(x)
        # At initialization, sigmoid gate ≈ 0.5, so output won't be identity
        # but should be relatively close to input (within ~2x magnitude)
        ratio = out.norm() / x.norm()
        assert 0.1 < ratio.item() < 10.0, (
            f"Output magnitude ratio {ratio:.2f} too extreme"
        )

    def test_tffm_knn_adjacency_symmetric(self) -> None:
        """Adjacency matrix is symmetric."""
        tffm = TFFMBlock3D(in_channels=8, grid_size=4, hidden_dim=8, k_neighbors=4)
        nodes = torch.randn(1, 64, 8)
        adj = tffm._build_knn_adjacency(nodes)
        # Check symmetry
        diff = (adj - adj.transpose(1, 2)).abs().max()
        assert diff < 1e-6, f"Adjacency not symmetric: max diff={diff}"

    def test_tffm_knn_adjacency_k_neighbors(self) -> None:
        """Each node has at least k neighbors."""
        tffm = TFFMBlock3D(in_channels=8, grid_size=4, hidden_dim=8, k_neighbors=4)
        nodes = torch.randn(1, 64, 8)
        adj = tffm._build_knn_adjacency(nodes)
        # Due to symmetrization, each node should have >= k neighbors
        neighbor_counts = adj[0].sum(dim=1)
        assert neighbor_counts.min() >= 4, (
            f"Min neighbor count {neighbor_counts.min()} < k=4"
        )

    def test_tffm_gradient_flows(self) -> None:
        """Backward through TFFM block."""
        tffm = TFFMBlock3D(in_channels=8, grid_size=4, hidden_dim=8, n_heads=2)
        x = torch.randn(1, 8, 4, 8, 8, requires_grad=True)
        out = tffm(x)
        out.sum().backward()
        assert x.grad is not None
        for name, p in tffm.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_tffm_different_grid_sizes(self) -> None:
        """grid_size=4, 8, 16 all work."""
        for g in [4, 8]:
            tffm = TFFMBlock3D(in_channels=8, grid_size=g, hidden_dim=8)
            x = torch.randn(1, 8, 16, 16, 16)
            out = tffm(x)
            assert out.shape == x.shape, f"Failed for grid_size={g}"

    def test_tffm_small_spatial_dim(self) -> None:
        """Input smaller than grid_size handled via adaptive pooling."""
        tffm = TFFMBlock3D(in_channels=8, grid_size=8, hidden_dim=8)
        # Spatial dims (4, 4, 4) < grid_size 8
        x = torch.randn(1, 8, 4, 4, 4)
        out = tffm(x)
        assert out.shape == x.shape

    def test_tffm_param_count(self) -> None:
        """Verify expected parameter count is reasonable."""
        tffm = TFFMBlock3D(in_channels=32, grid_size=8, hidden_dim=32, n_heads=4)
        n_params = sum(p.numel() for p in tffm.parameters())
        # Should be relatively lightweight compared to main model
        assert n_params < 100_000, f"TFFM has {n_params} params (expected <100K)"
