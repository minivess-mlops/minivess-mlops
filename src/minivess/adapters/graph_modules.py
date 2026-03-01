"""Graph attention modules for topology-aware feature fusion.

Provides dense graph attention layers and the TFFMBlock3D module that
applies graph-based reasoning to 3D feature maps. No torch_geometric
dependency — uses dense attention on pooled spatial grids.

Reference: Ahmed et al. (2026). "TFFM: Topology-Aware Feature Fusion Module
via Latent Graph Reasoning for Retinal Vessel Segmentation." WACV 2026.
Ported from https://github.com/tffm-module/tffm-code (2D → 3D adaptation).

Uses established libraries: PyTorch for dense attention, F.adaptive_avg_pool3d
for multi-scale pooling. See CLAUDE.md Critical Rule #3.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GraphAttentionLayer(nn.Module):  # type: ignore[misc]
    """Dense multi-head graph attention layer.

    Operates on dense tensors [B, N, F] with dense adjacency [B, N, N].
    Uses the standard GAT attention mechanism (Veličković et al., ICLR 2018)
    with multi-head support.

    Parameters
    ----------
    in_features:
        Input feature dimension per node.
    out_features:
        Output feature dimension per node.
    n_heads:
        Number of attention heads. out_features must be divisible by n_heads.
    dropout:
        Dropout probability on attention weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert out_features % n_heads == 0, "out_features must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads
        self.out_features = out_features

        # Linear projection for each head
        self.W = nn.Linear(in_features, out_features, bias=False)
        # Attention coefficients: [a_l || a_r] per head
        self.a_l = nn.Parameter(torch.empty(n_heads, self.head_dim))
        self.a_r = nn.Parameter(torch.empty(n_heads, self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_l.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_r.unsqueeze(0))

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """Apply graph attention.

        Parameters
        ----------
        x:
            Node features [B, N, F_in].
        adj:
            Adjacency matrix [B, N, N]. Non-zero entries indicate edges.

        Returns
        -------
        Updated node features [B, N, F_out].
        """
        B, N, _ = x.shape

        # Project: [B, N, F_in] → [B, N, out_features]
        h = self.W(x)
        # Reshape to heads: [B, N, n_heads, head_dim]
        h = h.view(B, N, self.n_heads, self.head_dim)

        # Compute attention scores
        # e_l[b, i, head] = h[b, i, head, :] · a_l[head, :]
        e_l = (h * self.a_l).sum(dim=-1)  # [B, N, n_heads]
        e_r = (h * self.a_r).sum(dim=-1)  # [B, N, n_heads]

        # Pairwise attention: e[b, i, j, head] = e_l[b,i,head] + e_r[b,j,head]
        # [B, N, 1, n_heads] + [B, 1, N, n_heads] → [B, N, N, n_heads]
        e = self.leaky_relu(e_l.unsqueeze(2) + e_r.unsqueeze(1))

        # Mask with adjacency (set non-edges to -inf)
        # adj: [B, N, N] → [B, N, N, 1]
        mask = adj.unsqueeze(-1)
        e = e.masked_fill(mask == 0, float("-inf"))

        # Softmax over neighbors
        alpha = F.softmax(e, dim=2)  # [B, N, N, n_heads]
        alpha = self.dropout(alpha)

        # Aggregate: weighted sum of neighbor features
        # h: [B, N, n_heads, head_dim] → [B, N_j, n_heads, head_dim]
        # alpha: [B, N_i, N_j, n_heads]
        # out: [B, N_i, n_heads, head_dim]
        out = torch.einsum("bijn,bjnh->binh", alpha, h)

        # Concatenate heads: [B, N, out_features]
        out = out.reshape(B, N, self.out_features)

        return out


class TFFMBlock3D(nn.Module):  # type: ignore[misc]
    """Topology Feature Fusion Module for 3D volumes.

    Takes a 3D feature map (B, C, D, H, W) and enhances it via graph-based
    reasoning over pooled spatial grid cells. Each grid cell becomes a node
    in a graph, and graph attention performs cross-region reasoning.

    Pipeline:
      1. Channel compression: C → hidden_dim via Conv3d(1x1x1)
      2. Spatial pooling to grid: (D,H,W) → (G,G,G) cells, each is a node
      3. kNN graph construction via cosine similarity
      4. Graph attention (2 layers)
      5. Upsample back to original spatial size
      6. Channel expansion: hidden_dim → C via Conv3d(1x1x1)
      7. Gated residual connection

    Parameters
    ----------
    in_channels:
        Number of input/output channels.
    grid_size:
        Size of the spatial grid for graph construction (G).
    hidden_dim:
        Hidden feature dimension for graph nodes.
    n_heads:
        Number of attention heads.
    k_neighbors:
        Number of neighbors in kNN graph. If >= grid_size^3, fully connected.
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        grid_size: int = 8,
        hidden_dim: int = 32,
        n_heads: int = 4,
        k_neighbors: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors

        # Channel compression
        self.compress = nn.Conv3d(in_channels, hidden_dim, kernel_size=1)

        # Graph attention layers
        self.gat1 = GraphAttentionLayer(hidden_dim, hidden_dim, n_heads, dropout)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim, n_heads, dropout)

        # Channel expansion
        self.expand = nn.Conv3d(hidden_dim, in_channels, kernel_size=1)

        # Gated residual
        self.gate = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Norm layers
        self.norm1 = nn.InstanceNorm3d(hidden_dim)
        self.norm2 = nn.InstanceNorm3d(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Enhance features via graph attention.

        Parameters
        ----------
        x:
            Feature map [B, C, D, H, W].

        Returns
        -------
        Enhanced feature map [B, C, D, H, W] (same shape as input).
        """
        B, C, D, H, W = x.shape
        residual = x

        # 1. Channel compression
        h = self.compress(x)  # [B, hidden_dim, D, H, W]

        # 2. Pool to spatial grid → graph nodes
        G = self.grid_size
        # Adaptive pool to (G, G, G) — uses F.adaptive_avg_pool3d (PyTorch built-in)
        h_pooled = F.adaptive_avg_pool3d(h, (G, G, G))  # [B, hidden_dim, G, G, G]

        # Reshape to node features: [B, N, hidden_dim] where N = G^3
        N = G * G * G
        nodes = h_pooled.reshape(B, self.hidden_dim, N).permute(
            0, 2, 1
        )  # [B, N, hidden_dim]

        # 3. Build kNN adjacency via cosine similarity
        adj = self._build_knn_adjacency(nodes)  # [B, N, N]

        # 4. Graph attention (2 layers with residual)
        nodes_gat = self.gat1(nodes, adj)
        # Reshape for norm: [B, hidden_dim, G, G, G]
        nodes_gat_vol = nodes_gat.permute(0, 2, 1).reshape(B, self.hidden_dim, G, G, G)
        nodes_gat_vol = self.norm1(nodes_gat_vol)
        nodes_gat = nodes_gat_vol.reshape(B, self.hidden_dim, N).permute(0, 2, 1)
        nodes_gat = F.gelu(nodes_gat)

        nodes_gat2 = self.gat2(nodes_gat, adj)
        nodes_gat2_vol = nodes_gat2.permute(0, 2, 1).reshape(
            B, self.hidden_dim, G, G, G
        )
        nodes_gat2_vol = self.norm2(nodes_gat2_vol)
        # Residual from first GAT layer
        nodes_out = nodes_gat2_vol + nodes_gat_vol

        # 5. Upsample back to original spatial size
        h_enhanced = F.interpolate(
            nodes_out, size=(D, H, W), mode="trilinear", align_corners=False
        )  # [B, hidden_dim, D, H, W]

        # 6. Channel expansion
        h_expanded = self.expand(h_enhanced)  # [B, C, D, H, W]

        # 7. Gated residual connection
        gate_val = self.gate(h_expanded)
        out: Tensor = residual + gate_val * h_expanded

        return out

    def _build_knn_adjacency(self, nodes: Tensor) -> Tensor:
        """Build kNN graph from node features using cosine similarity.

        Parameters
        ----------
        nodes:
            Node features [B, N, F].

        Returns
        -------
        Adjacency matrix [B, N, N] with 1s for kNN connections.
        """
        # Cosine similarity via F.cosine_similarity (PyTorch built-in)
        # Normalize features
        nodes_norm = F.normalize(nodes, p=2, dim=-1)  # [B, N, F]
        # Pairwise cosine similarity
        sim = torch.bmm(nodes_norm, nodes_norm.transpose(1, 2))  # [B, N, N]

        N = nodes.shape[1]
        k = min(self.k_neighbors, N)

        # Top-k neighbors per node
        _, topk_idx = sim.topk(k, dim=-1)  # [B, N, k]

        # Build binary adjacency
        adj = torch.zeros_like(sim)
        # Scatter 1s at top-k positions
        adj.scatter_(2, topk_idx, 1.0)
        # Symmetrize (undirected graph)
        adj = (adj + adj.transpose(1, 2)).clamp(max=1.0)

        return adj
