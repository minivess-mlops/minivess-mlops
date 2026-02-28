"""CAPE Loss — Connectivity-Aware Path Enforcement.

Penalises broken paths between endpoint pairs on the ground-truth skeleton.
Uses differentiable soft geodesic distance via iterative diffusion on the
predicted probability map.

Inspired by:
    Luo et al. "Connectivity-Aware Path Enforcement for Tubular
    Structure Segmentation" MICCAI 2025.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CAPELoss(nn.Module):  # type: ignore[misc]
    """Connectivity-Aware Path Enforcement loss.

    Samples endpoint pairs on the GT foreground, then checks whether
    the predicted probability map maintains connectivity between them
    via iterative diffusion (soft geodesic proxy).

    Parameters
    ----------
    n_pairs:
        Number of endpoint pairs to sample per batch element.
    n_diffusion_iters:
        Number of diffusion iterations for soft geodesic.
    softmax:
        Apply softmax to logits.
    smooth:
        Smoothing factor for numerical stability.
    """

    def __init__(
        self,
        n_pairs: int = 32,
        n_diffusion_iters: int = 10,
        softmax: bool = True,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.n_pairs = n_pairs
        self.n_diffusion_iters = n_diffusion_iters
        self.softmax = softmax
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute CAPE loss.

        Parameters
        ----------
        logits:
            (B, C, D, H, W) raw logits.
        labels:
            (B, 1, D, H, W) integer or float labels.

        Returns
        -------
        Scalar loss tensor.
        """
        if self.softmax:
            fg_prob = F.softmax(logits, dim=1)[:, 1:2]  # foreground prob
        else:
            fg_prob = torch.sigmoid(logits)

        gt = (labels > 0.5).float()
        batch_size = logits.shape[0]

        total_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        valid_count = 0

        for b in range(batch_size):
            gt_b = gt[b, 0]  # (D, H, W)
            prob_b = fg_prob[b, 0]  # (D, H, W)

            # Find foreground voxel coordinates
            fg_coords = torch.nonzero(gt_b > 0.5, as_tuple=False)  # (N, 3)

            if len(fg_coords) < 2:
                continue

            # Sample endpoint pairs
            n_fg = len(fg_coords)
            n_actual_pairs = min(self.n_pairs, n_fg * (n_fg - 1) // 2)
            if n_actual_pairs < 1:
                continue

            # Random pair indices
            idx_a = torch.randint(0, n_fg, (n_actual_pairs,), device=logits.device)
            idx_b = torch.randint(0, n_fg, (n_actual_pairs,), device=logits.device)
            # Avoid self-pairs
            same = idx_a == idx_b
            idx_b[same] = (idx_b[same] + 1) % n_fg

            endpoints_a = fg_coords[idx_a]  # (n_pairs, 3)
            endpoints_b = fg_coords[idx_b]  # (n_pairs, 3)

            # Compute soft connectivity score for each pair
            pair_loss = self._compute_pair_connectivity(
                prob_b, endpoints_a, endpoints_b
            )
            total_loss = total_loss + pair_loss
            valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return total_loss / valid_count

    def _compute_pair_connectivity(
        self,
        prob_map: torch.Tensor,
        endpoints_a: torch.Tensor,
        endpoints_b: torch.Tensor,
    ) -> torch.Tensor:
        """Check soft connectivity between endpoint pairs via diffusion.

        Uses the predicted probability map as a conductance field.
        Iteratively diffuses "heat" from source endpoints and measures
        how much reaches the target endpoints.

        Parameters
        ----------
        prob_map:
            (D, H, W) foreground probability.
        endpoints_a:
            (n_pairs, 3) source endpoints.
        endpoints_b:
            (n_pairs, 3) target endpoints.

        Returns
        -------
        Scalar loss: 1 - mean connectivity score.
        """
        n_pairs = endpoints_a.shape[0]

        # Initialize heat map at source endpoints
        heat = torch.zeros_like(prob_map)
        for i in range(n_pairs):
            z, y, x = endpoints_a[i]
            heat[z, y, x] = 1.0

        # Diffuse heat through probability map
        # Use 3D average pooling as diffusion + multiply by probability
        heat = heat.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        conductance = prob_map.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

        for _ in range(self.n_diffusion_iters):
            # Diffuse: average with neighbors
            diffused = F.avg_pool3d(heat, kernel_size=3, stride=1, padding=1)
            # Weight by conductance (probability)
            heat = diffused * conductance
            # Re-normalize to prevent vanishing
            heat_max = heat.max()
            if heat_max > 0:
                heat = heat / (heat_max + self.smooth)

        heat = heat.squeeze(0).squeeze(0)  # back to (D, H, W)

        # Measure heat at target endpoints
        connectivity_scores = torch.zeros(
            n_pairs, device=prob_map.device, dtype=prob_map.dtype
        )
        for i in range(n_pairs):
            z, y, x = endpoints_b[i]
            connectivity_scores[i] = heat[z, y, x]

        # Loss: penalize low connectivity
        return 1.0 - connectivity_scores.mean()
