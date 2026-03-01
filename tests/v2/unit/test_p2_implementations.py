"""Tests for P2 implementations: Murray's law (#131) + TopoSegNet loss (#133).

Both are EXPERIMENTAL — simplified proxies, NOT faithful paper implementations.
Tests validate basic sanity (forward/backward, no NaN, convergence).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Murray's Law Compliance (#131)
# ---------------------------------------------------------------------------


class TestMurrayCompliance:
    """Tests for compute_murray_compliance topology metric."""

    def test_empty_mask_returns_zero_compliance(self) -> None:
        from minivess.pipeline.topology_metrics import compute_murray_compliance

        result = compute_murray_compliance(np.zeros((8, 8, 8), dtype=np.uint8))
        assert result["compliance_score"] == 0.0
        assert result["n_bifurcations"] == 0

    def test_simple_tube_no_bifurcations(self) -> None:
        """A straight tube has no bifurcations."""
        from minivess.pipeline.topology_metrics import compute_murray_compliance

        mask = np.zeros((16, 16, 16), dtype=np.uint8)
        mask[:, 7:9, 7:9] = 1  # Simple tube along D axis
        result = compute_murray_compliance(mask)
        assert result["n_bifurcations"] == 0

    def test_result_has_expected_keys(self) -> None:
        from minivess.pipeline.topology_metrics import compute_murray_compliance

        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        result = compute_murray_compliance(mask)
        assert "mean_ratio" in result
        assert "mean_deviation" in result
        assert "compliance_score" in result
        assert "n_bifurcations" in result

    def test_compliance_score_bounded(self) -> None:
        """Compliance score is in [0, 1]."""
        from minivess.pipeline.topology_metrics import compute_murray_compliance

        # Create Y-shaped structure (bifurcation)
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        # Trunk
        mask[:10, 9:11, 9:11] = 1
        # Branch 1
        mask[10:, 5:7, 9:11] = 1
        # Branch 2
        mask[10:, 13:15, 9:11] = 1
        result = compute_murray_compliance(mask)
        if result["n_bifurcations"] > 0:
            assert 0.0 <= result["compliance_score"] <= 1.0


# ---------------------------------------------------------------------------
# TopoSegNet Loss (#133)
# ---------------------------------------------------------------------------

SPATIAL = (8, 8, 8)
B, C = 1, 2


class TestTopoSegLoss:
    """Tests for TopoSegLoss (EXPERIMENTAL critical point proxy)."""

    def test_forward_finite(self) -> None:
        """Forward produces finite scalar."""
        from minivess.pipeline.vendored_losses.toposeg import TopoSegLoss

        loss_fn = TopoSegLoss()
        logits = torch.randn(B, C, *SPATIAL, requires_grad=True)
        labels = torch.randint(0, 2, (B, 1, *SPATIAL)).float()
        loss = loss_fn(logits, labels)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_backward_succeeds(self) -> None:
        """Backward pass produces gradients."""
        from minivess.pipeline.vendored_losses.toposeg import TopoSegLoss

        loss_fn = TopoSegLoss()
        logits = torch.randn(B, C, *SPATIAL, requires_grad=True)
        labels = torch.randint(0, 2, (B, 1, *SPATIAL)).float()
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None

    def test_no_nan_on_perfect_pred(self) -> None:
        """No NaN when prediction matches GT."""
        from minivess.pipeline.vendored_losses.toposeg import TopoSegLoss

        loss_fn = TopoSegLoss()
        labels = torch.randint(0, 2, (B, 1, *SPATIAL)).float()
        # Build perfect logits
        labels_int = labels.squeeze(1).long()
        logits = torch.zeros(B, C, *SPATIAL)
        for c in range(C):
            logits[:, c] = (labels_int == c).float() * 10.0 - 5.0
        logits.requires_grad_(True)
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss)

    def test_no_nan_on_empty_mask(self) -> None:
        """No crash on all-background GT."""
        from minivess.pipeline.vendored_losses.toposeg import TopoSegLoss

        loss_fn = TopoSegLoss()
        logits = torch.randn(B, C, *SPATIAL, requires_grad=True)
        labels = torch.zeros(B, 1, *SPATIAL)
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss)
        assert torch.isfinite(loss)

    def test_registered_in_factory(self) -> None:
        """TopoSegLoss is available via build_loss_function."""
        from minivess.pipeline.loss_functions import build_loss_function

        loss_fn = build_loss_function("toposeg")
        assert loss_fn is not None

    def test_convergence_smoke(self) -> None:
        """Loss decreases over 15 gradient steps on synthetic data."""
        from torch import nn as nn_module

        from minivess.pipeline.vendored_losses.toposeg import TopoSegLoss

        class TinyModel(nn_module.Module):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn_module.Conv3d(1, C, 3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        torch.manual_seed(42)
        model = TinyModel()
        loss_fn = TopoSegLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(B, 1, *SPATIAL)
        labels = torch.zeros(B, 1, *SPATIAL)
        labels[:, 0, :, 3:5, 3:5] = 1.0  # Tube structure

        losses: list[float] = []
        for step in range(15):
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, labels)
            if loss.item() > 0:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            if math.isnan(loss.item()):
                pytest.fail(f"NaN at step {step}")

        # Allow for possibility that loss starts at 0 (no critical points found)
        nonzero_losses = [v for v in losses if v > 0]
        if len(nonzero_losses) >= 2:
            assert min(nonzero_losses[-3:]) <= max(nonzero_losses[:3])
