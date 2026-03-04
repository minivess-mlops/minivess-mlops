"""Tests for SWA (Stochastic Weight Averaging) model soup.

SWA averages state dicts across fold checkpoints to produce a single
model with lower generalization error and zero inference overhead.

Issue: #307 | Phase 1 | Plan: T1.1 (RED)
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn


def _make_simple_net(seed: int = 42) -> nn.Module:
    """Create a small deterministic network for testing."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv3d(1, 4, kernel_size=3, padding=1),
        nn.BatchNorm3d(4),
        nn.ReLU(),
        nn.Conv3d(4, 2, kernel_size=1),
    )


class TestSWAUtility:
    """Unit tests for uniform SWA averaging."""

    def test_swa_averages_state_dicts(self) -> None:
        """SWA of two state dicts should produce element-wise average."""
        from minivess.ensemble.model_soup import uniform_swa

        net1 = _make_simple_net(seed=1)
        net2 = _make_simple_net(seed=2)

        state_dicts = [net1.state_dict(), net2.state_dict()]
        averaged = uniform_swa(state_dicts)

        # Each floating-point parameter should be the mean of the two
        for key in averaged:
            if state_dicts[0][key].is_floating_point():
                expected = (state_dicts[0][key] + state_dicts[1][key]) / 2.0
                torch.testing.assert_close(averaged[key], expected)
            else:
                # Non-floating (e.g., num_batches_tracked): taken from first
                torch.testing.assert_close(averaged[key], state_dicts[0][key])

    def test_swa_preserves_architecture(self) -> None:
        """Averaged state dict must be loadable into the same architecture."""
        from minivess.ensemble.model_soup import uniform_swa

        net = _make_simple_net(seed=1)
        state_dicts = [_make_simple_net(seed=i).state_dict() for i in range(3)]
        averaged = uniform_swa(state_dicts)

        # Must load without errors
        net.load_state_dict(averaged)

    def test_swa_produces_single_model(self) -> None:
        """SWA result is a single state dict, not a list."""
        from minivess.ensemble.model_soup import uniform_swa

        state_dicts = [_make_simple_net(seed=i).state_dict() for i in range(4)]
        result = uniform_swa(state_dicts)

        assert isinstance(result, dict)
        assert all(isinstance(v, torch.Tensor) for v in result.values())

    def test_swa_from_fold_checkpoints(self) -> None:
        """SWA should work with checkpoints that have 'state_dict' key."""
        from minivess.ensemble.model_soup import swa_from_checkpoints

        # Simulate checkpoint files with state_dict + optimizer_state_dict
        checkpoints = []
        for i in range(3):
            net = _make_simple_net(seed=i)
            checkpoints.append(
                {
                    "state_dict": net.state_dict(),
                    "optimizer_state_dict": {},
                    "epoch": 100,
                    "fold": i,
                }
            )

        averaged = swa_from_checkpoints(checkpoints)
        assert isinstance(averaged, dict)

        # Verify keys match original architecture
        reference_keys = set(checkpoints[0]["state_dict"].keys())
        assert set(averaged.keys()) == reference_keys

    def test_swa_single_model_returns_clone(self) -> None:
        """SWA with a single state dict should return a clone of that dict."""
        from minivess.ensemble.model_soup import uniform_swa

        net = _make_simple_net(seed=42)
        original = net.state_dict()
        result = uniform_swa([copy.deepcopy(original)])

        for key in result:
            torch.testing.assert_close(result[key], original[key])

    def test_swa_empty_raises(self) -> None:
        """SWA with empty list should raise ValueError."""
        from minivess.ensemble.model_soup import uniform_swa

        with pytest.raises(ValueError, match="at least one"):
            uniform_swa([])

    def test_swa_output_valid_forward(self) -> None:
        """SWA-averaged model should produce valid forward pass output."""
        from minivess.ensemble.model_soup import uniform_swa

        nets = [_make_simple_net(seed=i) for i in range(3)]
        averaged = uniform_swa([n.state_dict() for n in nets])

        model = _make_simple_net(seed=0)
        model.load_state_dict(averaged)
        model.eval()

        x = torch.randn(1, 1, 4, 4, 4)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2, 4, 4, 4)
        assert torch.isfinite(out).all()
