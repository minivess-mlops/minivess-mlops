"""Tests for model merging utilities (linear, SLERP, layer-wise).

Training-free model merging: interpolate weights between loss-specialized
models in weight space. Based on MedSAMix (Yang et al. 2025) and
standard SLERP interpolation.

Issue: #309 | Phase 3 | Plan: T3.1 (RED)
"""

from __future__ import annotations

import torch
from torch import nn


def _make_net(seed: int = 42) -> nn.Module:
    """Create a simple deterministic 3D network."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv3d(1, 4, kernel_size=3, padding=1),
        nn.BatchNorm3d(4),
        nn.ReLU(),
        nn.Conv3d(4, 2, kernel_size=1),
    )


class TestLinearMerge:
    """Tests for linear (weighted average) model merging."""

    def test_linear_merge_two_state_dicts(self) -> None:
        """Linear merge at t=0.5 should produce element-wise average."""
        from minivess.ensemble.model_merging import linear_merge

        sd1 = _make_net(seed=1).state_dict()
        sd2 = _make_net(seed=2).state_dict()
        merged = linear_merge(sd1, sd2, t=0.5)

        for key in merged:
            if sd1[key].is_floating_point():
                expected = 0.5 * sd1[key] + 0.5 * sd2[key]
                torch.testing.assert_close(merged[key], expected, atol=1e-6, rtol=1e-5)

    def test_linear_merge_t0_returns_first(self) -> None:
        """Linear merge at t=0 should return the first model."""
        from minivess.ensemble.model_merging import linear_merge

        sd1 = _make_net(seed=1).state_dict()
        sd2 = _make_net(seed=2).state_dict()
        merged = linear_merge(sd1, sd2, t=0.0)

        for key in merged:
            if sd1[key].is_floating_point():
                torch.testing.assert_close(merged[key], sd1[key])

    def test_linear_merge_t1_returns_second(self) -> None:
        """Linear merge at t=1 should return the second model."""
        from minivess.ensemble.model_merging import linear_merge

        sd1 = _make_net(seed=1).state_dict()
        sd2 = _make_net(seed=2).state_dict()
        merged = linear_merge(sd1, sd2, t=1.0)

        for key in merged:
            if sd1[key].is_floating_point():
                torch.testing.assert_close(merged[key], sd2[key])


class TestSLERPMerge:
    """Tests for SLERP (Spherical Linear Interpolation) merging."""

    def test_slerp_merge_two_state_dicts(self) -> None:
        """SLERP merge should produce valid merged state dict."""
        from minivess.ensemble.model_merging import slerp_merge

        sd1 = _make_net(seed=1).state_dict()
        sd2 = _make_net(seed=2).state_dict()
        merged = slerp_merge(sd1, sd2, t=0.5)

        assert set(merged.keys()) == set(sd1.keys())
        for key in merged:
            assert merged[key].shape == sd1[key].shape

    def test_slerp_t0_returns_first_model(self) -> None:
        """SLERP at t=0 should return the first model."""
        from minivess.ensemble.model_merging import slerp_merge

        sd1 = _make_net(seed=1).state_dict()
        sd2 = _make_net(seed=2).state_dict()
        merged = slerp_merge(sd1, sd2, t=0.0)

        for key in merged:
            if sd1[key].is_floating_point():
                torch.testing.assert_close(merged[key], sd1[key], atol=1e-5, rtol=1e-4)

    def test_slerp_t1_returns_second_model(self) -> None:
        """SLERP at t=1 should return the second model."""
        from minivess.ensemble.model_merging import slerp_merge

        sd1 = _make_net(seed=1).state_dict()
        sd2 = _make_net(seed=2).state_dict()
        merged = slerp_merge(sd1, sd2, t=1.0)

        for key in merged:
            if sd1[key].is_floating_point():
                torch.testing.assert_close(merged[key], sd2[key], atol=1e-5, rtol=1e-4)


class TestLayerWiseMerge:
    """Tests for layer-wise merging with per-layer interpolation weights."""

    def test_layer_wise_merge_config(self) -> None:
        """Layer-wise merge should accept per-layer t values."""
        from minivess.ensemble.model_merging import layer_wise_merge

        sd1 = _make_net(seed=1).state_dict()
        sd2 = _make_net(seed=2).state_dict()

        # Use different t for each layer
        layer_weights = {key: 0.3 for key in sd1}
        merged = layer_wise_merge(sd1, sd2, layer_weights=layer_weights)

        assert set(merged.keys()) == set(sd1.keys())

    def test_merged_model_produces_valid_output(self) -> None:
        """Merged model should produce finite, correctly shaped output."""
        from minivess.ensemble.model_merging import linear_merge

        sd1 = _make_net(seed=1).state_dict()
        sd2 = _make_net(seed=2).state_dict()
        merged = linear_merge(sd1, sd2, t=0.5)

        model = _make_net(seed=0)
        model.load_state_dict(merged)
        model.eval()

        x = torch.randn(1, 1, 4, 4, 4)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2, 4, 4, 4)
        assert torch.isfinite(out).all()
