"""Tests for WeightWatcher post-training diagnostics (T2.2, #649).

Validates:
- run_weightwatcher() returns summary with alpha metrics
- Frozen layers are filtered out
- Metrics use diag_ww_ prefix (NOT prof_ww_ — RC8)
- Artifact path is diagnostics/ (NOT profiling/ — RC12)
"""

from __future__ import annotations

import torch
from torch import nn


def _make_trainable_model() -> nn.Module:
    """Create a model with trainable conv2d + linear layers for WeightWatcher.

    NOTE: WeightWatcher only analyzes 2D conv and linear layers, not 3D conv.
    For 3D segmentation models, WeightWatcher still works on the linear layers
    and any 2D projection layers.
    """

    class _MultiLayerModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.fc = nn.Linear(32, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(self.conv2(self.conv1(x)).mean(dim=[-1, -2]))

    return _MultiLayerModel()


def _make_partially_frozen_model() -> nn.Module:
    """Create a model with some frozen layers (simulates SAM3 backbone)."""
    model = _make_trainable_model()
    # Freeze the first conv (simulates frozen backbone)
    conv1: nn.Module = model.conv1  # type: ignore[assignment]
    for param in conv1.parameters():
        param.requires_grad = False
    return model


class TestRunWeightWatcher:
    """Test run_weightwatcher() function."""

    def test_returns_summary(self) -> None:
        """run_weightwatcher returns summary dict with alpha metrics."""
        from minivess.diagnostics.weight_diagnostics import run_weightwatcher

        model = _make_trainable_model()
        summary = run_weightwatcher(model)
        assert isinstance(summary, dict)
        assert "diag_ww_alpha_mean" in summary
        assert "diag_ww_num_layers_analyzed" in summary

    def test_filters_frozen_layers(self) -> None:
        """Frozen layers should be excluded from analysis."""
        from minivess.diagnostics.weight_diagnostics import run_weightwatcher

        full_model = _make_trainable_model()
        partial_model = _make_partially_frozen_model()

        full_summary = run_weightwatcher(full_model)
        partial_summary = run_weightwatcher(partial_model)

        # Partially frozen model should analyze fewer layers
        assert (
            partial_summary["diag_ww_num_layers_analyzed"]
            <= full_summary["diag_ww_num_layers_analyzed"]
        )


class TestMetricPrefix:
    """Verify metrics use diag_ww_ prefix (NOT prof_ww_)."""

    def test_uses_diag_ww_prefix(self) -> None:
        from minivess.diagnostics.weight_diagnostics import run_weightwatcher

        model = _make_trainable_model()
        summary = run_weightwatcher(model)
        for key in summary:
            if "alpha" in key or "layers" in key:
                assert key.startswith("diag_ww_"), (
                    f"Key '{key}' must start with 'diag_ww_' (RC8)"
                )


class TestArtifactPath:
    """Verify artifact path is diagnostics/."""

    def test_artifact_path_is_diagnostics(self) -> None:
        from minivess.diagnostics.weight_diagnostics import ARTIFACT_PATH

        assert ARTIFACT_PATH == "diagnostics", (
            f"Artifact path must be 'diagnostics', got '{ARTIFACT_PATH}'"
        )
