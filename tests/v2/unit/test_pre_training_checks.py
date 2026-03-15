"""Tests for pre-training sanity checks module (T2.1, #648).

Six pre-training checks that catch broken models before wasting GPU hours.
Diagnostics always run regardless of ProfilingConfig.enabled (RC17).
Artifact path: diagnostics/ (NOT profiling/ — RC12).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn


def _make_simple_model(out_channels: int = 2) -> nn.Module:
    """Create a minimal model for testing: conv3d → output."""

    class _SimpleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv3d(1, out_channels, kernel_size=3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    return _SimpleModel()


def _make_frozen_model() -> nn.Module:
    """Create a model where all params are frozen (no gradient flow)."""
    model = _make_simple_model()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _make_nan_model() -> nn.Module:
    """Create a model that outputs NaN."""

    class _NaNModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv3d(1, 2, kernel_size=3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.full_like(self.conv(x), float("nan"))

    return _NaNModel()


@pytest.fixture()
def sample_batch() -> dict[str, torch.Tensor]:
    """Create a small batch for testing."""
    return {
        "image": torch.rand(1, 1, 8, 8, 4),
        "label": torch.zeros(1, 1, 8, 8, 4, dtype=torch.long),
    }


class TestOutputShapeCheck:
    """Test output shape mismatch detection."""

    def test_passes_correct_shape(self, sample_batch: dict[str, torch.Tensor]) -> None:
        from minivess.diagnostics.pre_training_checks import check_output_shape

        model = _make_simple_model(out_channels=2)
        result = check_output_shape(model, sample_batch, expected_channels=2)
        assert result.passed is True

    def test_fails_wrong_channels(self, sample_batch: dict[str, torch.Tensor]) -> None:
        from minivess.diagnostics.pre_training_checks import check_output_shape

        model = _make_simple_model(out_channels=2)
        result = check_output_shape(model, sample_batch, expected_channels=5)
        assert result.passed is False


class TestGradientFlowCheck:
    """Test gradient flow detection."""

    def test_passes_unfrozen(self, sample_batch: dict[str, torch.Tensor]) -> None:
        from minivess.diagnostics.pre_training_checks import check_gradient_flow

        model = _make_simple_model()
        result = check_gradient_flow(model, sample_batch)
        assert result.passed is True

    def test_fails_frozen(self, sample_batch: dict[str, torch.Tensor]) -> None:
        from minivess.diagnostics.pre_training_checks import check_gradient_flow

        model = _make_frozen_model()
        result = check_gradient_flow(model, sample_batch)
        assert result.passed is False


class TestLossSanityCheck:
    """Test loss sanity check."""

    def test_passes_reasonable_loss(
        self, sample_batch: dict[str, torch.Tensor]
    ) -> None:
        from minivess.diagnostics.pre_training_checks import check_loss_sanity

        model = _make_simple_model()
        criterion = nn.CrossEntropyLoss()
        result = check_loss_sanity(model, sample_batch, criterion)
        assert result.passed is True


class TestNaNCheck:
    """Test NaN/Inf detection."""

    def test_fails_on_nan_output(self, sample_batch: dict[str, torch.Tensor]) -> None:
        from minivess.diagnostics.pre_training_checks import check_nan_inf

        model = _make_nan_model()
        result = check_nan_inf(model, sample_batch)
        assert result.passed is False


class TestRunAllChecks:
    """Test orchestrator function."""

    def test_returns_list_of_results(
        self, sample_batch: dict[str, torch.Tensor]
    ) -> None:
        from minivess.diagnostics.pre_training_checks import (
            CheckResult,
            run_pre_training_checks,
        )

        model = _make_simple_model()
        criterion = nn.CrossEntropyLoss()
        results = run_pre_training_checks(
            model=model,
            sample_batch=sample_batch,
            criterion=criterion,
            expected_channels=2,
        )
        assert isinstance(results, list)
        assert all(isinstance(r, CheckResult) for r in results)
        assert len(results) >= 4  # At least the 4 core checks

    def test_artifact_path_is_diagnostics(self) -> None:
        """Verify the module uses diagnostics/ artifact path (RC12)."""
        from minivess.diagnostics.pre_training_checks import ARTIFACT_PATH

        assert ARTIFACT_PATH == "diagnostics", (
            f"Artifact path must be 'diagnostics', got '{ARTIFACT_PATH}'"
        )
