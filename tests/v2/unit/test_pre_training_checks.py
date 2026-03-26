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


class TestAutocastInPreTrainingChecks:
    """Pre-training checks must use autocast for SAM3 VRAM safety.

    Without autocast, check_gradient_flow() runs a full FP32 forward+backward
    through SAM3's 648M-param encoder → OOM on L4 (24 GB). This was the root
    cause of zero SAM3 training completions across 9 passes.

    See: .claude/metalearning/2026-03-25-overconfident-oom-fixed-claim.md
    """

    def test_gradient_flow_accepts_mixed_precision(
        self, sample_batch: dict[str, torch.Tensor]
    ) -> None:
        """check_gradient_flow must accept mixed_precision kwarg."""
        from minivess.diagnostics.pre_training_checks import check_gradient_flow

        model = _make_simple_model()
        result = check_gradient_flow(model, sample_batch, mixed_precision=True)
        assert result.passed is True

    def test_gradient_flow_works_with_amp_disabled(
        self, sample_batch: dict[str, torch.Tensor]
    ) -> None:
        """FP32 mode still works for small models (backward compat)."""
        from minivess.diagnostics.pre_training_checks import check_gradient_flow

        model = _make_simple_model()
        result = check_gradient_flow(model, sample_batch, mixed_precision=False)
        assert result.passed is True

    def test_output_shape_accepts_mixed_precision(
        self, sample_batch: dict[str, torch.Tensor]
    ) -> None:
        from minivess.diagnostics.pre_training_checks import check_output_shape

        model = _make_simple_model(out_channels=2)
        result = check_output_shape(
            model, sample_batch, expected_channels=2, mixed_precision=True
        )
        assert result.passed is True

    def test_loss_sanity_accepts_mixed_precision(
        self, sample_batch: dict[str, torch.Tensor]
    ) -> None:
        from minivess.diagnostics.pre_training_checks import check_loss_sanity

        model = _make_simple_model()
        criterion = nn.CrossEntropyLoss()
        result = check_loss_sanity(
            model, sample_batch, criterion, mixed_precision=True
        )
        assert result.passed is True

    def test_nan_inf_accepts_mixed_precision(
        self, sample_batch: dict[str, torch.Tensor]
    ) -> None:
        from minivess.diagnostics.pre_training_checks import check_nan_inf

        model = _make_simple_model()
        result = check_nan_inf(model, sample_batch, mixed_precision=True)
        assert result.passed is True

    def test_run_all_checks_accepts_mixed_precision(
        self, sample_batch: dict[str, torch.Tensor]
    ) -> None:
        from minivess.diagnostics.pre_training_checks import run_pre_training_checks

        model = _make_simple_model()
        criterion = nn.CrossEntropyLoss()
        results = run_pre_training_checks(
            model=model,
            sample_batch=sample_batch,
            criterion=criterion,
            expected_channels=2,
            mixed_precision=True,
        )
        assert all(r.passed for r in results)


class TestSkipGradientFlow:
    """Tests for skip_gradient_flow parameter in run_pre_training_checks.

    SAM3 gradient checkpointing causes OOM in the diagnostic forward+backward
    (which does NOT use gradient checkpointing). When gradient checkpointing
    is enabled, we skip check_gradient_flow() and report as passed with skip message.

    Issue: #966, Plan: sam3-gradient-checkpointing-plan.xml Task 2
    """

    def test_skip_gradient_flow_returns_passed(
        self, sample_batch: dict[str, torch.Tensor]
    ) -> None:
        """When skip_gradient_flow=True, gradient_flow check must pass with skip message."""
        from minivess.diagnostics.pre_training_checks import run_pre_training_checks

        model = _make_simple_model()
        criterion = nn.CrossEntropyLoss()
        results = run_pre_training_checks(
            model=model,
            sample_batch=sample_batch,
            criterion=criterion,
            expected_channels=2,
            skip_gradient_flow=True,
        )
        grad_results = [r for r in results if r.name == "gradient_flow"]
        assert len(grad_results) == 1
        assert grad_results[0].passed is True
        assert "skip" in grad_results[0].message.lower()

    def test_skip_still_runs_other_checks(
        self, sample_batch: dict[str, torch.Tensor]
    ) -> None:
        """All other pre-training checks must still run when gradient_flow is skipped."""
        from minivess.diagnostics.pre_training_checks import run_pre_training_checks

        model = _make_simple_model()
        criterion = nn.CrossEntropyLoss()
        results = run_pre_training_checks(
            model=model,
            sample_batch=sample_batch,
            criterion=criterion,
            expected_channels=2,
            skip_gradient_flow=True,
        )
        names = [r.name for r in results]
        assert "output_shape" in names
        assert "loss_sanity" in names
        assert "nan_inf" in names

    def test_default_does_not_skip(
        self, sample_batch: dict[str, torch.Tensor]
    ) -> None:
        """By default, gradient_flow check should NOT be skipped."""
        from minivess.diagnostics.pre_training_checks import run_pre_training_checks

        model = _make_simple_model()
        criterion = nn.CrossEntropyLoss()
        results = run_pre_training_checks(
            model=model,
            sample_batch=sample_batch,
            criterion=criterion,
            expected_channels=2,
        )
        grad_results = [r for r in results if r.name == "gradient_flow"]
        assert len(grad_results) == 1
        assert "skip" not in grad_results[0].message.lower()


class TestGradientFlowCleanup:
    """T6: check_gradient_flow must clean up gradients after check.

    Bug: After forward+backward, check_gradient_flow() never calls
    model.zero_grad(set_to_none=True). Residual .grad tensors on all
    parameters consume VRAM that is never freed.
    """

    def test_check_gradient_flow_clears_gradients_after(
        self, sample_batch: dict[str, torch.Tensor]
    ) -> None:
        """After check_gradient_flow, model params should have no residual gradients."""
        from minivess.diagnostics.pre_training_checks import check_gradient_flow

        model = _make_simple_model()
        result = check_gradient_flow(model, sample_batch)
        assert result.passed is True

        # After the check, gradients should be cleared
        for name, param in model.named_parameters():
            assert param.grad is None, (
                f"Param '{name}' still has .grad after check_gradient_flow — "
                f"leaked ~{param.grad.numel() * 4 / 1e6:.1f} MB"
            )

    def test_check_gradient_flow_logs_runtime_error(
        self, sample_batch: dict[str, torch.Tensor]
    ) -> None:
        """RuntimeError during backward should be caught and reported."""
        from minivess.diagnostics.pre_training_checks import check_gradient_flow

        class _FailBackwardModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv3d(1, 2, 3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.conv(x)
                # Register a hook that raises during backward
                out.register_hook(lambda g: (_ for _ in ()).throw(RuntimeError("OOM")))
                return out

        model = _FailBackwardModel()
        result = check_gradient_flow(model, sample_batch)
        assert result.passed is False
        assert "backward" in result.message.lower() or "gradient" in result.message.lower()


class TestChecksToDictSerialization:
    """T14: checks_to_dict() serialization coverage."""

    def test_checks_to_dict_all_passed(self) -> None:
        """Serialize results where all checks passed."""
        from minivess.diagnostics.pre_training_checks import CheckResult, checks_to_dict

        results = [
            CheckResult(name="output_shape", passed=True, message="OK"),
            CheckResult(name="gradient_flow", passed=True, message="OK"),
            CheckResult(name="loss_sanity", passed=True, message="Loss: 0.69"),
            CheckResult(name="nan_inf", passed=True, message="No NaN"),
        ]
        d = checks_to_dict(results)
        assert d["summary"]["total"] == 4
        assert d["summary"]["passed"] == 4
        assert d["summary"]["failed"] == 0
        assert len(d["checks"]) == 4

    def test_checks_to_dict_with_failure(self) -> None:
        """Serialize results with some failures."""
        from minivess.diagnostics.pre_training_checks import CheckResult, checks_to_dict

        results = [
            CheckResult(name="output_shape", passed=True, message="OK"),
            CheckResult(name="gradient_flow", passed=False, message="No grad", severity="error"),
        ]
        d = checks_to_dict(results)
        assert d["summary"]["total"] == 2
        assert d["summary"]["passed"] == 1
        assert d["summary"]["failed"] == 1

    def test_checks_to_dict_with_skipped_gradient_flow(self) -> None:
        """Serialize results with skipped gradient_flow (skip_gradient_flow=True path)."""
        from minivess.diagnostics.pre_training_checks import CheckResult, checks_to_dict

        results = [
            CheckResult(name="output_shape", passed=True, message="OK"),
            CheckResult(
                name="gradient_flow",
                passed=True,
                message="Skipped — model uses gradient checkpointing",
            ),
            CheckResult(name="loss_sanity", passed=True, message="Loss: 0.69"),
            CheckResult(name="nan_inf", passed=True, message="No NaN"),
        ]
        d = checks_to_dict(results)
        assert d["summary"]["passed"] == 4
        # The skipped check should still appear in the serialized list
        grad_check = [c for c in d["checks"] if c["name"] == "gradient_flow"]
        assert len(grad_check) == 1
        assert "skipped" in grad_check[0]["message"].lower()

    def test_checks_to_dict_summary_counts(self) -> None:
        """Summary counts must be accurate."""
        from minivess.diagnostics.pre_training_checks import CheckResult, checks_to_dict

        results = [
            CheckResult(name="a", passed=True, message="OK"),
            CheckResult(name="b", passed=False, message="Fail", severity="error"),
            CheckResult(name="c", passed=True, message="OK"),
            CheckResult(name="d", passed=False, message="Fail", severity="warning"),
        ]
        d = checks_to_dict(results)
        assert d["summary"]["total"] == 4
        assert d["summary"]["passed"] == 2
        assert d["summary"]["failed"] == 2
