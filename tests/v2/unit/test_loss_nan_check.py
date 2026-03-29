from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
import torch

from minivess.pipeline.loss_functions import VesselCompoundLoss, build_loss_function


@pytest.fixture()
def synthetic_3d_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic 3D segmentation data: logits + labels."""
    torch.manual_seed(42)
    logits = torch.randn(1, 2, 8, 16, 16, requires_grad=True)
    # Create realistic labels (not all zeros, not all ones)
    labels = torch.zeros(1, 1, 8, 16, 16)
    # Put some foreground in the center
    labels[:, :, 2:6, 4:12, 4:12] = 1.0
    return logits, labels


@pytest.fixture()
def synthetic_3d_data_empty_labels() -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic 3D data with empty (all-background) labels."""
    torch.manual_seed(42)
    logits = torch.randn(1, 2, 8, 16, 16, requires_grad=True)
    labels = torch.zeros(1, 1, 8, 16, 16)
    return logits, labels


EXPERIMENT_LOSSES = [
    # LIBRARY (4)
    "dice_ce",
    "dice",
    "focal",
    "cldice",
    # LIBRARY-COMPOUND (2)
    "dice_ce_cldice",
    "cbdice_cldice",
    # HYBRID (3)
    "skeleton_recall",
    "cape",
    "betti_matching",
    # EXPERIMENTAL (9)
    "cb_dice",
    "cbdice",
    "centerline_ce",
    "warp",
    "topo",
    "betti",
    "full_topo",
    "graph_topology",
    "toposeg",
]


class TestAllLossesNoNaN:
    """Verify ALL experiment losses produce valid gradients — no NaN.

    This is a pre-training gate: if any loss produces NaN on synthetic data,
    training will diverge immediately and waste GPU time.
    """

    @pytest.mark.parametrize("loss_name", EXPERIMENT_LOSSES)
    def test_loss_not_nan(
        self, loss_name: str, synthetic_3d_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Loss must not be NaN on synthetic data with foreground."""
        logits, labels = synthetic_3d_data
        loss_fn = build_loss_function(loss_name)
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss), f"{loss_name} produced NaN loss"
        assert not torch.isinf(loss), f"{loss_name} produced Inf loss"

    @pytest.mark.parametrize("loss_name", EXPERIMENT_LOSSES)
    def test_gradient_not_nan(
        self, loss_name: str, synthetic_3d_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Gradients must not be NaN."""
        logits, labels = synthetic_3d_data
        loss_fn = build_loss_function(loss_name)
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None, f"{loss_name}: no gradient"
        assert not torch.any(torch.isnan(logits.grad)), f"{loss_name}: NaN in gradient"

    @pytest.mark.parametrize("loss_name", EXPERIMENT_LOSSES)
    def test_loss_not_nan_empty_labels(
        self,
        loss_name: str,
        synthetic_3d_data_empty_labels: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Loss must handle empty labels (all background) without NaN."""
        logits, labels = synthetic_3d_data_empty_labels
        loss_fn = build_loss_function(loss_name)
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss), f"{loss_name} produced NaN on empty labels"

    @pytest.mark.parametrize(
        "loss_name",
        ["dice_ce_cldice", "cbdice_cldice", "bce_dice_05cldice", "full_topo"],
    )
    def test_cldice_nan_guard_catches_degenerate(self, loss_name: str) -> None:
        """NaN guard must catch degenerate SoftclDiceLoss output and fall back to 0.0.

        Mocks SoftclDiceLoss.forward to return NaN (simulating soft skeleton
        degeneration). The VesselCompoundLoss NaN guard must:
        1. Replace NaN with 0.0 (clDice = 0 is semantically correct for no centerline)
        2. Log a warning (Rule 25: loud failures)
        3. Produce finite overall loss (DiceCE component provides gradient signal)
        """
        torch.manual_seed(42)
        logits = torch.randn(1, 2, 8, 16, 16, requires_grad=True)
        labels = torch.zeros(1, 1, 8, 16, 16)
        labels[:, :, 2:6, 4:12, 4:12] = 1.0

        loss_fn = build_loss_function(loss_name)

        # Mock SoftclDiceLoss.forward to return NaN
        nan_tensor = torch.tensor(float("nan"))
        with patch(
            "monai.losses.cldice.SoftclDiceLoss.forward", return_value=nan_tensor
        ):
            loss = loss_fn(logits, labels)
            assert torch.isfinite(loss), f"{loss_name}: NaN guard failed — loss is {loss}"

    @pytest.mark.parametrize(
        "loss_name",
        ["dice_ce_cldice", "cbdice_cldice", "bce_dice_05cldice", "full_topo"],
    )
    def test_cldice_nan_guard_logs_warning(
        self, loss_name: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """NaN guard must log a warning when clDice is NaN (Rule 25)."""
        torch.manual_seed(42)
        logits = torch.randn(1, 2, 8, 16, 16, requires_grad=True)
        labels = torch.zeros(1, 1, 8, 16, 16)

        loss_fn = build_loss_function(loss_name)

        nan_tensor = torch.tensor(float("nan"))
        with (
            patch(
                "monai.losses.cldice.SoftclDiceLoss.forward", return_value=nan_tensor
            ),
            caplog.at_level(logging.WARNING),
        ):
            loss_fn(logits, labels)
            assert any(
                "clDice" in r.message or "cldice" in r.message.lower()
                for r in caplog.records
            ), "NaN guard did not log a warning about degenerate clDice"

    def test_cldice_nan_guard_gradient_finite(self) -> None:
        """When NaN guard triggers, backward() must still produce finite gradients."""
        torch.manual_seed(42)
        logits = torch.randn(1, 2, 8, 16, 16, requires_grad=True)
        labels = torch.zeros(1, 1, 8, 16, 16)
        labels[:, :, 2:6, 4:12, 4:12] = 1.0

        loss_fn = VesselCompoundLoss()

        nan_tensor = torch.tensor(float("nan"))
        with patch(
            "monai.losses.cldice.SoftclDiceLoss.forward", return_value=nan_tensor
        ):
            loss = loss_fn(logits, labels)
            assert torch.isfinite(loss), f"Loss is not finite: {loss}"
            loss.backward()
            assert logits.grad is not None, "No gradient computed"
            assert not torch.any(
                torch.isnan(logits.grad)
            ), "NaN in gradients after guard"

    @pytest.mark.parametrize("loss_name", EXPERIMENT_LOSSES)
    def test_multi_step_no_nan(self, loss_name: str) -> None:
        """Simulate 5 training steps — loss must stay finite throughout."""
        torch.manual_seed(42)
        logits = torch.randn(1, 2, 8, 16, 16, requires_grad=True)
        labels = torch.zeros(1, 1, 8, 16, 16)
        labels[:, :, 2:6, 4:12, 4:12] = 1.0

        loss_fn = build_loss_function(loss_name)
        optimizer = torch.optim.Adam([logits], lr=1e-3)

        for step in range(5):
            optimizer.zero_grad()
            loss = loss_fn(logits, labels)
            assert not torch.isnan(loss), f"{loss_name}: NaN at step {step}"
            assert not torch.isinf(loss), f"{loss_name}: Inf at step {step}"
            loss.backward()
            optimizer.step()
