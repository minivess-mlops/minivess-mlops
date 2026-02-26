from __future__ import annotations

import pytest
import torch

from minivess.pipeline.loss_functions import build_loss_function


@pytest.fixture()
def synthetic_3d_data():
    """Synthetic 3D segmentation data: logits + labels."""
    torch.manual_seed(42)
    logits = torch.randn(1, 2, 8, 16, 16, requires_grad=True)
    # Create realistic labels (not all zeros, not all ones)
    labels = torch.zeros(1, 1, 8, 16, 16)
    # Put some foreground in the center
    labels[:, :, 2:6, 4:12, 4:12] = 1.0
    return logits, labels


@pytest.fixture()
def synthetic_3d_data_empty_labels():
    """Synthetic 3D data with empty (all-background) labels."""
    torch.manual_seed(42)
    logits = torch.randn(1, 2, 8, 16, 16, requires_grad=True)
    labels = torch.zeros(1, 1, 8, 16, 16)
    return logits, labels


EXPERIMENT_LOSSES = ["dice_ce", "cbdice", "dice_ce_cldice", "cbdice_cldice"]


class TestAllLossesNoNaN:
    """Verify ALL experiment losses produce valid gradients — no NaN.

    This is a pre-training gate: if any loss produces NaN on synthetic data,
    training will diverge immediately and waste GPU time.
    """

    @pytest.mark.parametrize("loss_name", EXPERIMENT_LOSSES)
    def test_loss_not_nan(self, loss_name, synthetic_3d_data):
        """Loss must not be NaN on synthetic data with foreground."""
        logits, labels = synthetic_3d_data
        loss_fn = build_loss_function(loss_name)
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss), f"{loss_name} produced NaN loss"
        assert not torch.isinf(loss), f"{loss_name} produced Inf loss"

    @pytest.mark.parametrize("loss_name", EXPERIMENT_LOSSES)
    def test_gradient_not_nan(self, loss_name, synthetic_3d_data):
        """Gradients must not be NaN."""
        logits, labels = synthetic_3d_data
        loss_fn = build_loss_function(loss_name)
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None, f"{loss_name}: no gradient"
        assert not torch.any(torch.isnan(logits.grad)), f"{loss_name}: NaN in gradient"

    @pytest.mark.parametrize("loss_name", EXPERIMENT_LOSSES)
    def test_loss_not_nan_empty_labels(self, loss_name, synthetic_3d_data_empty_labels):
        """Loss must handle empty labels (all background) without NaN."""
        logits, labels = synthetic_3d_data_empty_labels
        loss_fn = build_loss_function(loss_name)
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss), f"{loss_name} produced NaN on empty labels"

    @pytest.mark.parametrize("loss_name", EXPERIMENT_LOSSES)
    def test_multi_step_no_nan(self, loss_name):
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
