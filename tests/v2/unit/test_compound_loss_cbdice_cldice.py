from __future__ import annotations

import pytest
import torch

from minivess.pipeline.loss_functions import build_loss_function


@pytest.fixture()
def random_logits_labels():
    """Create random 3D logits and labels for loss testing."""
    torch.manual_seed(42)
    logits = torch.randn(1, 2, 8, 16, 16, requires_grad=True)
    labels = torch.randint(0, 2, (1, 1, 8, 16, 16)).float()
    return logits, labels


class TestCbDiceClDiceLoss:
    """Tests for CbDiceClDiceLoss compound loss."""

    def test_forward_returns_scalar(self, random_logits_labels):
        """Output must be a scalar tensor."""
        logits, labels = random_logits_labels
        loss_fn = build_loss_function("cbdice_cldice")
        loss = loss_fn(logits, labels)
        assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"

    def test_gradient_flow(self, random_logits_labels):
        """Gradients must flow back through both sub-losses."""
        logits, labels = random_logits_labels
        loss_fn = build_loss_function("cbdice_cldice")
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None, "No gradient on logits"
        assert not torch.all(logits.grad == 0), "Gradient is all zeros"

    def test_no_nan(self, random_logits_labels):
        """Loss must not produce NaN on random input."""
        logits, labels = random_logits_labels
        loss_fn = build_loss_function("cbdice_cldice")
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss), "Loss is NaN"

    def test_loss_is_positive(self, random_logits_labels):
        """Loss must be non-negative."""
        logits, labels = random_logits_labels
        loss_fn = build_loss_function("cbdice_cldice")
        loss = loss_fn(logits, labels)
        assert loss.item() >= 0.0, f"Loss is negative: {loss.item()}"

    def test_custom_weights(self, random_logits_labels):
        """Custom lambda weights should change the loss value."""
        from minivess.pipeline.loss_functions import CbDiceClDiceLoss

        logits, labels = random_logits_labels
        loss_equal = CbDiceClDiceLoss(lambda_cbdice=0.5, lambda_cldice=0.5)(
            logits, labels
        )
        loss_skewed = CbDiceClDiceLoss(lambda_cbdice=0.9, lambda_cldice=0.1)(
            logits, labels
        )
        # Different weights should produce different loss values
        assert not torch.isclose(loss_equal, loss_skewed, atol=1e-6), (
            "Different weights produced same loss"
        )

    def test_default_weights_are_half(self):
        """Default lambda weights should be 0.5 each."""
        from minivess.pipeline.loss_functions import CbDiceClDiceLoss

        loss_fn = CbDiceClDiceLoss()
        assert loss_fn.lambda_cbdice == 0.5
        assert loss_fn.lambda_cldice == 0.5

    def test_3d_input_shape(self):
        """Must work with standard 3D segmentation shapes."""
        from minivess.pipeline.loss_functions import CbDiceClDiceLoss

        torch.manual_seed(42)
        logits = torch.randn(2, 2, 16, 32, 32)  # Batch of 2
        labels = torch.randint(0, 2, (2, 1, 16, 32, 32)).float()
        loss_fn = CbDiceClDiceLoss()
        loss = loss_fn(logits, labels)
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_factory_returns_correct_class(self):
        """build_loss_function('cbdice_cldice') must return CbDiceClDiceLoss."""
        from minivess.pipeline.loss_functions import CbDiceClDiceLoss

        loss_fn = build_loss_function("cbdice_cldice")
        assert isinstance(loss_fn, CbDiceClDiceLoss)
