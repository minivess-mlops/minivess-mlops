"""Quasi-E2E: test all model×loss combinations (Phase 6, #338).

Each dynamically parametrized test verifies that:
1. The model instantiates without error
2. The loss function instantiates without error
3. A single forward+backward pass completes
4. Loss value is finite (not NaN/Inf)
5. Gradients flow (grad_norm > 0)

This is NOT a performance test — it verifies training mechanics only.
Run with: pytest tests/v2/quasi_e2e/ -x -v
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from minivess.testing.quasi_e2e_runner import (
    _get_patch_size_for_model,
    build_loss_for_test,
    build_model_for_test,
    run_single_forward_backward,
)

if TYPE_CHECKING:
    from minivess.testing.capability_discovery import TestCombination

# These tests instantiate PyTorch models — exclude from staging tier
pytestmark = pytest.mark.model_loading


class TestModelLossForwardBackward:
    """Each model×loss combination completes a forward+backward pass."""

    def test_model_instantiates(self, model_loss_combo: TestCombination) -> None:
        """Model can be instantiated from config."""
        patch = _get_patch_size_for_model(model_loss_combo.model)
        model = build_model_for_test(model_loss_combo.model, patch_size=patch)
        assert model is not None

    def test_loss_instantiates(self, model_loss_combo: TestCombination) -> None:
        """Loss function can be instantiated."""
        loss_fn = build_loss_for_test(model_loss_combo.loss)
        assert loss_fn is not None

    def test_forward_backward_completes(
        self, model_loss_combo: TestCombination
    ) -> None:
        """Single forward+backward pass completes without error."""
        patch = _get_patch_size_for_model(model_loss_combo.model)
        model = build_model_for_test(model_loss_combo.model, patch_size=patch)
        loss_fn = build_loss_for_test(model_loss_combo.loss)
        result = run_single_forward_backward(
            model=model,
            loss_fn=loss_fn,
            patch_size=patch,
            batch_size=1,
            in_channels=1,
            num_classes=2,
        )
        assert result["loss_value"] is not None

    def test_loss_is_finite(self, model_loss_combo: TestCombination) -> None:
        """Loss value is finite (not NaN or Inf)."""
        patch = _get_patch_size_for_model(model_loss_combo.model)
        model = build_model_for_test(model_loss_combo.model, patch_size=patch)
        loss_fn = build_loss_for_test(model_loss_combo.loss)
        result = run_single_forward_backward(
            model=model,
            loss_fn=loss_fn,
            patch_size=patch,
            batch_size=1,
            in_channels=1,
            num_classes=2,
        )
        loss_val = result["loss_value"]
        assert not torch.isnan(torch.tensor(loss_val)), (
            f"NaN loss for {model_loss_combo.model}+{model_loss_combo.loss}"
        )
        assert not torch.isinf(torch.tensor(loss_val)), (
            f"Inf loss for {model_loss_combo.model}+{model_loss_combo.loss}"
        )

    def test_gradients_flow(self, model_loss_combo: TestCombination) -> None:
        """Gradients are non-zero after backward pass."""
        patch = _get_patch_size_for_model(model_loss_combo.model)
        model = build_model_for_test(model_loss_combo.model, patch_size=patch)
        loss_fn = build_loss_for_test(model_loss_combo.loss)
        result = run_single_forward_backward(
            model=model,
            loss_fn=loss_fn,
            patch_size=patch,
            batch_size=1,
            in_channels=1,
            num_classes=2,
        )
        assert result["grad_norm"] > 0, (
            f"Zero gradients for {model_loss_combo.model}+{model_loss_combo.loss}"
        )
