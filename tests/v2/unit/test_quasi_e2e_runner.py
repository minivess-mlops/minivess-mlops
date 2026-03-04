"""Tests for quasi-E2E runner infrastructure (Phase 4, #336).

Verifies that the quasi-E2E runner correctly generates parametrized
test IDs, validates model instantiation, and loss construction.
"""

from __future__ import annotations

import pytest
from torch import nn

from minivess.testing.capability_discovery import (
    build_practical_combinations,
    discover_all_losses,
    discover_implemented_models,
)
from minivess.testing.quasi_e2e_runner import (
    build_loss_for_test,
    build_model_for_test,
    generate_test_ids,
    run_single_forward_backward,
)


class TestGenerateTestIds:
    """generate_test_ids produces deterministic, readable test IDs."""

    def test_returns_nonempty(self) -> None:
        combos = build_practical_combinations()
        ids = generate_test_ids(combos)
        assert len(ids) == len(combos)

    def test_ids_are_unique(self) -> None:
        combos = build_practical_combinations()
        ids = generate_test_ids(combos)
        assert len(set(ids)) == len(ids)

    def test_id_format(self) -> None:
        combos = build_practical_combinations()
        ids = generate_test_ids(combos)
        for test_id in ids:
            # Format: "model__loss"
            assert "__" in test_id
            parts = test_id.split("__")
            assert len(parts) == 2


class TestBuildModelForTest:
    """build_model_for_test creates a model adapter for any implemented model."""

    def test_builds_dynunet(self) -> None:
        model = build_model_for_test("dynunet")
        assert isinstance(model, nn.Module)

    def test_all_implemented_models_build(self) -> None:
        """Every discovered model can be instantiated."""
        models = discover_implemented_models()
        for model_name in models:
            model = build_model_for_test(model_name)
            assert isinstance(model, nn.Module), f"Failed to build {model_name}"

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError):
            build_model_for_test("nonexistent_model")


class TestBuildLossForTest:
    """build_loss_for_test creates a loss module for any discovered loss."""

    def test_builds_cbdice_cldice(self) -> None:
        loss = build_loss_for_test("cbdice_cldice")
        assert isinstance(loss, nn.Module)

    def test_builds_dice_ce(self) -> None:
        loss = build_loss_for_test("dice_ce")
        assert isinstance(loss, nn.Module)

    def test_all_discovered_losses_build(self) -> None:
        """Every discovered loss can be instantiated."""
        losses = discover_all_losses()
        for loss_name in losses:
            loss = build_loss_for_test(loss_name)
            assert isinstance(loss, nn.Module), f"Failed to build {loss_name}"


class TestRunSingleForwardBackward:
    """run_single_forward_backward does one fwd+bwd step on random data."""

    def test_dynunet_cbdice_cldice(self) -> None:
        model = build_model_for_test("dynunet")
        loss_fn = build_loss_for_test("cbdice_cldice")
        result = run_single_forward_backward(
            model=model,
            loss_fn=loss_fn,
            patch_size=(32, 32, 8),
            batch_size=1,
            in_channels=1,
            num_classes=2,
        )
        assert result["loss_value"] is not None
        assert isinstance(result["loss_value"], float)
        assert result["output_shape"] is not None

    def test_result_has_required_keys(self) -> None:
        model = build_model_for_test("dynunet")
        loss_fn = build_loss_for_test("dice_ce")
        result = run_single_forward_backward(
            model=model,
            loss_fn=loss_fn,
            patch_size=(32, 32, 8),
            batch_size=1,
            in_channels=1,
            num_classes=2,
        )
        assert "loss_value" in result
        assert "output_shape" in result
        assert "grad_norm" in result
