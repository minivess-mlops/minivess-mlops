"""Tests for quasi-E2E runner infrastructure (Phase 4, #336).

Verifies that the quasi-E2E runner correctly generates parametrized
test IDs, validates model instantiation, and loss construction.
"""

from __future__ import annotations

import pytest
from torch import nn

from minivess.adapters.model_builder import _sam3_package_available
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

_SAM3_FAMILIES = frozenset({"sam3_vanilla", "sam3_topolora", "sam3_hybrid"})
# SAM3 families that require ≥16 GB VRAM for training (VRAM check raises at build time)
_SAM3_HIGH_VRAM = frozenset({"sam3_topolora"})


def _gpu_vram_gb() -> float:
    """Return detected GPU VRAM in GB, or 0.0 if no GPU available."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        pass
    return 0.0


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
        """Every discovered model can be instantiated (SAM3 skipped when not installed)."""
        vram_gb = _gpu_vram_gb()
        models = discover_implemented_models()
        for model_name in models:
            if model_name in _SAM3_FAMILIES and not _sam3_package_available():
                continue  # SAM3 requires real pretrained weights — skip when not installed
            if model_name in _SAM3_HIGH_VRAM and vram_gb < 16.0:
                continue  # SAM3 LoRA requires ≥16 GB VRAM — skip on insufficient hardware
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
