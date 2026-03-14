"""VesselFM + diagnostics compatibility tests (T2.1).

Verifies that pre-training checks and WeightWatcher work with VesselFM
adapter BEFORE spending GPU credits on RunPod.

All tests use pretrained=False (random weights) to avoid HF download.
Marked with @model_loading — excluded from staging tier.
"""

from __future__ import annotations

import pytest
import torch

from minivess.adapters.vesselfm import VesselFMAdapter
from minivess.config.models import ModelConfig, ModelFamily
from minivess.diagnostics.pre_training_checks import (
    check_gradient_flow,
    check_loss_sanity,
    check_nan_inf,
    check_output_shape,
    run_pre_training_checks,
)


def _make_vesselfm_model() -> VesselFMAdapter:
    """Create VesselFM adapter with random weights (no HF download)."""
    config = ModelConfig(
        family=ModelFamily.VESSEL_FM,
        name="vesselfm-diag-test",
        in_channels=1,
        out_channels=2,
    )
    return VesselFMAdapter(config, pretrained=False)


def _make_sample_batch() -> dict[str, torch.Tensor]:
    """Create a minimal sample batch for diagnostics.

    VesselFM DynUNet has 6 levels with stride-2 in ALL 3 dimensions.
    All spatial dims must be divisible by 2^5=32. Use (1,1,64,64,64).
    """
    return {
        "image": torch.randn(1, 1, 64, 64, 64),
        "label": torch.randint(0, 2, (1, 1, 64, 64, 64)).float(),
    }


@pytest.mark.model_loading
class TestVesselFMPreTrainingChecks:
    """Verify pre-training checks work with VesselFM's binary→2-class output."""

    def test_output_shape_correct(self) -> None:
        """VesselFM produces 2-channel output (binary→cat[-logits, logits])."""
        model = _make_vesselfm_model()
        batch = _make_sample_batch()
        result = check_output_shape(model, batch, expected_channels=2)
        assert result.passed, f"Output shape check failed: {result.message}"

    def test_gradient_flow_works(self) -> None:
        """At least one VesselFM parameter receives gradients."""
        model = _make_vesselfm_model()
        batch = _make_sample_batch()
        result = check_gradient_flow(model, batch)
        assert result.passed, f"Gradient flow failed: {result.message}"

    def test_nan_inf_absent(self) -> None:
        """VesselFM output has no NaN or Inf with random weights."""
        model = _make_vesselfm_model()
        batch = _make_sample_batch()
        result = check_nan_inf(model, batch)
        assert result.passed, f"NaN/Inf check failed: {result.message}"

    def test_loss_sanity_with_dice_ce(self) -> None:
        """DiceCELoss at random init produces a finite loss value."""
        from monai.losses import DiceCELoss

        model = _make_vesselfm_model()
        batch = _make_sample_batch()
        criterion = DiceCELoss(to_onehot_y=True, softmax=True)
        result = check_loss_sanity(model, batch, criterion)
        assert result.passed, f"Loss sanity failed: {result.message}"

    def test_run_all_checks_pass(self) -> None:
        """run_pre_training_checks() returns all-pass for VesselFM."""
        from monai.losses import DiceCELoss

        model = _make_vesselfm_model()
        batch = _make_sample_batch()
        criterion = DiceCELoss(to_onehot_y=True, softmax=True)
        results = run_pre_training_checks(
            model=model,
            sample_batch=batch,
            criterion=criterion,
            expected_channels=2,
        )
        failures = [r for r in results if not r.passed]
        assert not failures, f"Checks failed: {[f.message for f in failures]}"


@pytest.mark.model_loading
class TestVesselFMWeightWatcher:
    """Verify WeightWatcher runs without crash on VesselFM DynUNet."""

    def test_weightwatcher_runs_without_crash(self) -> None:
        """WeightWatcher analysis completes (may return 0 layers for 3D conv)."""
        from minivess.diagnostics.weight_diagnostics import run_weightwatcher

        model = _make_vesselfm_model()
        result = run_weightwatcher(model)
        assert isinstance(result, dict)
        assert "diag_ww_num_layers_analyzed" in result
        assert "diag_ww_alpha_mean" in result

    def test_weightwatcher_returns_numeric_metrics(self) -> None:
        """WeightWatcher metrics are numeric (float or NaN, not None)."""
        from minivess.diagnostics.weight_diagnostics import run_weightwatcher

        model = _make_vesselfm_model()
        result = run_weightwatcher(model)
        # Alpha mean may be NaN if no layers analyzed (3D conv only)
        assert isinstance(result["diag_ww_alpha_mean"], float)
        assert isinstance(result["diag_ww_num_layers_analyzed"], int)
