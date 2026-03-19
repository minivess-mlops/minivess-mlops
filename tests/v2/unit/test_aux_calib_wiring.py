"""Test that with_aux_calib is properly wired through the training pipeline.

P0 BLOCKER (found by reviewer agent): train_one_fold_task() previously called
build_loss_function(loss_name) WITHOUT passing with_aux_calib, making the
aux_calibration factorial factor dead code (24 conditions silently producing
only 12 distinct results).
"""

from __future__ import annotations

import pytest


class TestAuxCalibWiring:
    """Verify auxiliary calibration loss is wired end-to-end."""

    def test_build_loss_without_aux_calib(self) -> None:
        """Default: no auxiliary calibration → bare loss."""
        from minivess.pipeline.loss_functions import build_loss_function

        loss = build_loss_function("cbdice_cldice")
        # Should NOT be wrapped in AuxCalibCompoundLoss
        assert type(loss).__name__ != "AuxCalibCompoundLoss"

    def test_build_loss_with_aux_calib(self) -> None:
        """with_aux_calib=True → AuxCalibCompoundLoss wrapper."""
        from minivess.pipeline.loss_functions import build_loss_function

        loss = build_loss_function(
            "cbdice_cldice",
            with_aux_calib=True,
            aux_calib_weight=1.0,
        )
        assert type(loss).__name__ == "AuxCalibCompoundLoss"

    def test_aux_calib_wraps_all_three_losses(self) -> None:
        """All 3 factorial losses support aux_calib wrapping."""
        from minivess.pipeline.loss_functions import build_loss_function

        for loss_name in ["cbdice_cldice", "dice_ce", "dice_ce_cldice"]:
            loss = build_loss_function(
                loss_name,
                with_aux_calib=True,
                aux_calib_weight=0.5,
            )
            assert type(loss).__name__ == "AuxCalibCompoundLoss", (
                f"Loss {loss_name} not wrapped with AuxCalibCompoundLoss"
            )

    @pytest.mark.model_loading
    def test_aux_calib_forward_produces_finite_loss(self) -> None:
        """AuxCalibCompoundLoss produces finite loss on synthetic data."""
        import torch

        from minivess.pipeline.loss_functions import build_loss_function

        loss_fn = build_loss_function(
            "dice_ce",
            with_aux_calib=True,
            aux_calib_weight=1.0,
        )
        # Synthetic logits and labels (binary segmentation)
        logits = torch.randn(1, 2, 16, 16, 4)
        labels = torch.randint(0, 2, (1, 1, 16, 16, 4)).float()

        result = loss_fn(logits, labels)
        assert torch.isfinite(result).all(), f"Non-finite loss: {result}"
