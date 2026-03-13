"""Tests for diagnostics wiring in train_flow.py (T2.3).

Validates:
- Pre-training checks called before trainer.fit()
- Pre-check severity=error aborts flow
- WeightWatcher called after trainer.fit()
- Diagnostics run even when profiling is disabled (RC17)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch


def _make_mock_config(*, profiling_enabled: bool = True) -> dict[str, Any]:
    """Minimal config dict for train_one_fold_task."""
    return {
        "loss_name": "dice_ce",
        "model_family": "dynunet",
        "debug": True,
        "max_epochs": 1,
        "num_folds": 1,
        "batch_size": 1,
        "experiment_name": "test_diagnostics",
        "tracking_uri": "mlruns",
        "profiling": {"enabled": profiling_enabled, "epochs": 1},
    }


def _make_mock_fold_split() -> dict[str, list[dict[str, str]]]:
    """Minimal fold split with fake volume paths."""
    return {
        "train": [{"image": "/fake/img.nii.gz", "label": "/fake/lbl.nii.gz"}],
        "val": [{"image": "/fake/img.nii.gz", "label": "/fake/lbl.nii.gz"}],
    }


def _make_mock_loader() -> list[dict[str, torch.Tensor]]:
    """Return an iterable that yields one sample batch."""
    return [
        {
            "image": torch.rand(1, 1, 8, 8, 4),
            "label": torch.zeros(1, 1, 8, 8, 4, dtype=torch.long),
        }
    ]


def _mock_tracker() -> MagicMock:
    """Create a mock ExperimentTracker with working start_run context."""
    tracker = MagicMock()
    tracker.start_run.return_value.__enter__ = MagicMock()
    tracker.start_run.return_value.__exit__ = MagicMock(return_value=False)
    return tracker


def _fit_result() -> dict[str, Any]:
    """Standard mock fit result."""
    return {
        "best_val_loss": 0.5,
        "final_epoch": 1,
        "history": {},
        "best_metrics": {},
        "mlflow_run_id": "fake",
    }


class TestPreChecksCalled:
    """Pre-training checks must be called before trainer.fit()."""

    @pytest.mark.model_loading
    def test_pre_checks_called_before_fit(self, tmp_path: Any) -> None:
        """run_pre_training_checks is invoked before trainer.fit()."""
        from minivess.orchestration.flows.train_flow import train_one_fold_task

        call_order: list[str] = []

        with (
            patch(
                "minivess.diagnostics.pre_training_checks.run_pre_training_checks",
            ) as mock_pre_checks,
            patch(
                "minivess.diagnostics.weight_diagnostics.run_weightwatcher",
                return_value={},
            ),
            patch(
                "minivess.pipeline.trainer.SegmentationTrainer",
            ) as mock_trainer_cls,
            patch(
                "minivess.adapters.model_builder.build_adapter",
                return_value=MagicMock(),
            ),
            patch(
                "minivess.pipeline.loss_functions.build_loss_function",
                return_value=MagicMock(),
            ),
            patch(
                "minivess.data.loader.build_train_loader",
                return_value=_make_mock_loader(),
            ),
            patch(
                "minivess.data.loader.build_val_loader",
                return_value=_make_mock_loader(),
            ),
            patch(
                "minivess.observability.tracking.ExperimentTracker",
                return_value=_mock_tracker(),
            ),
        ):

            def _pre_checks_side_effect(**kw: Any) -> list[Any]:
                call_order.append("pre_checks")
                return []

            mock_pre_checks.side_effect = _pre_checks_side_effect

            def _fit_side_effect(*a: Any, **kw: Any) -> dict[str, Any]:
                call_order.append("fit")
                return _fit_result()

            mock_trainer_cls.return_value.fit = MagicMock(side_effect=_fit_side_effect)

            train_one_fold_task.fn(
                fold_id=0,
                fold_split=_make_mock_fold_split(),
                config=_make_mock_config(),
                checkpoint_dir=tmp_path,
            )

        assert "pre_checks" in call_order, "run_pre_training_checks was not called"
        assert "fit" in call_order, "trainer.fit() was not called"
        pre_idx = call_order.index("pre_checks")
        fit_idx = call_order.index("fit")
        assert pre_idx < fit_idx, (
            f"pre_checks (idx={pre_idx}) must run before fit (idx={fit_idx})"
        )


class TestPreCheckErrorAborts:
    """Pre-check with severity=error should abort the training flow."""

    @pytest.mark.model_loading
    def test_pre_check_error_aborts_flow(self, tmp_path: Any) -> None:
        """A failed check with severity='error' raises RuntimeError."""
        from minivess.diagnostics.pre_training_checks import CheckResult
        from minivess.orchestration.flows.train_flow import train_one_fold_task

        failed_check = CheckResult(
            name="output_shape",
            passed=False,
            message="Expected 2 channels, got 5",
            severity="error",
        )

        with (
            patch(
                "minivess.diagnostics.pre_training_checks.run_pre_training_checks",
                return_value=[failed_check],
            ),
            patch(
                "minivess.adapters.model_builder.build_adapter",
                return_value=MagicMock(),
            ),
            patch(
                "minivess.pipeline.loss_functions.build_loss_function",
                return_value=MagicMock(),
            ),
            patch("minivess.pipeline.trainer.SegmentationTrainer"),
            patch(
                "minivess.data.loader.build_train_loader",
                return_value=_make_mock_loader(),
            ),
            patch(
                "minivess.data.loader.build_val_loader",
                return_value=_make_mock_loader(),
            ),
            patch(
                "minivess.observability.tracking.ExperimentTracker",
                return_value=_mock_tracker(),
            ),
            pytest.raises(RuntimeError, match="Pre-training check failed"),
        ):
            train_one_fold_task.fn(
                fold_id=0,
                fold_split=_make_mock_fold_split(),
                config=_make_mock_config(),
                checkpoint_dir=tmp_path,
            )


class TestWeightWatcherCalled:
    """WeightWatcher must be called after trainer.fit()."""

    @pytest.mark.model_loading
    def test_weightwatcher_called_after_fit(self, tmp_path: Any) -> None:
        """run_weightwatcher is invoked after trainer.fit()."""
        from minivess.orchestration.flows.train_flow import train_one_fold_task

        call_order: list[str] = []

        with (
            patch(
                "minivess.diagnostics.pre_training_checks.run_pre_training_checks",
                return_value=[],
            ),
            patch(
                "minivess.diagnostics.weight_diagnostics.run_weightwatcher",
            ) as mock_ww,
            patch(
                "minivess.pipeline.trainer.SegmentationTrainer",
            ) as mock_trainer_cls,
            patch(
                "minivess.adapters.model_builder.build_adapter",
                return_value=MagicMock(),
            ),
            patch(
                "minivess.pipeline.loss_functions.build_loss_function",
                return_value=MagicMock(),
            ),
            patch(
                "minivess.data.loader.build_train_loader",
                return_value=_make_mock_loader(),
            ),
            patch(
                "minivess.data.loader.build_val_loader",
                return_value=_make_mock_loader(),
            ),
            patch(
                "minivess.observability.tracking.ExperimentTracker",
                return_value=_mock_tracker(),
            ),
        ):

            def _fit_side_effect2(*a: Any, **kw: Any) -> dict[str, Any]:
                call_order.append("fit")
                return _fit_result()

            mock_trainer_cls.return_value.fit = MagicMock(side_effect=_fit_side_effect2)

            def _ww_side_effect(model: Any, **kw: Any) -> dict[str, Any]:
                call_order.append("weightwatcher")
                return {}

            mock_ww.side_effect = _ww_side_effect

            train_one_fold_task.fn(
                fold_id=0,
                fold_split=_make_mock_fold_split(),
                config=_make_mock_config(),
                checkpoint_dir=tmp_path,
            )

        assert "weightwatcher" in call_order, "run_weightwatcher was not called"
        fit_idx = call_order.index("fit")
        ww_idx = call_order.index("weightwatcher")
        assert fit_idx < ww_idx, (
            f"fit (idx={fit_idx}) must run before weightwatcher (idx={ww_idx})"
        )


class TestDiagnosticsUnconditional:
    """Diagnostics run even when profiling is disabled (RC17)."""

    @pytest.mark.model_loading
    def test_diagnostics_run_even_when_profiling_disabled(self, tmp_path: Any) -> None:
        """Both diagnostics run with profiling.enabled=False."""
        from minivess.orchestration.flows.train_flow import train_one_fold_task

        with (
            patch(
                "minivess.diagnostics.pre_training_checks.run_pre_training_checks",
                return_value=[],
            ) as mock_pre,
            patch(
                "minivess.diagnostics.weight_diagnostics.run_weightwatcher",
                return_value={"diag_ww_alpha_mean": 2.0},
            ) as mock_ww,
            patch(
                "minivess.pipeline.trainer.SegmentationTrainer",
            ) as mock_trainer_cls,
            patch(
                "minivess.adapters.model_builder.build_adapter",
                return_value=MagicMock(),
            ),
            patch(
                "minivess.pipeline.loss_functions.build_loss_function",
                return_value=MagicMock(),
            ),
            patch(
                "minivess.data.loader.build_train_loader",
                return_value=_make_mock_loader(),
            ),
            patch(
                "minivess.data.loader.build_val_loader",
                return_value=_make_mock_loader(),
            ),
            patch(
                "minivess.observability.tracking.ExperimentTracker",
                return_value=_mock_tracker(),
            ),
        ):
            mock_trainer_cls.return_value.fit.return_value = _fit_result()

            train_one_fold_task.fn(
                fold_id=0,
                fold_split=_make_mock_fold_split(),
                config=_make_mock_config(profiling_enabled=False),
                checkpoint_dir=tmp_path,
            )

        mock_pre.assert_called_once()
        mock_ww.assert_called_once()
