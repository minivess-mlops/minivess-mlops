"""Tests for T-06: training_flow() body — real SegmentationTrainer.fit() call.

Verifies the ORCHESTRATION structure of training_flow() using mocked fold
training. Heavy computation (actual trainer.fit()) is in the integration tests.
Uses yaml.safe_load/ast.parse — NO regex (CLAUDE.md Rule #16).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import patch

import pytest

_TRAIN_FLOW_SRC = Path("src/minivess/orchestration/flows/train_flow.py")

_FAKE_FOLD_RESULT: dict = {
    "best_val_loss": 0.42,
    "final_epoch": 1,
    "history": {"train_loss": [0.9], "val_loss": [0.42]},
    "best_metrics": {"val_dice": 0.6},
}


def _make_splits_dir(tmp_path: Path, n_folds: int = 1) -> Path:
    """Create a minimal splits.json in a temp SPLITS_DIR."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    splits = [
        {
            "train": [{"image": f"vol_{i:03d}.nii.gz", "label": f"vol_{i:03d}.nii.gz"}],
            "val": [
                {"image": f"vol_{i + 1:03d}.nii.gz", "label": f"vol_{i + 1:03d}.nii.gz"}
            ],
        }
        for i in range(n_folds)
    ]
    (splits_dir / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    return splits_dir


# ---------------------------------------------------------------------------
# AST-level: no argparse.Namespace in train_flow.py
# ---------------------------------------------------------------------------


class TestNoArgparseNamespace:
    def test_train_flow_no_argparse_namespace(self) -> None:
        """train_flow.py must not construct argparse.Namespace (stub pattern)."""
        source = _TRAIN_FLOW_SRC.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # Check for argparse.Namespace() call
            is_attr_namespace = (
                isinstance(func, ast.Attribute) and func.attr == "Namespace"
            )
            is_name_namespace = isinstance(func, ast.Name) and func.id == "Namespace"
            assert not (is_attr_namespace or is_name_namespace), (
                f"argparse.Namespace() still in train_flow.py line {node.lineno}. "
                "Remove the stub argparse pattern."
            )


# ---------------------------------------------------------------------------
# TrainingFlowResult: must be a dataclass (not a plain dict)
# ---------------------------------------------------------------------------


class TestTrainingFlowResult:
    def test_training_flow_result_is_importable(self) -> None:
        from minivess.orchestration.flows.train_flow import (
            TrainingFlowResult,  # noqa: F401
        )

    def test_training_flow_result_is_dataclass(self) -> None:
        import dataclasses

        from minivess.orchestration.flows.train_flow import TrainingFlowResult

        assert dataclasses.is_dataclass(TrainingFlowResult), (
            "TrainingFlowResult must be a @dataclass, not a plain dict."
        )

    def test_training_flow_result_has_status(self) -> None:
        from minivess.orchestration.flows.train_flow import TrainingFlowResult

        r = TrainingFlowResult()
        assert hasattr(r, "status")

    def test_training_flow_result_has_fold_results(self) -> None:
        from minivess.orchestration.flows.train_flow import TrainingFlowResult

        r = TrainingFlowResult()
        assert hasattr(r, "fold_results")
        assert isinstance(r.fold_results, list)

    def test_training_flow_result_has_mlflow_run_id(self) -> None:
        from minivess.orchestration.flows.train_flow import TrainingFlowResult

        r = TrainingFlowResult()
        assert hasattr(r, "mlflow_run_id")

    def test_training_flow_result_has_upstream_data_run_id(self) -> None:
        from minivess.orchestration.flows.train_flow import TrainingFlowResult

        r = TrainingFlowResult()
        assert hasattr(r, "upstream_data_run_id")


# ---------------------------------------------------------------------------
# run_training() must be deprecated
# ---------------------------------------------------------------------------


class TestRunTrainingDeprecated:
    def test_run_training_raises_not_implemented(self) -> None:
        """run_training() stub must raise NotImplementedError."""
        from minivess.orchestration.flows.train_flow import run_training

        with pytest.raises(NotImplementedError):
            run_training({"loss_name": "dice_ce"})


# ---------------------------------------------------------------------------
# Functional: training_flow() orchestration logic with mocked fold training
# ---------------------------------------------------------------------------


class TestTrainingFlowOrchestration:
    def test_training_flow_returns_training_flow_result(
        self, monkeypatch, tmp_path
    ) -> None:
        """training_flow() must return TrainingFlowResult (not stub dict)."""
        monkeypatch.setenv("PREFECT_DISABLED", "1")
        splits_dir = _make_splits_dir(tmp_path, n_folds=1)
        monkeypatch.setenv("SPLITS_DIR", str(splits_dir))
        monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))

        with patch(
            "minivess.orchestration.flows.train_flow.train_one_fold_task",
            return_value=_FAKE_FOLD_RESULT,
        ):
            from minivess.orchestration.flows.train_flow import (
                TrainingFlowResult,
                training_flow,
            )

            result = training_flow(
                loss_name="cbdice_cldice",
                num_folds=1,
                max_epochs=1,
                debug=True,
            )

        assert isinstance(result, TrainingFlowResult), (
            f"training_flow() returned {type(result)}, expected TrainingFlowResult. "
            "Replace the stub dict return with TrainingFlowResult."
        )
        assert result.status == "completed"

    def test_training_flow_checkpoint_dir_from_env(self, monkeypatch, tmp_path) -> None:
        """Checkpoint dir must come from CHECKPOINT_DIR env var (not hardcoded)."""
        monkeypatch.setenv("PREFECT_DISABLED", "1")
        splits_dir = _make_splits_dir(tmp_path, n_folds=1)
        monkeypatch.setenv("SPLITS_DIR", str(splits_dir))
        ckpt_dir = tmp_path / "vol_checkpoints"
        monkeypatch.setenv("CHECKPOINT_DIR", str(ckpt_dir))
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))

        captured: list[str] = []

        def _mock_fold(fold_id, fold_split, config, checkpoint_dir):
            captured.append(str(checkpoint_dir))
            return _FAKE_FOLD_RESULT

        with patch(
            "minivess.orchestration.flows.train_flow.train_one_fold_task",
            side_effect=_mock_fold,
        ):
            from minivess.orchestration.flows.train_flow import training_flow

            training_flow(
                loss_name="cbdice_cldice",
                num_folds=1,
                max_epochs=1,
                debug=True,
            )

        assert captured, "train_one_fold_task was never called"
        for d in captured:
            assert str(d).startswith(str(ckpt_dir)), (
                f"checkpoint_dir {d!r} does not start with CHECKPOINT_DIR={ckpt_dir!r}. "
                "training_flow() must read CHECKPOINT_DIR from env var."
            )

    def test_training_flow_reads_splits_from_env(self, monkeypatch, tmp_path) -> None:
        """training_flow() must read fold splits from SPLITS_DIR env var."""
        monkeypatch.setenv("PREFECT_DISABLED", "1")
        splits_dir = _make_splits_dir(tmp_path, n_folds=2)
        monkeypatch.setenv("SPLITS_DIR", str(splits_dir))
        monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))

        call_count = [0]

        def _mock_fold(fold_id, fold_split, config, checkpoint_dir):
            call_count[0] += 1
            return _FAKE_FOLD_RESULT

        with patch(
            "minivess.orchestration.flows.train_flow.train_one_fold_task",
            side_effect=_mock_fold,
        ):
            from minivess.orchestration.flows.train_flow import training_flow

            training_flow(
                loss_name="cbdice_cldice",
                num_folds=2,
                max_epochs=1,
                debug=True,
            )

        assert call_count[0] == 2, (
            f"Expected train_one_fold_task to be called 2 times (one per fold), "
            f"got {call_count[0]}"
        )

    def test_training_flow_fold_results_in_result(self, monkeypatch, tmp_path) -> None:
        """TrainingFlowResult.fold_results must contain one entry per fold."""
        monkeypatch.setenv("PREFECT_DISABLED", "1")
        splits_dir = _make_splits_dir(tmp_path, n_folds=1)
        monkeypatch.setenv("SPLITS_DIR", str(splits_dir))
        monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))

        with patch(
            "minivess.orchestration.flows.train_flow.train_one_fold_task",
            return_value=_FAKE_FOLD_RESULT,
        ):
            from minivess.orchestration.flows.train_flow import training_flow

            result = training_flow(
                loss_name="cbdice_cldice",
                num_folds=1,
                max_epochs=1,
                debug=True,
            )

        assert len(result.fold_results) == 1, (
            f"Expected 1 fold result, got {len(result.fold_results)}"
        )
