"""Tests for MLflow custom evaluators for segmentation metrics.

Phase 2 of MLflow serving integration (#81): Tests for custom mlflow.evaluate()
metrics using path-based indirection for 3D segmentation volumes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_npz_prediction(output_dir: Path, name: str, pred: np.ndarray) -> Path:
    """Save a prediction as .npz matching prediction_store format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.npz"
    np.savez_compressed(path, hard_pred=pred)
    return path


def _save_npz_label(output_dir: Path, name: str, label: np.ndarray) -> Path:
    """Save a label as .npz."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.npz"
    np.savez_compressed(path, arr_0=label)
    return path


def _make_test_volumes(
    tmp_path: Path,
    n_volumes: int = 3,
    shape: tuple[int, int, int] = (16, 16, 8),
    *,
    perfect: bool = False,
) -> tuple[Path, Path]:
    """Create matched prediction and label .npz files.

    Returns (predictions_dir, labels_dir).
    """
    pred_dir = tmp_path / "predictions"
    label_dir = tmp_path / "labels"

    rng = np.random.default_rng(42)
    for i in range(n_volumes):
        label = (rng.random(shape) > 0.5).astype(np.int64)
        if perfect:
            pred = label.copy()
        else:
            # ~70% overlap
            noise = (rng.random(shape) > 0.7).astype(np.int64)
            pred = np.logical_xor(label, noise).astype(np.int64)

        _save_npz_prediction(pred_dir, f"vol_{i:04d}", pred)
        _save_npz_label(label_dir, f"vol_{i:04d}", label)

    return pred_dir, label_dir


# ---------------------------------------------------------------------------
# Tests: build_evaluation_dataframe
# ---------------------------------------------------------------------------


class TestBuildEvaluationDataframe:
    """Tests for building pd.DataFrame with prediction/label file paths."""

    def test_builds_dataframe_from_dirs(self, tmp_path: Path) -> None:
        """build_evaluation_dataframe returns a pd.DataFrame."""
        from minivess.serving.mlflow_evaluators import build_evaluation_dataframe

        pred_dir, label_dir = _make_test_volumes(tmp_path, n_volumes=3)
        df = build_evaluation_dataframe(pred_dir, label_dir)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_dataframe_columns(self, tmp_path: Path) -> None:
        """DataFrame has prediction_path, label_path, and volume_name columns."""
        from minivess.serving.mlflow_evaluators import build_evaluation_dataframe

        pred_dir, label_dir = _make_test_volumes(tmp_path, n_volumes=2)
        df = build_evaluation_dataframe(pred_dir, label_dir)

        assert "prediction_path" in df.columns
        assert "label_path" in df.columns
        assert "volume_name" in df.columns

    def test_dataframe_matches_files(self, tmp_path: Path) -> None:
        """DataFrame entries point to existing .npz files."""
        from minivess.serving.mlflow_evaluators import build_evaluation_dataframe

        pred_dir, label_dir = _make_test_volumes(tmp_path, n_volumes=2)
        df = build_evaluation_dataframe(pred_dir, label_dir)

        for _, row in df.iterrows():
            assert Path(row["prediction_path"]).exists()
            assert Path(row["label_path"]).exists()

    def test_empty_dir_returns_empty_df(self, tmp_path: Path) -> None:
        """Empty directories return an empty DataFrame."""
        from minivess.serving.mlflow_evaluators import build_evaluation_dataframe

        pred_dir = tmp_path / "empty_pred"
        label_dir = tmp_path / "empty_label"
        pred_dir.mkdir()
        label_dir.mkdir()

        df = build_evaluation_dataframe(pred_dir, label_dir)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Tests: load helpers
# ---------------------------------------------------------------------------


class TestLoadHelpers:
    """Tests for .npz loading functions."""

    def test_load_npz_prediction(self, tmp_path: Path) -> None:
        """load_npz_prediction loads the hard_pred array from .npz."""
        from minivess.serving.mlflow_evaluators import load_npz_prediction

        pred = np.ones((8, 8, 4), dtype=np.int64)
        path = _save_npz_prediction(tmp_path, "test", pred)

        loaded = load_npz_prediction(str(path))
        np.testing.assert_array_equal(loaded, pred)

    def test_load_npz_label(self, tmp_path: Path) -> None:
        """load_npz_label loads the arr_0 array from .npz."""
        from minivess.serving.mlflow_evaluators import load_npz_label

        label = np.ones((8, 8, 4), dtype=np.int64)
        path = _save_npz_label(tmp_path, "test", label)

        loaded = load_npz_label(str(path))
        np.testing.assert_array_equal(loaded, label)


# ---------------------------------------------------------------------------
# Tests: dice metric
# ---------------------------------------------------------------------------


class TestDiceMetric:
    """Tests for the custom Dice metric."""

    def test_dice_metric_creation(self) -> None:
        """dice_metric is an EvaluationMetric created via make_metric."""
        from minivess.serving.mlflow_evaluators import dice_metric

        assert dice_metric is not None
        assert hasattr(dice_metric, "eval_fn")

    def test_dice_eval_fn_perfect_overlap(self, tmp_path: Path) -> None:
        """Perfect overlap gives Dice = 1.0."""
        from minivess.serving.mlflow_evaluators import dice_eval_fn

        pred = np.ones((8, 8, 4), dtype=np.int64)
        label = np.ones((8, 8, 4), dtype=np.int64)

        pred_path = _save_npz_prediction(tmp_path / "pred", "vol", pred)
        label_path = _save_npz_label(tmp_path / "label", "vol", label)

        predictions = pd.Series([str(pred_path)])
        targets = pd.Series([str(label_path)])

        result = dice_eval_fn(predictions, targets, {})
        # MetricValue or float
        if hasattr(result, "aggregate_results"):
            assert result.aggregate_results["mean_dice"] == pytest.approx(1.0, abs=1e-5)
        else:
            assert result == pytest.approx(1.0, abs=1e-5)

    def test_dice_eval_fn_no_overlap(self, tmp_path: Path) -> None:
        """No overlap gives Dice = 0.0."""
        from minivess.serving.mlflow_evaluators import dice_eval_fn

        pred = np.ones((8, 8, 4), dtype=np.int64)
        label = np.zeros((8, 8, 4), dtype=np.int64)

        pred_path = _save_npz_prediction(tmp_path / "pred", "vol", pred)
        label_path = _save_npz_label(tmp_path / "label", "vol", label)

        predictions = pd.Series([str(pred_path)])
        targets = pd.Series([str(label_path)])

        result = dice_eval_fn(predictions, targets, {})
        if hasattr(result, "aggregate_results"):
            assert result.aggregate_results["mean_dice"] == pytest.approx(0.0, abs=1e-5)
        else:
            assert result == pytest.approx(0.0, abs=1e-5)

    def test_dice_eval_fn_partial_overlap(self, tmp_path: Path) -> None:
        """Partial overlap gives 0 < Dice < 1."""
        from minivess.serving.mlflow_evaluators import dice_eval_fn

        rng = np.random.default_rng(42)
        label = (rng.random((8, 8, 4)) > 0.5).astype(np.int64)
        noise = (rng.random((8, 8, 4)) > 0.7).astype(np.int64)
        pred = np.logical_xor(label, noise).astype(np.int64)

        pred_path = _save_npz_prediction(tmp_path / "pred", "vol", pred)
        label_path = _save_npz_label(tmp_path / "label", "vol", label)

        predictions = pd.Series([str(pred_path)])
        targets = pd.Series([str(label_path)])

        result = dice_eval_fn(predictions, targets, {})
        if hasattr(result, "aggregate_results"):
            dice_val = result.aggregate_results["mean_dice"]
        else:
            dice_val = result

        assert 0.0 < dice_val < 1.0

    def test_dice_returns_metric_value_with_scores(self, tmp_path: Path) -> None:
        """dice_eval_fn returns MetricValue with per-volume scores."""
        from minivess.serving.mlflow_evaluators import dice_eval_fn

        pred_dir, label_dir = _make_test_volumes(tmp_path, n_volumes=3, perfect=True)
        pred_files = sorted(pred_dir.glob("*.npz"))
        label_files = sorted(label_dir.glob("*.npz"))

        predictions = pd.Series([str(p) for p in pred_files])
        targets = pd.Series([str(lf) for lf in label_files])

        result = dice_eval_fn(predictions, targets, {})
        assert hasattr(result, "scores")
        assert len(result.scores) == 3


# ---------------------------------------------------------------------------
# Tests: compound metric
# ---------------------------------------------------------------------------


class TestCompoundMetric:
    """Tests for the compound MASD+clDice metric."""

    def test_compound_metric_creation(self) -> None:
        """compound_metric is an EvaluationMetric."""
        from minivess.serving.mlflow_evaluators import compound_metric

        assert compound_metric is not None

    def test_compound_eval_fn_returns_aggregate(self, tmp_path: Path) -> None:
        """compound_eval_fn returns a MetricValue with aggregate_results."""
        from minivess.serving.mlflow_evaluators import compound_eval_fn

        pred_dir, label_dir = _make_test_volumes(tmp_path, n_volumes=2, perfect=True)
        pred_files = sorted(pred_dir.glob("*.npz"))
        label_files = sorted(label_dir.glob("*.npz"))

        predictions = pd.Series([str(p) for p in pred_files])
        targets = pd.Series([str(lf) for lf in label_files])

        result = compound_eval_fn(predictions, targets, {})
        assert hasattr(result, "aggregate_results")
        assert "mean_compound" in result.aggregate_results


# ---------------------------------------------------------------------------
# Tests: run_mlflow_evaluation
# ---------------------------------------------------------------------------


class TestRunMlflowEvaluation:
    """Tests for the top-level mlflow.evaluate() wrapper."""

    @patch("minivess.serving.mlflow_evaluators.mlflow")
    def test_calls_mlflow_evaluate(self, mock_mlflow: MagicMock, tmp_path: Path) -> None:
        """run_mlflow_evaluation calls mlflow.evaluate."""
        from minivess.serving.mlflow_evaluators import run_mlflow_evaluation

        pred_dir, label_dir = _make_test_volumes(tmp_path, n_volumes=2)

        mock_mlflow.evaluate.return_value = MagicMock(metrics={"dice": 0.8})
        run_mlflow_evaluation(pred_dir, label_dir)

        mock_mlflow.evaluate.assert_called_once()

    @patch("minivess.serving.mlflow_evaluators.mlflow")
    def test_passes_custom_metrics(self, mock_mlflow: MagicMock, tmp_path: Path) -> None:
        """run_mlflow_evaluation passes extra_metrics with our custom metrics."""
        from minivess.serving.mlflow_evaluators import run_mlflow_evaluation

        pred_dir, label_dir = _make_test_volumes(tmp_path, n_volumes=2)

        mock_mlflow.evaluate.return_value = MagicMock(metrics={"dice": 0.8})
        run_mlflow_evaluation(pred_dir, label_dir)

        call_kwargs = mock_mlflow.evaluate.call_args.kwargs
        assert "extra_metrics" in call_kwargs
        assert len(call_kwargs["extra_metrics"]) > 0

    @patch("minivess.serving.mlflow_evaluators.mlflow")
    def test_returns_result(self, mock_mlflow: MagicMock, tmp_path: Path) -> None:
        """run_mlflow_evaluation returns the mlflow EvaluationResult."""
        from minivess.serving.mlflow_evaluators import run_mlflow_evaluation

        pred_dir, label_dir = _make_test_volumes(tmp_path, n_volumes=2)
        expected = MagicMock(metrics={"dice": 0.8})
        mock_mlflow.evaluate.return_value = expected

        result = run_mlflow_evaluation(pred_dir, label_dir)
        assert result is expected
