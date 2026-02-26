"""Tests for UnifiedEvaluationRunner — Phase 6 evaluation pipeline.

Covers: EvaluationResult dataclass, runner init, single-subset evaluation,
full-model evaluation, prediction saving, markdown generation, MLflow logging,
and uniform interface for single models and ensembles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch import nn

from minivess.config.evaluation_config import EvaluationConfig
from minivess.pipeline.ci import ConfidenceInterval
from minivess.pipeline.evaluation import FoldResult
from minivess.pipeline.evaluation_runner import (
    EvaluationResult,
    UnifiedEvaluationRunner,
)
from minivess.pipeline.inference import SlidingWindowInferenceRunner

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class _MockSegModel(nn.Module):
    """Deterministic mock model returning soft predictions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _c, d, h, w = x.shape
        logits = torch.randn(b, 2, d, h, w)
        return torch.softmax(logits, dim=1)


class _MockEnsembleModel(nn.Module):
    """Mock ensemble (structurally identical interface to single model)."""

    def __init__(self) -> None:
        super().__init__()
        self.members = nn.ModuleList([_MockSegModel(), _MockSegModel()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [m(x) for m in self.members]
        return torch.stack(outputs).mean(dim=0)


class _FakeLoader:
    """Minimal iterable that yields image/label dicts."""

    def __init__(self, n_volumes: int = 3, spatial: tuple[int, ...] = (16, 16, 8)) -> None:
        self._n = n_volumes
        self._spatial = spatial

    def __iter__(self):  # noqa: ANN204
        rng = np.random.default_rng(42)
        for _ in range(self._n):
            yield {
                "image": torch.randn(1, 1, *self._spatial),
                "label": torch.from_numpy(
                    rng.integers(0, 2, size=(1, 1, *self._spatial)).astype(np.int64)
                ),
            }


def _make_fake_fold_result(n_volumes: int = 3) -> FoldResult:
    """Build a trivial FoldResult with synthetic data."""
    per_vol: dict[str, list[float]] = {
        "dsc": [0.8, 0.85, 0.9][:n_volumes],
        "centreline_dsc": [0.7, 0.75, 0.8][:n_volumes],
        "measured_masd": [2.0, 1.5, 1.0][:n_volumes],
    }
    aggregated: dict[str, ConfidenceInterval] = {}
    for name, vals in per_vol.items():
        arr = np.array(vals)
        aggregated[name] = ConfidenceInterval(
            point_estimate=float(np.mean(arr)),
            lower=float(np.min(arr)),
            upper=float(np.max(arr)),
            confidence_level=0.95,
            method="percentile_bootstrap",
        )
    return FoldResult(per_volume_metrics=per_vol, aggregated=aggregated)


def _build_runner() -> tuple[UnifiedEvaluationRunner, EvaluationConfig, SlidingWindowInferenceRunner]:
    """Construct a runner with mocked heavy dependencies."""
    config = EvaluationConfig()
    inference_runner = SlidingWindowInferenceRunner(
        roi_size=(16, 16, 8),
        num_classes=2,
    )
    runner = UnifiedEvaluationRunner(
        eval_config=config,
        inference_runner=inference_runner,
    )
    return runner, config, inference_runner


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvaluationResultDataclass:
    """EvaluationResult is a well-formed dataclass."""

    def test_evaluation_result_fields(self) -> None:
        fold = _make_fake_fold_result()
        result = EvaluationResult(
            model_name="test_model",
            dataset_name="minivess",
            subset_name="all",
            fold_result=fold,
            predictions_dir=None,
            uncertainty_maps_dir=None,
        )
        assert result.model_name == "test_model"
        assert result.dataset_name == "minivess"
        assert result.subset_name == "all"
        assert result.fold_result is fold
        assert result.predictions_dir is None
        assert result.uncertainty_maps_dir is None

    def test_evaluation_result_with_paths(self, tmp_path: Path) -> None:
        fold = _make_fake_fold_result()
        result = EvaluationResult(
            model_name="ens",
            dataset_name="minivess",
            subset_name="thin",
            fold_result=fold,
            predictions_dir=tmp_path / "preds",
            uncertainty_maps_dir=tmp_path / "uq",
        )
        assert result.predictions_dir == tmp_path / "preds"
        assert result.uncertainty_maps_dir == tmp_path / "uq"


class TestRunnerInit:
    """UnifiedEvaluationRunner initialises correctly."""

    def test_runner_init_with_config(self) -> None:
        runner, config, inference_runner = _build_runner()
        assert runner.eval_config is config
        assert runner.inference_runner is inference_runner


class TestEvaluateSingleSubset:
    """evaluate_single_subset returns an EvaluationResult."""

    def test_returns_evaluation_result(self) -> None:
        runner, _, _ = _build_runner()
        model = _MockSegModel()
        loader = _FakeLoader(n_volumes=2, spatial=(16, 16, 8))

        # Mock the expensive MetricsReloaded call
        fake_fold = _make_fake_fold_result(n_volumes=2)
        with patch.object(
            runner, "_run_evaluation", return_value=fake_fold
        ):
            result = runner.evaluate_single_subset(
                model,
                loader,
                model_name="mock",
                dataset_name="ds",
                subset_name="all",
            )
        assert isinstance(result, EvaluationResult)
        assert result.model_name == "mock"
        assert result.dataset_name == "ds"
        assert result.subset_name == "all"

    def test_has_fold_result(self) -> None:
        runner, _, _ = _build_runner()
        model = _MockSegModel()
        loader = _FakeLoader(n_volumes=2, spatial=(16, 16, 8))

        fake_fold = _make_fake_fold_result(n_volumes=2)
        with patch.object(
            runner, "_run_evaluation", return_value=fake_fold
        ):
            result = runner.evaluate_single_subset(
                model,
                loader,
                model_name="mock",
                dataset_name="ds",
                subset_name="all",
            )
        assert isinstance(result.fold_result, FoldResult)
        assert "dsc" in result.fold_result.aggregated

    def test_predictions_saved_when_output_dir_given(self, tmp_path: Path) -> None:
        runner, _, _ = _build_runner()
        model = _MockSegModel()
        loader = _FakeLoader(n_volumes=2, spatial=(16, 16, 8))

        fake_fold = _make_fake_fold_result(n_volumes=2)
        with patch.object(
            runner, "_run_evaluation", return_value=fake_fold
        ):
            result = runner.evaluate_single_subset(
                model,
                loader,
                model_name="mock",
                dataset_name="ds",
                subset_name="all",
                output_dir=tmp_path / "preds",
            )
        assert result.predictions_dir is not None
        assert result.predictions_dir.exists()
        npz_files = list(result.predictions_dir.glob("*.npz"))
        assert len(npz_files) == 2


class TestEvaluateModel:
    """evaluate_model returns a nested dict over datasets and subsets."""

    def test_returns_nested_dict(self) -> None:
        runner, _, _ = _build_runner()
        model = _MockSegModel()

        loaders: dict[str, dict[str, Any]] = {
            "minivess": {"all": _FakeLoader(n_volumes=2, spatial=(16, 16, 8))},
        }

        fake_fold = _make_fake_fold_result(n_volumes=2)
        with patch.object(
            runner, "_run_evaluation", return_value=fake_fold
        ):
            results = runner.evaluate_model(
                model,
                loaders,
                model_name="test",
            )

        assert isinstance(results, dict)
        assert "minivess" in results
        assert "all" in results["minivess"]
        assert isinstance(results["minivess"]["all"], EvaluationResult)

    def test_covers_all_datasets_and_subsets(self) -> None:
        runner, _, _ = _build_runner()
        model = _MockSegModel()

        loaders: dict[str, dict[str, Any]] = {
            "ds_a": {
                "all": _FakeLoader(n_volumes=2, spatial=(16, 16, 8)),
                "thin": _FakeLoader(n_volumes=1, spatial=(16, 16, 8)),
            },
            "ds_b": {
                "all": _FakeLoader(n_volumes=2, spatial=(16, 16, 8)),
            },
        }

        fake_fold = _make_fake_fold_result(n_volumes=2)
        with patch.object(
            runner, "_run_evaluation", return_value=fake_fold
        ):
            results = runner.evaluate_model(
                model,
                loaders,
                model_name="test",
            )

        assert set(results.keys()) == {"ds_a", "ds_b"}
        assert set(results["ds_a"].keys()) == {"all", "thin"}
        assert set(results["ds_b"].keys()) == {"all"}


class TestMarkdownSummary:
    """generate_summary_markdown produces formatted output."""

    def _make_results(self) -> dict[str, dict[str, EvaluationResult]]:
        fold = _make_fake_fold_result()
        return {
            "minivess": {
                "all": EvaluationResult(
                    model_name="test",
                    dataset_name="minivess",
                    subset_name="all",
                    fold_result=fold,
                    predictions_dir=None,
                    uncertainty_maps_dir=None,
                ),
            },
        }

    def test_format_is_markdown(self) -> None:
        runner, _, _ = _build_runner()
        results = self._make_results()
        md = runner.generate_summary_markdown(results, model_name="test")
        assert isinstance(md, str)
        assert "|" in md  # markdown table

    def test_contains_metrics(self) -> None:
        runner, _, _ = _build_runner()
        results = self._make_results()
        md = runner.generate_summary_markdown(results, model_name="test")
        assert "dsc" in md.lower()
        assert "minivess" in md.lower()


class TestMLflowLogging:
    """log_results_to_mlflow creates a run with correct tags."""

    def _make_results(self) -> dict[str, dict[str, EvaluationResult]]:
        fold = _make_fake_fold_result()
        return {
            "minivess": {
                "all": EvaluationResult(
                    model_name="model_a",
                    dataset_name="minivess",
                    subset_name="all",
                    fold_result=fold,
                    predictions_dir=None,
                    uncertainty_maps_dir=None,
                ),
            },
        }

    @patch("minivess.pipeline.evaluation_runner.mlflow")
    def test_creates_run(self, mock_mlflow: MagicMock) -> None:
        runner, _, _ = _build_runner()
        results = self._make_results()

        mock_run = MagicMock()
        mock_run.info.run_id = "abc123"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        run_id = runner.log_results_to_mlflow(
            results,
            model_name="model_a",
        )
        assert run_id == "abc123"
        mock_mlflow.set_experiment.assert_called_once()
        mock_mlflow.start_run.assert_called_once()

    @patch("minivess.pipeline.evaluation_runner.mlflow")
    def test_tags_correct(self, mock_mlflow: MagicMock) -> None:
        runner, _, _ = _build_runner()
        results = self._make_results()

        mock_run = MagicMock()
        mock_run.info.run_id = "run42"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        runner.log_results_to_mlflow(
            results,
            model_name="model_a",
            model_tags={"model_type": "single", "loss_type": "dice_ce"},
        )

        # Check set_tags was called with at least model_name
        set_tags_calls = mock_mlflow.set_tags.call_args_list
        assert len(set_tags_calls) >= 1
        all_tags: dict[str, str] = {}
        for call in set_tags_calls:
            all_tags.update(call[0][0])
        assert all_tags["model_name"] == "model_a"
        assert all_tags["model_type"] == "single"


class TestSameInterfaceSingleAndEnsemble:
    """Single models and ensemble models use the same evaluate_model call."""

    def test_same_interface(self) -> None:
        runner, _, _ = _build_runner()
        single = _MockSegModel()
        ensemble = _MockEnsembleModel()

        loaders: dict[str, dict[str, Any]] = {
            "ds": {"all": _FakeLoader(n_volumes=2, spatial=(16, 16, 8))},
        }

        fake_fold = _make_fake_fold_result(n_volumes=2)

        with patch.object(
            runner, "_run_evaluation", return_value=fake_fold
        ):
            single_results = runner.evaluate_model(
                single, loaders, model_name="single"
            )
            ensemble_results = runner.evaluate_model(
                ensemble, loaders, model_name="ensemble"
            )

        # Both produce identical structure
        assert set(single_results.keys()) == set(ensemble_results.keys())
        for ds_name in single_results:
            assert set(single_results[ds_name].keys()) == set(
                ensemble_results[ds_name].keys()
            )
            for subset_name in single_results[ds_name]:
                assert isinstance(
                    single_results[ds_name][subset_name], EvaluationResult
                )
                assert isinstance(
                    ensemble_results[ds_name][subset_name], EvaluationResult
                )
