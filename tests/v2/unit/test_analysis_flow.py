"""Tests for the Analysis Prefect Flow (Flow 3).

All tests run with PREFECT_DISABLED=1 so the @flow/@task decorators are no-ops,
exercising the pure-Python orchestration logic without a Prefect server.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Ensure Prefect is disabled before any minivess imports
os.environ["PREFECT_DISABLED"] = "1"

from minivess.config.evaluation_config import (  # noqa: E402
    EnsembleStrategyName,
    EvaluationConfig,
    MetricDirection,
)
from minivess.ensemble.builder import EnsembleMember, EnsembleSpec  # noqa: E402
from minivess.pipeline.ci import ConfidenceInterval  # noqa: E402
from minivess.pipeline.evaluation import FoldResult  # noqa: E402
from minivess.pipeline.evaluation_runner import EvaluationResult  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers: build lightweight mock data structures
# ---------------------------------------------------------------------------


def _make_ci(value: float) -> ConfidenceInterval:
    """Create a mock ConfidenceInterval with given point estimate."""
    return ConfidenceInterval(
        point_estimate=value,
        lower=value - 0.05,
        upper=value + 0.05,
        confidence_level=0.95,
        method="percentile",
    )


def _make_fold_result(dice: float = 0.80, cldice: float = 0.70) -> FoldResult:
    """Create a FoldResult with common metrics."""
    return FoldResult(
        per_volume_metrics={
            "dsc": [dice],
            "centreline_dsc": [cldice],
            "measured_masd": [1.5],
        },
        aggregated={
            "dsc": _make_ci(dice),
            "centreline_dsc": _make_ci(cldice),
            "measured_masd": _make_ci(1.5),
        },
    )


def _make_eval_result(
    model_name: str = "test_model",
    dataset: str = "minivess",
    subset: str = "all",
    dice: float = 0.80,
) -> EvaluationResult:
    """Create a mock EvaluationResult."""
    return EvaluationResult(
        model_name=model_name,
        dataset_name=dataset,
        subset_name=subset,
        fold_result=_make_fold_result(dice=dice),
        predictions_dir=None,
        uncertainty_maps_dir=None,
    )


def _make_eval_config() -> EvaluationConfig:
    """Create a minimal EvaluationConfig for testing."""
    return EvaluationConfig(
        primary_metric="val_compound_masd_cldice",
        primary_metric_direction=MetricDirection.MAXIMIZE,
        ensemble_strategies=[EnsembleStrategyName.PER_LOSS_SINGLE_BEST],
        bootstrap_n_resamples=100,
    )


def _make_mock_run(
    run_id: str = "run_001",
    loss_type: str = "dice_ce",
    fold_id: int = 0,
) -> dict[str, Any]:
    """Create a mock training run info dict."""
    return {
        "run_id": run_id,
        "loss_type": loss_type,
        "fold_id": fold_id,
        "artifact_dir": "/tmp/fake/artifacts",
        "metrics": {"val_dice": 0.80, "val_loss": 0.20},
    }


def _make_mock_ensemble_spec(name: str = "test_ensemble") -> EnsembleSpec:
    """Create a mock EnsembleSpec with a dummy member."""
    import torch

    dummy_net = torch.nn.Linear(1, 1)
    member = EnsembleMember(
        checkpoint_path=Path("/tmp/fake/best_val_dice.pth"),
        run_id="run_001",
        loss_type="dice_ce",
        fold_id=0,
        metric_name="val_dice",
        net=dummy_net,
    )
    return EnsembleSpec(
        name=name,
        strategy=EnsembleStrategyName.PER_LOSS_SINGLE_BEST,
        members=[member],
        description="Test ensemble with 1 member",
    )


def _make_mock_dataloaders() -> dict[str, dict[str, Any]]:
    """Create a mock HierarchicalDataLoaderDict (empty loaders for shape)."""
    return {"minivess": {"all": MagicMock()}}


def _make_all_results() -> dict[str, dict[str, dict[str, EvaluationResult]]]:
    """Create a mock all_results dict: {model: {dataset: {subset: result}}}."""
    return {
        "dice_ce_fold0": {
            "minivess": {"all": _make_eval_result("dice_ce_fold0", dice=0.82)},
        },
        "dice_ce_cldice_fold0": {
            "minivess": {"all": _make_eval_result("dice_ce_cldice_fold0", dice=0.78)},
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadTrainingArtifacts:
    """Tests for load_training_artifacts task."""

    def test_callable(self) -> None:
        """load_training_artifacts is a callable function."""
        from minivess.orchestration.flows.analysis_flow import (
            load_training_artifacts,
        )

        assert callable(load_training_artifacts)

    @patch("minivess.orchestration.flows.analysis_flow._discover_runs")
    def test_returns_list_of_dicts(self, mock_discover: MagicMock) -> None:
        """load_training_artifacts returns a list of run info dicts."""
        from minivess.orchestration.flows.analysis_flow import (
            load_training_artifacts,
        )

        mock_discover.return_value = [_make_mock_run()]
        config = _make_eval_config()
        model_config: dict[str, Any] = {"model_name": "DynUNet"}

        result = load_training_artifacts(config, model_config)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["run_id"] == "run_001"


class TestBuildEnsembles:
    """Tests for build_ensembles task."""

    def test_callable(self) -> None:
        """build_ensembles is a callable function."""
        from minivess.orchestration.flows.analysis_flow import build_ensembles

        assert callable(build_ensembles)

    @patch("minivess.orchestration.flows.analysis_flow.EnsembleBuilder")
    def test_returns_dict_of_ensemble_specs(
        self, mock_builder_cls: MagicMock
    ) -> None:
        """build_ensembles returns a dict mapping names to EnsembleSpec."""
        from minivess.orchestration.flows.analysis_flow import build_ensembles

        mock_builder = MagicMock()
        mock_builder.build_all.return_value = {
            "ensemble_a": _make_mock_ensemble_spec("ensemble_a"),
        }
        mock_builder_cls.return_value = mock_builder

        config = _make_eval_config()
        runs = [_make_mock_run()]
        model_config: dict[str, Any] = {"model_name": "DynUNet"}

        result = build_ensembles(runs, config, model_config)
        assert isinstance(result, dict)
        assert "ensemble_a" in result
        assert isinstance(result["ensemble_a"], EnsembleSpec)


class TestEvaluateAllModels:
    """Tests for evaluate_all_models task."""

    def test_callable(self) -> None:
        """evaluate_all_models is a callable function."""
        from minivess.orchestration.flows.analysis_flow import (
            evaluate_all_models,
        )

        assert callable(evaluate_all_models)

    @patch(
        "minivess.orchestration.flows.analysis_flow._evaluate_single_model_on_all"
    )
    def test_returns_nested_results_dict(
        self, mock_eval: MagicMock
    ) -> None:
        """evaluate_all_models returns {model: {dataset: {subset: result}}}."""
        from minivess.orchestration.flows.analysis_flow import (
            evaluate_all_models,
        )

        # Mock _evaluate_single_model_on_all to return per-model results
        mock_eval.return_value = {
            "minivess": {"all": _make_eval_result()},
        }

        import torch

        single_models = {"model_a": torch.nn.Linear(1, 1)}
        ensembles: dict[str, EnsembleSpec] = {}
        dataloaders = _make_mock_dataloaders()
        config = _make_eval_config()

        result = evaluate_all_models(
            single_models, ensembles, dataloaders, config
        )
        assert isinstance(result, dict)
        assert "model_a" in result
        assert "minivess" in result["model_a"]


class TestGenerateComparison:
    """Tests for generate_comparison task."""

    def test_returns_markdown(self) -> None:
        """generate_comparison returns a markdown string."""
        from minivess.orchestration.flows.analysis_flow import (
            generate_comparison,
        )

        all_results = _make_all_results()
        markdown = generate_comparison(all_results)
        assert isinstance(markdown, str)
        assert len(markdown) > 0

    def test_empty_results_returns_string(self) -> None:
        """generate_comparison handles empty results gracefully."""
        from minivess.orchestration.flows.analysis_flow import (
            generate_comparison,
        )

        markdown = generate_comparison({})
        assert isinstance(markdown, str)


class TestRegisterChampionTask:
    """Tests for register_champion_task."""

    def test_callable(self) -> None:
        """register_champion_task is a callable function."""
        from minivess.orchestration.flows.analysis_flow import (
            register_champion_task,
        )

        assert callable(register_champion_task)

    @patch("minivess.orchestration.flows.analysis_flow.ModelPromoter")
    def test_returns_promotion_dict(self, mock_promoter_cls: MagicMock) -> None:
        """register_champion_task returns a dict with champion info."""
        from minivess.orchestration.flows.analysis_flow import (
            register_champion_task,
        )

        mock_promoter = MagicMock()
        mock_promoter.find_best_model.return_value = ("dice_ce_fold0", 0.82)
        mock_promoter.rank_models.return_value = [
            ("dice_ce_fold0", 0.82),
            ("dice_ce_cldice_fold0", 0.78),
        ]
        mock_promoter.generate_promotion_report.return_value = (
            "# Promotion Report\n..."
        )
        mock_promoter_cls.return_value = mock_promoter

        all_results = _make_all_results()
        config = _make_eval_config()

        result = register_champion_task(all_results, config)
        assert isinstance(result, dict)
        assert "champion_name" in result
        assert "champion_score" in result
        assert "promotion_report" in result


class TestGenerateReport:
    """Tests for generate_report task."""

    def test_returns_string(self) -> None:
        """generate_report returns a markdown string."""
        from minivess.orchestration.flows.analysis_flow import generate_report

        all_results = _make_all_results()
        comparison_md = "## Comparison\n| Model | DSC |\n"
        promotion_info = {
            "champion_name": "dice_ce_fold0",
            "champion_score": 0.82,
            "promotion_report": "# Report",
        }

        report = generate_report(all_results, comparison_md, promotion_info)
        assert isinstance(report, str)
        assert "Analysis" in report or "analysis" in report.lower()

    def test_contains_sections(self) -> None:
        """generate_report contains comparison and promotion sections."""
        from minivess.orchestration.flows.analysis_flow import generate_report

        all_results = _make_all_results()
        comparison_md = "## Cross-Model Comparison\nsome table"
        promotion_info = {
            "champion_name": "dice_ce_fold0",
            "champion_score": 0.82,
            "promotion_report": "# Promotion\nranking table",
        }

        report = generate_report(all_results, comparison_md, promotion_info)
        assert "Comparison" in report or "comparison" in report.lower()
        assert "dice_ce_fold0" in report


class TestRunAnalysisFlow:
    """Tests for run_analysis_flow (the @flow)."""

    @patch("minivess.orchestration.flows.analysis_flow.generate_report")
    @patch("minivess.orchestration.flows.analysis_flow.register_champion_task")
    @patch("minivess.orchestration.flows.analysis_flow.generate_comparison")
    @patch("minivess.orchestration.flows.analysis_flow.evaluate_all_models")
    @patch("minivess.orchestration.flows.analysis_flow.build_ensembles")
    @patch("minivess.orchestration.flows.analysis_flow.load_training_artifacts")
    def test_returns_dict_with_expected_keys(
        self,
        mock_load: MagicMock,
        mock_build: MagicMock,
        mock_eval: MagicMock,
        mock_compare: MagicMock,
        mock_register: MagicMock,
        mock_report: MagicMock,
    ) -> None:
        """run_analysis_flow returns dict with results, comparison, promotion, report."""
        from minivess.orchestration.flows.analysis_flow import (
            run_analysis_flow,
        )

        mock_load.return_value = [_make_mock_run()]
        mock_build.return_value = {}
        mock_eval.return_value = _make_all_results()
        mock_compare.return_value = "## Comparison"
        mock_register.return_value = {
            "champion_name": "dice_ce_fold0",
            "champion_score": 0.82,
            "promotion_report": "# Report",
        }
        mock_report.return_value = "# Full Report"

        config = _make_eval_config()
        model_config: dict[str, Any] = {"model_name": "DynUNet"}
        dataloaders = _make_mock_dataloaders()

        result = run_analysis_flow(config, model_config, dataloaders)

        assert isinstance(result, dict)
        expected_keys = {"results", "comparison", "promotion", "report", "mlflow_evaluation"}
        assert set(result.keys()) == expected_keys

    def test_flow_works_without_prefect(self) -> None:
        """Verify the flow is usable with PREFECT_DISABLED=1."""
        from minivess.orchestration.flows.analysis_flow import (
            run_analysis_flow,
        )

        # The flow decorator should be a no-op, so run_analysis_flow
        # should be a regular callable
        assert callable(run_analysis_flow)
        # Verify it is not a Prefect Flow object (since Prefect is disabled)
        assert not hasattr(run_analysis_flow, "fn")


class TestEachTaskIndependentlyCallable:
    """Verify all tasks can be imported and are callable."""

    def test_all_tasks_importable(self) -> None:
        """All task functions can be imported from the module."""
        from minivess.orchestration.flows.analysis_flow import (
            build_ensembles,
            evaluate_all_models,
            generate_comparison,
            generate_report,
            load_training_artifacts,
            register_champion_task,
            run_analysis_flow,
        )

        tasks = [
            load_training_artifacts,
            build_ensembles,
            evaluate_all_models,
            generate_comparison,
            register_champion_task,
            generate_report,
            run_analysis_flow,
        ]
        for t in tasks:
            assert callable(t), f"{t} is not callable"


class TestAnalysisFlowWithMockData:
    """End-to-end test of the analysis flow with mocked dependencies."""

    @patch("minivess.orchestration.flows.analysis_flow.ModelPromoter")
    @patch("minivess.orchestration.flows.analysis_flow.EnsembleBuilder")
    @patch("minivess.orchestration.flows.analysis_flow._discover_runs")
    @patch(
        "minivess.orchestration.flows.analysis_flow._evaluate_single_model_on_all"
    )
    def test_full_flow_with_mocks(
        self,
        mock_eval_single: MagicMock,
        mock_discover: MagicMock,
        mock_builder_cls: MagicMock,
        mock_promoter_cls: MagicMock,
    ) -> None:
        """Full flow executes all tasks and returns complete result."""
        from minivess.orchestration.flows.analysis_flow import (
            run_analysis_flow,
        )

        # Setup mock discovery
        mock_discover.return_value = [
            _make_mock_run("run_001", "dice_ce", 0),
            _make_mock_run("run_002", "dice_ce", 1),
        ]

        # Setup mock ensemble builder
        mock_builder = MagicMock()
        mock_builder.build_all.return_value = {
            "ensemble_test": _make_mock_ensemble_spec("ensemble_test"),
        }
        mock_builder_cls.return_value = mock_builder

        # Setup mock evaluation
        mock_eval_single.return_value = {
            "minivess": {"all": _make_eval_result()},
        }

        # Setup mock promoter
        mock_promoter = MagicMock()
        mock_promoter.find_best_model.return_value = ("dice_ce_fold0", 0.82)
        mock_promoter.rank_models.return_value = [
            ("dice_ce_fold0", 0.82),
        ]
        mock_promoter.generate_promotion_report.return_value = (
            "# Promotion Report"
        )
        mock_promoter_cls.return_value = mock_promoter

        config = _make_eval_config()
        model_config: dict[str, Any] = {"model_name": "DynUNet"}
        dataloaders = _make_mock_dataloaders()

        result = run_analysis_flow(config, model_config, dataloaders)

        assert isinstance(result, dict)
        assert "results" in result
        assert "comparison" in result
        assert "promotion" in result
        assert "report" in result
        assert isinstance(result["report"], str)
        assert len(result["report"]) > 0


# ---------------------------------------------------------------------------
# Phase 3 additions: MLflow serving integration (#81, #84)
# ---------------------------------------------------------------------------


class TestLogModelsToMlflow:
    """Tests for log_models_to_mlflow task."""

    def test_callable(self) -> None:
        """log_models_to_mlflow is a callable function."""
        from minivess.orchestration.flows.analysis_flow import (
            log_models_to_mlflow,
        )

        assert callable(log_models_to_mlflow)

    @patch("minivess.orchestration.flows.analysis_flow._log_single_model_safe")
    @patch("minivess.orchestration.flows.analysis_flow._log_ensemble_model_safe")
    def test_logs_single_and_ensemble_models(
        self,
        mock_log_ensemble: MagicMock,
        mock_log_single: MagicMock,
    ) -> None:
        """log_models_to_mlflow logs both single and ensemble models."""
        from minivess.orchestration.flows.analysis_flow import (
            log_models_to_mlflow,
        )

        mock_log_single.return_value = "runs:/run_001/model"
        mock_log_ensemble.return_value = "runs:/ens_001/ensemble_model"

        runs = [_make_mock_run("run_001"), _make_mock_run("run_002")]
        ensembles = {"ens_a": _make_mock_ensemble_spec("ens_a")}
        config = _make_eval_config()
        model_config: dict[str, Any] = {"family": "test"}

        result = log_models_to_mlflow(
            runs, ensembles, config, model_config
        )

        assert isinstance(result, dict)
        # Should have entries for single models + ensembles
        assert mock_log_single.called or mock_log_ensemble.called

    @patch("minivess.orchestration.flows.analysis_flow._log_single_model_safe")
    @patch("minivess.orchestration.flows.analysis_flow._log_ensemble_model_safe")
    def test_returns_model_uris(
        self,
        mock_log_ensemble: MagicMock,
        mock_log_single: MagicMock,
    ) -> None:
        """log_models_to_mlflow returns a mapping of model_name -> model_uri."""
        from minivess.orchestration.flows.analysis_flow import (
            log_models_to_mlflow,
        )

        mock_log_single.return_value = "runs:/run_001/model"
        mock_log_ensemble.return_value = "runs:/ens_001/ensemble_model"

        runs = [_make_mock_run("run_001")]
        ensembles = {"ens_a": _make_mock_ensemble_spec("ens_a")}
        config = _make_eval_config()
        model_config: dict[str, Any] = {"family": "test"}

        result = log_models_to_mlflow(
            runs, ensembles, config, model_config
        )

        # Result should be a dict of model names to URIs
        assert isinstance(result, dict)


class TestEnsembleInferenceWrapper:
    """Tests for _EnsembleInferenceWrapper that wraps DeepEnsemblePredictor."""

    def test_wrapper_is_nn_module(self) -> None:
        """_EnsembleInferenceWrapper is an nn.Module."""
        import torch

        from minivess.orchestration.flows.analysis_flow import (
            _EnsembleInferenceWrapper,
        )

        nets = [torch.nn.Linear(1, 1)]
        wrapper = _EnsembleInferenceWrapper(nets)
        assert isinstance(wrapper, torch.nn.Module)

    def test_wrapper_uses_all_members(self) -> None:
        """Wrapper stores all member networks."""
        import torch

        from minivess.orchestration.flows.analysis_flow import (
            _EnsembleInferenceWrapper,
        )

        nets = [torch.nn.Linear(1, 1) for _ in range(3)]
        wrapper = _EnsembleInferenceWrapper(nets)
        assert wrapper.n_members == 3

    def test_wrapper_forward_returns_tensor(self) -> None:
        """Wrapper forward() returns logits as a Tensor (for inference runner)."""
        import torch
        from torch import nn

        from minivess.orchestration.flows.analysis_flow import (
            _EnsembleInferenceWrapper,
        )

        # Simple mock models that produce (B, 2, D, H, W)
        class _TinyNet(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, _, d, h, w = x.shape
                return torch.randn(b, 2, d, h, w)

        nets = [_TinyNet() for _ in range(2)]
        wrapper = _EnsembleInferenceWrapper(nets)
        x = torch.randn(1, 1, 4, 4, 4)
        out = wrapper(x)

        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 2, 4, 4, 4)


class TestExtractSingleModelsFromRuns:
    """Tests for extracting single-fold models from ensemble builder members."""

    def test_extract_returns_nn_modules(self) -> None:
        """_extract_single_models_as_modules returns {name: nn.Module}."""
        import torch

        from minivess.orchestration.flows.analysis_flow import (
            _extract_single_models_as_modules,
        )

        ensembles = {
            "per_loss_single_best_dice_ce": _make_mock_ensemble_spec(
                "per_loss_single_best_dice_ce"
            ),
        }

        result = _extract_single_models_as_modules(ensembles)
        assert isinstance(result, dict)
        for name, module in result.items():
            assert isinstance(name, str)
            assert isinstance(module, torch.nn.Module)

    def test_deduplicates_by_run_id(self) -> None:
        """Models with same run_id are not duplicated."""

        from minivess.orchestration.flows.analysis_flow import (
            _extract_single_models_as_modules,
        )

        # Make two ensembles that share a member
        spec1 = _make_mock_ensemble_spec("ens1")
        spec2 = _make_mock_ensemble_spec("ens2")

        ensembles = {"ens1": spec1, "ens2": spec2}
        result = _extract_single_models_as_modules(ensembles)
        # Both share run_001 so only one entry
        assert len(result) == 1


class TestEvaluateWithMlflow:
    """Tests for evaluate_with_mlflow task."""

    def test_callable(self) -> None:
        """evaluate_with_mlflow is a callable function."""
        from minivess.orchestration.flows.analysis_flow import (
            evaluate_with_mlflow,
        )

        assert callable(evaluate_with_mlflow)

    @patch("minivess.orchestration.flows.analysis_flow._run_mlflow_eval_safe")
    def test_returns_dict(self, mock_eval: MagicMock) -> None:
        """evaluate_with_mlflow returns a dict of model_name -> eval results."""
        from minivess.orchestration.flows.analysis_flow import (
            evaluate_with_mlflow,
        )

        mock_eval.return_value = {"dice_coefficient": 0.85}

        all_results = _make_all_results()
        config = _make_eval_config()

        result = evaluate_with_mlflow(all_results, config)
        assert isinstance(result, dict)


class TestUpdatedFlowWithMlflowSteps:
    """Tests for the updated run_analysis_flow with MLflow steps."""

    @patch("minivess.orchestration.flows.analysis_flow.generate_report")
    @patch("minivess.orchestration.flows.analysis_flow.register_champion_task")
    @patch("minivess.orchestration.flows.analysis_flow.generate_comparison")
    @patch("minivess.orchestration.flows.analysis_flow.evaluate_with_mlflow")
    @patch("minivess.orchestration.flows.analysis_flow.evaluate_all_models")
    @patch("minivess.orchestration.flows.analysis_flow.log_models_to_mlflow")
    @patch("minivess.orchestration.flows.analysis_flow.build_ensembles")
    @patch("minivess.orchestration.flows.analysis_flow.load_training_artifacts")
    def test_flow_calls_mlflow_steps(
        self,
        mock_load: MagicMock,
        mock_build: MagicMock,
        mock_log_models: MagicMock,
        mock_eval: MagicMock,
        mock_mlflow_eval: MagicMock,
        mock_compare: MagicMock,
        mock_register: MagicMock,
        mock_report: MagicMock,
    ) -> None:
        """Updated flow calls log_models_to_mlflow and evaluate_with_mlflow."""
        from minivess.orchestration.flows.analysis_flow import (
            run_analysis_flow,
        )

        mock_load.return_value = [_make_mock_run()]
        mock_build.return_value = {}
        mock_log_models.return_value = {}
        mock_eval.return_value = _make_all_results()
        mock_mlflow_eval.return_value = {}
        mock_compare.return_value = "## Comparison"
        mock_register.return_value = {
            "champion_name": "model_a",
            "champion_score": 0.82,
            "promotion_report": "# Report",
        }
        mock_report.return_value = "# Full Report"

        config = _make_eval_config()
        model_config: dict[str, Any] = {"model_name": "DynUNet"}
        dataloaders = _make_mock_dataloaders()

        result = run_analysis_flow(config, model_config, dataloaders)

        assert isinstance(result, dict)
        mock_log_models.assert_called_once()
        mock_mlflow_eval.assert_called_once()
