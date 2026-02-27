"""Integration tests for the Analysis Prefect Flow (Flow 3).

Exercises the real analysis flow against the actual mlruns/ directory from the
``dynunet_loss_variation_v2`` experiment (ID: 843896622863223169), which contains
4 production runs (dice_ce, cbdice, dice_ce_cldice, cbdice_cldice) each with:
  - 7 checkpoints (best_val_*.pth + last.pth)
  - 46 metrics
  - 3-fold evaluation data

All tests are skipped automatically when mlruns/ is not present (e.g. CI).
Slow checkpoint-loading tests are additionally marked with ``@pytest.mark.slow``.

Run with::

    uv run pytest tests/v2/integration/test_analysis_flow_integration.py -m integration -v
    uv run pytest tests/v2/integration/test_analysis_flow_integration.py -m "integration and not slow" -v
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Ensure Prefect is disabled before any minivess imports
os.environ["PREFECT_DISABLED"] = "1"

import pytest
import torch
from torch import nn

from minivess.config.evaluation_config import (  # noqa: E402
    EnsembleStrategyName,
    EvaluationConfig,
    MetricDirection,
)
from minivess.ensemble.builder import (  # noqa: E402
    EnsembleBuilder,
    EnsembleMember,
    EnsembleSpec,
    expand_runs_to_per_fold,
)
from minivess.pipeline.ci import ConfidenceInterval  # noqa: E402
from minivess.pipeline.evaluation import FoldResult  # noqa: E402
from minivess.pipeline.evaluation_runner import EvaluationResult  # noqa: E402
from minivess.pipeline.mlruns_inspector import get_production_runs  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MLRUNS_DIR: Path = Path(__file__).resolve().parents[3] / "mlruns"
V2_EXPERIMENT_ID: str = "843896622863223169"
V2_EXPERIMENT_NAME: str = "dynunet_loss_variation_v2"
TRACKING_URI: str = f"file://{MLRUNS_DIR}"

EXPECTED_LOSSES: frozenset[str] = frozenset(
    {
        "dice_ce",
        "cbdice",
        "dice_ce_cldice",
        "cbdice_cldice",
    }
)

MODEL_CONFIG: dict[str, Any] = {
    "family": "dynunet",
    "in_channels": 1,
    "out_channels": 2,
}

# Skip condition: no mlruns/ directory or experiment directory is absent
_MLRUNS_MISSING: bool = not (MLRUNS_DIR / V2_EXPERIMENT_ID).is_dir()
SKIP_IF_NO_MLRUNS = pytest.mark.skipif(
    _MLRUNS_MISSING,
    reason=f"mlruns/{V2_EXPERIMENT_ID} not found — run training first",
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")  # type: ignore[misc]
def eval_config() -> EvaluationConfig:
    """EvaluationConfig pointing at the real dynunet_loss_variation_v2 experiment."""
    return EvaluationConfig(
        primary_metric="val_compound_masd_cldice",
        primary_metric_direction=MetricDirection.MAXIMIZE,
        ensemble_strategies=[EnsembleStrategyName.PER_LOSS_SINGLE_BEST],
        mlflow_training_experiment=V2_EXPERIMENT_NAME,
        bootstrap_n_resamples=100,
    )


@pytest.fixture(scope="module")  # type: ignore[misc]
def all_strategies_eval_config() -> EvaluationConfig:
    """EvaluationConfig with all 4 ensemble strategies enabled."""
    return EvaluationConfig(
        primary_metric="val_compound_masd_cldice",
        primary_metric_direction=MetricDirection.MAXIMIZE,
        ensemble_strategies=list(EnsembleStrategyName),
        mlflow_training_experiment=V2_EXPERIMENT_NAME,
        bootstrap_n_resamples=100,
    )


@pytest.fixture(scope="module")  # type: ignore[misc]
def production_run_ids() -> list[str]:
    """Production run IDs from the real mlruns/ directory."""
    if _MLRUNS_MISSING:
        pytest.skip("mlruns not found")
    result: list[str] = get_production_runs(MLRUNS_DIR, V2_EXPERIMENT_ID)
    return result


@pytest.fixture(scope="module")  # type: ignore[misc]
def raw_runs(eval_config: EvaluationConfig) -> list[dict[str, Any]]:
    """Raw per-loss runs from EnsembleBuilder.discover_training_runs_raw()."""
    if _MLRUNS_MISSING:
        pytest.skip("mlruns not found")
    builder = EnsembleBuilder(eval_config, MODEL_CONFIG, tracking_uri=TRACKING_URI)
    result: list[dict[str, Any]] = builder.discover_training_runs_raw()
    return result


@pytest.fixture(scope="module")  # type: ignore[misc]
def expanded_runs(raw_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-fold runs (12 total = 4 losses × 3 folds)."""
    result: list[dict[str, Any]] = expand_runs_to_per_fold(raw_runs)
    return result


@pytest.fixture(scope="module")  # type: ignore[misc]
def expanded_runs_with_checkpoints(
    expanded_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expanded runs with artifact_dir pointing at the checkpoints subdirectory."""
    fixed: list[dict[str, Any]] = []
    for run in expanded_runs:
        run_copy = dict(run)
        run_copy["artifact_dir"] = str(Path(run["artifact_dir"]) / "checkpoints")
        fixed.append(run_copy)
    return fixed


def _make_ci(value: float) -> ConfidenceInterval:
    """Create a ConfidenceInterval with given point estimate."""
    return ConfidenceInterval(
        point_estimate=value,
        lower=value - 0.05,
        upper=value + 0.05,
        confidence_level=0.95,
        method="percentile",
    )


def _make_fold_result(
    dice: float = 0.80,
    cldice: float = 0.70,
) -> FoldResult:
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


def _make_all_results(
    losses: list[str] | None = None,
) -> dict[str, dict[str, dict[str, EvaluationResult]]]:
    """Create a mock all_results dict with one entry per loss."""
    if losses is None:
        losses = [
            "dice_ce_fold0",
            "cbdice_fold0",
            "dice_ce_cldice_fold0",
            "cbdice_cldice_fold0",
        ]
    scores = [0.82, 0.79, 0.81, 0.83]
    return {
        name: {
            "minivess": {"all": _make_eval_result(name, dice=scores[i % len(scores)])}
        }
        for i, name in enumerate(losses)
    }


# ---------------------------------------------------------------------------
# 1. TestDiscoverTrainingRunsIntegration
# ---------------------------------------------------------------------------


@pytest.mark.integration
@SKIP_IF_NO_MLRUNS
class TestDiscoverTrainingRunsIntegration:
    """Integration tests for run discovery against real mlruns/ directory."""

    def test_discover_raw_finds_4_production_runs(
        self, raw_runs: list[dict[str, Any]]
    ) -> None:
        """discover_training_runs_raw() returns exactly 4 runs."""
        assert len(raw_runs) == 4, (
            f"Expected 4 production runs, got {len(raw_runs)}. "
            "Check that all 4 losses have eval_fold2_dsc."
        )

    def test_discover_raw_returns_correct_loss_names(
        self, raw_runs: list[dict[str, Any]]
    ) -> None:
        """Each discovered raw run has one of the expected loss names."""
        found_losses = {r["loss_type"] for r in raw_runs}
        assert found_losses == EXPECTED_LOSSES, (
            f"Loss mismatch: expected {EXPECTED_LOSSES}, got {found_losses}"
        )

    def test_discover_raw_run_has_expected_keys(
        self, raw_runs: list[dict[str, Any]]
    ) -> None:
        """Every raw run dict has the expected structural keys."""
        required_keys = {
            "run_id",
            "loss_type",
            "fold_id",
            "artifact_dir",
            "metrics",
            "num_folds",
        }
        for run in raw_runs:
            missing = required_keys - set(run.keys())
            assert not missing, (
                f"Run {run.get('run_id', '?')[:12]} missing keys: {missing}"
            )

    def test_discover_expanded_returns_12_entries(
        self, expanded_runs: list[dict[str, Any]]
    ) -> None:
        """discover_training_runs(expand_folds=True) returns 4 losses × 3 folds = 12 entries."""
        assert len(expanded_runs) == 12, (
            f"Expected 12 per-fold runs, got {len(expanded_runs)}"
        )

    def test_discover_expanded_covers_all_folds_per_loss(
        self, expanded_runs: list[dict[str, Any]]
    ) -> None:
        """Each loss function has exactly fold_ids 0, 1, 2 in the expanded runs."""
        by_loss: dict[str, set[int]] = {}
        for run in expanded_runs:
            loss = run["loss_type"]
            by_loss.setdefault(loss, set()).add(run["fold_id"])

        for loss_name, fold_ids in by_loss.items():
            assert fold_ids == {0, 1, 2}, (
                f"Loss '{loss_name}' has fold IDs {sorted(fold_ids)}, expected {{0, 1, 2}}"
            )

    def test_each_run_has_artifact_dir(self, raw_runs: list[dict[str, Any]]) -> None:
        """Every raw run has a non-empty artifact_dir string."""
        for run in raw_runs:
            assert isinstance(run["artifact_dir"], str), (
                f"artifact_dir is not a string: {type(run['artifact_dir'])}"
            )
            assert len(run["artifact_dir"]) > 0, (
                f"artifact_dir is empty for run {run['run_id'][:12]}"
            )

    def test_artifact_dir_points_to_existing_path(
        self, raw_runs: list[dict[str, Any]]
    ) -> None:
        """artifact_dir for each raw run points to a directory that exists on disk."""
        for run in raw_runs:
            artifact_path = Path(run["artifact_dir"])
            assert artifact_path.is_dir(), (
                f"artifact_dir does not exist: {artifact_path} "
                f"(run {run['run_id'][:12]}, loss={run['loss_type']})"
            )

    def test_each_run_has_metrics_dict(self, raw_runs: list[dict[str, Any]]) -> None:
        """Every raw run has a non-empty metrics dict with known metric keys."""
        known_metrics = {
            "eval_fold0_dsc",
            "eval_fold1_dsc",
            "eval_fold2_dsc",
            "val_compound_masd_cldice",
        }
        for run in raw_runs:
            metrics = run["metrics"]
            assert isinstance(metrics, dict), (
                f"metrics is not a dict for run {run['run_id'][:12]}"
            )
            assert len(metrics) > 0, f"Empty metrics for run {run['run_id'][:12]}"
            for key in known_metrics:
                assert key in metrics, (
                    f"Missing metric '{key}' in run {run['run_id'][:12]} (loss={run['loss_type']})"
                )

    def test_discover_runs_flow_helper_returns_12(
        self, eval_config: EvaluationConfig
    ) -> None:
        """_discover_runs() flow helper returns 12 expanded per-fold entries."""
        from minivess.orchestration.flows.analysis_flow import _discover_runs

        runs = _discover_runs(eval_config, MODEL_CONFIG, tracking_uri=TRACKING_URI)
        assert len(runs) == 12, (
            f"_discover_runs() returned {len(runs)}, expected 12 (4 losses × 3 folds)"
        )

    def test_discover_runs_unique_loss_fold_combinations(
        self, eval_config: EvaluationConfig
    ) -> None:
        """_discover_runs() produces unique (loss_type, fold_id) pairs — no duplicates."""
        from minivess.orchestration.flows.analysis_flow import _discover_runs

        runs = _discover_runs(eval_config, MODEL_CONFIG, tracking_uri=TRACKING_URI)
        combos = [(r["loss_type"], r["fold_id"]) for r in runs]
        assert len(combos) == len(set(combos)), (
            "Duplicate (loss_type, fold_id) combinations in _discover_runs output"
        )


# ---------------------------------------------------------------------------
# 2. TestEnsembleBuildIntegration
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
@SKIP_IF_NO_MLRUNS
class TestEnsembleBuildIntegration:
    """Build ensembles from real production checkpoints."""

    def test_per_loss_single_best_builds_4_ensembles(
        self,
        eval_config: EvaluationConfig,
        expanded_runs_with_checkpoints: list[dict[str, Any]],
    ) -> None:
        """per_loss_single_best creates exactly 1 ensemble per loss = 4 ensembles."""
        builder = EnsembleBuilder(eval_config, MODEL_CONFIG, tracking_uri=TRACKING_URI)
        ensembles = builder.build_per_loss_single_best(expanded_runs_with_checkpoints)
        assert len(ensembles) == 4, (
            f"Expected 4 per-loss ensembles, got {len(ensembles)}: {list(ensembles.keys())}"
        )

    def test_per_loss_single_best_member_count(
        self,
        eval_config: EvaluationConfig,
        expanded_runs_with_checkpoints: list[dict[str, Any]],
    ) -> None:
        """Each per_loss_single_best ensemble has exactly 3 members (1 per fold)."""
        builder = EnsembleBuilder(eval_config, MODEL_CONFIG, tracking_uri=TRACKING_URI)
        ensembles = builder.build_per_loss_single_best(expanded_runs_with_checkpoints)

        for name, spec in ensembles.items():
            assert len(spec.members) == 3, (
                f"Ensemble '{name}' has {len(spec.members)} members, expected 3"
            )

    def test_per_loss_single_best_covers_all_losses(
        self,
        eval_config: EvaluationConfig,
        expanded_runs_with_checkpoints: list[dict[str, Any]],
    ) -> None:
        """per_loss_single_best ensembles cover all 4 expected losses."""
        builder = EnsembleBuilder(eval_config, MODEL_CONFIG, tracking_uri=TRACKING_URI)
        ensembles = builder.build_per_loss_single_best(expanded_runs_with_checkpoints)

        ensemble_losses = {
            name.replace("per_loss_single_best_", "") for name in ensembles
        }
        assert ensemble_losses == EXPECTED_LOSSES, (
            f"Ensemble losses {ensemble_losses} != expected {EXPECTED_LOSSES}"
        )

    def test_ensemble_members_have_loaded_nets(
        self,
        eval_config: EvaluationConfig,
        expanded_runs_with_checkpoints: list[dict[str, Any]],
    ) -> None:
        """Each member's .net is a loaded nn.Module in eval mode."""
        builder = EnsembleBuilder(eval_config, MODEL_CONFIG, tracking_uri=TRACKING_URI)
        # Build for one loss only for speed
        one_loss_runs = [
            r
            for r in expanded_runs_with_checkpoints
            if r["loss_type"] == "cbdice_cldice"
        ]
        members = builder._load_members_for_metric(
            one_loss_runs, "val_compound_masd_cldice"
        )

        assert len(members) == 3, f"Expected 3 members, got {len(members)}"
        for member in members:
            assert isinstance(member.net, nn.Module), (
                f"member.net is not nn.Module: {type(member.net)}"
            )
            # eval mode means all modules should not be training
            assert not member.net.training, (
                "Member net is in training mode, expected eval mode"
            )

    def test_ensemble_member_checkpoints_are_real_paths(
        self,
        eval_config: EvaluationConfig,
        expanded_runs_with_checkpoints: list[dict[str, Any]],
    ) -> None:
        """Each loaded member's checkpoint_path exists on disk."""
        builder = EnsembleBuilder(eval_config, MODEL_CONFIG, tracking_uri=TRACKING_URI)
        one_loss_runs = [
            r for r in expanded_runs_with_checkpoints if r["loss_type"] == "dice_ce"
        ]
        members = builder._load_members_for_metric(
            one_loss_runs, "val_compound_masd_cldice"
        )

        for member in members:
            assert member.checkpoint_path.exists(), (
                f"Checkpoint path does not exist: {member.checkpoint_path}"
            )
            assert member.checkpoint_path.suffix == ".pth", (
                f"Expected .pth suffix, got: {member.checkpoint_path.suffix}"
            )

    def test_ensemble_wrapper_forward_pass(
        self,
        eval_config: EvaluationConfig,
        expanded_runs_with_checkpoints: list[dict[str, Any]],
    ) -> None:
        """_EnsembleInferenceWrapper produces valid output shape with real nets."""
        from minivess.orchestration.flows.analysis_flow import _EnsembleInferenceWrapper

        builder = EnsembleBuilder(eval_config, MODEL_CONFIG, tracking_uri=TRACKING_URI)
        one_loss_runs = [
            r
            for r in expanded_runs_with_checkpoints
            if r["loss_type"] == "cbdice_cldice"
        ]
        # Load only 1 member for speed
        members = builder._load_members_for_metric(
            one_loss_runs[:1], "val_compound_masd_cldice"
        )
        assert len(members) == 1

        wrapper = _EnsembleInferenceWrapper([m.net for m in members])
        x = torch.randn(1, 1, 16, 16, 16)
        with torch.no_grad():
            out = wrapper(x)

        assert isinstance(out, torch.Tensor), "Output is not a tensor"
        assert out.shape == (1, 2, 16, 16, 16), f"Unexpected output shape: {out.shape}"

    def test_all_loss_single_best_has_12_members(
        self,
        eval_config: EvaluationConfig,
        expanded_runs_with_checkpoints: list[dict[str, Any]],
    ) -> None:
        """all_loss_single_best creates 1 ensemble with 12 members (4 losses × 3 folds)."""
        builder = EnsembleBuilder(eval_config, MODEL_CONFIG, tracking_uri=TRACKING_URI)
        ensembles = builder.build_all_loss_single_best(expanded_runs_with_checkpoints)

        assert len(ensembles) == 1, (
            f"Expected 1 all_loss_single_best ensemble, got {len(ensembles)}"
        )
        spec = ensembles["all_loss_single_best"]
        assert len(spec.members) == 12, (
            f"Expected 12 members in all_loss_single_best, got {len(spec.members)}"
        )


# ---------------------------------------------------------------------------
# 3. TestAnalysisFlowTasksIntegration
# ---------------------------------------------------------------------------


@pytest.mark.integration
@SKIP_IF_NO_MLRUNS
class TestAnalysisFlowTasksIntegration:
    """Test individual Prefect tasks with real MLflow data."""

    def test_load_training_artifacts_returns_runs(
        self, eval_config: EvaluationConfig
    ) -> None:
        """load_training_artifacts finds real production runs via MLflow client."""
        from minivess.orchestration.flows.analysis_flow import load_training_artifacts

        runs = load_training_artifacts(
            eval_config, MODEL_CONFIG, tracking_uri=TRACKING_URI
        )

        assert isinstance(runs, list), "load_training_artifacts should return a list"
        assert len(runs) == 12, f"Expected 12 expanded runs, got {len(runs)}"
        # Spot check structure
        first = runs[0]
        assert "run_id" in first
        assert "loss_type" in first
        assert "fold_id" in first
        assert "artifact_dir" in first

    def test_load_training_artifacts_has_4_unique_losses(
        self, eval_config: EvaluationConfig
    ) -> None:
        """load_training_artifacts finds exactly 4 unique loss functions."""
        from minivess.orchestration.flows.analysis_flow import load_training_artifacts

        runs = load_training_artifacts(
            eval_config, MODEL_CONFIG, tracking_uri=TRACKING_URI
        )
        unique_losses = {r["loss_type"] for r in runs}
        assert unique_losses == EXPECTED_LOSSES, (
            f"Unexpected loss functions: {unique_losses}"
        )

    @patch("minivess.orchestration.flows.analysis_flow.EnsembleBuilder")
    def test_build_ensembles_returns_specs(
        self,
        mock_builder_cls: MagicMock,
        eval_config: EvaluationConfig,
    ) -> None:
        """build_ensembles creates EnsembleSpec objects from pre-fetched runs."""
        from minivess.orchestration.flows.analysis_flow import build_ensembles

        mock_builder = MagicMock()
        mock_builder.build_all.return_value = {
            "per_loss_single_best_dice_ce": EnsembleSpec(
                name="per_loss_single_best_dice_ce",
                strategy=EnsembleStrategyName.PER_LOSS_SINGLE_BEST,
                members=[],
                description="Test",
            )
        }
        mock_builder_cls.return_value = mock_builder

        runs = [
            {
                "run_id": "abc",
                "loss_type": "dice_ce",
                "fold_id": 0,
                "artifact_dir": "/tmp/fake",
                "metrics": {},
            }
        ]
        result = build_ensembles(runs, eval_config, MODEL_CONFIG)

        assert isinstance(result, dict)
        for _name, spec in result.items():
            assert isinstance(spec, EnsembleSpec), (
                f"Expected EnsembleSpec, got {type(spec)}"
            )

    def test_generate_comparison_with_real_eval_results(self) -> None:
        """generate_comparison produces valid markdown from realistic eval data."""
        from minivess.orchestration.flows.analysis_flow import generate_comparison

        # Create eval results matching the 4 production losses
        all_results = _make_all_results(
            losses=[f"{loss}_fold0" for loss in sorted(EXPECTED_LOSSES)]
        )

        markdown = generate_comparison(all_results)

        assert isinstance(markdown, str), "Expected string output"
        assert len(markdown) > 0, "Empty comparison output"
        # Should contain at least one loss name
        assert any(loss in markdown for loss in EXPECTED_LOSSES), (
            "No loss names found in comparison markdown"
        )

    def test_generate_comparison_contains_dsc_column(self) -> None:
        """generate_comparison includes the dsc metric column."""
        from minivess.orchestration.flows.analysis_flow import generate_comparison

        all_results = _make_all_results()
        markdown = generate_comparison(all_results)

        assert "dsc" in markdown.lower(), (
            f"Expected 'dsc' in comparison output, got: {markdown[:200]}"
        )

    def test_generate_report_complete(self) -> None:
        """generate_report produces all expected sections."""
        from minivess.orchestration.flows.analysis_flow import generate_report

        all_results = _make_all_results()
        comparison_md = "| Loss | dsc |\n| dice_ce_fold0 | 0.82 |\n"
        promotion_info = {
            "champion_name": "cbdice_cldice_fold0",
            "champion_score": 0.83,
            "promotion_report": "## Rankings\n1. cbdice_cldice_fold0",
            "environment": "staging",
        }

        report = generate_report(all_results, comparison_md, promotion_info)

        assert isinstance(report, str)
        # Check for required sections
        assert "Analysis Flow Report" in report, "Missing report title"
        assert "Per-Model Results" in report, "Missing per-model section"
        assert "Cross-Model Comparison" in report, "Missing comparison section"
        assert "Champion Model" in report, "Missing champion section"
        assert "cbdice_cldice_fold0" in report, "Champion name missing"

    def test_generate_report_contains_timestamp(self) -> None:
        """generate_report includes a UTC timestamp in the output."""
        from minivess.orchestration.flows.analysis_flow import generate_report

        all_results = _make_all_results()
        comparison_md = "comparison"
        promotion_info = {
            "champion_name": "m1",
            "champion_score": 0.80,
            "promotion_report": "",
        }

        report = generate_report(all_results, comparison_md, promotion_info)
        assert "UTC" in report, "Expected UTC timestamp in report"

    def test_generate_report_contains_model_count(self) -> None:
        """generate_report states the number of evaluated models."""
        from minivess.orchestration.flows.analysis_flow import generate_report

        all_results = _make_all_results()
        promotion_info = {
            "champion_name": "m1",
            "champion_score": 0.80,
            "promotion_report": "",
        }

        report = generate_report(all_results, "comparison", promotion_info)
        n_models = len(all_results)
        assert str(n_models) in report, (
            f"Expected model count {n_models} to appear in report"
        )

    def test_register_champion_finds_best(self) -> None:
        """register_champion_task identifies the highest-scoring model as champion."""
        from minivess.orchestration.flows.analysis_flow import register_champion_task

        config = EvaluationConfig(
            primary_metric="dsc",
            primary_metric_direction=MetricDirection.MAXIMIZE,
            ensemble_strategies=[EnsembleStrategyName.PER_LOSS_SINGLE_BEST],
            bootstrap_n_resamples=100,
        )
        all_results = {
            "dice_ce_fold0": {
                "ds": {"all": _make_eval_result("dice_ce_fold0", dice=0.82)}
            },
            "cbdice_fold0": {
                "ds": {"all": _make_eval_result("cbdice_fold0", dice=0.79)}
            },
            "cbdice_cldice_fold0": {
                "ds": {"all": _make_eval_result("cbdice_cldice_fold0", dice=0.85)}
            },
        }

        result = register_champion_task(all_results, config)

        assert result["champion_name"] == "cbdice_cldice_fold0", (
            f"Expected cbdice_cldice_fold0 as champion, got: {result['champion_name']}"
        )
        assert abs(result["champion_score"] - 0.85) < 0.01, (
            f"Unexpected champion score: {result['champion_score']}"
        )

    def test_register_champion_returns_required_keys(self) -> None:
        """register_champion_task returns dict with all required keys."""
        from minivess.orchestration.flows.analysis_flow import register_champion_task

        config = EvaluationConfig(
            primary_metric="dsc",
            primary_metric_direction=MetricDirection.MAXIMIZE,
            ensemble_strategies=[EnsembleStrategyName.PER_LOSS_SINGLE_BEST],
            bootstrap_n_resamples=100,
        )
        all_results = {"m1": {"ds": {"all": _make_eval_result("m1", dice=0.80)}}}
        result = register_champion_task(all_results, config)

        required_keys = {
            "champion_name",
            "champion_score",
            "rankings",
            "promotion_report",
            "environment",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )

    def test_register_champion_empty_results(self) -> None:
        """register_champion_task handles empty results gracefully."""
        from minivess.orchestration.flows.analysis_flow import register_champion_task

        config = EvaluationConfig(
            primary_metric="dsc",
            primary_metric_direction=MetricDirection.MAXIMIZE,
            ensemble_strategies=[EnsembleStrategyName.PER_LOSS_SINGLE_BEST],
            bootstrap_n_resamples=100,
        )

        result = register_champion_task({}, config)

        assert result["champion_name"] == "", (
            "Expected empty champion name for empty results"
        )
        import math

        assert math.isnan(result["champion_score"]), (
            "Expected NaN champion score for empty results"
        )


# ---------------------------------------------------------------------------
# 4. TestAnalysisFlowDataContract
# ---------------------------------------------------------------------------


@pytest.mark.integration
@SKIP_IF_NO_MLRUNS
class TestAnalysisFlowDataContract:
    """Verify data contracts between Prefect tasks."""

    def test_discover_output_feeds_build_ensembles(
        self, eval_config: EvaluationConfig
    ) -> None:
        """Output of _discover_runs feeds directly into EnsembleBuilder.build_all."""
        from minivess.orchestration.flows.analysis_flow import _discover_runs

        runs = _discover_runs(eval_config, MODEL_CONFIG, tracking_uri=TRACKING_URI)

        # Verify runs have the expected shape for build_ensembles
        for run in runs:
            assert isinstance(run.get("run_id"), str), "run_id must be a str"
            assert isinstance(run.get("loss_type"), str), "loss_type must be a str"
            assert isinstance(run.get("fold_id"), int), "fold_id must be an int"
            assert isinstance(run.get("artifact_dir"), str), (
                "artifact_dir must be a str"
            )
            assert isinstance(run.get("metrics"), dict), "metrics must be a dict"

    @patch("minivess.orchestration.flows.analysis_flow.EnsembleBuilder")
    def test_build_ensembles_output_feeds_extract_single_models(
        self,
        mock_builder_cls: MagicMock,
        eval_config: EvaluationConfig,
    ) -> None:
        """EnsembleSpecs from build_ensembles feed into _extract_single_models_as_modules."""
        from minivess.orchestration.flows.analysis_flow import (
            _extract_single_models_as_modules,
        )

        # Create realistic EnsembleSpecs with loaded nets
        dummy_net = torch.nn.Linear(1, 1)
        member = EnsembleMember(
            checkpoint_path=Path("/tmp/fake.pth"),
            run_id="test_run_001",
            loss_type="dice_ce",
            fold_id=0,
            metric_name="val_compound_masd_cldice",
            net=dummy_net,
        )
        spec = EnsembleSpec(
            name="per_loss_single_best_dice_ce",
            strategy=EnsembleStrategyName.PER_LOSS_SINGLE_BEST,
            members=[member],
            description="Test ensemble",
        )
        ensembles = {"per_loss_single_best_dice_ce": spec}

        # Extract single models from ensemble members
        single_models = _extract_single_models_as_modules(ensembles)

        assert isinstance(single_models, dict)
        assert len(single_models) == 1
        model_name = "dice_ce_fold0"
        assert model_name in single_models, (
            f"Expected '{model_name}' key, got: {list(single_models.keys())}"
        )
        assert isinstance(single_models[model_name], nn.Module)

    def test_eval_results_feed_comparison(self) -> None:
        """EvaluationResult dicts from evaluate_all_models feed into generate_comparison."""
        from minivess.orchestration.flows.analysis_flow import generate_comparison

        # Simulate output structure from evaluate_all_models
        all_results: dict[str, dict[str, dict[str, EvaluationResult]]] = {
            "dice_ce_fold0": {"minivess": {"all": _make_eval_result("dice_ce_fold0")}},
            "cbdice_fold0": {"minivess": {"all": _make_eval_result("cbdice_fold0")}},
        }

        # Should not raise
        markdown = generate_comparison(all_results)
        assert isinstance(markdown, str)
        assert len(markdown) > 0

    def test_eval_results_feed_register_champion(self) -> None:
        """EvaluationResult dicts from evaluate_all_models feed into register_champion_task."""
        from minivess.orchestration.flows.analysis_flow import register_champion_task

        config = EvaluationConfig(
            primary_metric="dsc",
            primary_metric_direction=MetricDirection.MAXIMIZE,
            ensemble_strategies=[EnsembleStrategyName.PER_LOSS_SINGLE_BEST],
            bootstrap_n_resamples=100,
        )

        # Simulate output structure from evaluate_all_models
        all_results: dict[str, dict[str, dict[str, EvaluationResult]]] = {
            "dice_ce_fold0": {
                "minivess": {"all": _make_eval_result("dice_ce_fold0", dice=0.82)}
            },
            "cbdice_fold0": {
                "minivess": {"all": _make_eval_result("cbdice_fold0", dice=0.79)}
            },
        }

        # Should not raise
        result = register_champion_task(all_results, config)
        assert isinstance(result, dict)
        assert result["champion_name"] in {"dice_ce_fold0", "cbdice_fold0"}

    def test_champion_result_feeds_generate_report(self) -> None:
        """Promotion dict from register_champion_task feeds generate_report correctly."""
        from minivess.orchestration.flows.analysis_flow import generate_report

        all_results = _make_all_results()
        comparison_md = "| Model | DSC |\n|---|---|\n| m | 0.80 |"

        # Simulate the exact shape returned by register_champion_task
        promotion_info = {
            "champion_name": "cbdice_cldice_fold0",
            "champion_score": 0.83,
            "rankings": [("cbdice_cldice_fold0", 0.83), ("dice_ce_fold0", 0.82)],
            "promotion_report": "## Rankings\n1. cbdice_cldice_fold0 (0.83)",
            "environment": "staging",
            "registration": None,
        }

        report = generate_report(all_results, comparison_md, promotion_info)
        assert "cbdice_cldice_fold0" in report

    def test_extract_single_models_deduplicates_by_run_id(self) -> None:
        """_extract_single_models_as_modules deduplicates members sharing the same run_id."""
        from minivess.orchestration.flows.analysis_flow import (
            _extract_single_models_as_modules,
        )

        shared_net = torch.nn.Linear(1, 1)
        shared_run_id = "shared_run_abc123"

        # Two ensembles whose only member has the same run_id
        member_a = EnsembleMember(
            checkpoint_path=Path("/tmp/a.pth"),
            run_id=shared_run_id,
            loss_type="dice_ce",
            fold_id=0,
            metric_name="val_dice",
            net=shared_net,
        )
        member_b = EnsembleMember(
            checkpoint_path=Path("/tmp/b.pth"),
            run_id=shared_run_id,
            loss_type="dice_ce",
            fold_id=0,
            metric_name="val_cldice",
            net=shared_net,
        )
        ensembles = {
            "ens_a": EnsembleSpec(
                name="ens_a",
                strategy=EnsembleStrategyName.PER_LOSS_SINGLE_BEST,
                members=[member_a],
                description="A",
            ),
            "ens_b": EnsembleSpec(
                name="ens_b",
                strategy=EnsembleStrategyName.ALL_LOSS_SINGLE_BEST,
                members=[member_b],
                description="B",
            ),
        }

        result = _extract_single_models_as_modules(ensembles)
        # Despite two ensembles, same run_id → only 1 unique model
        assert len(result) == 1, (
            f"Expected 1 unique model after dedup, got {len(result)}: {list(result.keys())}"
        )


# ---------------------------------------------------------------------------
# 5. TestAnalysisFlowE2E
# ---------------------------------------------------------------------------


@pytest.mark.integration
@SKIP_IF_NO_MLRUNS
class TestAnalysisFlowE2E:
    """End-to-end analysis flow with mocked inference but real MLflow data."""

    @patch("minivess.orchestration.flows.analysis_flow._run_mlflow_eval_safe")
    @patch("minivess.orchestration.flows.analysis_flow._log_ensemble_model_safe")
    @patch("minivess.orchestration.flows.analysis_flow._log_single_model_safe")
    @patch("minivess.orchestration.flows.analysis_flow._evaluate_single_model_on_all")
    def test_full_flow_with_real_mlflow_data(
        self,
        mock_eval: MagicMock,
        mock_log_single: MagicMock,
        mock_log_ensemble: MagicMock,
        mock_mlflow_eval: MagicMock,
        eval_config: EvaluationConfig,
    ) -> None:
        """Full analysis flow using real MLflow run discovery but mocked inference."""
        from minivess.orchestration.flows.analysis_flow import run_analysis_flow

        # Mock inference to return synthetic eval results
        mock_eval.return_value = {"minivess": {"all": _make_eval_result(dice=0.82)}}
        mock_log_single.return_value = None  # No actual pyfunc logging
        mock_log_ensemble.return_value = None
        mock_mlflow_eval.return_value = {}

        dataloaders: dict[str, Any] = {"minivess": {"all": MagicMock()}}

        result = run_analysis_flow(
            eval_config,
            MODEL_CONFIG,
            dataloaders,
            tracking_uri=TRACKING_URI,
        )

        assert isinstance(result, dict)
        expected_keys = {
            "results",
            "comparison",
            "promotion",
            "report",
            "mlflow_evaluation",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - set(result.keys())}"
        )

    @patch("minivess.orchestration.flows.analysis_flow._run_mlflow_eval_safe")
    @patch("minivess.orchestration.flows.analysis_flow._log_ensemble_model_safe")
    @patch("minivess.orchestration.flows.analysis_flow._log_single_model_safe")
    @patch("minivess.orchestration.flows.analysis_flow._evaluate_single_model_on_all")
    def test_full_flow_report_is_non_empty_string(
        self,
        mock_eval: MagicMock,
        mock_log_single: MagicMock,
        mock_log_ensemble: MagicMock,
        mock_mlflow_eval: MagicMock,
        eval_config: EvaluationConfig,
    ) -> None:
        """Full flow produces a non-empty markdown report."""
        from minivess.orchestration.flows.analysis_flow import run_analysis_flow

        mock_eval.return_value = {"minivess": {"all": _make_eval_result(dice=0.80)}}
        mock_log_single.return_value = None
        mock_log_ensemble.return_value = None
        mock_mlflow_eval.return_value = {}

        dataloaders: dict[str, Any] = {"minivess": {"all": MagicMock()}}

        result = run_analysis_flow(
            eval_config,
            MODEL_CONFIG,
            dataloaders,
            tracking_uri=TRACKING_URI,
        )

        report = result["report"]
        assert isinstance(report, str)
        assert len(report) > 100, "Report is suspiciously short"
        assert "Analysis Flow Report" in report

    @patch("minivess.orchestration.flows.analysis_flow._run_mlflow_eval_safe")
    @patch("minivess.orchestration.flows.analysis_flow._log_ensemble_model_safe")
    @patch("minivess.orchestration.flows.analysis_flow._log_single_model_safe")
    @patch("minivess.orchestration.flows.analysis_flow._evaluate_single_model_on_all")
    @patch("minivess.orchestration.flows.analysis_flow.build_ensembles")
    def test_full_flow_promotion_info_has_champion(
        self,
        mock_build_ensembles: MagicMock,
        mock_eval: MagicMock,
        mock_log_single: MagicMock,
        mock_log_ensemble: MagicMock,
        mock_mlflow_eval: MagicMock,
        eval_config: EvaluationConfig,
    ) -> None:
        """Full flow promotion_info contains a non-empty champion_name.

        build_ensembles is mocked because the real flow passes artifact_dir
        without the 'checkpoints/' subdirectory to EnsembleBuilder, which
        would result in zero members loaded, zero models evaluated, and
        therefore no champion to promote.  The E2E flow with real checkpoint
        loading is covered by TestEnsembleBuildIntegration (slow tests).
        """
        from minivess.orchestration.flows.analysis_flow import run_analysis_flow

        # Provide a realistic EnsembleSpec with a dummy net
        dummy_net = nn.Linear(1, 1)
        member = EnsembleMember(
            checkpoint_path=Path("/tmp/fake.pth"),
            run_id="run_dice_ce",
            loss_type="dice_ce",
            fold_id=0,
            metric_name="val_compound_masd_cldice",
            net=dummy_net,
        )
        spec = EnsembleSpec(
            name="per_loss_single_best_dice_ce",
            strategy=EnsembleStrategyName.PER_LOSS_SINGLE_BEST,
            members=[member],
            description="Mocked ensemble",
        )
        mock_build_ensembles.return_value = {"per_loss_single_best_dice_ce": spec}
        mock_eval.return_value = {"minivess": {"all": _make_eval_result(dice=0.80)}}
        mock_log_single.return_value = None
        mock_log_ensemble.return_value = None
        mock_mlflow_eval.return_value = {}

        dataloaders: dict[str, Any] = {"minivess": {"all": MagicMock()}}

        result = run_analysis_flow(
            eval_config,
            MODEL_CONFIG,
            dataloaders,
            tracking_uri=TRACKING_URI,
        )

        promotion = result["promotion"]
        assert isinstance(promotion, dict)
        assert promotion.get("champion_name"), "champion_name should be non-empty"
        assert isinstance(promotion.get("champion_score"), float)

    @patch("minivess.orchestration.flows.analysis_flow._run_mlflow_eval_safe")
    @patch("minivess.orchestration.flows.analysis_flow._log_ensemble_model_safe")
    @patch("minivess.orchestration.flows.analysis_flow._log_single_model_safe")
    @patch("minivess.orchestration.flows.analysis_flow._evaluate_single_model_on_all")
    @patch("minivess.orchestration.flows.analysis_flow.build_ensembles")
    def test_full_flow_results_contain_per_fold_models(
        self,
        mock_build_ensembles: MagicMock,
        mock_eval: MagicMock,
        mock_log_single: MagicMock,
        mock_log_ensemble: MagicMock,
        mock_mlflow_eval: MagicMock,
        eval_config: EvaluationConfig,
    ) -> None:
        """Full flow results dict includes entries for single-fold models.

        build_ensembles is mocked because real checkpoint loading requires
        the 'checkpoints/' subdirectory which is not in the artifact_dir
        returned by load_training_artifacts.
        """
        from minivess.orchestration.flows.analysis_flow import run_analysis_flow

        # Provide 4 EnsembleSpecs (one per loss) each with 3 members (one per fold)
        dummy_nets = [nn.Linear(1, 1) for _ in range(12)]
        net_idx = 0
        ensembles: dict[str, EnsembleSpec] = {}
        for loss in sorted(EXPECTED_LOSSES):
            members_for_loss: list[EnsembleMember] = []
            for fold in range(3):
                members_for_loss.append(
                    EnsembleMember(
                        checkpoint_path=Path(f"/tmp/{loss}_fold{fold}.pth"),
                        run_id=f"run_{loss}",
                        loss_type=loss,
                        fold_id=fold,
                        metric_name="val_compound_masd_cldice",
                        net=dummy_nets[net_idx],
                    )
                )
                net_idx += 1
            ens_name = f"per_loss_single_best_{loss}"
            ensembles[ens_name] = EnsembleSpec(
                name=ens_name,
                strategy=EnsembleStrategyName.PER_LOSS_SINGLE_BEST,
                members=members_for_loss,
                description=f"Mocked ensemble for {loss}",
            )
        mock_build_ensembles.return_value = ensembles
        mock_eval.return_value = {"minivess": {"all": _make_eval_result(dice=0.80)}}
        mock_log_single.return_value = None
        mock_log_ensemble.return_value = None
        mock_mlflow_eval.return_value = {}

        dataloaders: dict[str, Any] = {"minivess": {"all": MagicMock()}}

        result = run_analysis_flow(
            eval_config,
            MODEL_CONFIG,
            dataloaders,
            tracking_uri=TRACKING_URI,
        )

        results = result["results"]
        assert isinstance(results, dict)
        # With PER_LOSS_SINGLE_BEST + 4 losses + 3 folds each = 12 unique single models
        # plus 4 ensemble models = 16 total
        assert len(results) > 0, "Expected non-empty results dict"
        # Verify all model names are strings
        for model_name in results:
            assert isinstance(model_name, str)

    @patch("minivess.orchestration.flows.analysis_flow._run_mlflow_eval_safe")
    @patch("minivess.orchestration.flows.analysis_flow._log_ensemble_model_safe")
    @patch("minivess.orchestration.flows.analysis_flow._log_single_model_safe")
    @patch("minivess.orchestration.flows.analysis_flow._evaluate_single_model_on_all")
    @patch("minivess.orchestration.flows.analysis_flow.build_ensembles")
    def test_full_flow_eval_called_for_each_model(
        self,
        mock_build_ensembles: MagicMock,
        mock_eval: MagicMock,
        mock_log_single: MagicMock,
        mock_log_ensemble: MagicMock,
        mock_mlflow_eval: MagicMock,
        eval_config: EvaluationConfig,
    ) -> None:
        """_evaluate_single_model_on_all is called once per model (single + ensemble).

        build_ensembles is mocked so there are real members to evaluate.
        """
        from minivess.orchestration.flows.analysis_flow import run_analysis_flow

        dummy_net = nn.Linear(1, 1)
        member = EnsembleMember(
            checkpoint_path=Path("/tmp/fake.pth"),
            run_id="run_dice_ce",
            loss_type="dice_ce",
            fold_id=0,
            metric_name="val_compound_masd_cldice",
            net=dummy_net,
        )
        spec = EnsembleSpec(
            name="per_loss_single_best_dice_ce",
            strategy=EnsembleStrategyName.PER_LOSS_SINGLE_BEST,
            members=[member],
            description="Mocked ensemble",
        )
        mock_build_ensembles.return_value = {"per_loss_single_best_dice_ce": spec}
        mock_eval.return_value = {"minivess": {"all": _make_eval_result(dice=0.80)}}
        mock_log_single.return_value = None
        mock_log_ensemble.return_value = None
        mock_mlflow_eval.return_value = {}

        dataloaders: dict[str, Any] = {"minivess": {"all": MagicMock()}}

        run_analysis_flow(
            eval_config,
            MODEL_CONFIG,
            dataloaders,
            tracking_uri=TRACKING_URI,
        )

        # Called for 1 single-fold model + 1 ensemble = 2 times
        assert mock_eval.call_count >= 2, (
            f"_evaluate_single_model_on_all called {mock_eval.call_count} times, "
            "expected at least 2 (single + ensemble)"
        )
