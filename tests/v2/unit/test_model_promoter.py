"""Tests for ModelPromoter -- Phase 7 champion/challenger tagging.

Covers: promoter init, find_best_model (maximize & minimize), rank_models
(sorted order, multi-dataset averaging), generate_promotion_report (content
and markdown format), register_champion (MLflow mock, aliases), per-loss
aliases, and empty-results error handling.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from minivess.config.evaluation_config import EvaluationConfig, MetricDirection
from minivess.pipeline.ci import ConfidenceInterval
from minivess.pipeline.evaluation import FoldResult
from minivess.pipeline.evaluation_runner import EvaluationResult
from minivess.pipeline.model_promoter import ModelPromoter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ci(
    point_estimate: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> ConfidenceInterval:
    """Create a ConfidenceInterval with sensible defaults."""
    lo = lower if lower is not None else point_estimate - 0.05
    hi = upper if upper is not None else point_estimate + 0.05
    return ConfidenceInterval(
        point_estimate=point_estimate,
        lower=lo,
        upper=hi,
        confidence_level=0.95,
        method="percentile_bootstrap",
    )


def _make_fold_result(
    dsc: float = 0.8,
    centreline_dsc: float = 0.7,
    measured_masd: float = 2.0,
) -> FoldResult:
    """Build a FoldResult with specified metric point estimates."""
    per_vol: dict[str, list[float]] = {
        "dsc": [dsc],
        "centreline_dsc": [centreline_dsc],
        "measured_masd": [measured_masd],
    }
    aggregated: dict[str, ConfidenceInterval] = {
        "dsc": _make_ci(dsc),
        "centreline_dsc": _make_ci(centreline_dsc),
        "measured_masd": _make_ci(
            measured_masd, lower=measured_masd - 0.5, upper=measured_masd + 0.5
        ),
    }
    return FoldResult(per_volume_metrics=per_vol, aggregated=aggregated)


def _make_eval_result(
    model_name: str,
    dataset_name: str = "minivess",
    subset_name: str = "all",
    dsc: float = 0.8,
    centreline_dsc: float = 0.7,
    measured_masd: float = 2.0,
) -> EvaluationResult:
    """Build an EvaluationResult with specified metric values."""
    fold = _make_fold_result(
        dsc=dsc, centreline_dsc=centreline_dsc, measured_masd=measured_masd
    )
    return EvaluationResult(
        model_name=model_name,
        dataset_name=dataset_name,
        subset_name=subset_name,
        fold_result=fold,
        predictions_dir=None,
        uncertainty_maps_dir=None,
    )


def _build_results_dict(
    model_results: dict[str, dict[str, dict[str, EvaluationResult]]],
) -> dict[str, dict[str, dict[str, EvaluationResult]]]:
    """Passthrough helper for clarity: model_name -> dataset -> subset -> result."""
    return model_results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPromoterInit:
    """ModelPromoter initialisation."""

    def test_promoter_init_default_config(self) -> None:
        config = EvaluationConfig()
        promoter = ModelPromoter(config)
        assert promoter.eval_config is config

    def test_promoter_init_custom_metric(self) -> None:
        config = EvaluationConfig(
            primary_metric="val_dice",
            primary_metric_direction=MetricDirection.MAXIMIZE,
        )
        promoter = ModelPromoter(config)
        assert promoter.eval_config.primary_metric == "val_dice"


class TestFindBestModel:
    """find_best_model selects the correct model by primary metric."""

    def test_find_best_model_maximize(self) -> None:
        """When maximizing, model with highest compound metric wins."""
        config = EvaluationConfig(
            primary_metric="val_compound_masd_cldice",
            primary_metric_direction=MetricDirection.MAXIMIZE,
        )
        promoter = ModelPromoter(config)

        # model_a: dsc=0.8, cldice=0.7, masd=2.0 => compound ~0.83
        # model_b: dsc=0.9, cldice=0.9, masd=0.5 => compound ~0.945
        results: dict[str, dict[str, dict[str, EvaluationResult]]] = {
            "model_a": {
                "minivess": {
                    "all": _make_eval_result(
                        "model_a", dsc=0.8, centreline_dsc=0.7, measured_masd=2.0
                    ),
                },
            },
            "model_b": {
                "minivess": {
                    "all": _make_eval_result(
                        "model_b", dsc=0.9, centreline_dsc=0.9, measured_masd=0.5
                    ),
                },
            },
        }

        best_name, best_value = promoter.find_best_model(results)
        assert best_name == "model_b"
        assert best_value > 0.9  # compound of model_b should be > 0.9

    def test_find_best_model_minimize(self) -> None:
        """When minimizing (e.g. measured_masd), model with lowest metric wins."""
        config = EvaluationConfig(
            primary_metric="measured_masd",
            primary_metric_direction=MetricDirection.MINIMIZE,
        )
        promoter = ModelPromoter(config)

        results: dict[str, dict[str, dict[str, EvaluationResult]]] = {
            "model_a": {
                "minivess": {
                    "all": _make_eval_result("model_a", measured_masd=5.0),
                },
            },
            "model_b": {
                "minivess": {
                    "all": _make_eval_result("model_b", measured_masd=1.5),
                },
            },
        }

        best_name, best_value = promoter.find_best_model(results)
        assert best_name == "model_b"
        assert best_value < 2.0

    def test_find_best_model_uses_dsc_when_configured(self) -> None:
        """When primary metric is dsc, selects by dsc point estimate."""
        config = EvaluationConfig(
            primary_metric="dsc",
            primary_metric_direction=MetricDirection.MAXIMIZE,
        )
        promoter = ModelPromoter(config)

        results: dict[str, dict[str, dict[str, EvaluationResult]]] = {
            "model_low": {
                "ds": {"all": _make_eval_result("model_low", dsc=0.6)},
            },
            "model_high": {
                "ds": {"all": _make_eval_result("model_high", dsc=0.95)},
            },
        }

        best_name, _ = promoter.find_best_model(results)
        assert best_name == "model_high"


class TestRankModels:
    """rank_models returns models sorted by primary metric."""

    def test_rank_models_sorted_descending_maximize(self) -> None:
        config = EvaluationConfig(
            primary_metric="dsc",
            primary_metric_direction=MetricDirection.MAXIMIZE,
        )
        promoter = ModelPromoter(config)

        results: dict[str, dict[str, dict[str, EvaluationResult]]] = {
            "worst": {"ds": {"all": _make_eval_result("worst", dsc=0.5)}},
            "mid": {"ds": {"all": _make_eval_result("mid", dsc=0.7)}},
            "best": {"ds": {"all": _make_eval_result("best", dsc=0.95)}},
        }

        rankings = promoter.rank_models(results)
        assert len(rankings) == 3
        assert rankings[0][0] == "best"
        assert rankings[1][0] == "mid"
        assert rankings[2][0] == "worst"
        # Values should be descending
        assert rankings[0][1] > rankings[1][1] > rankings[2][1]

    def test_rank_models_sorted_ascending_minimize(self) -> None:
        """When minimizing, best (lowest value) comes first."""
        config = EvaluationConfig(
            primary_metric="measured_masd",
            primary_metric_direction=MetricDirection.MINIMIZE,
        )
        promoter = ModelPromoter(config)

        results: dict[str, dict[str, dict[str, EvaluationResult]]] = {
            "worst": {"ds": {"all": _make_eval_result("worst", measured_masd=10.0)}},
            "best": {"ds": {"all": _make_eval_result("best", measured_masd=0.5)}},
        }

        rankings = promoter.rank_models(results)
        assert rankings[0][0] == "best"
        assert rankings[1][0] == "worst"

    def test_rank_models_with_multiple_datasets(self) -> None:
        """Average across datasets and subsets to produce final ranking."""
        config = EvaluationConfig(
            primary_metric="dsc",
            primary_metric_direction=MetricDirection.MAXIMIZE,
        )
        promoter = ModelPromoter(config)

        # model_a: dsc=0.8 on ds_1, dsc=0.6 on ds_2 => mean 0.7
        # model_b: dsc=0.75 on ds_1, dsc=0.75 on ds_2 => mean 0.75
        results: dict[str, dict[str, dict[str, EvaluationResult]]] = {
            "model_a": {
                "ds_1": {
                    "all": _make_eval_result("model_a", dataset_name="ds_1", dsc=0.8)
                },
                "ds_2": {
                    "all": _make_eval_result("model_a", dataset_name="ds_2", dsc=0.6)
                },
            },
            "model_b": {
                "ds_1": {
                    "all": _make_eval_result("model_b", dataset_name="ds_1", dsc=0.75)
                },
                "ds_2": {
                    "all": _make_eval_result("model_b", dataset_name="ds_2", dsc=0.75)
                },
            },
        }

        rankings = promoter.rank_models(results)
        assert rankings[0][0] == "model_b"  # mean 0.75 > 0.7
        assert abs(rankings[0][1] - 0.75) < 1e-6
        assert abs(rankings[1][1] - 0.70) < 1e-6


class TestGeneratePromotionReport:
    """generate_promotion_report produces valid markdown."""

    def _make_rankings(self) -> list[tuple[str, float]]:
        return [
            ("ensemble_all_loss_all_best", 0.8542),
            ("ensemble_per_loss_dice_ce", 0.8321),
            ("single_fold0_dice_ce", 0.7890),
        ]

    def test_report_contains_champion(self) -> None:
        config = EvaluationConfig()
        promoter = ModelPromoter(config)
        rankings = self._make_rankings()
        report = promoter.generate_promotion_report(
            rankings, champion_name="ensemble_all_loss_all_best"
        )
        assert "Champion" in report
        assert "ensemble_all_loss_all_best" in report

    def test_report_contains_challenger(self) -> None:
        config = EvaluationConfig()
        promoter = ModelPromoter(config)
        rankings = self._make_rankings()
        report = promoter.generate_promotion_report(
            rankings, champion_name="ensemble_all_loss_all_best"
        )
        assert "Challenger" in report
        assert "ensemble_per_loss_dice_ce" in report

    def test_report_markdown_format(self) -> None:
        config = EvaluationConfig()
        promoter = ModelPromoter(config)
        rankings = self._make_rankings()
        report = promoter.generate_promotion_report(
            rankings, champion_name="ensemble_all_loss_all_best"
        )
        # Should have table header separators
        assert "---" in report
        # Should have table rows with pipes
        assert "| 1 |" in report or "|1|" in report or "| 1|" in report
        # Should have header
        assert "Model Promotion Report" in report

    def test_report_includes_all_models(self) -> None:
        config = EvaluationConfig()
        promoter = ModelPromoter(config)
        rankings = self._make_rankings()
        report = promoter.generate_promotion_report(
            rankings, champion_name="ensemble_all_loss_all_best"
        )
        for name, _ in rankings:
            assert name in report


class TestRegisterChampion:
    """register_champion interacts correctly with MLflow."""

    @patch("minivess.pipeline.model_promoter.MlflowClient")
    @patch("minivess.pipeline.model_promoter.mlflow")
    def test_register_champion_calls_mlflow(
        self, mock_mlflow: MagicMock, mock_client_cls: MagicMock
    ) -> None:
        config = EvaluationConfig()
        promoter = ModelPromoter(config)

        # Mock register_model return value
        mock_model_version = MagicMock()
        mock_model_version.version = "1"
        mock_mlflow.register_model.return_value = mock_model_version

        result = promoter.register_champion(
            "best_model",
            run_id="run-123",
            registry_name="MiniVess-Segmentor",
            environment="staging",
            tracking_uri="mlruns",
        )

        mock_mlflow.register_model.assert_called_once()
        assert isinstance(result, dict)
        assert "model_name" in result

    @patch("minivess.pipeline.model_promoter.MlflowClient")
    @patch("minivess.pipeline.model_promoter.mlflow")
    def test_register_champion_sets_aliases(
        self, mock_mlflow: MagicMock, mock_client_cls: MagicMock
    ) -> None:
        config = EvaluationConfig()
        promoter = ModelPromoter(config)

        mock_model_version = MagicMock()
        mock_model_version.version = "3"
        mock_mlflow.register_model.return_value = mock_model_version

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        promoter.register_champion(
            "best_model",
            run_id="run-456",
            registry_name="MiniVess-Segmentor",
            environment="staging",
            tracking_uri="mlruns",
        )

        # Should set both "staging-champion" and "champion" aliases
        alias_calls = mock_client.set_registered_model_alias.call_args_list
        aliases_set = {call[0][1] for call in alias_calls}
        assert "staging-champion" in aliases_set
        assert "champion" in aliases_set

    @patch("minivess.pipeline.model_promoter.MlflowClient")
    @patch("minivess.pipeline.model_promoter.mlflow")
    def test_register_champion_prod_environment(
        self, mock_mlflow: MagicMock, mock_client_cls: MagicMock
    ) -> None:
        config = EvaluationConfig()
        promoter = ModelPromoter(config)

        mock_model_version = MagicMock()
        mock_model_version.version = "5"
        mock_mlflow.register_model.return_value = mock_model_version
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        promoter.register_champion(
            "best_model",
            run_id="run-789",
            registry_name="MiniVess-Segmentor",
            environment="production",
            tracking_uri="mlruns",
        )

        alias_calls = mock_client.set_registered_model_alias.call_args_list
        aliases_set = {call[0][1] for call in alias_calls}
        assert "production-champion" in aliases_set
        assert "champion" in aliases_set

    @patch("minivess.pipeline.model_promoter.MlflowClient")
    @patch("minivess.pipeline.model_promoter.mlflow")
    def test_register_champion_uses_default_registry_name(
        self, mock_mlflow: MagicMock, mock_client_cls: MagicMock
    ) -> None:
        config = EvaluationConfig(model_registry_name="Custom-Segmentor")
        promoter = ModelPromoter(config)

        mock_model_version = MagicMock()
        mock_model_version.version = "1"
        mock_mlflow.register_model.return_value = mock_model_version

        promoter.register_champion(
            "best_model",
            run_id="run-abc",
            # no registry_name => uses config default
        )

        # Should use config's model_registry_name
        call_args = mock_mlflow.register_model.call_args
        assert "Custom-Segmentor" in str(call_args)


class TestSetPerLossAliases:
    """set_per_loss_aliases creates best-{loss} aliases."""

    @patch("minivess.pipeline.model_promoter.MlflowClient")
    @patch("minivess.pipeline.model_promoter.mlflow")
    def test_sets_per_loss_aliases(
        self, mock_mlflow: MagicMock, mock_client_cls: MagicMock
    ) -> None:
        config = EvaluationConfig()
        promoter = ModelPromoter(config)

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Mock search_model_versions to return a fake version list
        mock_version = MagicMock()
        mock_version.version = "2"
        mock_client.search_model_versions.return_value = [mock_version]

        loss_rankings: dict[str, list[tuple[str, float]]] = {
            "dice_ce": [("fold0_dice_ce", 0.85), ("fold1_dice_ce", 0.80)],
            "cldice": [("fold0_cldice", 0.90), ("fold1_cldice", 0.88)],
        }

        promoter.set_per_loss_aliases(
            loss_rankings,
            tracking_uri="mlruns",
        )

        # Should have called set_registered_model_alias for each loss
        alias_calls = mock_client.set_registered_model_alias.call_args_list
        aliases_set = {call[0][1] for call in alias_calls}
        assert "best-dice_ce" in aliases_set
        assert "best-cldice" in aliases_set


class TestEmptyResults:
    """Empty results are handled gracefully."""

    def test_empty_results_raises(self) -> None:
        config = EvaluationConfig()
        promoter = ModelPromoter(config)
        empty_results: dict[str, dict[str, dict[str, EvaluationResult]]] = {}

        import pytest

        with pytest.raises(ValueError, match="[Nn]o.*results|[Ee]mpty"):
            promoter.find_best_model(empty_results)

    def test_empty_results_rank_raises(self) -> None:
        config = EvaluationConfig()
        promoter = ModelPromoter(config)
        empty_results: dict[str, dict[str, dict[str, EvaluationResult]]] = {}

        import pytest

        with pytest.raises(ValueError, match="[Nn]o.*results|[Ee]mpty"):
            promoter.rank_models(empty_results)
