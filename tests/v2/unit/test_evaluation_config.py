from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from minivess.config.evaluation_config import (
    EnsembleStrategyName,
    EvaluationConfig,
    MetricDirection,
)


class TestDefaultEvaluationConfig:
    """Default construction produces a valid, sensible config."""

    def test_default_evaluation_config_valid(self) -> None:
        """Default EvaluationConfig should construct without errors."""
        cfg = EvaluationConfig()
        assert isinstance(cfg, EvaluationConfig)
        assert cfg.mlflow_evaluation_experiment == "minivess_evaluation"
        assert cfg.mlflow_training_experiment == "minivess_training"
        assert cfg.include_expensive_metrics is True
        assert cfg.bootstrap_n_resamples == 10_000
        assert cfg.confidence_level == 0.95
        assert cfg.model_registry_name == "MiniVess-Segmentor"
        assert cfg.datasets_config is None

    def test_primary_metric_default_is_compound(self) -> None:
        """Primary metric should default to the compound MASD+clDice metric."""
        cfg = EvaluationConfig()
        assert cfg.primary_metric == "val_compound_masd_cldice"

    def test_primary_metric_direction_default_maximize(self) -> None:
        """Compound metric is higher-is-better, so direction must be maximize."""
        cfg = EvaluationConfig()
        assert cfg.primary_metric_direction == MetricDirection.MAXIMIZE

    def test_ensemble_strategies_default_all_four(self) -> None:
        """All four ensemble strategy names should be present by default."""
        cfg = EvaluationConfig()
        assert len(cfg.ensemble_strategies) == 4
        expected = {
            EnsembleStrategyName.PER_LOSS_SINGLE_BEST,
            EnsembleStrategyName.ALL_LOSS_SINGLE_BEST,
            EnsembleStrategyName.PER_LOSS_ALL_BEST,
            EnsembleStrategyName.ALL_LOSS_ALL_BEST,
        }
        assert set(cfg.ensemble_strategies) == expected


class TestLoadFromYaml:
    """EvaluationConfig should be loadable from a YAML file."""

    def test_load_from_yaml(self, tmp_path) -> None:
        """Round-trip: write YAML, load it, check values match."""
        yaml_content = {
            "primary_metric": "val_dice",
            "primary_metric_direction": "maximize",
            "mlflow_evaluation_experiment": "test_eval",
            "mlflow_training_experiment": "test_train",
            "include_expensive_metrics": False,
            "bootstrap_n_resamples": 5000,
            "confidence_level": 0.90,
            "ensemble_strategies": [
                "per_loss_single_best",
                "all_loss_single_best",
            ],
            "model_registry_name": "TestModel",
        }
        yaml_file = tmp_path / "eval_config.yaml"
        yaml_file.write_text(
            yaml.dump(yaml_content, default_flow_style=False),
            encoding="utf-8",
        )

        raw = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        cfg = EvaluationConfig.model_validate(raw)

        assert cfg.primary_metric == "val_dice"
        assert cfg.primary_metric_direction == MetricDirection.MAXIMIZE
        assert cfg.mlflow_evaluation_experiment == "test_eval"
        assert cfg.include_expensive_metrics is False
        assert cfg.bootstrap_n_resamples == 5000
        assert cfg.confidence_level == 0.90
        assert len(cfg.ensemble_strategies) == 2
        assert cfg.model_registry_name == "TestModel"


class TestCustomPrimaryMetric:
    """Users can override the primary metric to any string."""

    def test_custom_primary_metric(self) -> None:
        cfg = EvaluationConfig(
            primary_metric="val_masd",
            primary_metric_direction=MetricDirection.MINIMIZE,
        )
        assert cfg.primary_metric == "val_masd"
        assert cfg.primary_metric_direction == MetricDirection.MINIMIZE


class TestInvalidDirection:
    """Bogus direction strings must be rejected."""

    def test_invalid_direction_raises(self) -> None:
        with pytest.raises(ValidationError, match="(?i)direction"):
            EvaluationConfig(primary_metric_direction="upward")  # type: ignore[arg-type]


class TestCheckpointFilename:
    """checkpoint_filename() derives a .pth name from primary_metric."""

    def test_checkpoint_filename_from_primary_metric(self) -> None:
        cfg = EvaluationConfig()  # default primary_metric
        assert cfg.checkpoint_filename() == "best_val_compound_masd_cldice.pth"

    def test_checkpoint_filename_custom_metric(self) -> None:
        cfg = EvaluationConfig(primary_metric="val_f1_foreground")
        assert cfg.checkpoint_filename() == "best_val_f1_foreground.pth"


class TestEvaluationExperimentName:
    """Default MLflow experiment name for evaluation."""

    def test_evaluation_experiment_name_default(self) -> None:
        cfg = EvaluationConfig()
        assert cfg.mlflow_evaluation_experiment == "minivess_evaluation"


class TestSerializationRoundtrip:
    """model_dump / model_validate round-trip preserves all fields."""

    def test_serialization_roundtrip(self) -> None:
        cfg = EvaluationConfig(
            primary_metric="val_cldice",
            primary_metric_direction=MetricDirection.MAXIMIZE,
            bootstrap_n_resamples=2000,
            confidence_level=0.99,
            ensemble_strategies=[
                EnsembleStrategyName.PER_LOSS_SINGLE_BEST,
                EnsembleStrategyName.ALL_LOSS_ALL_BEST,
            ],
            model_registry_name="Custom-Segmentor",
        )
        dumped = cfg.model_dump()
        restored = EvaluationConfig.model_validate(dumped)

        assert restored.primary_metric == cfg.primary_metric
        assert restored.primary_metric_direction == cfg.primary_metric_direction
        assert restored.bootstrap_n_resamples == cfg.bootstrap_n_resamples
        assert restored.confidence_level == cfg.confidence_level
        assert restored.ensemble_strategies == cfg.ensemble_strategies
        assert restored.model_registry_name == cfg.model_registry_name

    def test_json_roundtrip(self) -> None:
        cfg = EvaluationConfig()
        json_str = cfg.model_dump_json()
        restored = EvaluationConfig.model_validate_json(json_str)
        assert restored == cfg


class TestIsBetter:
    """is_better() respects the metric direction."""

    def test_maximize_higher_is_better(self) -> None:
        cfg = EvaluationConfig(primary_metric_direction=MetricDirection.MAXIMIZE)
        assert cfg.is_better(current=0.85, best=0.80) is True
        assert cfg.is_better(current=0.75, best=0.80) is False

    def test_minimize_lower_is_better(self) -> None:
        cfg = EvaluationConfig(primary_metric_direction=MetricDirection.MINIMIZE)
        assert cfg.is_better(current=0.10, best=0.15) is True
        assert cfg.is_better(current=0.20, best=0.15) is False

    def test_equal_values_not_better(self) -> None:
        cfg_max = EvaluationConfig(primary_metric_direction=MetricDirection.MAXIMIZE)
        cfg_min = EvaluationConfig(primary_metric_direction=MetricDirection.MINIMIZE)
        assert cfg_max.is_better(current=0.5, best=0.5) is False
        assert cfg_min.is_better(current=0.5, best=0.5) is False


class TestFieldValidation:
    """Pydantic field constraints should be enforced."""

    def test_bootstrap_below_minimum_raises(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationConfig(bootstrap_n_resamples=50)

    def test_confidence_level_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationConfig(confidence_level=0.0)

    def test_confidence_level_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationConfig(confidence_level=1.0)
