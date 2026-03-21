"""Tests for Analysis Flow MLflow experiment and factorial tag schema (T4.1-T4.3).

Validates that:
- Analysis creates runs in minivess_evaluation (SEPARATE experiment)
- EXPERIMENT_EVALUATION constant is correct
- The 6-factor tag schema enables Biostatistics layered ANOVA discovery

References: XML plan T4.1-T4.3, synthesis Part 2.3-2.4.
"""

from __future__ import annotations


class TestAnalysisExperimentName:
    """T4.1: Analysis creates runs in minivess_evaluation (separate)."""

    def test_evaluation_experiment_is_separate_from_training(self) -> None:
        """EXPERIMENT_EVALUATION must differ from EXPERIMENT_TRAINING."""
        from minivess.orchestration.constants import (
            EXPERIMENT_EVALUATION,
            EXPERIMENT_TRAINING,
        )

        assert EXPERIMENT_EVALUATION != EXPERIMENT_TRAINING, (
            "Analysis must log to a SEPARATE experiment from training. "
            "EXPERIMENT_EVALUATION must not equal EXPERIMENT_TRAINING."
        )

    def test_evaluation_experiment_name(self) -> None:
        """EXPERIMENT_EVALUATION must be 'minivess_evaluation'."""
        from minivess.orchestration.constants import EXPERIMENT_EVALUATION

        assert EXPERIMENT_EVALUATION == "minivess_evaluation"

    def test_eval_config_default_matches_constant(self) -> None:
        """EvaluationConfig.mlflow_evaluation_experiment defaults to EXPERIMENT_EVALUATION."""
        from minivess.config.evaluation_config import EvaluationConfig
        from minivess.orchestration.constants import EXPERIMENT_EVALUATION

        config = EvaluationConfig()
        assert config.mlflow_evaluation_experiment == EXPERIMENT_EVALUATION

    def test_eval_config_training_experiment_matches_constant(self) -> None:
        """EvaluationConfig.mlflow_training_experiment for upstream discovery."""
        from minivess.config.evaluation_config import EvaluationConfig
        from minivess.orchestration.constants import EXPERIMENT_TRAINING

        config = EvaluationConfig()
        assert config.mlflow_training_experiment == EXPERIMENT_TRAINING


class TestAnalysisTagSchema:
    """T4.2: Verify tag schema supports 6-factor ANOVA discovery."""

    def test_flow_name_constant_is_analysis_flow(self) -> None:
        """FLOW_NAME_ANALYSIS must be 'analysis-flow'."""
        from minivess.orchestration.constants import FLOW_NAME_ANALYSIS

        assert FLOW_NAME_ANALYSIS == "analysis-flow"

    def test_ensemble_strategy_enum_has_all_values(self) -> None:
        """EnsembleStrategyName must contain all 4 named strategies."""
        from minivess.config.evaluation_config import EnsembleStrategyName

        expected = {
            "per_loss_single_best",
            "all_loss_single_best",
            "per_loss_all_best",
            "all_loss_all_best",
        }
        actual = {s.value for s in EnsembleStrategyName}
        assert actual == expected, (
            f"EnsembleStrategyName values mismatch. Expected {expected}, got {actual}"
        )

    def test_factorial_yaml_analysis_strategies_match_enum(self) -> None:
        """Production factorial YAML ensemble strategies must be valid EnsembleStrategyName values."""
        from pathlib import Path

        from minivess.config.evaluation_config import EnsembleStrategyName
        from minivess.config.factorial_config import parse_factorial_yaml

        yaml_path = Path("configs/factorial/paper_full.yaml")
        design = parse_factorial_yaml(yaml_path)
        levels = design.factor_levels()
        yaml_strategies = set(levels["ensemble_strategy"])
        enum_values = {s.value for s in EnsembleStrategyName}

        # All YAML strategies (except "none") must be valid enum values
        non_none = yaml_strategies - {"none"}
        invalid = non_none - enum_values
        assert not invalid, (
            f"Factorial YAML has strategies not in EnsembleStrategyName: {invalid}"
        )


class TestBiostatisticsDiscoverability:
    """T4.3: Biostatistics can discover analysis runs for layered ANOVA."""

    def test_six_factor_names_available(self) -> None:
        """All 6 experimental factors must be defined in debug factorial YAML."""
        from pathlib import Path

        from minivess.config.factorial_config import parse_factorial_yaml

        yaml_path = Path("configs/factorial/debug.yaml")
        design = parse_factorial_yaml(yaml_path)
        all_names = design.factor_names()

        # Layer A (training)
        assert "model_family" in all_names
        assert "loss_name" in all_names
        assert "aux_calibration" in all_names
        # Layer B (post-training)
        assert "method" in all_names  # post_training method
        assert "recalibration" in all_names
        # Layer C (analysis)
        assert "ensemble_strategy" in all_names

    def test_layered_anova_factor_groups(self) -> None:
        """Factors must be grouped by layer for layered ANOVA."""
        from pathlib import Path

        from minivess.config.factorial_config import parse_factorial_yaml

        yaml_path = Path("configs/factorial/debug.yaml")
        design = parse_factorial_yaml(yaml_path)

        # Layer A: 3-way ANOVA
        layer_a = design.factor_names(layer="training")
        assert len(layer_a) == 3

        # Layer B: extends to 5-way
        layer_b = design.factor_names(layer="post_training")
        assert len(layer_b) == 2

        # Layer C: full 6-factor
        layer_c = design.factor_names(layer="analysis")
        assert len(layer_c) == 1
        assert layer_c[0] == "ensemble_strategy"

        # Total: 3 + 2 + 1 = 6 experimental factors
        total = len(layer_a) + len(layer_b) + len(layer_c)
        assert total == 6
