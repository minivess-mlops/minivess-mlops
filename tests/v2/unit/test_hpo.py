"""Tests for Optuna HPO integration (Issue #43)."""

from __future__ import annotations

from unittest.mock import MagicMock

import optuna
import pytest

from minivess.config.models import TrainingConfig

# ---------------------------------------------------------------------------
# T1: SearchSpace dataclass
# ---------------------------------------------------------------------------


class TestSearchSpace:
    """Test HPO search space definition."""

    def test_search_space_default_ranges(self) -> None:
        """Default SearchSpace should have sensible DynUNet ranges."""
        from minivess.pipeline.hpo import SearchSpace

        space = SearchSpace()
        assert space.lr_low < space.lr_high
        assert space.weight_decay_low < space.weight_decay_high
        assert space.batch_size_low <= space.batch_size_high
        assert len(space.optimizers) >= 2

    def test_search_space_custom_ranges(self) -> None:
        """SearchSpace should accept custom parameter ranges."""
        from minivess.pipeline.hpo import SearchSpace

        space = SearchSpace(
            lr_low=1e-6,
            lr_high=1e-1,
            batch_size_low=2,
            batch_size_high=8,
        )
        assert space.lr_low == 1e-6
        assert space.lr_high == 1e-1
        assert space.batch_size_low == 2
        assert space.batch_size_high == 8

    def test_search_space_optimizers_list(self) -> None:
        """SearchSpace should list optimizer choices."""
        from minivess.pipeline.hpo import SearchSpace

        space = SearchSpace(optimizers=["adamw", "sgd", "adam"])
        assert "adamw" in space.optimizers
        assert "sgd" in space.optimizers
        assert len(space.optimizers) == 3


# ---------------------------------------------------------------------------
# T2: create_study factory
# ---------------------------------------------------------------------------


class TestCreateStudy:
    """Test Optuna study factory."""

    def test_create_study_returns_study(self) -> None:
        """create_study should return an optuna.Study."""
        from minivess.pipeline.hpo import create_study

        study = create_study(study_name="test_study")
        assert isinstance(study, optuna.Study)

    def test_create_study_direction_minimize(self) -> None:
        """Default direction should be minimize (we minimize val_loss)."""
        from minivess.pipeline.hpo import create_study

        study = create_study(study_name="test_minimize")
        assert study.direction == optuna.study.StudyDirection.MINIMIZE

    def test_create_study_with_pruner(self) -> None:
        """Study should have a MedianPruner by default."""
        from minivess.pipeline.hpo import create_study

        study = create_study(study_name="test_pruner")
        assert isinstance(study.pruner, optuna.pruners.MedianPruner)

    def test_create_study_custom_storage(self, tmp_path: object) -> None:
        """create_study should accept a storage URL."""
        from pathlib import Path

        from minivess.pipeline.hpo import create_study

        db_path = Path(str(tmp_path)) / "test_hpo.db"
        study = create_study(
            study_name="test_storage",
            storage=f"sqlite:///{db_path}",
        )
        assert isinstance(study, optuna.Study)


# ---------------------------------------------------------------------------
# T3: build_trial_config
# ---------------------------------------------------------------------------


class TestBuildTrialConfig:
    """Test converting Optuna trial to TrainingConfig."""

    def test_build_trial_config_returns_training_config(self) -> None:
        """build_trial_config should return a valid TrainingConfig."""
        from minivess.pipeline.hpo import SearchSpace, build_trial_config

        space = SearchSpace()
        study = optuna.create_study()
        trial = study.ask()

        config = build_trial_config(trial, space)
        assert isinstance(config, TrainingConfig)

    def test_build_trial_config_respects_search_space(self) -> None:
        """Generated config values must be within search space bounds."""
        from minivess.pipeline.hpo import SearchSpace, build_trial_config

        space = SearchSpace(
            lr_low=1e-4,
            lr_high=1e-2,
            batch_size_low=1,
            batch_size_high=4,
        )
        study = optuna.create_study()
        trial = study.ask()

        config = build_trial_config(trial, space)
        assert space.lr_low <= config.learning_rate <= space.lr_high
        assert space.batch_size_low <= config.batch_size <= space.batch_size_high
        assert config.optimizer in space.optimizers

    def test_build_trial_config_with_base_config(self) -> None:
        """build_trial_config should merge trial params with a base config."""
        from minivess.pipeline.hpo import SearchSpace, build_trial_config

        space = SearchSpace()
        base = TrainingConfig(max_epochs=50, seed=123)
        study = optuna.create_study()
        trial = study.ask()

        config = build_trial_config(trial, space, base_config=base)
        assert config.max_epochs == 50
        assert config.seed == 123

    def test_build_trial_config_weight_decay_range(self) -> None:
        """Weight decay should be within search space bounds."""
        from minivess.pipeline.hpo import SearchSpace, build_trial_config

        space = SearchSpace(weight_decay_low=1e-6, weight_decay_high=1e-3)
        study = optuna.create_study()
        trial = study.ask()

        config = build_trial_config(trial, space)
        assert space.weight_decay_low <= config.weight_decay <= space.weight_decay_high


# ---------------------------------------------------------------------------
# T4: run_hpo orchestrator
# ---------------------------------------------------------------------------


class TestRunHPO:
    """Test HPO orchestration."""

    def test_run_hpo_returns_best_params(self) -> None:
        """run_hpo should return a dict with best_params and best_value."""
        from minivess.pipeline.hpo import SearchSpace, run_hpo

        mock_objective = MagicMock(return_value=0.5)

        result = run_hpo(
            objective_fn=mock_objective,
            search_space=SearchSpace(),
            n_trials=3,
        )
        assert "best_params" in result
        assert "best_value" in result
        assert isinstance(result["best_value"], float)

    def test_run_hpo_calls_objective_n_times(self) -> None:
        """Objective function should be called n_trials times."""
        from minivess.pipeline.hpo import SearchSpace, run_hpo

        call_count = 0

        def counting_objective(trial: optuna.Trial) -> float:
            nonlocal call_count
            call_count += 1
            return 1.0 / (call_count + 1)

        run_hpo(
            objective_fn=counting_objective,
            search_space=SearchSpace(),
            n_trials=5,
        )
        assert call_count == 5

    def test_run_hpo_best_value_is_minimum(self) -> None:
        """Best value should be the minimum across trials."""
        from minivess.pipeline.hpo import SearchSpace, run_hpo

        values = iter([0.8, 0.3, 0.6])

        def fixed_objective(trial: optuna.Trial) -> float:
            return next(values)

        result = run_hpo(
            objective_fn=fixed_objective,
            search_space=SearchSpace(),
            n_trials=3,
        )
        assert result["best_value"] == pytest.approx(0.3)

    def test_run_hpo_returns_study(self) -> None:
        """run_hpo should include the study object in results."""
        from minivess.pipeline.hpo import SearchSpace, run_hpo

        result = run_hpo(
            objective_fn=lambda trial: 0.5,
            search_space=SearchSpace(),
            n_trials=2,
        )
        assert "study" in result
        assert isinstance(result["study"], optuna.Study)


# ---------------------------------------------------------------------------
# T5: make_objective factory
# ---------------------------------------------------------------------------


class TestMakeObjective:
    """Test objective function factory for SegmentationTrainer integration."""

    def test_make_objective_returns_callable(self) -> None:
        """make_objective should return a callable."""
        from minivess.pipeline.hpo import SearchSpace, make_objective

        mock_train_fn = MagicMock(return_value={"best_val_loss": 0.5})
        objective = make_objective(
            train_fn=mock_train_fn,
            search_space=SearchSpace(),
        )
        assert callable(objective)

    def test_make_objective_calls_train_fn(self) -> None:
        """The objective function should call train_fn with a TrainingConfig."""
        from minivess.pipeline.hpo import SearchSpace, make_objective

        mock_train_fn = MagicMock(return_value={"best_val_loss": 0.42})
        objective = make_objective(
            train_fn=mock_train_fn,
            search_space=SearchSpace(),
        )

        study = optuna.create_study()
        trial = study.ask()
        val = objective(trial)

        mock_train_fn.assert_called_once()
        assert val == pytest.approx(0.42)

    def test_make_objective_passes_config(self) -> None:
        """The objective should pass a valid TrainingConfig to train_fn."""
        from minivess.pipeline.hpo import SearchSpace, make_objective

        received_configs: list[TrainingConfig] = []

        def capture_train_fn(config: TrainingConfig) -> dict:
            received_configs.append(config)
            return {"best_val_loss": 0.5}

        objective = make_objective(
            train_fn=capture_train_fn,
            search_space=SearchSpace(),
        )

        study = optuna.create_study()
        trial = study.ask()
        objective(trial)

        assert len(received_configs) == 1
        assert isinstance(received_configs[0], TrainingConfig)
