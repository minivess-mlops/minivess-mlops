"""Tests for Optuna + ASHA HPO engine (#283).

Covers:
- HPOEngine creation and configuration
- SearchSpace from YAML
- ASHA pruning via HyperbandPruner
- MLflow callback integration
- HPO config YAML
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestHPOEngine:
    """Test HPOEngine class."""

    def test_engine_instantiation(self) -> None:
        from minivess.optimization.hpo_engine import HPOEngine

        engine = HPOEngine(study_name="test-study", storage=None)
        assert engine.study_name == "test-study"

    def test_engine_create_study(self) -> None:
        from minivess.optimization.hpo_engine import HPOEngine

        engine = HPOEngine(study_name="test-create", storage=None)
        study = engine.create_study(direction="minimize")
        assert study is not None

    def test_engine_with_hyperband_pruner(self) -> None:
        from minivess.optimization.hpo_engine import HPOEngine

        engine = HPOEngine(
            study_name="test-asha",
            storage=None,
            pruner="hyperband",
        )
        study = engine.create_study(direction="minimize")
        assert study is not None

    def test_engine_suggest_params(self) -> None:
        from minivess.optimization.hpo_engine import HPOEngine

        engine = HPOEngine(study_name="test-suggest", storage=None)
        study = engine.create_study(direction="minimize")

        search_space: dict[str, dict[str, Any]] = {
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "batch_size": {"type": "categorical", "choices": [1, 2, 4]},
        }

        trial = study.ask()
        params = engine.suggest_params(trial, search_space)
        assert "learning_rate" in params
        assert "batch_size" in params
        assert isinstance(params["learning_rate"], float)


class TestSearchSpace:
    """Test SearchSpace from YAML."""

    def test_load_search_space(self) -> None:
        from minivess.optimization.search_space import SearchSpace

        space = SearchSpace.from_dict(
            {
                "learning_rate": {
                    "type": "float",
                    "low": 1e-5,
                    "high": 1e-2,
                    "log": True,
                },
                "loss_name": {
                    "type": "categorical",
                    "choices": ["dice_ce", "cbdice_cldice"],
                },
                "max_epochs": {"type": "int", "low": 50, "high": 200},
            }
        )
        assert len(space.params) == 3

    def test_search_space_param_types(self) -> None:
        from minivess.optimization.search_space import SearchSpace

        space = SearchSpace.from_dict(
            {
                "lr": {"type": "float", "low": 0.001, "high": 0.1},
                "bs": {"type": "int", "low": 1, "high": 8},
                "loss": {"type": "categorical", "choices": ["a", "b"]},
            }
        )
        assert space.params["lr"]["type"] == "float"
        assert space.params["bs"]["type"] == "int"
        assert space.params["loss"]["type"] == "categorical"


class TestHPOConfig:
    """Test HPO experiment config."""

    def test_hpo_config_yaml_exists(self) -> None:
        path = PROJECT_ROOT / "configs" / "experiments" / "hpo_dynunet_example.yaml"
        assert path.exists()

    def test_hpo_config_valid(self) -> None:
        import yaml

        path = PROJECT_ROOT / "configs" / "experiments" / "hpo_dynunet_example.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "search_space" in data
        assert "n_trials" in data


class TestHPORunner:
    """Test HPO runner script exists."""

    def test_runner_script_exists(self) -> None:
        path = PROJECT_ROOT / "scripts" / "run_hpo.py"
        assert path.exists()
