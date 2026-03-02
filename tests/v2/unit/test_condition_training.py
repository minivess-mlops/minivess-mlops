"""Tests for condition-aware training integration (T8a — topology real-data plan)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch


class TestRunFoldSafeWithCondition:
    """Test that run_fold_safe accepts and uses condition config."""

    def test_accepts_condition_param(self) -> None:
        """run_fold_safe accepts condition kwarg without error."""
        import sys

        sys.path.insert(0, "scripts")
        # Verify function signature accepts condition
        import inspect

        import train_monitored

        sig = inspect.signature(train_monitored.run_fold_safe)
        assert "condition" in sig.parameters

    def test_condition_none_uses_standard_path(self) -> None:
        """When condition=None, uses standard model/loss build path."""
        import sys

        sys.path.insert(0, "scripts")
        # Verify the function exists and can be called with condition=None
        import inspect

        import train_monitored

        sig = inspect.signature(train_monitored.run_fold_safe)
        param = sig.parameters["condition"]
        assert param.default is None


class TestRunConditionsMode:
    """Test conditions-mode execution in run_experiment.py."""

    def test_run_conditions_mode_exists(self) -> None:
        """run_conditions_mode function exists in run_experiment."""
        import sys

        sys.path.insert(0, "scripts")
        from run_experiment import run_conditions_mode

        assert callable(run_conditions_mode)

    def test_run_conditions_mode_iterates_conditions(self) -> None:
        """run_conditions_mode iterates over conditions in config."""
        import sys

        sys.path.insert(0, "scripts")
        from run_experiment import run_conditions_mode

        config: dict[str, Any] = {
            "experiment_name": "test",
            "model_family": "dynunet",
            "loss": "cbdice_cldice",
            "conditions": [
                {"name": "baseline", "wrappers": [], "d2c_enabled": False},
                {
                    "name": "d2c_only",
                    "wrappers": [],
                    "d2c_enabled": True,
                    "d2c_probability": 0.3,
                },
            ],
            "data_dir": "data/raw/minivess",
            "num_folds": 3,
            "max_epochs": 6,
            "seed": 42,
        }

        # Mock the actual training to avoid running real training
        with patch("run_experiment._run_single_condition") as mock_run:
            mock_run.return_value = {"status": "completed"}
            result = run_conditions_mode(config)

        # Should have been called once per condition
        assert mock_run.call_count == 2
        assert result["completed"] + result["failed"] == 2
