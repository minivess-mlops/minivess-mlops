"""Tests for T-08: hpo_flow.py — Prefect flow wrapping HPOEngine.

Uses ast.parse() for source inspection — NO regex (CLAUDE.md Rule #16).
training_flow() is mocked so no real GPU training is needed for unit tests.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

_HPO_FLOW_SRC = Path("src/minivess/orchestration/flows/hpo_flow.py")

# Default search space for tests: single float param to minimize
_TEST_SEARCH_SPACE = {
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True}
}


def _mock_training_flow(**kwargs):
    """Return a fake TrainingFlowResult so unit tests skip real GPU training."""
    from minivess.orchestration.flows.train_flow import TrainingFlowResult

    lr = kwargs.get("learning_rate", 1e-3)
    # Simple convex objective: distance from 1e-3
    fake_val_loss = abs(lr - 1e-3) + 0.1
    return TrainingFlowResult(
        status="completed",
        fold_results=[{"best_val_loss": fake_val_loss, "final_epoch": 1}],
    )


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


class TestHpoFlowStructure:
    def test_hpo_flow_module_exists(self) -> None:
        """hpo_flow.py must exist in the flows directory."""
        assert _HPO_FLOW_SRC.exists(), (
            "src/minivess/orchestration/flows/hpo_flow.py does not exist. "
            "Create a Prefect flow wrapping HPOEngine."
        )

    def test_hpo_flow_decorated(self) -> None:
        """hpo_flow function must be importable and callable (has @flow decorator)."""
        from minivess.orchestration.flows import (
            hpo_flow as hpo_module,  # type: ignore[import]
        )

        assert callable(hpo_module.hpo_flow), "hpo_flow must be callable with @flow"

    def test_hpo_flow_no_subprocess_script(self) -> None:
        """hpo_flow.py must not invoke scripts/run_hpo.py via subprocess."""
        source = _HPO_FLOW_SRC.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert "scripts/run_hpo.py" not in node.value, (
                    f"hpo_flow.py references scripts/run_hpo.py at line {node.lineno}. "
                    "Use HPOEngine directly instead of subprocess invocation."
                )


# ---------------------------------------------------------------------------
# Functional tests (with mocked training_flow)
# ---------------------------------------------------------------------------


class TestHpoFlowFunctional:
    def test_hpo_flow_returns_dict_with_best_params(
        self, monkeypatch, tmp_path
    ) -> None:
        """hpo_flow() must return a dict with 'best_params' key."""
        monkeypatch.setenv("PREFECT_DISABLED", "1")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))

        with patch(
            "minivess.orchestration.flows.hpo_flow.training_flow",
            side_effect=_mock_training_flow,
        ):
            from minivess.orchestration.flows import (
                hpo_flow as hpo_module,  # type: ignore[import]
            )

            result = hpo_module.hpo_flow(
                n_trials=1,
                study_name="test_hpo",
                sampler="tpe",
                search_space=_TEST_SEARCH_SPACE,
            )

        assert isinstance(result, dict), (
            f"hpo_flow() returned {type(result)}, expected dict"
        )
        assert "best_params" in result, (
            f"hpo_flow() result missing 'best_params'. Got: {list(result.keys())}"
        )

    def test_hpo_flow_returns_best_value(self, monkeypatch, tmp_path) -> None:
        """hpo_flow() result must contain 'best_value' key."""
        monkeypatch.setenv("PREFECT_DISABLED", "1")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))

        with patch(
            "minivess.orchestration.flows.hpo_flow.training_flow",
            side_effect=_mock_training_flow,
        ):
            from minivess.orchestration.flows import (
                hpo_flow as hpo_module,  # type: ignore[import]
            )

            result = hpo_module.hpo_flow(
                n_trials=1,
                study_name="test_hpo_val",
                sampler="tpe",
                search_space=_TEST_SEARCH_SPACE,
            )

        assert "best_value" in result, (
            f"hpo_flow() result missing 'best_value'. Got: {list(result.keys())}"
        )

    def test_hpo_flow_returns_n_trials(self, monkeypatch, tmp_path) -> None:
        """hpo_flow() result must contain 'n_trials' key."""
        monkeypatch.setenv("PREFECT_DISABLED", "1")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))

        with patch(
            "minivess.orchestration.flows.hpo_flow.training_flow",
            side_effect=_mock_training_flow,
        ):
            from minivess.orchestration.flows import (
                hpo_flow as hpo_module,  # type: ignore[import]
            )

            result = hpo_module.hpo_flow(
                n_trials=2,
                study_name="test_hpo_trials",
                sampler="tpe",
                search_space=_TEST_SEARCH_SPACE,
            )

        assert "n_trials" in result, (
            f"hpo_flow() result missing 'n_trials'. Got: {list(result.keys())}"
        )
        assert result["n_trials"] == 2
