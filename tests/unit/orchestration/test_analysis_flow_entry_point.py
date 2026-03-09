"""Tests for analysis_flow Docker entry point (Phase -1, T-1.3).

RED phase: these tests FAIL against the current implementation which has
an unconditional `raise SystemExit` in __main__ and no _entry_point_from_env.
After T-1.4 GREEN they must all pass.

Rules applied:
  - Rule #16: ast.parse() + ast.walk() — no regex for code analysis
  - Rule #22: env vars are the single source of truth for runtime config values
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FLOW_FILE = Path("src/minivess/orchestration/flows/analysis_flow.py")


def _parse_flow_ast() -> ast.Module:
    """Parse analysis_flow.py into an AST module."""
    return ast.parse(_FLOW_FILE.read_text(encoding="utf-8"))


def _find_main_block(tree: ast.Module) -> ast.If | None:
    """Return the ``if __name__ == '__main__':`` block, or None."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        # Match: __name__ == "__main__"  or  "__main__" == __name__
        if isinstance(test, ast.Compare):
            left = test.left
            ops = test.ops
            comps = test.comparators
            if (
                len(ops) == 1
                and isinstance(ops[0], ast.Eq)
                and len(comps) == 1
                and (
                    (
                        isinstance(left, ast.Name)
                        and left.id == "__name__"
                        and isinstance(comps[0], ast.Constant)
                        and comps[0].value == "__main__"
                    )
                    or (
                        isinstance(left, ast.Constant)
                        and left.value == "__main__"
                        and isinstance(comps[0], ast.Name)
                        and comps[0].id == "__name__"
                    )
                )
            ):
                return node
    return None


# ---------------------------------------------------------------------------
# T-1.3.A  Structural tests (AST — no subprocess, no runtime)
# ---------------------------------------------------------------------------


class TestAnalysisFlowEntryPointStructure:
    """Structural AST checks for analysis_flow __main__ block and entry point."""

    def test_entry_point_from_env_exists(self) -> None:
        """analysis_flow module must expose _entry_point_from_env function.

        Will FAIL before fix: function does not exist.
        Must PASS after fix: function is defined at module level.
        """
        import minivess.orchestration.flows.analysis_flow as af

        assert hasattr(af, "_entry_point_from_env"), (
            "_entry_point_from_env not found in analysis_flow module. "
            "Implement it so the Docker entry point can call analysis without hardcoded params."
        )

    def test_main_block_does_not_raise_system_exit_unconditionally(self) -> None:
        """The __main__ block must NOT unconditionally raise SystemExit.

        Current (broken) code raises SystemExit in both branches of if/else.
        After fix: __main__ calls _entry_point_from_env() + run_analysis_flow().
        """
        tree = _parse_flow_ast()
        main_block = _find_main_block(tree)
        assert main_block is not None, "__main__ block not found in analysis_flow.py"

        # Check: there must NOT be a bare `raise SystemExit(...)` at the top level
        # of the __main__ body (unconditional raises block Docker invocation)
        unconditional_system_exits = []
        for stmt in main_block.body:
            if isinstance(stmt, ast.Raise) and stmt.exc is not None:
                exc = stmt.exc
                if isinstance(exc, ast.Call):
                    fn = exc.func
                    if (isinstance(fn, ast.Name) and fn.id == "SystemExit") or (
                        isinstance(fn, ast.Attribute) and fn.attr == "SystemExit"
                    ):
                        unconditional_system_exits.append(stmt)

        assert not unconditional_system_exits, (
            f"__main__ block has {len(unconditional_system_exits)} unconditional "
            "raise SystemExit statement(s). Replace with _entry_point_from_env() call."
        )

    def test_main_block_calls_entry_point_from_env(self) -> None:
        """The __main__ block must call _entry_point_from_env().

        Will FAIL before fix: __main__ raises SystemExit instead of calling the function.
        Must PASS after fix: __main__ calls _entry_point_from_env().
        """
        tree = _parse_flow_ast()
        main_block = _find_main_block(tree)
        assert main_block is not None, "__main__ block not found in analysis_flow.py"

        # Walk the __main__ body looking for a call to _entry_point_from_env
        calls_to_entry_point = [
            node
            for node in ast.walk(main_block)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_entry_point_from_env"
        ]

        assert calls_to_entry_point, (
            "_entry_point_from_env() is not called in the __main__ block. "
            "The Docker entry point must discover analysis params via env vars, not raise."
        )

    def test_entry_point_references_upstream_experiment_env_var(self) -> None:
        """_entry_point_from_env must reference 'UPSTREAM_EXPERIMENT' env var.

        Will FAIL before fix: function does not exist.
        Must PASS after fix: function reads os.environ.get('UPSTREAM_EXPERIMENT', ...).
        """
        tree = _parse_flow_ast()

        entry_func: ast.FunctionDef | None = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "_entry_point_from_env"
            ):
                entry_func = node
                break

        assert entry_func is not None, (
            "_entry_point_from_env function not found in analysis_flow.py"
        )

        upstream_refs = [
            node
            for node in ast.walk(entry_func)
            if isinstance(node, ast.Constant) and node.value == "UPSTREAM_EXPERIMENT"
        ]
        assert upstream_refs, (
            "'UPSTREAM_EXPERIMENT' not referenced inside _entry_point_from_env(). "
            "The function must read os.environ.get('UPSTREAM_EXPERIMENT') to discover "
            "the correct MLflow training experiment."
        )


# ---------------------------------------------------------------------------
# T-1.3.B  Behavioural tests
# ---------------------------------------------------------------------------


class TestAnalysisFlowEntryPointBehavioural:
    """Runtime behavioural checks for _entry_point_from_env."""

    def test_entry_point_raises_runtime_error_without_upstream_experiment(
        self, monkeypatch: object
    ) -> None:
        """_entry_point_from_env raises RuntimeError when UPSTREAM_EXPERIMENT is unset.

        Will FAIL before fix: function does not exist (AttributeError).
        Must PASS after fix: RuntimeError with useful message.
        """
        monkeypatch.delenv("UPSTREAM_EXPERIMENT", raising=False)  # type: ignore[attr-defined]

        import minivess.orchestration.flows.analysis_flow as af

        with pytest.raises(RuntimeError, match="UPSTREAM_EXPERIMENT"):
            af._entry_point_from_env()

    def test_entry_point_returns_dict_with_required_keys(
        self, monkeypatch: object, tmp_path: object
    ) -> None:
        """_entry_point_from_env returns dict with required keys when env var is set.

        Will FAIL before fix: function does not exist.
        Must PASS after fix: returns dict with eval_config, model_config_dict, etc.
        """
        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")  # type: ignore[attr-defined]
        monkeypatch.setenv("UPSTREAM_EXPERIMENT", "my_debug_exp")  # type: ignore[attr-defined]

        import minivess.orchestration.flows.analysis_flow as af

        # Stub out the upstream discovery to avoid needing a real MLflow server
        def _fake_find_upstream(
            tracking_uri: str, experiment_name: str, upstream_flow: str
        ) -> dict:
            return {"run_id": "fake-run-id", "status": "FINISHED"}

        monkeypatch.setattr(af, "find_upstream_safely", _fake_find_upstream)  # type: ignore[attr-defined]

        # Stub the heavy config-building helpers (they don't exist yet either)
        monkeypatch.setattr(  # type: ignore[attr-defined]
            af,
            "_load_config_from_mlflow",
            lambda run_id, tracking_uri: {},
        )
        monkeypatch.setattr(  # type: ignore[attr-defined]
            af,
            "_build_eval_config_from_dict",
            lambda d: object(),
        )
        monkeypatch.setattr(  # type: ignore[attr-defined]
            af,
            "_build_model_config_from_dict",
            lambda d: {},
        )
        monkeypatch.setattr(  # type: ignore[attr-defined]
            af,
            "_build_dataloaders_from_config",
            lambda d: {},
        )

        result = af._entry_point_from_env()

        assert result is not None, "_entry_point_from_env returned None"
        assert isinstance(result, dict), "_entry_point_from_env must return a dict"
        required_keys = {
            "eval_config",
            "model_config_dict",
            "dataloaders_dict",
            "upstream_training_run_id",
        }
        missing = required_keys - result.keys()
        assert not missing, (
            f"_entry_point_from_env result missing keys: {missing}. "
            f"Got: {set(result.keys())}"
        )
