"""Tests that post_training_flow reads UPSTREAM_EXPERIMENT from env (Phase -1, T-1.1).

RED phase: these tests FAIL against the current implementation which hardcodes
'minivess_training'. After the GREEN fix (T-1.2) they must all pass.

Rules applied:
  - Rule #16: ast.parse() + ast.walk() — no regex for code analysis
  - Rule #22: env vars are the single source of truth for runtime config values
"""

from __future__ import annotations

import ast
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FLOW_FILE = Path("src/minivess/orchestration/flows/post_training_flow.py")


def _parse_flow_ast() -> ast.Module:
    """Parse post_training_flow.py into an AST module."""
    return ast.parse(_FLOW_FILE.read_text(encoding="utf-8"))


def _find_call_nodes(tree: ast.AST, func_name: str) -> list[ast.Call]:
    """Return all Call nodes whose function resolves to *func_name*.

    Matches both bare calls ``func_name(...)`` and attribute calls
    ``obj.func_name(...)``.
    """
    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if (
            isinstance(fn, ast.Name)
            and fn.id == func_name
            or isinstance(fn, ast.Attribute)
            and fn.attr == func_name
        ):
            results.append(node)
    return results


def _keyword_value(call: ast.Call, name: str) -> ast.expr | None:
    """Return the value AST node for keyword *name* in *call*, or None."""
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


# ---------------------------------------------------------------------------
# T-1.1.A  Structural tests (AST — no subprocess, no runtime)
# ---------------------------------------------------------------------------


class TestNoHardcodedExperimentName:
    """post_training_flow must not hardcode 'minivess_training' in call sites.

    Current (broken) code:
      find_upstream_safely(..., experiment_name="minivess_training", ...)
      mlflow.set_experiment("minivess_training")

    After fix:
      upstream_exp = os.environ.get("UPSTREAM_EXPERIMENT", ...)
      find_upstream_safely(..., experiment_name=upstream_exp, ...)
      mlflow.set_experiment(os.environ.get("EXPERIMENT", ...))
    """

    def test_find_upstream_safely_experiment_name_not_string_literal(self) -> None:
        """find_upstream_safely() must NOT pass 'minivess_training' as a string literal.

        Will FAIL before fix: line with experiment_name="minivess_training"
        Must PASS after fix: experiment_name is read from os.environ
        """
        tree = _parse_flow_ast()
        calls = _find_call_nodes(tree, "find_upstream_safely")
        assert calls, "find_upstream_safely not found in post_training_flow.py"

        for call in calls:
            val = _keyword_value(call, "experiment_name")
            if val is None:
                continue
            assert not (
                isinstance(val, ast.Constant) and val.value == "minivess_training"
            ), (
                "find_upstream_safely() has hardcoded experiment_name='minivess_training'. "
                "Use os.environ.get('UPSTREAM_EXPERIMENT', ...) instead."
            )

    def test_set_experiment_not_string_literal(self) -> None:
        """mlflow.set_experiment() must NOT be called with 'minivess_training' directly.

        Will FAIL before fix: mlflow.set_experiment("minivess_training")
        Must PASS after fix: uses env var
        """
        tree = _parse_flow_ast()
        calls = _find_call_nodes(tree, "set_experiment")
        assert calls, "mlflow.set_experiment not found in post_training_flow.py"

        for call in calls:
            if not call.args:
                continue
            arg = call.args[0]
            assert not (
                isinstance(arg, ast.Constant) and arg.value == "minivess_training"
            ), (
                "mlflow.set_experiment() called with hardcoded 'minivess_training'. "
                "Use os.environ.get('EXPERIMENT', ...) instead."
            )

    def test_upstream_experiment_env_var_referenced_in_flow_body(self) -> None:
        """The flow function body must reference 'UPSTREAM_EXPERIMENT' env var.

        Currently, UPSTREAM_EXPERIMENT is only read in __main__ (dead code).
        After fix: the flow body (or a helper called from it) reads the env var.
        """
        tree = _parse_flow_ast()

        # Find the post_training_flow function definition
        flow_func: ast.FunctionDef | None = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "post_training_flow":
                flow_func = node
                break

        assert flow_func is not None, "post_training_flow function not found"

        # Walk the function body looking for "UPSTREAM_EXPERIMENT" string constant
        upstream_refs = [
            node
            for node in ast.walk(flow_func)
            if isinstance(node, ast.Constant) and node.value == "UPSTREAM_EXPERIMENT"
        ]
        assert upstream_refs, (
            "'UPSTREAM_EXPERIMENT' not referenced inside post_training_flow() body. "
            "The flow must call os.environ.get('UPSTREAM_EXPERIMENT', ...) internally."
        )

    def test_experiment_env_var_referenced_for_mlflow_logging(self) -> None:
        """The 'EXPERIMENT' env var must be read for mlflow.set_experiment() call.

        After fix: os.environ.get('EXPERIMENT', ...) drives set_experiment.
        """
        tree = _parse_flow_ast()

        flow_func: ast.FunctionDef | None = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "post_training_flow":
                flow_func = node
                break

        assert flow_func is not None, "post_training_flow function not found"

        # Check for 'EXPERIMENT' constant (env var key) inside the flow function body
        experiment_refs = [
            node
            for node in ast.walk(flow_func)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in ("EXPERIMENT", "UPSTREAM_EXPERIMENT")
        ]
        assert experiment_refs, (
            "Neither 'EXPERIMENT' nor 'UPSTREAM_EXPERIMENT' is referenced inside "
            "post_training_flow() body for setting the MLflow experiment name."
        )


# ---------------------------------------------------------------------------
# T-1.1.B  Behavioural test (monkeypatch — verifies runtime path)
# ---------------------------------------------------------------------------


class TestPostTrainingUpstreamExperimentBehavioural:
    """Verify that UPSTREAM_EXPERIMENT env var is observed at runtime."""

    def test_find_upstream_safely_receives_env_var_value(
        self, monkeypatch: object, tmp_path: object
    ) -> None:
        """find_upstream_safely is called with the value from UPSTREAM_EXPERIMENT.

        Will FAIL before fix: hardcoded "minivess_training" is used.
        Must PASS after fix: env var value "my_debug_exp" is forwarded.
        """

        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")  # type: ignore[attr-defined]
        monkeypatch.setenv("PREFECT_DISABLED", "1")  # type: ignore[attr-defined]
        monkeypatch.setenv("UPSTREAM_EXPERIMENT", "my_debug_exp")  # type: ignore[attr-defined]
        monkeypatch.setenv("EXPERIMENT", "my_debug_exp")  # type: ignore[attr-defined]

        from pathlib import Path as _Path

        # Must patch find_upstream_safely in the flow module's namespace
        # (it's imported directly: `from mlflow_helpers import find_upstream_safely`).
        import minivess.orchestration.flows.post_training_flow as ptf

        captured: dict[str, str] = {}

        def _fake_find_upstream(
            tracking_uri: str, experiment_name: str, upstream_flow: str
        ) -> None:
            captured["experiment_name"] = experiment_name
            return None

        monkeypatch.setattr(ptf, "find_upstream_safely", _fake_find_upstream)  # type: ignore[attr-defined]
        monkeypatch.setattr(ptf, "log_completion_safe", lambda **kw: None)  # type: ignore[attr-defined]

        from minivess.config.post_training_config import PostTrainingConfig
        from minivess.orchestration.flows.post_training_flow import post_training_flow

        def _one_plugin_enabled(self: PostTrainingConfig) -> list[str]:
            return ["swa"]

        monkeypatch.setattr(
            PostTrainingConfig, "enabled_plugin_names", _one_plugin_enabled
        )  # type: ignore[attr-defined]
        monkeypatch.setattr(ptf, "_build_registry", lambda: _StubRegistry())  # type: ignore[attr-defined]
        monkeypatch.setattr(  # type: ignore[attr-defined]
            ptf,
            "run_plugin_task",
            lambda plugin, pi: {"status": "success", "model_paths": [], "metrics": {}},
        )

        # Stub mlflow to avoid needing a real server
        import mlflow

        monkeypatch.setattr(mlflow, "set_tracking_uri", lambda *a, **k: None)  # type: ignore[attr-defined]
        monkeypatch.setattr(mlflow, "set_experiment", lambda *a, **k: None)  # type: ignore[attr-defined]

        class _FakeRun:
            class info:
                run_id = "fake-run-id"

        from contextlib import contextmanager

        @contextmanager
        def _fake_start_run(**kw: object):  # type: ignore[misc]
            yield _FakeRun()

        monkeypatch.setattr(mlflow, "start_run", _fake_start_run)  # type: ignore[attr-defined]
        monkeypatch.setattr(mlflow, "log_metric", lambda *a, **k: None)  # type: ignore[attr-defined]

        # Run the flow — pass tmp_path so output_dir can be created on host
        post_training_flow(
            config=PostTrainingConfig(),
            output_dir=_Path(str(tmp_path)) / "post_training",
        )

        # The important assertion: upstream discovery used the env var
        assert captured.get("experiment_name") == "my_debug_exp", (
            f"find_upstream_safely was called with experiment_name="
            f"'{captured.get('experiment_name')}' instead of 'my_debug_exp' from env. "
            "Fix: read UPSTREAM_EXPERIMENT inside post_training_flow()."
        )


# ---------------------------------------------------------------------------
# Stub helpers for behavioural test
# ---------------------------------------------------------------------------


class _StubPlugin:
    name = "swa"

    def validate_inputs(self, pi: object) -> list[str]:
        return []

    def execute(self, pi: object) -> object:
        class _Out:
            model_paths: list = []
            metrics: dict = {}
            artifacts: dict = {}

        return _Out()


class _StubRegistry:
    def get(self, name: str) -> _StubPlugin:
        return _StubPlugin()
