from __future__ import annotations

import os
from unittest.mock import patch


class TestPrefectCompat:
    def test_noop_task_preserves_function(self):
        """When PREFECT_DISABLED=1, @task decorator is a no-op passthrough."""
        with patch.dict(os.environ, {"PREFECT_DISABLED": "1"}):
            # Re-import to pick up env var
            import importlib

            import minivess.orchestration._prefect_compat as mod

            importlib.reload(mod)

            @mod.task
            def my_func(x: int) -> int:
                return x * 2

            assert my_func(5) == 10

    def test_noop_flow_preserves_function(self):
        """When PREFECT_DISABLED=1, @flow decorator is a no-op passthrough."""
        with patch.dict(os.environ, {"PREFECT_DISABLED": "1"}):
            import importlib

            import minivess.orchestration._prefect_compat as mod

            importlib.reload(mod)

            @mod.flow
            def my_flow(name: str) -> str:
                return f"hello {name}"

            assert my_flow("world") == "hello world"

    def test_prefect_disabled_env_var(self):
        """PREFECT_DISABLED=1 prevents Prefect import."""
        with patch.dict(os.environ, {"PREFECT_DISABLED": "1"}):
            import importlib

            import minivess.orchestration._prefect_compat as mod

            importlib.reload(mod)

            assert mod.PREFECT_AVAILABLE is False

    def test_get_run_logger_fallback(self):
        """get_run_logger returns stdlib logger when Prefect is disabled."""
        with patch.dict(os.environ, {"PREFECT_DISABLED": "1"}):
            import importlib

            import minivess.orchestration._prefect_compat as mod

            importlib.reload(mod)

            logger = mod.get_run_logger()
            assert hasattr(logger, "info")
            assert hasattr(logger, "warning")

    def test_decorated_function_callable(self):
        """Decorated functions remain callable with correct signatures."""
        with patch.dict(os.environ, {"PREFECT_DISABLED": "1"}):
            import importlib

            import minivess.orchestration._prefect_compat as mod

            importlib.reload(mod)

            @mod.task(name="add")
            def add(a: int, b: int) -> int:
                return a + b

            @mod.flow(name="pipeline")
            def pipeline() -> int:
                return add(1, 2)

            assert pipeline() == 3

    def test_task_with_kwargs(self):
        """@task decorator works with keyword arguments like cache_key_fn."""
        with patch.dict(os.environ, {"PREFECT_DISABLED": "1"}):
            import importlib

            import minivess.orchestration._prefect_compat as mod

            importlib.reload(mod)

            @mod.task(name="compute", retries=3, cache_key_fn=None)
            def compute(x: int) -> int:
                return x**2

            assert compute(4) == 16


class TestPrefectCompatInit:
    def test_orchestration_package_importable(self):
        """The orchestration package is importable."""
        import minivess.orchestration

        assert hasattr(minivess.orchestration, "__name__")

    def test_public_exports(self):
        """Key symbols are re-exported from __init__."""
        with patch.dict(os.environ, {"PREFECT_DISABLED": "1"}):
            import importlib

            import minivess.orchestration as orch

            importlib.reload(orch)
            # Should export task, flow, get_run_logger
            assert hasattr(orch, "task")
            assert hasattr(orch, "flow")
            assert hasattr(orch, "get_run_logger")
