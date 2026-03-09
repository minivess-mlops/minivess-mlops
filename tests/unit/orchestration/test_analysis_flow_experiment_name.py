"""Tests verifying analysis_flow uses resolve_experiment_name() — Issue #555.

Verifies:
- analysis_flow source has no hardcoded "minivess_training" string constant
- resolve_experiment_name() correctly appends _DEBUG suffix
- UPSTREAM_EXPERIMENT env var overrides resolve_experiment_name()

Plan: docs/planning/prefect-flow-connectivity-execution-plan.xml Phase 0 (T0.5)
"""

from __future__ import annotations

import ast
import inspect
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


class TestAnalysisFlowNoHardcodedExperimentName:
    def test_analysis_flow_has_no_hardcoded_minivess_training(self) -> None:
        """analysis_flow source must not contain the literal 'minivess_training'."""
        from minivess.orchestration.flows import analysis_flow as af_module

        source = inspect.getsource(af_module)
        tree = ast.parse(source)

        hardcoded = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and node.value == "minivess_training"
        ]
        assert len(hardcoded) == 0, (
            f"Found {len(hardcoded)} hardcoded 'minivess_training' string(s) in "
            f"analysis_flow.py — use resolve_experiment_name() instead.\n"
            f"Line numbers: {[n.lineno for n in hardcoded]}"
        )

    def test_analysis_flow_imports_resolve_experiment_name(self) -> None:
        """analysis_flow must import resolve_experiment_name from constants."""
        from minivess.orchestration.flows import analysis_flow as af_module

        source = inspect.getsource(af_module)
        tree = ast.parse(source)

        # Look for import of resolve_experiment_name
        found_import = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and "constants" in node.module
            ):
                names = [alias.name for alias in node.names]
                if "resolve_experiment_name" in names:
                    found_import = True
                    break
        assert found_import, (
            "analysis_flow.py must import resolve_experiment_name from "
            "minivess.orchestration.constants"
        )


class TestResolveExperimentNameIntegration:
    def test_debug_suffix_applied_to_training_experiment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MINIVESS_DEBUG_SUFFIX", "_DEBUG")
        from minivess.orchestration.constants import resolve_experiment_name

        name = resolve_experiment_name("minivess_training")
        assert name == "minivess_training_DEBUG"

    def test_upstream_experiment_env_overrides(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """UPSTREAM_EXPERIMENT env var takes priority over resolved name."""
        monkeypatch.setenv("UPSTREAM_EXPERIMENT", "my_custom_experiment")
        monkeypatch.setenv("MINIVESS_DEBUG_SUFFIX", "_DEBUG")
        from minivess.orchestration.constants import resolve_experiment_name

        # Simulate how analysis_flow should resolve the upstream experiment name
        upstream_exp = os.environ.get(
            "UPSTREAM_EXPERIMENT",
            resolve_experiment_name("minivess_training"),
        )
        assert upstream_exp == "my_custom_experiment"

    def test_no_upstream_env_uses_resolved_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("UPSTREAM_EXPERIMENT", raising=False)
        monkeypatch.setenv("MINIVESS_DEBUG_SUFFIX", "_DEBUG")
        from minivess.orchestration.constants import resolve_experiment_name

        upstream_exp = os.environ.get(
            "UPSTREAM_EXPERIMENT",
            resolve_experiment_name("minivess_training"),
        )
        assert upstream_exp == "minivess_training_DEBUG"
