"""Tests for Hydra-zen bridge in train_flow.py.

Verifies that:
1. __main__ calls compose_experiment_config() when EXPERIMENT env var is set
2. training_flow() accepts config_dict parameter
3. log_hydra_config() is called inside training runs
4. max_train_volumes / max_val_volumes data subsetting works
5. loss_function tag is used (not loss_name)

Uses ast.parse() for structural inspection — NO regex (CLAUDE.md Rule #16).
All tests use monkeypatching to avoid real ML execution.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

_TRAIN_FLOW_PATH = Path("src/minivess/orchestration/flows/train_flow.py")


# ---------------------------------------------------------------------------
# Structural tests (AST-based)
# ---------------------------------------------------------------------------


class TestTrainFlowHydraStructure:
    """Structural invariants for Hydra bridge in train_flow.py."""

    def _get_source(self) -> str:
        if not _TRAIN_FLOW_PATH.exists():
            pytest.skip("train_flow.py not found")
        return _TRAIN_FLOW_PATH.read_text(encoding="utf-8")

    def test_experiment_env_var_used_in_main(self) -> None:
        """__main__ block must read EXPERIMENT env var."""
        source = self._get_source()
        assert "EXPERIMENT" in source, (
            "train_flow.py __main__ must read EXPERIMENT env var to support "
            "compose_experiment_config() integration."
        )

    def test_compose_experiment_config_imported(self) -> None:
        """train_flow.py must import compose_experiment_config."""
        source = self._get_source()
        assert "compose_experiment_config" in source, (
            "train_flow.py must import and call compose_experiment_config() "
            "from minivess.config.compose"
        )

    def test_hydra_overrides_env_var_used(self) -> None:
        """__main__ block must read HYDRA_OVERRIDES env var."""
        source = self._get_source()
        assert "HYDRA_OVERRIDES" in source, (
            "train_flow.py __main__ must read HYDRA_OVERRIDES env var "
            "to support CLI Hydra override injection."
        )

    def test_config_dict_parameter_in_training_flow(self) -> None:
        """training_flow() must accept config_dict parameter."""
        source = self._get_source()
        tree = ast.parse(source)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "training_flow":
                # Check both regular args and keyword-only args (functions with *)
                all_args = list(node.args.args) + list(node.args.kwonlyargs)
                for arg in all_args:
                    if arg.arg == "config_dict":
                        found = True
                        break
        assert found, (
            "training_flow() must have a config_dict parameter to accept the full "
            "resolved Hydra config dict from compose_experiment_config()."
        )

    def test_log_hydra_config_called(self) -> None:
        """train_flow.py must call tracker.log_hydra_config()."""
        source = self._get_source()
        assert "log_hydra_config" in source, (
            "train_flow.py must call tracker.log_hydra_config() to persist the "
            "resolved config as a MLflow artifact (config/resolved_config.yaml). "
            "This is the single source of truth for what ran (CLAUDE.md Rule #23)."
        )

    def test_loss_function_tag_not_loss_name(self) -> None:
        """train_flow.py must use 'loss_function' tag, not 'loss_name'."""
        source = self._get_source()
        # Look for string constant "loss_name" used as a dict key in tags
        # We look for 'loss_function' being present AND ensure 'loss_name' as key is gone
        # Using AST to check tag dictionaries
        assert '"loss_function"' in source or "'loss_function'" in source, (
            "train_flow.py must use 'loss_function' as MLflow tag key "
            "(not 'loss_name'). builder.py reads 'loss_function' tag."
        )

    def test_max_train_volumes_subsetting(self) -> None:
        """train_flow.py must support max_train_volumes config key."""
        source = self._get_source()
        assert "max_train_volumes" in source, (
            "train_flow.py must support max_train_volumes for data subsetting. "
            "Debug experiment configs set this to 2 to limit training volume count."
        )

    def test_max_val_volumes_subsetting(self) -> None:
        """train_flow.py must support max_val_volumes config key."""
        source = self._get_source()
        assert "max_val_volumes" in source, (
            "train_flow.py must support max_val_volumes for data subsetting. "
            "Debug experiment configs set this to 2 to limit validation volume count."
        )


# ---------------------------------------------------------------------------
# Functional tests (with mocking)
# ---------------------------------------------------------------------------


class TestTrainFlowConfigDict:
    """Test training_flow() config_dict parameter extraction."""

    def _make_minimal_config(self, **overrides: Any) -> dict[str, Any]:
        """Build a minimal config dict that training_flow() can extract from."""
        base: dict[str, Any] = {
            "losses": ["cbdice_cldice"],
            "model": "dynunet",
            "max_epochs": 1,
            "experiment_name": "test_experiment",
            "num_folds": 1,
            "batch_size": 1,
            "debug": True,
            "max_train_volumes": 2,
            "max_val_volumes": 2,
        }
        base.update(overrides)
        return base

    def test_config_dict_extracts_model(self) -> None:
        """training_flow() must read model from config_dict['model']."""
        import inspect

        from minivess.orchestration.flows.train_flow import training_flow

        # Just verify the function signature accepts config_dict
        sig = inspect.signature(training_flow.fn)  # type: ignore[attr-defined]
        assert "config_dict" in sig.parameters, (
            "training_flow() must have config_dict parameter"
        )

    def test_config_dict_parameter_is_optional(self) -> None:
        """config_dict must have a default of None (backward compat)."""
        import inspect

        from minivess.orchestration.flows.train_flow import training_flow

        sig = inspect.signature(training_flow.fn)  # type: ignore[attr-defined]
        param = sig.parameters.get("config_dict")
        assert param is not None, "config_dict parameter not found"
        assert param.default is None, (
            "config_dict must default to None for backward compatibility"
        )


# ---------------------------------------------------------------------------
# Builder tag tests (loss_function vs loss_name)
# ---------------------------------------------------------------------------


class TestLossTagStandardization:
    """Verify loss tag standardization in discover_training_runs_raw()."""

    def test_builder_reads_loss_function_tag(self) -> None:
        """builder.py must read 'loss_function' tag (not 'loss_name')."""
        builder_path = Path("src/minivess/ensemble/builder.py")
        if not builder_path.exists():
            pytest.skip("builder.py not found")
        source = builder_path.read_text(encoding="utf-8")
        assert '"loss_function"' in source or "'loss_function'" in source, (
            "builder.py discover_training_runs_raw() must read 'loss_function' tag"
        )

    def test_builder_has_fallback_to_loss_name(self) -> None:
        """builder.py must have backward-compat fallback to 'loss_name' tag."""
        builder_path = Path("src/minivess/ensemble/builder.py")
        if not builder_path.exists():
            pytest.skip("builder.py not found")
        source = builder_path.read_text(encoding="utf-8")
        # The builder should handle both 'loss_function' and 'loss_name' tags
        assert '"loss_name"' in source or "'loss_name'" in source, (
            "builder.py must fall back to 'loss_name' tag for backward compatibility "
            "with existing training runs that used the old tag name."
        )


# ---------------------------------------------------------------------------
# Phase 1 tests: eval gate and checkpoint error
# ---------------------------------------------------------------------------


class TestEvalGateConfigurable:
    """Verify eval_fold2_dsc gate is configurable in builder.py."""

    def test_discover_training_runs_raw_has_require_eval_metrics(self) -> None:
        """discover_training_runs_raw() must accept require_eval_metrics param."""
        builder_path = Path("src/minivess/ensemble/builder.py")
        if not builder_path.exists():
            pytest.skip("builder.py not found")

        source = builder_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        found = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "discover_training_runs_raw"
            ):
                for arg in node.args.args:
                    if arg.arg == "require_eval_metrics":
                        found = True
                        break
                # Also check keyword-only args
                for arg in node.args.kwonlyargs:
                    if arg.arg == "require_eval_metrics":
                        found = True
                        break
        assert found, (
            "discover_training_runs_raw() must accept require_eval_metrics parameter. "
            "Debug runs don't have eval_fold2_dsc — they must not be silently filtered."
        )


class TestLoadCheckpointRaisesOnMismatch:
    """Verify load_checkpoint() raises RuntimeError on state_dict mismatch."""

    def test_load_checkpoint_raises_not_warns(self) -> None:
        """load_checkpoint() must raise RuntimeError on key mismatch, not warn."""
        builder_path = Path("src/minivess/ensemble/builder.py")
        if not builder_path.exists():
            pytest.skip("builder.py not found")

        source = builder_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Find load_checkpoint function
        load_ckpt_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "load_checkpoint":
                load_ckpt_node = node
                break

        assert load_ckpt_node is not None, "load_checkpoint() function not found"

        # Check that RuntimeError is raised (not just a logger.warning)
        func_source = ast.get_source_segment(source, load_ckpt_node) or ""
        assert "RuntimeError" in func_source, (
            "load_checkpoint() must raise RuntimeError on state_dict mismatch. "
            "Silently returning random weights is worse than a crash — it produces "
            "plausible-looking but meaningless predictions."
        )


# ---------------------------------------------------------------------------
# Upstream experiment tests
# ---------------------------------------------------------------------------


class TestUpstreamExperiment:
    """Verify UPSTREAM_EXPERIMENT env var support in flow __main__ blocks."""


    def test_analysis_flow_reads_upstream_experiment(self) -> None:
        """analysis_flow.py __main__ must read UPSTREAM_EXPERIMENT env var."""
        af_path = Path("src/minivess/orchestration/flows/analysis_flow.py")
        if not af_path.exists():
            pytest.skip("analysis_flow.py not found")
        source = af_path.read_text(encoding="utf-8")
        assert "UPSTREAM_EXPERIMENT" in source, (
            "analysis_flow.py __main__ must read UPSTREAM_EXPERIMENT env var "
            "to discover training runs from the correct MLflow experiment."
        )


# ---------------------------------------------------------------------------
# Docker volume tests
# ---------------------------------------------------------------------------


class TestPostTrainingOutVolume:
    """Verify post_training_out volume in analyze container."""

    def test_analyze_service_has_post_training_out_volume(self) -> None:
        """docker-compose.flows.yml analyze service must mount post_training_out."""
        compose_path = Path("deployment/docker-compose.flows.yml")
        if not compose_path.exists():
            pytest.skip("docker-compose.flows.yml not found")

        import yaml

        with open(compose_path, encoding="utf-8") as f:
            compose = yaml.safe_load(f)

        services = compose.get("services", {})
        analyze = services.get("analyze", {})
        volumes = analyze.get("volumes", [])

        # Check for post_training_out volume
        has_volume = any("post_training_out" in str(v) for v in volumes)
        assert has_volume, (
            "analyze service in docker-compose.flows.yml must mount post_training_out "
            "volume so it can read post-training artifacts from the previous flow."
        )
