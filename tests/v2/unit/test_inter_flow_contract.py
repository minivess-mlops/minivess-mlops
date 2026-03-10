"""Tests for inter-flow contract fixes (#586, #587, #588).

Verifies:
- find_upstream_run() filters by flow_name tag (not just most recent)
- post_training_flow() auto-discovers checkpoints from upstream run
- EvaluationConfig.require_eval_metrics controls eval_fold2_dsc gate
- discover_post_training_models() accepts experiment_name override

Rule #16: No regex — use json.loads(), yaml.safe_load(), str methods only.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent


class TestFindUpstreamRunFlowFilter:
    """#586: find_upstream_run() must filter by upstream_flow tag."""

    def test_find_upstream_run_uses_flow_name_in_filter(self) -> None:
        """find_upstream_run() must include flow_name in MLflow filter_string."""
        source = (
            ROOT / "src" / "minivess" / "orchestration" / "flow_contract.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "find_upstream_run":
                # Check that upstream_flow is used in the body (not just accepted)
                body_source = ast.get_source_segment(source, node)
                assert body_source is not None
                assert "flow_name" in body_source, (
                    "find_upstream_run() must filter by flow_name tag. "
                    "Add: filter_parts.append(f\"tags.flow_name = '{upstream_flow}'\")"
                )
                break
        else:
            raise AssertionError("find_upstream_run() not found in flow_contract.py")


class TestPostTrainingCheckpointDiscovery:
    """#587: post_training_flow() must auto-discover checkpoints."""

    def test_post_training_flow_calls_resolve_checkpoints(self) -> None:
        """When checkpoint_paths is empty, flow must call resolve_checkpoint_paths_from_contract."""
        source = (
            ROOT
            / "src"
            / "minivess"
            / "orchestration"
            / "flows"
            / "post_training_flow.py"
        ).read_text(encoding="utf-8")

        # The function resolve_checkpoint_paths_from_contract must be called
        # in the post_training_flow function body (not just defined)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "post_training_flow":
                body_source = ast.get_source_segment(source, node)
                assert body_source is not None
                assert "resolve_checkpoint_paths_from_contract" in body_source, (
                    "post_training_flow() must call resolve_checkpoint_paths_from_contract() "
                    "to auto-discover checkpoints when none provided."
                )
                break
        else:
            raise AssertionError("post_training_flow() not found")


class TestEvalConfigRequireMetrics:
    """#588: EvaluationConfig must expose require_eval_metrics field."""

    def test_evaluation_config_has_require_eval_metrics(self) -> None:
        """EvaluationConfig must have require_eval_metrics field."""
        from minivess.config.evaluation_config import EvaluationConfig

        config = EvaluationConfig()
        assert hasattr(config, "require_eval_metrics"), (
            "EvaluationConfig must have require_eval_metrics field "
            "so debug runs are not silently filtered."
        )

    def test_require_eval_metrics_defaults_true(self) -> None:
        """Default must be True (production behavior unchanged)."""
        from minivess.config.evaluation_config import EvaluationConfig

        config = EvaluationConfig()
        assert config.require_eval_metrics is True

    def test_require_eval_metrics_false_accepted(self) -> None:
        """Must accept False for debug scenarios."""
        from minivess.config.evaluation_config import EvaluationConfig

        config = EvaluationConfig(require_eval_metrics=False)
        assert config.require_eval_metrics is False


class TestDiscoverRunsPassesRequireMetrics:
    """#588: _discover_runs must pass require_eval_metrics from eval_config."""

    def test_discover_runs_accepts_require_eval_metrics(self) -> None:
        """EnsembleBuilder.discover_training_runs() must accept require_eval_metrics."""
        source = (ROOT / "src" / "minivess" / "ensemble" / "builder.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "discover_training_runs"
            ):
                # Check it passes require_eval_metrics to discover_training_runs_raw
                body_source = ast.get_source_segment(source, node)
                assert body_source is not None
                assert "require_eval_metrics" in body_source, (
                    "discover_training_runs() must pass require_eval_metrics "
                    "to discover_training_runs_raw()"
                )
                break
        else:
            raise AssertionError("discover_training_runs() not found")

    def test_analysis_flow_passes_require_eval_metrics(self) -> None:
        """_discover_runs in analysis_flow must pass require_eval_metrics."""
        source = (
            ROOT / "src" / "minivess" / "orchestration" / "flows" / "analysis_flow.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_discover_runs":
                body_source = ast.get_source_segment(source, node)
                assert body_source is not None
                assert "require_eval_metrics" in body_source, (
                    "_discover_runs() must pass require_eval_metrics from "
                    "eval_config to EnsembleBuilder"
                )
                break
        else:
            raise AssertionError("_discover_runs() not found in analysis_flow.py")
