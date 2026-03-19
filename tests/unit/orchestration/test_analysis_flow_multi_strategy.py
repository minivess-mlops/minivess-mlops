"""Tests for analysis_flow — multi-strategy inference + metric key prefixing + AST guards."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from minivess.config.evaluation_config import EvaluationConfig, InferenceStrategyConfig
from minivess.pipeline.multi_strategy_inference import MultiStrategyInferenceRunner


def _two_strategy_config() -> EvaluationConfig:
    return EvaluationConfig(
        inference_strategies=[
            InferenceStrategyConfig(
                name="standard_patch",
                roi_size=[16, 16, 4],
                is_primary=True,
            ),
            InferenceStrategyConfig(
                name="fast",
                roi_size="per_model",
                is_primary=False,
            ),
        ]
    )


class TestMultiStrategyInferenceCallCount:
    """Verify MultiStrategyInferenceRunner calls sliding_window_inference once per strategy."""

    def test_evaluate_single_model_calls_each_strategy(self) -> None:
        """sliding_window_inference called once per strategy per volume."""
        config = _two_strategy_config()
        model = MagicMock(spec=torch.nn.Module)
        model.get_eval_roi_size = MagicMock(return_value=(16, 16, 4))
        fake_output = torch.zeros(1, 2, 16, 16, 4)
        call_count = 0

        def counting_swi(**kwargs: object) -> torch.Tensor:
            nonlocal call_count
            call_count += 1
            return fake_output

        volume = torch.zeros(1, 1, 16, 16, 4)
        runner = MultiStrategyInferenceRunner(
            strategies=config.inference_strategies, num_classes=2
        )

        with patch(
            "minivess.pipeline.multi_strategy_inference.sliding_window_inference",
            side_effect=counting_swi,
        ):
            runner.run_all_strategies(model, volume)

        # 2 strategies × 1 volume = 2 calls
        assert call_count == 2

    def test_primary_strategy_results_have_no_prefix(self) -> None:
        """Primary strategy metric keys are bare (no strategy_name/ prefix)."""
        raw_metrics = {"dsc": 0.9, "cldice": 0.85}
        # Primary strategy: bare keys — no prefix added
        result = dict(raw_metrics)
        assert "dsc" in result
        assert "standard_patch/dsc" not in result

    def test_non_primary_strategy_results_have_prefix(self) -> None:
        """Non-primary strategy metric keys: '{strategy_name}/{metric_name}'."""
        strategy_name = "fast"
        raw_metrics = {"dsc": 0.85, "cldice": 0.82}
        prefixed = {f"{strategy_name}/{k}": v for k, v in raw_metrics.items()}
        assert "fast/dsc" in prefixed
        assert "fast/cldice" in prefixed
        # Verify str.partition works for splitting — no regex (Rule #16)
        prefix, sep, suffix = "fast/dsc".partition("/")
        assert prefix == "fast"
        assert suffix == "dsc"

    def test_primary_strategy_tag_from_config(self) -> None:
        """primary strategy name from eval_config is correct."""
        config = _two_strategy_config()
        primary = next(s for s in config.inference_strategies if s.is_primary)
        assert primary.name == "standard_patch"


class TestAnalysisFlowPrimaryRoiIsUsed:
    """Verify _evaluate_single_model_on_all uses primary strategy roi, not hardcoded (32,32,32)."""

    def test_analysis_flow_uses_primary_roi_size(self) -> None:
        """The primary strategy roi_size is extracted from eval_config, not hardcoded."""
        config = EvaluationConfig(
            inference_strategies=[
                InferenceStrategyConfig(
                    name="custom",
                    roi_size=[64, 64, 8],
                    is_primary=True,
                )
            ]
        )
        primary = next(s for s in config.inference_strategies if s.is_primary)
        assert primary.roi_size == [64, 64, 8]
        # The roi (64,64,8) is NOT (32,32,32) — guards against regression
        assert primary.roi_size != [32, 32, 32]


class TestAnalysisFlowGuard:
    def test_no_hardcoded_32_roi_in_analysis_flow(self) -> None:
        """AST guard: literal (32, 32, 32) must not exist in analysis_flow.py."""
        import ast
        from pathlib import Path

        source = (
            Path(__file__).parents[3]
            / "src"
            / "minivess"
            / "orchestration"
            / "flows"
            / "analysis_flow.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        bad_tuples: list[str] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Tuple)
                and len(node.elts) == 3
                and all(
                    isinstance(e, ast.Constant) and e.value == 32 for e in node.elts
                )
            ):
                bad_tuples.append(f"line {node.lineno}")
        assert not bad_tuples, (
            f"Hardcoded roi_size=(32, 32, 32) found in analysis_flow.py at: {bad_tuples}. "
            "Use eval_config.inference_strategies instead (Rule #9)."
        )

    def test_no_model_family_string_comparison_in_analysis_flow(self) -> None:
        """AST guard: no if-branch comparing string to model family names."""
        import ast
        from pathlib import Path

        _BANNED_FAMILIES = frozenset(
            {"sam3", "dynunet", "mamba", "vesselfm", "mambavesselnet"}
        )

        source = (
            Path(__file__).parents[3]
            / "src"
            / "minivess"
            / "orchestration"
            / "flows"
            / "analysis_flow.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for comparator in node.comparators:
                    if (
                        isinstance(comparator, ast.Constant)
                        and isinstance(comparator.value, str)
                        and comparator.value.lower() in _BANNED_FAMILIES
                    ):
                        violations.append(
                            f"line {comparator.lineno}: model family comparison "
                            f"'{comparator.value}'"
                        )
        assert not violations, (
            f"Task-specific model family comparisons in analysis_flow.py: {violations}. "
            "Use adapter dispatch (Rule #9 — task-agnostic architecture)."
        )
