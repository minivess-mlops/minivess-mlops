"""Tests for zero-shot baseline discovery and tagging (#888).

Validates that discover_zero_shot_baselines() reads from factorial YAML,
tags with is_zero_shot=true, and enforces VesselFM DeepVess-only constraint.
"""

from __future__ import annotations

from pathlib import Path


class TestDiscoverZeroShotBaselines:
    """discover_zero_shot_baselines must read from factorial YAML."""

    def test_returns_empty_without_yaml(self) -> None:
        """Without factorial YAML, return empty list."""
        from minivess.orchestration.flows.analysis_flow import (
            discover_zero_shot_baselines,
        )

        result = discover_zero_shot_baselines(factorial_yaml=None)
        assert result == []

    def test_returns_empty_for_missing_yaml(self, tmp_path: Path) -> None:
        """Non-existent YAML file returns empty list (graceful)."""
        from minivess.orchestration.flows.analysis_flow import (
            discover_zero_shot_baselines,
        )

        result = discover_zero_shot_baselines(
            factorial_yaml=tmp_path / "nonexistent.yaml"
        )
        assert result == []

    def test_discovers_baselines_from_debug_yaml(self) -> None:
        """Debug factorial YAML should define 2 zero-shot baselines."""
        from minivess.orchestration.flows.analysis_flow import (
            discover_zero_shot_baselines,
        )

        yaml_path = Path("configs/factorial/debug.yaml")
        baselines = discover_zero_shot_baselines(factorial_yaml=yaml_path)
        assert len(baselines) == 2

    def test_sam3_vanilla_baseline(self) -> None:
        """SAM3 Vanilla must be one of the zero-shot baselines."""
        from minivess.orchestration.flows.analysis_flow import (
            discover_zero_shot_baselines,
        )

        yaml_path = Path("configs/factorial/debug.yaml")
        baselines = discover_zero_shot_baselines(factorial_yaml=yaml_path)
        models = {b["model"] for b in baselines}
        assert "sam3_vanilla" in models

    def test_vesselfm_baseline_deepvess_only(self) -> None:
        """VesselFM must be constrained to DeepVess dataset only."""
        from minivess.orchestration.flows.analysis_flow import (
            discover_zero_shot_baselines,
        )

        yaml_path = Path("configs/factorial/debug.yaml")
        baselines = discover_zero_shot_baselines(factorial_yaml=yaml_path)
        vesselfm = [b for b in baselines if b["model"] == "vesselfm"]
        assert len(vesselfm) == 1
        assert vesselfm[0]["dataset"] == "deepvess"

    def test_is_zero_shot_tag(self) -> None:
        """Every baseline must have is_zero_shot=True and tag."""
        from minivess.orchestration.flows.analysis_flow import (
            discover_zero_shot_baselines,
        )

        yaml_path = Path("configs/factorial/debug.yaml")
        baselines = discover_zero_shot_baselines(factorial_yaml=yaml_path)
        for baseline in baselines:
            assert baseline["is_zero_shot"] is True
            assert baseline["tags"]["is_zero_shot"] == "true"

    def test_tags_include_model_family(self) -> None:
        """Tags must include model_family for MLflow discovery."""
        from minivess.orchestration.flows.analysis_flow import (
            discover_zero_shot_baselines,
        )

        yaml_path = Path("configs/factorial/debug.yaml")
        baselines = discover_zero_shot_baselines(factorial_yaml=yaml_path)
        for baseline in baselines:
            assert "model_family" in baseline["tags"]
            assert baseline["tags"]["model_family"] == baseline["model"]

    def test_tags_include_zero_shot_strategy(self) -> None:
        """Tags must include zero_shot_strategy for downstream filtering."""
        from minivess.orchestration.flows.analysis_flow import (
            discover_zero_shot_baselines,
        )

        yaml_path = Path("configs/factorial/debug.yaml")
        baselines = discover_zero_shot_baselines(factorial_yaml=yaml_path)
        strategies = {b["tags"]["zero_shot_strategy"] for b in baselines}
        assert "frozen_encoder_eval" in strategies
        assert "zero_shot_only" in strategies
