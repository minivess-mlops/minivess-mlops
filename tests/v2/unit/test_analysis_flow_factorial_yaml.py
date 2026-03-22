"""Tests for factorial YAML-driven ensemble strategy resolution (T1.7).

Validates that the Analysis Flow reads ensemble strategies from the
composable factorial YAML (configs/factorial/*.yaml::factors.analysis.ensemble_strategy),
NOT from hardcoded values. References: XML plan T1.7, synthesis Part 1.2 Layer C.
"""

from __future__ import annotations

from pathlib import Path

import yaml


class TestFactorialYamlEnsembleStrategies:
    """Ensemble strategies must be parsed from factorial YAML."""

    def test_debug_yaml_has_analysis_section(self) -> None:
        """configs/factorial/debug.yaml must define factors.analysis."""
        from minivess.config.factorial_config import parse_factorial_yaml

        yaml_path = Path("configs/factorial/debug.yaml")
        design = parse_factorial_yaml(yaml_path)
        analysis_names = design.factor_names(layer="analysis")
        assert "ensemble_strategy" in analysis_names, (
            f"Expected 'ensemble_strategy' in analysis factors, got {analysis_names}"
        )

    def test_debug_yaml_ensemble_levels(self) -> None:
        """Debug YAML has 5 ensemble strategies (IDENTICAL to production, Rule 27)."""
        from minivess.config.factorial_config import parse_factorial_yaml

        yaml_path = Path("configs/factorial/debug.yaml")
        design = parse_factorial_yaml(yaml_path)
        levels = design.factor_levels()
        assert "ensemble_strategy" in levels
        debug_strategies = set(levels["ensemble_strategy"])
        assert len(debug_strategies) == 5, (
            f"Expected 5 debug strategies (same as production), got {len(debug_strategies)}: {debug_strategies}"
        )
        # Debug = production: all 5 strategies including "none" (CLAUDE.md Rule 27)
        assert "none" in debug_strategies

    def test_paper_full_yaml_has_5_ensemble_strategies(self) -> None:
        """Production YAML has 5 ensemble strategies including none."""
        from minivess.config.factorial_config import parse_factorial_yaml

        yaml_path = Path("configs/factorial/paper_full.yaml")
        design = parse_factorial_yaml(yaml_path)
        levels = design.factor_levels()
        assert "ensemble_strategy" in levels
        prod_strategies = set(levels["ensemble_strategy"])
        assert len(prod_strategies) == 5, (
            f"Expected 5 production strategies, got {len(prod_strategies)}: {prod_strategies}"
        )
        assert "none" in prod_strategies

    def test_resolve_ensemble_strategies_from_yaml(self, tmp_path: Path) -> None:
        """_resolve_ensemble_strategies() reads from factorial YAML if provided."""
        from minivess.orchestration.flows.analysis_flow import (
            _resolve_ensemble_strategies,
        )

        # Create a minimal factorial YAML
        factorial = {
            "factors": {
                "analysis": {
                    "ensemble_strategy": ["per_loss_single_best", "all_loss_all_best"],
                }
            }
        }
        yaml_path = tmp_path / "custom_factorial.yaml"
        yaml_path.write_text(yaml.dump(factorial), encoding="utf-8")

        strategies = _resolve_ensemble_strategies(factorial_yaml=yaml_path)
        assert strategies == ["per_loss_single_best", "all_loss_all_best"]

    def test_resolve_ensemble_strategies_fallback_without_yaml(self) -> None:
        """Without factorial_yaml, returns None (caller uses config defaults)."""
        from minivess.orchestration.flows.analysis_flow import (
            _resolve_ensemble_strategies,
        )

        result = _resolve_ensemble_strategies(factorial_yaml=None)
        assert result is None

    def test_resolve_ensemble_strategies_invalid_yaml_returns_none(
        self, tmp_path: Path
    ) -> None:
        """Invalid factorial YAML returns None (graceful fallback)."""
        from minivess.orchestration.flows.analysis_flow import (
            _resolve_ensemble_strategies,
        )

        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("not: valid: factorial", encoding="utf-8")
        result = _resolve_ensemble_strategies(factorial_yaml=bad_yaml)
        assert result is None
