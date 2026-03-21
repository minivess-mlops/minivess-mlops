"""Tests for factorial YAML-driven post-training factor discovery.

Validates that post-training factor levels come from the composable factorial
YAML (configs/factorial/*.yaml), NOT from hardcoded Python values.
References: XML plan T1.5, T4.3, synthesis Part 1.2 Layer B.
"""

from __future__ import annotations

from pathlib import Path


class TestFactorialYamlPostTrainingFactors:
    """Post-training factors must be parsed from factorial YAML."""

    def test_debug_yaml_has_post_training_section(self) -> None:
        """configs/factorial/debug.yaml must define factors.post_training."""
        from minivess.config.factorial_config import parse_factorial_yaml

        yaml_path = Path("configs/factorial/debug.yaml")
        design = parse_factorial_yaml(yaml_path)
        post_training_names = design.factor_names(layer="post_training")
        assert len(post_training_names) >= 2, (
            f"Expected at least 2 post-training factors, got {post_training_names}"
        )
        assert "method" in post_training_names
        assert "recalibration" in post_training_names

    def test_debug_yaml_post_training_method_levels(self) -> None:
        """Debug YAML must have {none, checkpoint_averaging} as post_training method levels."""
        from minivess.config.factorial_config import parse_factorial_yaml

        yaml_path = Path("configs/factorial/debug.yaml")
        design = parse_factorial_yaml(yaml_path)
        levels = design.factor_levels()
        assert "method" in levels, (
            f"'method' not in factor levels: {list(levels.keys())}"
        )
        assert set(levels["method"]) == {"none", "checkpoint_averaging"}, (
            f"Expected {{none, checkpoint_averaging}} for debug, got {levels['method']}"
        )

    def test_debug_yaml_recalibration_levels(self) -> None:
        """Debug YAML must have {none, temperature_scaling} recalibration levels."""
        from minivess.config.factorial_config import parse_factorial_yaml

        yaml_path = Path("configs/factorial/debug.yaml")
        design = parse_factorial_yaml(yaml_path)
        levels = design.factor_levels()
        assert "recalibration" in levels
        assert set(levels["recalibration"]) == {"none", "temperature_scaling"}

    def test_paper_full_yaml_has_more_post_training_levels(self) -> None:
        """Production YAML must have 3 post-training method levels."""
        from minivess.config.factorial_config import parse_factorial_yaml

        yaml_path = Path("configs/factorial/paper_full.yaml")
        design = parse_factorial_yaml(yaml_path)
        levels = design.factor_levels()
        assert "method" in levels
        assert set(levels["method"]) == {
            "none",
            "checkpoint_averaging",
            "swag",
        }, (
            f"Expected {{none, checkpoint_averaging, swag}} for production, got {levels['method']}"
        )

    def test_factor_names_auto_derived_not_hardcoded(self) -> None:
        """Factor names must be auto-derived from YAML keys, not hardcoded."""
        from minivess.config.factorial_config import parse_factorial_yaml

        yaml_path = Path("configs/factorial/debug.yaml")
        design = parse_factorial_yaml(yaml_path)
        all_names = design.factor_names()
        # All 6 factor names should come from the YAML
        assert len(all_names) >= 6, (
            f"Expected at least 6 factors from YAML, got {len(all_names)}: {all_names}"
        )

    def test_n_conditions_matches_factorial_product(self) -> None:
        """Debug design must have 4×3×2×2×2×4 = 384 conditions."""
        from minivess.config.factorial_config import parse_factorial_yaml

        yaml_path = Path("configs/factorial/debug.yaml")
        design = parse_factorial_yaml(yaml_path)
        # 4 models × 3 losses × 2 aux_calib × 2 method × 2 recalib × 4 ensemble
        assert design.n_conditions == 384, (
            f"Expected 384 debug conditions, got {design.n_conditions}"
        )
