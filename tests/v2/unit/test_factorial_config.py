"""Tests for composable factorial experiment configuration.

Validates that factorial YAMLs are parsed correctly with zero hardcoded
factor names. Users can create arbitrary factorial designs.

Source: docs/planning/intermedia-plan-synthesis-pre-debug-run.md Part 1.3-1.4
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minivess.config.factorial_config import (
    parse_factorial_yaml,
)


class TestFactorialConfigParsing:
    """Factorial YAML parsing with zero hardcoded factor names."""

    def test_parse_paper_full_yaml(self) -> None:
        """paper_full.yaml should parse to 640 conditions (4×4×2×2×2×5)."""
        yaml_path = Path("configs/factorial/paper_full.yaml")
        if not yaml_path.exists():
            pytest.skip("paper_full.yaml not found")
        design = parse_factorial_yaml(yaml_path)
        assert design.experiment_name == "paper_factorial"
        assert design.n_conditions == 640  # 4×4×2×2×2×5
        assert design.n_training_conditions == 32  # 4×4×2
        assert design.n_training_runs == 96  # 32×3 folds

    def test_parse_debug_yaml(self) -> None:
        """debug.yaml should parse to 640 conditions (IDENTICAL factors to production, Rule 27)."""
        yaml_path = Path("configs/factorial/debug.yaml")
        if not yaml_path.exists():
            pytest.skip("debug.yaml not found")
        design = parse_factorial_yaml(yaml_path)
        assert design.experiment_name == "debug_factorial"
        assert design.n_conditions == 640  # 4×4×2×2×2×5 (same as production)
        assert design.n_training_conditions == 32  # 4×4×2
        assert design.n_training_runs == 32  # 32×1 fold
        assert design.debug is True

    def test_parse_smoke_test_yaml(self) -> None:
        """smoke_test.yaml should parse to 1 condition."""
        yaml_path = Path("configs/factorial/smoke_test.yaml")
        if not yaml_path.exists():
            pytest.skip("smoke_test.yaml not found")
        design = parse_factorial_yaml(yaml_path)
        assert design.n_conditions == 1  # 1×1×1×1×1×1
        assert design.n_training_runs == 1

    def test_auto_derives_factor_names(self) -> None:
        """Factor names must be auto-derived from YAML keys, never hardcoded."""
        yaml_path = Path("configs/factorial/paper_full.yaml")
        if not yaml_path.exists():
            pytest.skip("paper_full.yaml not found")
        design = parse_factorial_yaml(yaml_path)

        # Get all factor names
        names = design.factor_names()
        # These should be derived from the YAML, not hardcoded
        assert "model_family" in names
        assert "loss_name" in names
        assert "aux_calibration" in names
        assert "method" in names
        assert "recalibration" in names
        assert "ensemble_strategy" in names

    def test_factors_grouped_by_layer(self) -> None:
        """Factors should be grouped by layer (training/post_training/analysis)."""
        yaml_path = Path("configs/factorial/paper_full.yaml")
        if not yaml_path.exists():
            pytest.skip("paper_full.yaml not found")
        design = parse_factorial_yaml(yaml_path)

        training = design.factor_names(layer="training")
        post_training = design.factor_names(layer="post_training")
        analysis = design.factor_names(layer="analysis")

        assert len(training) == 3  # model, loss, calib
        assert len(post_training) == 2  # method, recalibration
        assert len(analysis) == 1  # ensemble_strategy

    def test_training_conditions_cartesian_product(self) -> None:
        """training_conditions() should return Layer A Cartesian product."""
        yaml_path = Path("configs/factorial/debug.yaml")
        if not yaml_path.exists():
            pytest.skip("debug.yaml not found")
        design = parse_factorial_yaml(yaml_path)
        conditions = design.training_conditions()
        assert len(conditions) == 32  # 4×4×2

    def test_all_conditions_full_cartesian(self) -> None:
        """all_conditions() should return ALL-layer Cartesian product."""
        yaml_path = Path("configs/factorial/smoke_test.yaml")
        if not yaml_path.exists():
            pytest.skip("smoke_test.yaml not found")
        design = parse_factorial_yaml(yaml_path)
        conditions = design.all_conditions()
        assert len(conditions) == 1

    def test_factor_levels_dict(self) -> None:
        """factor_levels() should return name → levels mapping."""
        yaml_path = Path("configs/factorial/paper_full.yaml")
        if not yaml_path.exists():
            pytest.skip("paper_full.yaml not found")
        design = parse_factorial_yaml(yaml_path)
        levels = design.factor_levels()
        assert len(levels["model_family"]) == 4
        assert len(levels["loss_name"]) == 4
        assert len(levels["ensemble_strategy"]) == 5


class TestFactorialConfigMLflow:
    """MLflow experiment names from factorial config."""

    def test_mlflow_experiment_names(self) -> None:
        yaml_path = Path("configs/factorial/paper_full.yaml")
        if not yaml_path.exists():
            pytest.skip("paper_full.yaml not found")
        design = parse_factorial_yaml(yaml_path)
        assert "training_experiment" in design.mlflow
        assert "evaluation_experiment" in design.mlflow
        assert "biostatistics_experiment" in design.mlflow


class TestFactorialConfigZeroShot:
    """Zero-shot baselines from factorial config."""

    def test_zero_shot_baselines_present(self) -> None:
        yaml_path = Path("configs/factorial/paper_full.yaml")
        if not yaml_path.exists():
            pytest.skip("paper_full.yaml not found")
        design = parse_factorial_yaml(yaml_path)
        assert len(design.zero_shot_baselines) == 2
        models = [b["model"] for b in design.zero_shot_baselines]
        assert "sam3_vanilla" in models
        assert "vesselfm" in models

    def test_smoke_test_no_baselines(self) -> None:
        yaml_path = Path("configs/factorial/smoke_test.yaml")
        if not yaml_path.exists():
            pytest.skip("smoke_test.yaml not found")
        design = parse_factorial_yaml(yaml_path)
        assert len(design.zero_shot_baselines) == 0


class TestFactorialConfigCustomLab:
    """A hypothetical my_lab.yaml with arbitrary factor subsets should work."""

    def test_custom_2x1_design(self, tmp_path: Path) -> None:
        """A lab with 2 models × 1 loss should produce 2 conditions."""
        import yaml

        custom = {
            "experiment_name": "my_lab_test",
            "factors": {
                "training": {
                    "model_family": ["dynunet", "sam3_hybrid"],
                    "loss_name": ["cbdice_cldice"],
                },
                "post_training": {
                    "method": ["none"],
                },
                "analysis": {
                    "ensemble_strategy": ["per_loss_single_best"],
                },
            },
            "fixed": {"max_epochs": 10, "num_folds": 1},
            "mlflow": {"training_experiment": "my_lab"},
        }
        yaml_path = tmp_path / "my_lab.yaml"
        yaml_path.write_text(yaml.dump(custom), encoding="utf-8")

        design = parse_factorial_yaml(yaml_path)
        assert design.n_conditions == 2  # 2×1×1×1
        assert design.n_training_conditions == 2  # 2×1
        assert len(design.factor_names()) == 4  # model, loss, method, ensemble
