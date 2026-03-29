"""Tests for smoke_mini factorial and biostatistics configs (Phase 0).

Validates that the mini-experiment configs exist, are well-formed, and
match the plan specification: 2 losses × 2 post-training methods × 1 ensemble
= 4 conditions, 3 folds, 20 epochs, with DeepVess external test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
FACTORIAL_PATH = REPO_ROOT / "configs" / "factorial" / "smoke_mini.yaml"
BIOSTAT_PATH = REPO_ROOT / "configs" / "biostatistics" / "smoke_mini.yaml"


# ── Factorial YAML existence and structure ──────────────────────────────


class TestSmokeMiniFactorialExists:
    def test_factorial_yaml_exists(self) -> None:
        assert FACTORIAL_PATH.exists(), f"Missing: {FACTORIAL_PATH}"

    def test_factorial_yaml_is_valid(self) -> None:
        with FACTORIAL_PATH.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict)


class TestSmokeMiniFactorialDesign:
    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        with FACTORIAL_PATH.open(encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def test_debug_mode_true(self) -> None:
        assert self.cfg["debug"] is True

    def test_experiment_name(self) -> None:
        assert self.cfg["experiment_name"] == "smoke_mini"

    # ── Layer A: Training ──────────────────────────────────────────────

    def test_model_family_is_dynunet_only(self) -> None:
        models = self.cfg["factors"]["training"]["model_family"]
        assert models == ["dynunet"]

    def test_loss_names_two(self) -> None:
        losses = self.cfg["factors"]["training"]["loss_name"]
        assert set(losses) == {"dice_ce", "cbdice_cldice"}

    def test_aux_calibration_false_only(self) -> None:
        calib = self.cfg["factors"]["training"]["aux_calibration"]
        assert calib == [False]

    def test_layer_a_cell_count(self) -> None:
        t = self.cfg["factors"]["training"]
        cells = len(t["model_family"]) * len(t["loss_name"]) * len(t["aux_calibration"])
        assert cells == 2, f"Expected 2 Layer A cells, got {cells}"

    # ── Layer B: Post-Training ─────────────────────────────────────────

    def test_post_training_methods(self) -> None:
        methods = self.cfg["factors"]["post_training"]["method"]
        assert set(methods) == {"none", "checkpoint_averaging"}

    def test_recalibration_none_only(self) -> None:
        recalib = self.cfg["factors"]["post_training"]["recalibration"]
        assert recalib == ["none"]

    def test_layer_b_cell_count(self) -> None:
        pt = self.cfg["factors"]["post_training"]
        cells = len(pt["method"]) * len(pt["recalibration"])
        assert cells == 2, f"Expected 2 Layer B cells, got {cells}"

    # ── Layer C: Analysis ──────────────────────────────────────────────

    def test_ensemble_strategy_single(self) -> None:
        strategies = self.cfg["factors"]["analysis"]["ensemble_strategy"]
        assert strategies == ["per_loss_single_best"]

    def test_layer_c_cell_count(self) -> None:
        a = self.cfg["factors"]["analysis"]
        cells = len(a["ensemble_strategy"])
        assert cells == 1

    # ── Total conditions ───────────────────────────────────────────────

    def test_total_conditions_is_4(self) -> None:
        t = self.cfg["factors"]["training"]
        pt = self.cfg["factors"]["post_training"]
        a = self.cfg["factors"]["analysis"]
        total = (
            len(t["model_family"])
            * len(t["loss_name"])
            * len(t["aux_calibration"])
            * len(pt["method"])
            * len(pt["recalibration"])
            * len(a["ensemble_strategy"])
        )
        assert total == 4, f"Expected 4 total conditions, got {total}"

    # ── Fixed settings ─────────────────────────────────────────────────

    def test_max_epochs_20(self) -> None:
        assert self.cfg["fixed"]["max_epochs"] == 20

    def test_num_folds_3(self) -> None:
        assert self.cfg["fixed"]["num_folds"] == 3

    # ── MLflow experiments ─────────────────────────────────────────────

    def test_mlflow_training_experiment(self) -> None:
        assert "training_experiment" in self.cfg["mlflow"]

    def test_mlflow_evaluation_experiment(self) -> None:
        assert self.cfg["mlflow"]["evaluation_experiment"] == "smoke_mini_evaluation"

    def test_mlflow_biostatistics_experiment(self) -> None:
        assert self.cfg["mlflow"]["biostatistics_experiment"] == "smoke_mini_biostatistics"


# ── Biostatistics YAML existence and structure ──────────────────────────


class TestSmokeMinibiostatsExists:
    def test_biostatistics_yaml_exists(self) -> None:
        assert BIOSTAT_PATH.exists(), f"Missing: {BIOSTAT_PATH}"

    def test_biostatistics_yaml_is_valid(self) -> None:
        with BIOSTAT_PATH.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict)


class TestSmokeMinibiostatsConfig:
    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        with BIOSTAT_PATH.open(encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def test_experiment_name(self) -> None:
        assert "smoke_mini_evaluation" in self.cfg["experiment_names"]

    def test_output_dir(self) -> None:
        assert self.cfg["output_dir"] == "outputs/biostatistics_smoke_mini"

    def test_factorial_yaml_reference(self) -> None:
        assert self.cfg["factorial_yaml"] == "configs/factorial/smoke_mini.yaml"

    def test_metrics_include_core(self) -> None:
        metrics = self.cfg["metrics"]
        assert "dsc" in metrics
        assert "cldice" in metrics
        assert "masd" in metrics

    def test_primary_metric_is_cldice(self) -> None:
        assert self.cfg["primary_metric"] == "cldice"

    def test_alpha_from_config(self) -> None:
        from minivess.config.biostatistics_config import BiostatisticsConfig

        expected_alpha = BiostatisticsConfig().alpha
        assert self.cfg["alpha"] == expected_alpha

    def test_n_bootstrap_reduced(self) -> None:
        # Mini experiment uses reduced bootstrap for speed
        assert self.cfg["n_bootstrap"] <= 1000

    def test_seed(self) -> None:
        assert self.cfg["seed"] == 42

    def test_min_folds_per_condition_3(self) -> None:
        # Full 3-fold CV for mini experiment
        assert self.cfg["min_folds_per_condition"] == 3

    def test_rope_values_present(self) -> None:
        rope = self.cfg["rope_values"]
        assert "dsc" in rope
        assert "cldice" in rope
        assert all(v > 0 for v in rope.values())

    def test_splits_include_test(self) -> None:
        """Biostatistics config should include test split for DeepVess."""
        splits = self.cfg.get("splits", ["trainval"])
        assert "test" in splits


# ── Cross-config consistency ───────────────────────────────────────────


class TestCrossConfigConsistency:
    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        with FACTORIAL_PATH.open(encoding="utf-8") as f:
            self.factorial = yaml.safe_load(f)
        with BIOSTAT_PATH.open(encoding="utf-8") as f:
            self.biostat = yaml.safe_load(f)

    def test_biostat_references_correct_factorial(self) -> None:
        assert self.biostat["factorial_yaml"] == "configs/factorial/smoke_mini.yaml"

    def test_biostat_experiment_matches_factorial_eval(self) -> None:
        assert (
            self.factorial["mlflow"]["evaluation_experiment"]
            in self.biostat["experiment_names"]
        )

    def test_co_primary_metrics_in_biostat_metrics(self) -> None:
        """Co-primary metrics from factorial must be in biostat metric list."""
        co_primaries = self.factorial["biostatistics"]["co_primary_metrics"]
        biostat_metrics = self.biostat["metrics"]
        for metric in co_primaries:
            assert metric in biostat_metrics, f"Co-primary {metric} not in biostat metrics"
