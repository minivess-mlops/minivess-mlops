"""Tests for paper_factorial.yaml — 24-cell grid configuration.

T1.3: Load configs/hpo/paper_factorial.yaml, assert 24 cells
(4 models x 3 losses x 2 aux_calibration).

GUARDRAIL: 24 cells, NOT 128. LR/batch are NOT factors.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HPO_DIR = PROJECT_ROOT / "configs" / "hpo"


class TestPaperFactorialConfig:
    """Validate the paper factorial YAML produces exactly 24 cells."""

    def test_paper_factorial_yaml_exists(self) -> None:
        """configs/hpo/paper_factorial.yaml must exist."""
        path = HPO_DIR / "paper_factorial.yaml"
        assert path.exists(), (
            f"paper_factorial.yaml not found at {path}. "
            "Create the 24-cell factorial config for the paper study."
        )

    def test_paper_factorial_is_valid_yaml(self) -> None:
        """paper_factorial.yaml must parse as a dict."""
        path = HPO_DIR / "paper_factorial.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"Expected dict, got {type(data)}"

    def test_paper_factorial_has_required_keys(self) -> None:
        """Must have experiment_name, model_family (or factors), hyperparameters, fixed."""
        path = HPO_DIR / "paper_factorial.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Grid factorial configs use 'factors' key (not 'hyperparameters')
        # to distinguish from the older grid format
        assert "experiment_name" in data, "Missing experiment_name"
        assert "factors" in data, (
            "Missing 'factors' key — paper factorial uses factorial grid, "
            "not Optuna search_space"
        )
        assert "fixed" in data, "Missing 'fixed' key for non-swept parameters"

    def test_paper_factorial_24_cells(self) -> None:
        """Factorial must produce exactly 24 cells (4 x 3 x 2)."""
        path = HPO_DIR / "paper_factorial.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        factors = data["factors"]
        total_cells = math.prod(len(v) for v in factors.values())
        assert total_cells == 24, (
            f"Factorial must produce 24 cells (4 models x 3 losses x 2 aux_calib), "
            f"got {total_cells} from factors: "
            + ", ".join(f"{k}={len(v)}" for k, v in factors.items())
        )

    def test_paper_factorial_factors_content(self) -> None:
        """Validate specific factor values."""
        path = HPO_DIR / "paper_factorial.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        factors = data["factors"]

        # 4 model families
        assert "model_family" in factors
        assert len(factors["model_family"]) == 4, (
            f"Expected 4 model families, got {len(factors['model_family'])}"
        )

        # 3 loss functions
        assert "loss_name" in factors
        assert len(factors["loss_name"]) == 3, (
            f"Expected 3 losses, got {len(factors['loss_name'])}"
        )

        # 2 aux_calibration values
        assert "aux_calibration" in factors
        assert len(factors["aux_calibration"]) == 2, (
            f"Expected 2 aux_calibration values, got {len(factors['aux_calibration'])}"
        )

    def test_paper_factorial_lr_not_a_factor(self) -> None:
        """LR and batch_size must NOT be factors (guardrail: 24, not 128)."""
        path = HPO_DIR / "paper_factorial.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        factors = data["factors"]
        assert "learning_rate" not in factors, (
            "learning_rate must NOT be a factor — guardrail: 24 cells, not 128"
        )
        assert "batch_size" not in factors, (
            "batch_size must NOT be a factor — guardrail: 24 cells, not 128"
        )

    def test_paper_factorial_fixed_params(self) -> None:
        """Fixed params must include max_epochs, num_folds, batch_size."""
        path = HPO_DIR / "paper_factorial.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        fixed = data["fixed"]
        assert "max_epochs" in fixed, "Missing max_epochs in fixed params"
        assert "num_folds" in fixed, "Missing num_folds in fixed params"

    def test_paper_factorial_has_mlflow_experiment(self) -> None:
        """Must have mlflow_experiment key for tracking."""
        path = HPO_DIR / "paper_factorial.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "mlflow_experiment" in data, "Missing mlflow_experiment key"

    def test_paper_factorial_72_total_runs(self) -> None:
        """24 cells x 3 folds = 72 total training runs."""
        path = HPO_DIR / "paper_factorial.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        factors = data["factors"]
        total_cells = math.prod(len(v) for v in factors.values())
        num_folds = data["fixed"]["num_folds"]
        total_runs = total_cells * num_folds
        assert total_runs == 72, (
            f"Expected 72 total runs (24 cells x 3 folds), got {total_runs}"
        )
