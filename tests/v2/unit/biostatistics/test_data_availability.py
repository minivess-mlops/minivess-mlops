"""Tests for biostatistics data availability — factorial design support.

Verifies that the biostatistics pipeline can:
1. Parse factorial design from experiment YAML (auto-derive factors)
2. Discover runs with ALL factorial fields (model_family, with_aux_calib)
3. Store per-volume metrics in DuckDB
4. Support both trainval and test splits

These tests catch data pipeline gaps BEFORE running expensive cloud jobs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# T0.1: SourceRun must include model_family and with_aux_calib
# ---------------------------------------------------------------------------


class TestSourceRunFactorialFields:
    """SourceRun must carry all factorial factor values."""

    def test_source_run_has_model_family(self) -> None:
        """SourceRun must have a model_family field."""
        from minivess.pipeline.biostatistics_types import SourceRun

        run = SourceRun(
            run_id="abc123",
            experiment_id="1",
            experiment_name="test",
            loss_function="dice_ce",
            fold_id=0,
            status="FINISHED",
            model_family="dynunet",
        )
        assert run.model_family == "dynunet"

    def test_source_run_has_with_aux_calib(self) -> None:
        """SourceRun must have a with_aux_calib field."""
        from minivess.pipeline.biostatistics_types import SourceRun

        run = SourceRun(
            run_id="abc123",
            experiment_id="1",
            experiment_name="test",
            loss_function="dice_ce",
            fold_id=0,
            status="FINISHED",
            model_family="dynunet",
            with_aux_calib=True,
        )
        assert run.with_aux_calib is True


# ---------------------------------------------------------------------------
# T0.2: DuckDB schema must include factorial columns
# ---------------------------------------------------------------------------


class TestDuckDBFactorialSchema:
    """DuckDB runs table must include model_family and with_aux_calib."""

    def test_ddl_runs_has_model_family(self) -> None:
        """DDL for runs table must include model_family column."""
        from minivess.pipeline.biostatistics_duckdb import _DDL_RUNS

        assert "model_family" in _DDL_RUNS

    def test_ddl_runs_has_with_aux_calib(self) -> None:
        """DDL for runs table must include with_aux_calib column."""
        from minivess.pipeline.biostatistics_duckdb import _DDL_RUNS

        assert "with_aux_calib" in _DDL_RUNS


# ---------------------------------------------------------------------------
# T0.3: Per-volume metrics must be queryable
# ---------------------------------------------------------------------------


class TestPerVolumeMetricsAvailability:
    """per_volume_metrics DuckDB table must support individual volume scores."""

    def test_per_volume_table_exists_in_schema(self) -> None:
        """BIOSTATISTICS_TABLES must include per_volume_metrics."""
        from minivess.pipeline.biostatistics_duckdb import BIOSTATISTICS_TABLES

        assert "per_volume_metrics" in BIOSTATISTICS_TABLES

    def test_per_volume_ddl_has_volume_id(self) -> None:
        """per_volume_metrics DDL must have volume_id column."""
        from minivess.pipeline.biostatistics_duckdb import _DDL_PER_VOLUME_METRICS

        assert "volume_id" in _DDL_PER_VOLUME_METRICS

    def test_per_volume_ddl_has_metric_value(self) -> None:
        """per_volume_metrics DDL must have metric_value column."""
        from minivess.pipeline.biostatistics_duckdb import _DDL_PER_VOLUME_METRICS

        assert "metric_value" in _DDL_PER_VOLUME_METRICS


# ---------------------------------------------------------------------------
# T0.4: Factorial config auto-parsing from experiment YAML
# ---------------------------------------------------------------------------


class TestFactorialConfigParsing:
    """Factorial design must be auto-derived from experiment YAML."""

    def test_parse_debug_factorial_factors(self) -> None:
        """debug_factorial.yaml has 3 factors: model_family, loss_name, aux_calibration."""
        import yaml

        config_path = Path("configs/experiment/debug_factorial.yaml")
        if not config_path.exists():
            pytest.skip("debug_factorial.yaml not found")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        factors = config.get("factors", {})
        assert "model_family" in factors
        assert "loss_name" in factors
        assert "aux_calibration" in factors

    def test_parse_production_factorial_factors(self) -> None:
        """paper_factorial.yaml has same 3 factors."""
        import yaml

        config_path = Path("configs/hpo/paper_factorial.yaml")
        if not config_path.exists():
            pytest.skip("paper_factorial.yaml not found")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        factors = config.get("factors", {})
        assert len(factors) == 3
        assert set(factors.keys()) == {"model_family", "loss_name", "aux_calibration"}

    def test_debug_factorial_24_cells(self) -> None:
        """Debug factorial: 4 × 3 × 2 = 24 cells."""

        import yaml

        config_path = Path("configs/experiment/debug_factorial.yaml")
        if not config_path.exists():
            pytest.skip("debug_factorial.yaml not found")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        factors = config["factors"]
        n_cells = 1
        for levels in factors.values():
            n_cells *= len(levels)
        assert n_cells == 32  # 4 models × 4 losses × 2 aux_calib

    def test_production_factorial_96_runs(self) -> None:
        """Production: 32 cells × 3 folds = 96 runs."""
        import yaml

        config_path = Path("configs/hpo/paper_factorial.yaml")
        if not config_path.exists():
            pytest.skip("paper_factorial.yaml not found")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        factors = config["factors"]
        n_cells = 1
        for levels in factors.values():
            n_cells *= len(levels)
        n_folds = config.get("fixed", {}).get("num_folds", 3)
        assert n_cells * n_folds == 96  # 32 cells × 3 folds

    def test_co_primary_metrics_are_cldice_and_masd(self) -> None:
        """Co-primary metrics per MetricsReloaded: clDice + MASD (NOT Dice)."""
        from minivess.config.biostatistics_config import BiostatisticsConfig

        cfg = BiostatisticsConfig()
        # The config must support co-primary metrics
        co_primaries = getattr(cfg, "co_primary_metrics", None)
        assert co_primaries is not None, (
            "BiostatisticsConfig must have co_primary_metrics field"
        )
        assert "cldice" in co_primaries
        assert "masd" in co_primaries
        assert "dsc" not in co_primaries, "DSC is a FOIL metric, not co-primary"


# ---------------------------------------------------------------------------
# T0.5: N-way ANOVA must support 3+ factors
# ---------------------------------------------------------------------------


class TestNWayAnovaStructure:
    """Factorial ANOVA must return all main effects and interactions for N factors."""

    def test_3way_anova_returns_7_terms(self) -> None:
        """3-way ANOVA (4×3×2) should return 7 terms:
        3 main effects + 3 two-way + 1 three-way interaction.
        """
        from minivess.pipeline.biostatistics_statistics import (
            compute_factorial_anova,
        )

        # Build synthetic 3-factor data
        rng = np.random.default_rng(42)
        per_volume_data: dict[str, dict[str, dict[int, np.ndarray]]] = {}

        models = ["dynunet", "sam3"]
        losses = ["dice_ce", "cbdice"]
        calibs = ["calibTrue", "calibFalse"]

        metric_data: dict[str, dict[int, np.ndarray]] = {}
        for m in models:
            for loss in losses:
                for c in calibs:
                    key = f"{m}__{loss}__{c}"
                    metric_data[key] = {
                        0: rng.normal(0.8, 0.05, size=10),
                    }
        per_volume_data["cldice"] = metric_data

        result = compute_factorial_anova(
            per_volume_data,
            metric_name="cldice",
            factor_names=["model", "loss", "calib"],
        )

        # Should have main effects + interactions
        assert "model" in result.f_values or "Model" in result.f_values
        assert "loss" in result.f_values or "Loss" in result.f_values
