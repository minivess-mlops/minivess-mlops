"""Tests for BiostatisticsConfig 2-fold experiment override — Plan Task 1.4.

Verifies:
- analysis_duckdb_path field exists and is optional (None default)
- Config validates with min_folds_per_condition=2
- Config loads from YAML file with experiment-specific overrides
- validate_source_completeness passes with 2 folds

Pure unit tests — no Docker, no Prefect, no DagsHub.
"""

from __future__ import annotations

from pathlib import Path

from minivess.config.biostatistics_config import BiostatisticsConfig


class TestAnalysisDuckdbPathField:
    """Tests for the new analysis_duckdb_path field."""

    def test_field_exists_and_none_by_default(self) -> None:
        """analysis_duckdb_path should default to None."""
        config = BiostatisticsConfig()
        assert config.analysis_duckdb_path is None

    def test_field_accepts_path(self) -> None:
        """analysis_duckdb_path should accept a Path value."""
        config = BiostatisticsConfig(
            analysis_duckdb_path=Path("/tmp/test/analysis.duckdb")  # noqa: S108
        )
        assert config.analysis_duckdb_path == Path("/tmp/test/analysis.duckdb")  # noqa: S108


class TestTwoFoldConfig:
    """Tests for 2-fold experiment config overrides."""

    def test_min_folds_2_validates(self) -> None:
        """Config with min_folds_per_condition=2 should validate."""
        config = BiostatisticsConfig(min_folds_per_condition=2)
        assert config.min_folds_per_condition == 2

    def test_validation_passes_with_2_folds(self) -> None:
        """validate_source_completeness must pass with 2 folds when config allows it."""
        from minivess.pipeline.biostatistics_discovery import (
            validate_source_completeness,
        )
        from minivess.pipeline.biostatistics_types import (
            SourceRun,
            SourceRunManifest,
        )

        runs = [
            SourceRun(
                run_id=f"run_{loss}_{fold}",
                experiment_id="exp1",
                experiment_name="test",
                loss_function=loss,
                fold_id=fold,
                status="FINISHED",
            )
            for loss in ["dice_ce", "cbdice_cldice"]
            for fold in range(2)
        ]
        manifest = SourceRunManifest.from_runs(runs)

        result = validate_source_completeness(
            manifest,
            min_folds=2,  # Allow 2 folds
            min_conditions=2,
        )
        assert result.valid is True
        assert result.n_conditions == 2
        assert result.n_folds_per_condition == 2

    def test_yaml_config_loads(self, tmp_path: Path) -> None:
        """YAML config file with 2-fold overrides should load correctly."""
        import yaml

        config_data = {
            "min_folds_per_condition": 2,
            "experiment_names": ["local_dynunet_mechanics_training"],
            "co_primary_metrics": ["cldice", "assd"],
            "foil_metrics": ["dsc"],
            "calibration_co_primary_metrics": ["cal_ece", "cal_ba_ece"],
            "splits": ["trainval", "test"],
        }

        yaml_path = tmp_path / "local_dynunet_debug.yaml"
        yaml_path.write_text(
            yaml.dump(config_data, default_flow_style=False),
            encoding="utf-8",
        )

        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        config = BiostatisticsConfig(**loaded)

        assert config.min_folds_per_condition == 2
        assert config.experiment_names == ["local_dynunet_mechanics_training"]
        assert config.co_primary_metrics == ["cldice", "assd"]
        assert config.foil_metrics == ["dsc"]
