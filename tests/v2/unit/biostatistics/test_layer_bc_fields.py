"""Tests for Layer B+C factorial fields in biostatistics pipeline.

Phases 0.5 + 1 of the final QA plan:
- SourceRun must accept post_training_method, recalibration, ensemble_strategy, is_zero_shot
- DuckDB schema must have 4 new columns
- Discovery must read 4 new tags from MLflow
- Post-training flow must write recalibration tag
- Analysis flow must write ensemble_strategy tag
- Factor name mapping: YAML 'method' → tag 'post_training_method'
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Phase 1: SourceRun Layer B+C fields
# ---------------------------------------------------------------------------


class TestSourceRunLayerBCFields:
    """SourceRun must accept Layer B+C factorial fields."""

    def test_accepts_post_training_method(self) -> None:
        from minivess.pipeline.biostatistics_types import SourceRun

        run = SourceRun(
            run_id="r1",
            experiment_id="e1",
            experiment_name="test",
            loss_function="dice_ce",
            fold_id=0,
            status="FINISHED",
            post_training_method="swag",
        )
        assert run.post_training_method == "swag"

    def test_defaults_post_training_method_to_none(self) -> None:
        from minivess.pipeline.biostatistics_types import SourceRun

        run = SourceRun(
            run_id="r1",
            experiment_id="e1",
            experiment_name="test",
            loss_function="dice_ce",
            fold_id=0,
            status="FINISHED",
        )
        assert run.post_training_method == "none"

    def test_accepts_recalibration(self) -> None:
        from minivess.pipeline.biostatistics_types import SourceRun

        run = SourceRun(
            run_id="r1",
            experiment_id="e1",
            experiment_name="test",
            loss_function="dice_ce",
            fold_id=0,
            status="FINISHED",
            recalibration="temperature_scaling",
        )
        assert run.recalibration == "temperature_scaling"

    def test_defaults_recalibration_to_none(self) -> None:
        from minivess.pipeline.biostatistics_types import SourceRun

        run = SourceRun(
            run_id="r1",
            experiment_id="e1",
            experiment_name="test",
            loss_function="dice_ce",
            fold_id=0,
            status="FINISHED",
        )
        assert run.recalibration == "none"

    def test_accepts_ensemble_strategy(self) -> None:
        from minivess.pipeline.biostatistics_types import SourceRun

        run = SourceRun(
            run_id="r1",
            experiment_id="e1",
            experiment_name="test",
            loss_function="dice_ce",
            fold_id=0,
            status="FINISHED",
            ensemble_strategy="all_loss_all_best",
        )
        assert run.ensemble_strategy == "all_loss_all_best"

    def test_accepts_is_zero_shot(self) -> None:
        from minivess.pipeline.biostatistics_types import SourceRun

        run = SourceRun(
            run_id="r1",
            experiment_id="e1",
            experiment_name="test",
            loss_function="dice_ce",
            fold_id=0,
            status="FINISHED",
            is_zero_shot=True,
        )
        assert run.is_zero_shot is True

    def test_defaults_is_zero_shot_to_false(self) -> None:
        from minivess.pipeline.biostatistics_types import SourceRun

        run = SourceRun(
            run_id="r1",
            experiment_id="e1",
            experiment_name="test",
            loss_function="dice_ce",
            fold_id=0,
            status="FINISHED",
        )
        assert run.is_zero_shot is False


# ---------------------------------------------------------------------------
# Phase 1: DuckDB schema Layer B+C columns
# ---------------------------------------------------------------------------


class TestDuckDBLayerBCColumns:
    """DuckDB runs table DDL must include Layer B+C columns."""

    def test_ddl_has_post_training_method(self) -> None:
        from minivess.pipeline.biostatistics_duckdb import _DDL_RUNS

        assert "post_training_method" in _DDL_RUNS

    def test_ddl_has_recalibration(self) -> None:
        from minivess.pipeline.biostatistics_duckdb import _DDL_RUNS

        assert "recalibration" in _DDL_RUNS

    def test_ddl_has_ensemble_strategy(self) -> None:
        from minivess.pipeline.biostatistics_duckdb import _DDL_RUNS

        assert "ensemble_strategy" in _DDL_RUNS

    def test_ddl_has_is_zero_shot(self) -> None:
        from minivess.pipeline.biostatistics_duckdb import _DDL_RUNS

        assert "is_zero_shot" in _DDL_RUNS

    def test_insert_run_populates_new_columns(self, tmp_path: Path) -> None:
        """Full round-trip: insert SourceRun, query DuckDB, verify new columns."""
        import duckdb

        from minivess.pipeline.biostatistics_duckdb import _ALL_DDL

        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        for ddl in _ALL_DDL:
            conn.execute(ddl)

        # Insert with new columns via direct SQL to verify schema
        conn.execute(
            """INSERT INTO runs (run_id, experiment_id, experiment_name,
               loss_function, fold_id, model_family, with_aux_calib, status,
               start_time, post_training_method, recalibration,
               ensemble_strategy, is_zero_shot)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                "r1",
                "e1",
                "test",
                "dice_ce",
                0,
                "dynunet",
                False,
                "FINISHED",
                "",
                "swag",
                "temperature_scaling",
                "all_loss_all_best",
                True,
            ],
        )

        result = conn.execute(
            "SELECT post_training_method, recalibration, ensemble_strategy, "
            "is_zero_shot FROM runs WHERE run_id = 'r1'"
        ).fetchone()
        conn.close()

        assert result is not None
        assert result[0] == "swag"
        assert result[1] == "temperature_scaling"
        assert result[2] == "all_loss_all_best"
        assert result[3] is True


# ---------------------------------------------------------------------------
# Phase 1: Discovery reads Layer B+C tags
# ---------------------------------------------------------------------------


class TestDiscoveryLayerBCTags:
    """_parse_run_dir must read Layer B+C tags from MLflow."""

    def _make_run_dir(
        self,
        tmp_path: Path,
        *,
        tags: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> Path:
        """Create a minimal MLflow run directory with tags and params."""
        import yaml

        run_dir = tmp_path / "run1"
        run_dir.mkdir()

        # meta.yaml
        (run_dir / "meta.yaml").write_text(
            yaml.dump({"status": "FINISHED"}),
            encoding="utf-8",
        )

        # params directory
        params_dir = run_dir / "params"
        params_dir.mkdir()
        default_params = {"loss_name": "dice_ce", "model_family": "dynunet"}
        if params:
            default_params.update(params)
        for k, v in default_params.items():
            (params_dir / k).write_text(v, encoding="utf-8")

        # tags directory
        tags_dir = run_dir / "tags"
        tags_dir.mkdir()
        if tags:
            for k, v in tags.items():
                (tags_dir / k).write_text(v, encoding="utf-8")

        return run_dir

    def test_reads_post_training_method_tag(self, tmp_path: Path) -> None:
        from minivess.pipeline.biostatistics_discovery import _parse_run_dir

        run_dir = self._make_run_dir(
            tmp_path,
            tags={"post_training_method": "swag"},
        )
        run = _parse_run_dir(run_dir, "e1", "test")
        assert run is not None
        assert run.post_training_method == "swag"

    def test_reads_recalibration_tag(self, tmp_path: Path) -> None:
        from minivess.pipeline.biostatistics_discovery import _parse_run_dir

        run_dir = self._make_run_dir(
            tmp_path,
            tags={"recalibration": "temperature_scaling"},
        )
        run = _parse_run_dir(run_dir, "e1", "test")
        assert run is not None
        assert run.recalibration == "temperature_scaling"

    def test_reads_ensemble_strategy_tag(self, tmp_path: Path) -> None:
        from minivess.pipeline.biostatistics_discovery import _parse_run_dir

        run_dir = self._make_run_dir(
            tmp_path,
            tags={"ensemble_strategy": "all_loss_all_best"},
        )
        run = _parse_run_dir(run_dir, "e1", "test")
        assert run is not None
        assert run.ensemble_strategy == "all_loss_all_best"

    def test_reads_is_zero_shot_tag(self, tmp_path: Path) -> None:
        from minivess.pipeline.biostatistics_discovery import _parse_run_dir

        run_dir = self._make_run_dir(
            tmp_path,
            tags={"is_zero_shot": "true"},
        )
        run = _parse_run_dir(run_dir, "e1", "test")
        assert run is not None
        assert run.is_zero_shot is True

    def test_defaults_to_none_without_tags(self, tmp_path: Path) -> None:
        from minivess.pipeline.biostatistics_discovery import _parse_run_dir

        run_dir = self._make_run_dir(tmp_path)
        run = _parse_run_dir(run_dir, "e1", "test")
        assert run is not None
        assert run.post_training_method == "none"
        assert run.recalibration == "none"
        assert run.ensemble_strategy == "none"
        assert run.is_zero_shot is False


# ---------------------------------------------------------------------------
# Phase 0.5: Factor name mapping
# ---------------------------------------------------------------------------


class TestFactorNameMapping:
    """Factorial YAML 'method' must map to tag 'post_training_method'."""

    def test_factorial_yaml_method_maps_to_post_training_method(self) -> None:
        """The YAML factor 'method' under post_training must be understood
        as 'post_training_method' in MLflow tags and DuckDB."""
        from minivess.config.factorial_config import (
            FACTOR_NAME_MAPPING,
        )

        assert "method" in FACTOR_NAME_MAPPING
        assert FACTOR_NAME_MAPPING["method"] == "post_training_method"

    def test_mapping_has_aux_calibration(self) -> None:
        from minivess.config.factorial_config import FACTOR_NAME_MAPPING

        assert "aux_calibration" in FACTOR_NAME_MAPPING
        assert FACTOR_NAME_MAPPING["aux_calibration"] == "with_aux_calib"
