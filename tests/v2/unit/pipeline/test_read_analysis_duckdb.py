"""Tests for read_analysis_duckdb() — Plan Task 2.1.

Verifies reading analysis_results.duckdb into SourceRunManifest + PerVolumeData.
Round-trip: build → read → correct data structure.

Pure unit tests — no Docker, no Prefect, no DagsHub.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from minivess.pipeline.biostatistics_types import SourceRun, SourceRunManifest


def _build_test_duckdb(tmp_path: Path) -> Path:
    """Build a test analysis_results.duckdb with known data."""
    from minivess.pipeline.biostatistics_duckdb import build_analysis_results_duckdb

    runs = []
    configs = [
        ("dice_ce", False),
        ("dice_ce", True),
        ("cbdice_cldice", False),
        ("cbdice_cldice", True),
    ]
    for loss, aux in configs:
        for fold in range(2):
            runs.append(
                SourceRun(
                    run_id=f"run_{loss}_{aux}_{fold}",
                    experiment_id="exp1",
                    experiment_name="test_exp",
                    loss_function=loss,
                    fold_id=fold,
                    model_family="dynunet",
                    with_aux_calib=aux,
                    status="FINISHED",
                )
            )

    manifest = SourceRunManifest.from_runs(runs)

    rng = np.random.default_rng(42)
    pv_records = []
    fold_records = []
    for run in runs:
        for vol_idx in range(5):
            vol_id = f"vol_{vol_idx:03d}"
            for metric in ["dsc", "hd95", "cldice", "cal_ece"]:
                pv_records.append({
                    "run_id": run.run_id,
                    "fold_id": run.fold_id,
                    "split": "trainval",
                    "dataset": "minivess",
                    "volume_id": vol_id,
                    "metric_name": metric,
                    "metric_value": float(rng.random()),
                })
        for metric in ["dsc", "hd95", "cldice"]:
            fold_records.append({
                "run_id": run.run_id,
                "fold_id": run.fold_id,
                "split": "trainval",
                "metric_name": metric,
                "metric_value": float(rng.random()),
            })

    db_path = tmp_path / "analysis_results.duckdb"
    build_analysis_results_duckdb(
        manifest=manifest,
        per_volume_records=pv_records,
        fold_metric_records=fold_records,
        output_path=db_path,
        metadata={"git_sha": "test123"},
    )
    return db_path


class TestReadAnalysisDuckdb:
    """Tests for read_analysis_duckdb() → (SourceRunManifest, PerVolumeData)."""

    def test_returns_correct_manifest(self, tmp_path: Path) -> None:
        """Manifest should have 8 runs with correct metadata."""
        from minivess.pipeline.biostatistics_duckdb import read_analysis_duckdb

        db_path = _build_test_duckdb(tmp_path)
        manifest, _pv_data = read_analysis_duckdb(db_path)

        assert len(manifest.runs) == 8
        assert all(r.status == "FINISHED" for r in manifest.runs)
        assert all(r.model_family == "dynunet" for r in manifest.runs)

    def test_four_conditions(self, tmp_path: Path) -> None:
        """PerVolumeData should have 4 distinct conditions."""
        from minivess.pipeline.biostatistics_duckdb import read_analysis_duckdb

        db_path = _build_test_duckdb(tmp_path)
        _manifest, pv_data = read_analysis_duckdb(db_path)

        # pv_data: {metric: {condition_key: {fold_id: np.ndarray}}}
        assert "dsc" in pv_data
        dsc_conditions = pv_data["dsc"]
        assert len(dsc_conditions) == 4

    def test_two_folds_per_condition(self, tmp_path: Path) -> None:
        """Each condition should have 2 folds."""
        from minivess.pipeline.biostatistics_duckdb import read_analysis_duckdb

        db_path = _build_test_duckdb(tmp_path)
        _manifest, pv_data = read_analysis_duckdb(db_path)

        for _metric, conditions in pv_data.items():
            for _cond, folds in conditions.items():
                assert len(folds) == 2
                assert 0 in folds
                assert 1 in folds

    def test_five_volumes_per_fold(self, tmp_path: Path) -> None:
        """Each fold should have 5 per-volume scores."""
        from minivess.pipeline.biostatistics_duckdb import read_analysis_duckdb

        db_path = _build_test_duckdb(tmp_path)
        _manifest, pv_data = read_analysis_duckdb(db_path)

        for _metric, conditions in pv_data.items():
            for _cond, folds in conditions.items():
                for _fold_id, scores in folds.items():
                    assert isinstance(scores, np.ndarray)
                    assert len(scores) == 5

    def test_has_calibration_metrics(self, tmp_path: Path) -> None:
        """PerVolumeData should include cal_ prefix metrics."""
        from minivess.pipeline.biostatistics_duckdb import read_analysis_duckdb

        db_path = _build_test_duckdb(tmp_path)
        _manifest, pv_data = read_analysis_duckdb(db_path)

        assert "cal_ece" in pv_data

    def test_manifest_with_aux_calib(self, tmp_path: Path) -> None:
        """Manifest runs should correctly reflect with_aux_calib flag."""
        from minivess.pipeline.biostatistics_duckdb import read_analysis_duckdb

        db_path = _build_test_duckdb(tmp_path)
        manifest, _pv_data = read_analysis_duckdb(db_path)

        aux_true = [r for r in manifest.runs if r.with_aux_calib]
        aux_false = [r for r in manifest.runs if not r.with_aux_calib]
        assert len(aux_true) == 4  # 2 losses × 2 folds
        assert len(aux_false) == 4
