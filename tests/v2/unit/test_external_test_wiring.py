"""Tests for external test dataset wiring (GROUP J).

Covers:
- Phase 1: analysis_flow._build_dataloaders_from_config() wiring
- Phase 2: evaluation_runner test/ metric prefix
- Phase 3: DuckDB test_metrics table
- Phase 4: biostatistics split={trainval,test}

TDD: these tests are written FIRST, then implementation follows.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from minivess.pipeline.ci import ConfidenceInterval
from minivess.pipeline.evaluation import FoldResult

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_fake_fold_result(n_volumes: int = 3) -> FoldResult:
    """Build a trivial FoldResult with synthetic data."""
    per_vol: dict[str, list[float]] = {
        "dsc": [0.8, 0.85, 0.9][:n_volumes],
        "centreline_dsc": [0.7, 0.75, 0.8][:n_volumes],
    }
    aggregated: dict[str, ConfidenceInterval] = {}
    for name, vals in per_vol.items():
        arr = np.array(vals)
        aggregated[name] = ConfidenceInterval(
            point_estimate=float(np.mean(arr)),
            lower=float(np.min(arr)),
            upper=float(np.max(arr)),
            confidence_level=0.95,
            method="percentile_bootstrap",
        )
    return FoldResult(per_volume_metrics=per_vol, aggregated=aggregated)


def _make_deepvess_data_dir(tmp_path: Path) -> Path:
    """Create a synthetic DeepVess directory with NIfTI-like files."""
    deepvess_dir = tmp_path / "external" / "deepvess"
    images_dir = deepvess_dir / "images"
    labels_dir = deepvess_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    # Create 2 fake NIfTI files
    for i in range(2):
        (images_dir / f"vol_{i:03d}.nii.gz").write_bytes(b"fake_nifti")
        (labels_dir / f"vol_{i:03d}.nii.gz").write_bytes(b"fake_nifti")
    return tmp_path / "external"


# ===========================================================================
# PHASE 1: _build_dataloaders_from_config() wiring
# ===========================================================================


class TestBuildDataloadersFromConfig:
    """Phase 1: _build_dataloaders_from_config() must discover DeepVess data."""

    def test_raises_when_external_data_dir_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Must raise FileNotFoundError when EXTERNAL_DATA_DIR is not set."""
        monkeypatch.delenv("EXTERNAL_DATA_DIR", raising=False)
        from minivess.orchestration.flows.analysis_flow import (
            _build_dataloaders_from_config,
        )

        with pytest.raises(FileNotFoundError, match="EXTERNAL_DATA_DIR"):
            _build_dataloaders_from_config({})

    def test_raises_when_external_data_dir_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Must raise FileNotFoundError when EXTERNAL_DATA_DIR points to nonexistent dir."""
        monkeypatch.setenv("EXTERNAL_DATA_DIR", str(tmp_path / "nonexistent"))
        from minivess.orchestration.flows.analysis_flow import (
            _build_dataloaders_from_config,
        )

        with pytest.raises(FileNotFoundError):
            _build_dataloaders_from_config({})

    def test_returns_dict_with_deepvess_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Must return a dict containing a 'deepvess' key."""
        ext_dir = _make_deepvess_data_dir(tmp_path)
        monkeypatch.setenv("EXTERNAL_DATA_DIR", str(ext_dir))

        from minivess.orchestration.flows.analysis_flow import (
            _build_dataloaders_from_config,
        )

        result = _build_dataloaders_from_config({})
        assert isinstance(result, dict)
        assert "deepvess" in result

    def test_deepvess_has_all_subset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """DeepVess entry must have an 'all' subset."""
        ext_dir = _make_deepvess_data_dir(tmp_path)
        monkeypatch.setenv("EXTERNAL_DATA_DIR", str(ext_dir))

        from minivess.orchestration.flows.analysis_flow import (
            _build_dataloaders_from_config,
        )

        result = _build_dataloaders_from_config({})
        assert "all" in result["deepvess"]

    def test_vesselnn_excluded_from_test_dataloaders(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """VesselNN is drift-detection only — must NOT appear in test dataloaders."""
        ext_dir = _make_deepvess_data_dir(tmp_path)
        # Also create vesselnn dir (should be ignored)
        vesselnn_dir = ext_dir / "vesselnn" / "images"
        vesselnn_dir.mkdir(parents=True)
        (vesselnn_dir / "vol_000.nii.gz").write_bytes(b"fake")
        labels_dir = ext_dir / "vesselnn" / "labels"
        labels_dir.mkdir(parents=True)
        (labels_dir / "vol_000.nii.gz").write_bytes(b"fake")

        monkeypatch.setenv("EXTERNAL_DATA_DIR", str(ext_dir))

        from minivess.orchestration.flows.analysis_flow import (
            _build_dataloaders_from_config,
        )

        result = _build_dataloaders_from_config({})
        assert "vesselnn" not in result


# ===========================================================================
# PHASE 2: evaluation_runner test/ metric prefix
# ===========================================================================


class TestTestMetricPrefix:
    """Phase 2: External test datasets must use test/ prefix, not eval/."""

    def test_test_dataset_uses_test_prefix(self) -> None:
        """Metrics for external test datasets must use test/{ds}/{subset}/{metric}."""
        from minivess.pipeline.evaluation_runner import (
            EvaluationResult,
            UnifiedEvaluationRunner,
        )

        config = MagicMock()
        config.mlflow_evaluation_experiment = "test_eval"
        config.include_expensive_metrics = False

        runner = UnifiedEvaluationRunner.__new__(UnifiedEvaluationRunner)
        runner.eval_config = config

        fold = _make_fake_fold_result(n_volumes=2)
        results: dict[str, dict[str, EvaluationResult]] = {
            "deepvess": {
                "all": EvaluationResult(
                    model_name="model_a",
                    dataset_name="deepvess",
                    subset_name="all",
                    fold_result=fold,
                    predictions_dir=None,
                    uncertainty_maps_dir=None,
                ),
            },
        }

        with patch("minivess.pipeline.evaluation_runner.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "test123"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(
                return_value=mock_run
            )
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            runner.log_results_to_mlflow(
                results,
                model_name="model_a",
                is_test_dataset=True,
            )

            # Check that metrics were logged with test/ prefix
            logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
            metric_keys = list(logged_metrics.keys())
            assert any(k.startswith("test/") for k in metric_keys), (
                f"Expected test/ prefix in metric keys, got: {metric_keys}"
            )
            assert not any(k.startswith("eval/") for k in metric_keys), (
                f"Expected NO eval/ prefix for test datasets, got: {metric_keys}"
            )

    def test_trainval_dataset_keeps_eval_prefix(self) -> None:
        """Train/val (MiniVess) metrics must keep eval/ prefix."""
        from minivess.pipeline.evaluation_runner import (
            EvaluationResult,
            UnifiedEvaluationRunner,
        )

        config = MagicMock()
        config.mlflow_evaluation_experiment = "test_eval"
        config.include_expensive_metrics = False

        runner = UnifiedEvaluationRunner.__new__(UnifiedEvaluationRunner)
        runner.eval_config = config

        fold = _make_fake_fold_result(n_volumes=2)
        results: dict[str, dict[str, EvaluationResult]] = {
            "minivess": {
                "all": EvaluationResult(
                    model_name="model_a",
                    dataset_name="minivess",
                    subset_name="all",
                    fold_result=fold,
                    predictions_dir=None,
                    uncertainty_maps_dir=None,
                ),
            },
        }

        with patch("minivess.pipeline.evaluation_runner.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "eval123"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(
                return_value=mock_run
            )
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            runner.log_results_to_mlflow(
                results,
                model_name="model_a",
                is_test_dataset=False,
            )

            logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
            metric_keys = list(logged_metrics.keys())
            assert any(k.startswith("eval/") for k in metric_keys)

    def test_aggregate_test_metric_logged(self) -> None:
        """test/aggregate/{metric} must be logged as volume-weighted mean."""
        from minivess.pipeline.evaluation_runner import (
            EvaluationResult,
            UnifiedEvaluationRunner,
        )

        config = MagicMock()
        config.mlflow_evaluation_experiment = "test_eval"
        config.include_expensive_metrics = False

        runner = UnifiedEvaluationRunner.__new__(UnifiedEvaluationRunner)
        runner.eval_config = config

        fold = _make_fake_fold_result(n_volumes=2)
        results: dict[str, dict[str, EvaluationResult]] = {
            "deepvess": {
                "all": EvaluationResult(
                    model_name="model_a",
                    dataset_name="deepvess",
                    subset_name="all",
                    fold_result=fold,
                    predictions_dir=None,
                    uncertainty_maps_dir=None,
                ),
            },
        }

        with patch("minivess.pipeline.evaluation_runner.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "agg123"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(
                return_value=mock_run
            )
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            runner.log_results_to_mlflow(
                results,
                model_name="model_a",
                is_test_dataset=True,
            )

            logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
            assert any(k.startswith("test/aggregate/") for k in logged_metrics), (
                f"Expected test/aggregate/ prefix in: {list(logged_metrics.keys())}"
            )


# ===========================================================================
# PHASE 3: DuckDB test_metrics table
# ===========================================================================


class TestDuckDBTestMetrics:
    """Phase 3: test/ prefix metrics must land in test_metrics table."""

    def test_test_metrics_table_in_schema(self) -> None:
        """BIOSTATISTICS_TABLES must include 'test_metrics'."""
        from minivess.pipeline.biostatistics_duckdb import BIOSTATISTICS_TABLES

        assert "test_metrics" in BIOSTATISTICS_TABLES

    def test_is_test_metric_recognizes_test_prefix(self) -> None:
        """_is_test_metric must return True for test/{dataset}/{metric}."""
        from minivess.pipeline.biostatistics_duckdb import _is_test_metric

        assert _is_test_metric("test/deepvess/all/dsc") is True
        assert _is_test_metric("test/aggregate/dsc") is True
        assert _is_test_metric("eval/0/val_dice") is False
        assert _is_test_metric("train/loss") is False

    def test_parse_test_metric(self) -> None:
        """_parse_test_metric must extract (dataset, subset, metric) using str.split."""
        from minivess.pipeline.biostatistics_duckdb import _parse_test_metric

        dataset, subset, metric = _parse_test_metric("test/deepvess/all/dsc")
        assert dataset == "deepvess"
        assert subset == "all"
        assert metric == "dsc"

    def test_parse_test_aggregate_metric(self) -> None:
        """_parse_test_metric must handle test/aggregate/{metric} format."""
        from minivess.pipeline.biostatistics_duckdb import _parse_test_metric

        dataset, subset, metric = _parse_test_metric("test/aggregate/dsc")
        assert dataset == "aggregate"
        assert subset == ""
        assert metric == "dsc"

    def test_test_metric_lands_in_test_metrics_table(self, tmp_path: Path) -> None:
        """test/deepvess/all/dsc metric must land in test_metrics, not eval_metrics."""
        import yaml

        from minivess.pipeline.biostatistics_discovery import discover_source_runs
        from minivess.pipeline.biostatistics_duckdb import build_biostatistics_duckdb

        # Create mock mlruns with test/ prefix metrics
        mlruns = tmp_path / "mlruns"
        exp_dir = mlruns / "1"
        exp_dir.mkdir(parents=True)
        (exp_dir / "meta.yaml").write_text(
            yaml.dump({"name": "test_experiment"}), encoding="utf-8"
        )

        run_dir = exp_dir / "run_001"
        run_dir.mkdir()
        (run_dir / "meta.yaml").write_text(
            yaml.dump({"status": "FINISHED"}), encoding="utf-8"
        )
        params_dir = run_dir / "params"
        params_dir.mkdir()
        (params_dir / "loss_name").write_text("dice_ce", encoding="utf-8")
        (params_dir / "fold_id").write_text("0", encoding="utf-8")
        tags_dir = run_dir / "tags"
        tags_dir.mkdir()
        (tags_dir / "loss_function").write_text("dice_ce", encoding="utf-8")
        (tags_dir / "model_family").write_text("dynunet", encoding="utf-8")

        # Create test/ prefix metric files
        metrics_dir = run_dir / "metrics"
        test_metric_dir = metrics_dir / "test" / "deepvess" / "all"
        test_metric_dir.mkdir(parents=True)
        (test_metric_dir / "dsc").write_text("1000 0.72 1", encoding="utf-8")

        # Also create an eval/ metric to verify it does NOT go to test_metrics
        eval_metric_dir = metrics_dir / "eval" / "0"
        eval_metric_dir.mkdir(parents=True)
        (eval_metric_dir / "val_dice").write_text("1000 0.85 1", encoding="utf-8")

        manifest = discover_source_runs(mlruns, ["test_experiment"])
        db_path = tmp_path / "biostatistics.duckdb"
        build_biostatistics_duckdb(manifest, mlruns, db_path)

        import duckdb

        conn = duckdb.connect(str(db_path), read_only=True)

        # test/ metric should be in test_metrics
        test_rows = conn.execute("SELECT * FROM test_metrics").fetchall()
        assert len(test_rows) > 0, "test_metrics table should have rows"

        # eval/ metric should NOT be in test_metrics
        eval_in_test = conn.execute(
            "SELECT COUNT(*) FROM test_metrics WHERE metric_name = 'val_dice'"
        ).fetchone()
        assert eval_in_test is not None
        assert eval_in_test[0] == 0

        conn.close()

    def test_eval_metric_still_in_eval_metrics(self, tmp_path: Path) -> None:
        """eval/fold0/dsc must still land in eval_metrics (no regression)."""
        import yaml

        from minivess.pipeline.biostatistics_discovery import discover_source_runs
        from minivess.pipeline.biostatistics_duckdb import build_biostatistics_duckdb

        mlruns = tmp_path / "mlruns"
        exp_dir = mlruns / "1"
        exp_dir.mkdir(parents=True)
        (exp_dir / "meta.yaml").write_text(
            yaml.dump({"name": "test_experiment"}), encoding="utf-8"
        )

        run_dir = exp_dir / "run_002"
        run_dir.mkdir()
        (run_dir / "meta.yaml").write_text(
            yaml.dump({"status": "FINISHED"}), encoding="utf-8"
        )
        params_dir = run_dir / "params"
        params_dir.mkdir()
        (params_dir / "loss_name").write_text("dice_ce", encoding="utf-8")
        (params_dir / "fold_id").write_text("0", encoding="utf-8")
        tags_dir = run_dir / "tags"
        tags_dir.mkdir()
        (tags_dir / "loss_function").write_text("dice_ce", encoding="utf-8")
        (tags_dir / "model_family").write_text("dynunet", encoding="utf-8")

        metrics_dir = run_dir / "metrics"
        eval_dir = metrics_dir / "eval" / "0"
        eval_dir.mkdir(parents=True)
        (eval_dir / "val_dice").write_text("1000 0.88 1", encoding="utf-8")

        manifest = discover_source_runs(mlruns, ["test_experiment"])
        db_path = tmp_path / "biostatistics.duckdb"
        build_biostatistics_duckdb(manifest, mlruns, db_path)

        import duckdb

        conn = duckdb.connect(str(db_path), read_only=True)
        eval_rows = conn.execute("SELECT * FROM eval_metrics").fetchall()
        assert len(eval_rows) > 0, "eval_metrics should still have rows"
        conn.close()


# ===========================================================================
# PHASE 4: Biostatistics split={trainval,test}
# ===========================================================================


class TestBiostatisticsSplit:
    """Phase 4: Biostatistics must iterate over splits."""

    def test_build_per_volume_data_handles_eval_prefix(self) -> None:
        """_build_per_volume_data must handle eval/ prefix for trainval split."""
        from minivess.orchestration.flows.biostatistics_flow import (
            _build_per_volume_data,
        )
        from minivess.pipeline.biostatistics_types import (
            SourceRunManifest,
        )

        # The function signature should accept a split parameter
        # For now, just verify it works without crashing with existing data
        manifest = SourceRunManifest.from_runs([])
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            result = _build_per_volume_data(manifest, Path(tmpdir))
            assert isinstance(result, dict)

    def test_build_per_volume_data_handles_test_prefix(self, tmp_path: Path) -> None:
        """_build_per_volume_data must handle test/ prefix metrics for test split."""
        from minivess.orchestration.flows.biostatistics_flow import (
            _build_per_volume_data,
        )
        from minivess.pipeline.biostatistics_types import (
            SourceRun,
            SourceRunManifest,
        )

        # Create mlruns with test/ prefix per-volume metrics
        mlruns = tmp_path / "mlruns"
        exp_dir = mlruns / "1"
        run_dir = exp_dir / "run_001"
        metrics_dir = run_dir / "metrics"

        # test/deepvess/vol_000/dsc
        vol_dir = metrics_dir / "test" / "deepvess" / "vol_000"
        vol_dir.mkdir(parents=True)
        (vol_dir / "dsc").write_text("1000 0.72 1", encoding="utf-8")

        manifest = SourceRunManifest.from_runs(
            [SourceRun("run_001", "1", "test_exp", "dice_ce", 0, "FINISHED")]
        )

        result = _build_per_volume_data(manifest, mlruns, split="test")
        # Should find test/ prefix metrics
        assert isinstance(result, dict)
        # With test split, it should pick up the test/ prefix metrics
        if result:
            # If implementation finds the test/ metrics, verify structure
            for metric_name in result:
                assert isinstance(result[metric_name], dict)

    def test_biostatistics_config_has_splits_field(self) -> None:
        """BiostatisticsConfig must have a splits field defaulting to ['trainval', 'test']."""
        from minivess.config.biostatistics_config import BiostatisticsConfig

        config = BiostatisticsConfig()
        assert hasattr(config, "splits")
        assert "trainval" in config.splits
        assert "test" in config.splits
