"""Tests for per-volume metric persistence.

Covers:
- FoldResult.volume_ids field
- JSON serialization roundtrip
- Best/worst volume identification
- NaN handling
- MLflow artifact path
- save/load roundtrip

Closes #186.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from minivess.pipeline.ci import ConfidenceInterval
from minivess.pipeline.evaluation import FoldResult


class TestFoldResultVolumeIds:
    """Test volume_ids field on FoldResult."""

    def test_default_empty(self) -> None:
        fr = FoldResult()
        assert fr.volume_ids == []

    def test_set_volume_ids(self) -> None:
        fr = FoldResult(volume_ids=["mv01", "mv02", "mv03"])
        assert fr.volume_ids == ["mv01", "mv02", "mv03"]

    def test_backward_compatible(self) -> None:
        """Existing code that creates FoldResult without volume_ids still works."""
        fr = FoldResult(
            per_volume_metrics={"dsc": [0.8, 0.9]},
            aggregated={},
        )
        assert fr.volume_ids == []


class TestPerVolumeJson:
    """Test JSON serialization of per-volume metrics."""

    def _make_fold_result(self) -> FoldResult:
        return FoldResult(
            volume_ids=["mv01", "mv02", "mv03"],
            per_volume_metrics={
                "dsc": [0.85, 0.90, 0.78],
                "centreline_dsc": [0.70, 0.82, 0.65],
            },
            aggregated={
                "dsc": ConfidenceInterval(
                    point_estimate=0.843,
                    lower=0.78,
                    upper=0.90,
                    confidence_level=0.95,
                    method="percentile",
                ),
            },
        )

    def test_to_per_volume_json_structure(self) -> None:
        fr = self._make_fold_result()
        result = fr.to_per_volume_json()
        assert isinstance(result, list)
        assert len(result) == 3

    def test_to_per_volume_json_content(self) -> None:
        fr = self._make_fold_result()
        result = fr.to_per_volume_json()
        assert result[0]["volume_id"] == "mv01"
        assert result[0]["dsc"] == pytest.approx(0.85)
        assert result[1]["centreline_dsc"] == pytest.approx(0.82)

    def test_to_per_volume_json_sorted_by_volume_id(self) -> None:
        fr = FoldResult(
            volume_ids=["mv03", "mv01", "mv02"],
            per_volume_metrics={"dsc": [0.78, 0.85, 0.90]},
        )
        result = fr.to_per_volume_json()
        # Should preserve original order (not sort)
        assert result[0]["volume_id"] == "mv03"

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        fr = self._make_fold_result()
        out_path = fr.save_per_volume_json(tmp_path / "metrics.json")
        assert out_path.exists()

        loaded = FoldResult.load_per_volume_json(out_path)
        assert len(loaded) == 3
        assert loaded[0]["volume_id"] == "mv01"

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        fr = self._make_fold_result()
        deep_path = tmp_path / "a" / "b" / "c" / "metrics.json"
        out = fr.save_per_volume_json(deep_path)
        assert out.exists()

    def test_json_is_valid(self, tmp_path: Path) -> None:
        fr = self._make_fold_result()
        out_path = fr.save_per_volume_json(tmp_path / "metrics.json")
        # Parse and validate JSON
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert all("volume_id" in entry for entry in data)


class TestNanHandling:
    """Test NaN values in per-volume metrics."""

    def test_nan_serializes_to_null(self, tmp_path: Path) -> None:
        fr = FoldResult(
            volume_ids=["mv01", "mv02"],
            per_volume_metrics={"dsc": [0.85, float("nan")]},
        )
        out_path = fr.save_per_volume_json(tmp_path / "metrics.json")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data[0]["dsc"] == pytest.approx(0.85)
        assert data[1]["dsc"] is None

    def test_to_per_volume_json_with_nan(self) -> None:
        fr = FoldResult(
            volume_ids=["mv01"],
            per_volume_metrics={"dsc": [float("nan")]},
        )
        result = fr.to_per_volume_json()
        assert result[0]["dsc"] is None


class TestBestWorstVolume:
    """Test best/worst volume identification."""

    def test_best_volume(self) -> None:
        fr = FoldResult(
            volume_ids=["mv01", "mv02", "mv03"],
            per_volume_metrics={"dsc": [0.85, 0.90, 0.78]},
        )
        best_id, best_val = fr.best_volume("dsc")
        assert best_id == "mv02"
        assert best_val == pytest.approx(0.90)

    def test_worst_volume(self) -> None:
        fr = FoldResult(
            volume_ids=["mv01", "mv02", "mv03"],
            per_volume_metrics={"dsc": [0.85, 0.90, 0.78]},
        )
        worst_id, worst_val = fr.worst_volume("dsc")
        assert worst_id == "mv03"
        assert worst_val == pytest.approx(0.78)

    def test_best_volume_with_nan_skipped(self) -> None:
        fr = FoldResult(
            volume_ids=["mv01", "mv02"],
            per_volume_metrics={"dsc": [float("nan"), 0.80]},
        )
        best_id, best_val = fr.best_volume("dsc")
        assert best_id == "mv02"

    def test_best_volume_missing_metric_raises(self) -> None:
        fr = FoldResult(
            volume_ids=["mv01"],
            per_volume_metrics={"dsc": [0.85]},
        )
        with pytest.raises(KeyError):
            fr.best_volume("nonexistent")


class TestMlflowArtifact:
    """Test MLflow artifact path construction."""

    def test_log_per_volume_metrics_artifact_path(self) -> None:
        """Verify log_per_volume_metrics uses correct artifact path."""
        from minivess.observability.tracking import ExperimentTracker

        fr = FoldResult(
            volume_ids=["mv01", "mv02"],
            per_volume_metrics={"dsc": [0.85, 0.90]},
        )

        with patch.object(ExperimentTracker, "__init__", return_value=None):
            tracker = ExperimentTracker.__new__(ExperimentTracker)
            tracker._run_id = "test_run_123"

            with patch("mlflow.log_artifact") as mock_log:
                tracker.log_per_volume_metrics(fr, fold_id=0, loss_name="dice_ce")

                mock_log.assert_called_once()
                call_kwargs = mock_log.call_args
                artifact_path = call_kwargs.kwargs.get("artifact_path") or call_kwargs[
                    1
                ].get("artifact_path")
                assert artifact_path == "per_volume_metrics"

    def test_log_per_volume_metrics_filename(self) -> None:
        """Verify filename follows fold{id}_{loss}.json pattern."""
        from minivess.observability.tracking import ExperimentTracker

        fr = FoldResult(
            volume_ids=["mv01"],
            per_volume_metrics={"dsc": [0.85]},
        )

        with patch.object(ExperimentTracker, "__init__", return_value=None):
            tracker = ExperimentTracker.__new__(ExperimentTracker)
            tracker._run_id = "test_run_123"

            with patch("mlflow.log_artifact") as mock_log:
                tracker.log_per_volume_metrics(fr, fold_id=2, loss_name="cbdice")

                mock_log.assert_called_once()
                local_path = Path(mock_log.call_args[0][0])
                assert local_path.name == "fold2_cbdice.json"
