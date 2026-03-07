"""Tests for biostatistics MLflow logging (Phase 7, Task 7.2)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from minivess.pipeline.biostatistics_mlflow import log_biostatistics_run
from minivess.pipeline.biostatistics_types import (
    SourceRun,
    SourceRunManifest,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_manifest() -> SourceRunManifest:
    runs = [
        SourceRun(
            run_id="run_001",
            experiment_id="1",
            experiment_name="test_exp",
            loss_function="dice_ce",
            fold_id=0,
            status="FINISHED",
        ),
    ]
    return SourceRunManifest.from_runs(runs)


class TestLogBiostatisticsRun:
    @patch("minivess.pipeline.biostatistics_mlflow.mlflow")
    def test_creates_mlflow_run_with_correct_experiment(
        self, mock_mlflow: MagicMock
    ) -> None:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        manifest = _make_manifest()
        log_biostatistics_run(
            manifest=manifest,
            lineage={"fingerprint": manifest.fingerprint},
            figures=[],
            tables=[],
        )
        mock_mlflow.set_experiment.assert_called_once_with("minivess_biostatistics")

    @patch("minivess.pipeline.biostatistics_mlflow.mlflow")
    def test_logs_lineage_artifact(
        self, mock_mlflow: MagicMock, tmp_path: Path
    ) -> None:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        manifest = _make_manifest()
        log_biostatistics_run(
            manifest=manifest,
            lineage={"fingerprint": manifest.fingerprint},
            figures=[],
            tables=[],
        )
        # Should have called log_param or set_tag with fingerprint
        mock_mlflow.set_tag.assert_any_call(
            "upstream_fingerprint", manifest.fingerprint
        )

    @patch("minivess.pipeline.biostatistics_mlflow.mlflow")
    def test_tags_include_upstream_fingerprint(self, mock_mlflow: MagicMock) -> None:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        manifest = _make_manifest()
        log_biostatistics_run(
            manifest=manifest,
            lineage={"fingerprint": manifest.fingerprint},
            figures=[],
            tables=[],
        )

        tag_calls = {
            call.args[0]: call.args[1] for call in mock_mlflow.set_tag.call_args_list
        }
        assert "upstream_fingerprint" in tag_calls
        assert tag_calls["upstream_fingerprint"] == manifest.fingerprint

    @patch("minivess.pipeline.biostatistics_mlflow.mlflow")
    def test_mocked_mlflow_no_side_effects(self, mock_mlflow: MagicMock) -> None:
        """Ensure tests use mocked MLflow and don't create real runs."""
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_run = MagicMock()
        mock_run.info.run_id = "mock_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        manifest = _make_manifest()
        log_biostatistics_run(
            manifest=manifest,
            lineage={},
            figures=[],
            tables=[],
        )
        # Verify it used the mock
        assert mock_mlflow.set_experiment.called
