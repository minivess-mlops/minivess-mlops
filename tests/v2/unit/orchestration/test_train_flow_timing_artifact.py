"""Tests for timing JSONL artifact wiring into train_flow (issue #683).

Verifies that generate_timing_jsonl() output is logged as an MLflow artifact.
"""

from __future__ import annotations

import json
from unittest.mock import patch


class TestTimingJSONLArtifact:
    """Timing JSONL is logged as an MLflow artifact after training."""

    def test_timing_jsonl_logged_as_artifact(self) -> None:
        """After training with setup_seconds > 0, timing JSONL is logged."""
        from minivess.orchestration.flows.train_flow import log_timing_jsonl_artifact

        with patch("minivess.orchestration.flows.train_flow.mlflow") as mock_mlflow:
            log_timing_jsonl_artifact(
                setup_durations={"setup_total": 300.0, "dvc_pull": 200.0},
                training_seconds=600.0,
                epoch_count=10,
                hourly_rate_usd=0.19,
            )
            mock_mlflow.log_text.assert_called_once()
            call_args = mock_mlflow.log_text.call_args
            artifact_path = call_args[0][1]
            assert "timing/timing_report.jsonl" in artifact_path

    def test_timing_jsonl_content_is_valid(self) -> None:
        """The JSONL content contains at least one cost_summary line."""
        from minivess.orchestration.flows.train_flow import log_timing_jsonl_artifact

        with patch("minivess.orchestration.flows.train_flow.mlflow") as mock_mlflow:
            log_timing_jsonl_artifact(
                setup_durations={"setup_total": 300.0, "dvc_pull": 200.0},
                training_seconds=600.0,
                epoch_count=10,
                hourly_rate_usd=0.19,
            )
            jsonl_content = mock_mlflow.log_text.call_args[0][0]
            lines = jsonl_content.strip().split("\n")
            found_cost = False
            for line in lines:
                record = json.loads(line)
                if (
                    record.get("phase") == "cost"
                    and record.get("operation") == "cost_summary"
                ):
                    found_cost = True
            assert found_cost, "No cost_summary line found in JSONL output"

    def test_timing_jsonl_skipped_without_timing_data(self) -> None:
        """When no setup or training data, no JSONL artifact is logged."""
        from minivess.orchestration.flows.train_flow import log_timing_jsonl_artifact

        with patch("minivess.orchestration.flows.train_flow.mlflow") as mock_mlflow:
            log_timing_jsonl_artifact(
                setup_durations={},
                training_seconds=0.0,
                epoch_count=0,
                hourly_rate_usd=0.0,
            )
            mock_mlflow.log_text.assert_not_called()
