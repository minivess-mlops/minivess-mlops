"""Tests for MLflow stall detection (Phase 8, Task 8.1)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from minivess.observability.stall_detection import StallResult, detect_mlflow_metric_stall


class TestDetectMlflowMetricStall:
    def test_function_exists(self) -> None:
        assert callable(detect_mlflow_metric_stall)

    def test_returns_stall_result_when_mlflow_unavailable(self) -> None:
        with patch.dict("sys.modules", {"mlflow": None}):
            result = detect_mlflow_metric_stall("fake_run_id")
        assert isinstance(result, StallResult)
        assert result.stale is False

    def test_not_stale_when_recent_activity(self) -> None:
        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_run.data.metrics = {"train_loss": 0.5}
        mock_run.info.start_time = int(time.time() * 1000)
        mock_run.info.end_time = None
        mock_mlflow.MlflowClient.return_value.get_run.return_value = mock_run

        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            result = detect_mlflow_metric_stall("test_run", threshold_minutes=30)
        assert result.stale is False

    def test_stale_when_old_activity(self) -> None:
        mock_mlflow = MagicMock()
        old_time = int((time.time() - 3600) * 1000)
        mock_run = MagicMock()
        mock_run.data.metrics = {"train_loss": 0.5}
        mock_run.info.start_time = old_time
        mock_run.info.end_time = None
        mock_mlflow.MlflowClient.return_value.get_run.return_value = mock_run

        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            result = detect_mlflow_metric_stall("test_run", threshold_minutes=30)
        assert result.stale is True
        assert result.minutes_since > 30

    def test_no_metrics_not_stale(self) -> None:
        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_run.data.metrics = {}
        mock_mlflow.MlflowClient.return_value.get_run.return_value = mock_run

        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            result = detect_mlflow_metric_stall("test_run")
        assert result.stale is False
        assert "No metrics" in result.message

    def test_graceful_when_mlflow_unavailable(self) -> None:
        with patch.dict("sys.modules", {"mlflow": None}):
            result = detect_mlflow_metric_stall("fake_id")
        assert result.stale is False
