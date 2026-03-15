"""Tests for infrastructure timing parser and cost analysis module.

Tests the parse/compute/log pipeline for cloud GPU infrastructure timing:
- Shell-generated key=value timestamp parsing
- Cost analysis computation (amortization, break-even, effective rate)
- MLflow integration (params, metrics, artifacts)

Issue: #683
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestParseSetupTiming:
    """Tests for parse_setup_timing() — reads key=value timestamp files."""

    def test_parse_setup_timing_basic(self, tmp_path: Path) -> None:
        """Parse a sample timing file with 3 operations, verify durations."""
        from minivess.observability.infrastructure_timing import parse_setup_timing

        timing_file = tmp_path / "timing_setup.txt"
        timing_file.write_text(
            "setup_start=1710412800.000\n"
            "python_install_start=1710412800.100\n"
            "python_install_end=1710412835.300\n"
            "uv_install_start=1710412835.400\n"
            "uv_install_end=1710412842.500\n"
            "uv_sync_start=1710412842.600\n"
            "uv_sync_end=1710413130.600\n"
            "setup_end=1710413130.700\n",
            encoding="utf-8",
        )

        result = parse_setup_timing(timing_file)

        assert abs(result["python_install"] - 35.2) < 0.01
        assert abs(result["uv_install"] - 7.1) < 0.01
        assert abs(result["uv_sync"] - 288.0) < 0.01
        assert abs(result["setup_total"] - 330.7) < 0.01

    def test_parse_setup_timing_missing_end(self, tmp_path: Path) -> None:
        """Missing end timestamp for an operation -> duration is None, no crash."""
        from minivess.observability.infrastructure_timing import parse_setup_timing

        timing_file = tmp_path / "timing_setup.txt"
        timing_file.write_text(
            "setup_start=1710412800.000\n"
            "python_install_start=1710412800.100\n"
            # No python_install_end!
            "uv_install_start=1710412835.400\n"
            "uv_install_end=1710412842.500\n"
            "setup_end=1710413000.000\n",
            encoding="utf-8",
        )

        result = parse_setup_timing(timing_file)

        assert "python_install" not in result  # Missing end -> not included
        assert abs(result["uv_install"] - 7.1) < 0.01
        assert "setup_total" in result

    def test_parse_setup_timing_empty_file(self, tmp_path: Path) -> None:
        """Empty file -> empty dict, no crash."""
        from minivess.observability.infrastructure_timing import parse_setup_timing

        timing_file = tmp_path / "timing_setup.txt"
        timing_file.write_text("", encoding="utf-8")

        result = parse_setup_timing(timing_file)
        assert result == {}

    def test_parse_setup_timing_file_not_found(self, tmp_path: Path) -> None:
        """Non-existent file -> empty dict, no crash."""
        from minivess.observability.infrastructure_timing import parse_setup_timing

        result = parse_setup_timing(tmp_path / "nonexistent.txt")
        assert result == {}


class TestComputeCostAnalysis:
    """Tests for compute_cost_analysis() — cost computation from timing data."""

    def test_compute_cost_analysis_basic(self) -> None:
        """600s setup, 1750s training, 50 epochs, $0.40/hr."""
        from minivess.observability.infrastructure_timing import compute_cost_analysis

        result = compute_cost_analysis(
            setup_seconds=600.0,
            training_seconds=1750.0,
            epoch_count=50,
            hourly_rate_usd=0.40,
        )

        # total = (600 + 1750) / 3600 * 0.40 = 2350/3600 * 0.40 = 0.2611
        assert abs(result["cost_total_usd"] - 0.2611) < 0.01
        # setup_fraction = 600 / 2350 = 0.2553
        assert abs(result["cost_setup_fraction"] - 0.2553) < 0.01
        # effective = 0.2611 / (1750/3600) = 0.2611 / 0.4861 = 0.5371
        assert abs(result["cost_effective_gpu_rate"] - 0.537) < 0.01

    def test_compute_cost_analysis_zero_training(self) -> None:
        """0s training -> effective_gpu_rate = -1.0 (sentinel), no ZeroDivisionError."""
        from minivess.observability.infrastructure_timing import compute_cost_analysis

        result = compute_cost_analysis(
            setup_seconds=600.0,
            training_seconds=0.0,
            epoch_count=0,
            hourly_rate_usd=0.40,
        )

        assert result["cost_effective_gpu_rate"] == -1.0
        assert result["cost_total_usd"] > 0

    def test_compute_cost_analysis_zero_hourly_rate(self) -> None:
        """$0.0/hr (local) -> all costs = 0.0, no errors."""
        from minivess.observability.infrastructure_timing import compute_cost_analysis

        result = compute_cost_analysis(
            setup_seconds=600.0,
            training_seconds=1750.0,
            epoch_count=50,
            hourly_rate_usd=0.0,
        )

        assert result["cost_total_usd"] == 0.0
        assert result["cost_setup_usd"] == 0.0
        assert result["cost_training_usd"] == 0.0

    def test_compute_cost_analysis_amortization(self) -> None:
        """600s setup, 30s/epoch: epochs_to_amortize = 181 (9*600/30 + 1)."""
        from minivess.observability.infrastructure_timing import compute_cost_analysis

        result = compute_cost_analysis(
            setup_seconds=600.0,
            training_seconds=300.0,  # 10 epochs * 30s
            epoch_count=10,
            hourly_rate_usd=0.40,
        )

        assert result["cost_epochs_to_amortize_setup"] == 181

    def test_compute_cost_analysis_break_even(self) -> None:
        """600s setup, 35s/epoch: break_even = 18 (600/35 + 1)."""
        from minivess.observability.infrastructure_timing import compute_cost_analysis

        result = compute_cost_analysis(
            setup_seconds=600.0,
            training_seconds=350.0,  # 10 epochs * 35s
            epoch_count=10,
            hourly_rate_usd=0.40,
        )

        assert result["cost_break_even_epochs"] == 18


class TestGenerateTimingJsonl:
    """Tests for generate_timing_jsonl() — JSONL artifact generation."""

    def test_generate_timing_jsonl_format(self) -> None:
        """Output is valid JSONL (each line parses as JSON)."""
        from minivess.observability.infrastructure_timing import generate_timing_jsonl

        setup_durations = {"python_install": 35.2, "uv_sync": 288.0}
        output = generate_timing_jsonl(
            setup_durations=setup_durations,
            training_seconds=80.0,
            epoch_count=2,
            hourly_rate_usd=0.40,
        )

        lines = [line for line in output.strip().split("\n") if line.strip()]
        assert len(lines) >= 3  # At least: python_install, uv_sync, cost_summary

        for line in lines:
            parsed = json.loads(line)
            assert "operation" in parsed

    def test_generate_timing_jsonl_has_cost_summary(self) -> None:
        """Last line has 'operation': 'cost_summary' with all cost fields."""
        from minivess.observability.infrastructure_timing import generate_timing_jsonl

        output = generate_timing_jsonl(
            setup_durations={"uv_sync": 288.0},
            training_seconds=80.0,
            epoch_count=2,
            hourly_rate_usd=0.40,
        )

        lines = [line for line in output.strip().split("\n") if line.strip()]
        last = json.loads(lines[-1])

        assert last["operation"] == "cost_summary"
        assert "total_cost_usd" in last
        assert "effective_gpu_rate_usd" in last
        assert "setup_fraction" in last


class TestMlflowIntegration:
    """Tests for MLflow logging functions."""

    @patch("minivess.observability.infrastructure_timing.mlflow")
    def test_log_infrastructure_timing_no_file(
        self, mock_mlflow: MagicMock, tmp_path: Path
    ) -> None:
        """log_infrastructure_timing() with no timing file -> no-op, no crash."""
        from minivess.observability.infrastructure_timing import (
            log_infrastructure_timing,
        )

        tracker = MagicMock()
        log_infrastructure_timing(tracker, timing_dir=tmp_path)
        mock_mlflow.log_params.assert_not_called()

    @patch("minivess.observability.infrastructure_timing.mlflow")
    def test_log_infrastructure_timing_logs_params(
        self, mock_mlflow: MagicMock, tmp_path: Path
    ) -> None:
        """Timing data logged as setup_* params via mlflow.log_params."""
        from minivess.observability.infrastructure_timing import (
            log_infrastructure_timing,
        )

        timing_file = tmp_path / "timing_setup.txt"
        timing_file.write_text(
            "setup_start=1710412800.000\n"
            "uv_sync_start=1710412842.600\n"
            "uv_sync_end=1710413130.600\n"
            "setup_end=1710413130.700\n",
            encoding="utf-8",
        )

        tracker = MagicMock()
        log_infrastructure_timing(tracker, timing_dir=tmp_path)

        # Verify mlflow.log_params was called with setup_* keys
        mock_mlflow.log_params.assert_called_once()
        logged_params = mock_mlflow.log_params.call_args.args[0]
        assert "setup_uv_sync_seconds" in logged_params
        assert abs(logged_params["setup_uv_sync_seconds"] - 288.0) < 0.01

    @patch("minivess.observability.infrastructure_timing.mlflow")
    def test_log_cost_analysis_logs_metrics(self, mock_mlflow: MagicMock) -> None:
        """Cost data logged as mlflow.log_metric at step=0."""
        from minivess.observability.infrastructure_timing import log_cost_analysis

        tracker = MagicMock()
        log_cost_analysis(
            tracker,
            setup_seconds=600.0,
            training_seconds=1750.0,
            epoch_count=50,
            hourly_rate_usd=0.40,
        )

        # Verify mlflow.log_metric was called with cost_* keys
        metric_calls = {
            call.args[0]: call.args[1] for call in mock_mlflow.log_metric.call_args_list
        }
        assert "cost_total_usd" in metric_calls
        assert "cost_effective_gpu_rate" in metric_calls
        assert "cost_setup_fraction" in metric_calls


class TestHourlyRateFromEnv:
    """Tests for INSTANCE_HOURLY_USD env var reading."""

    def test_hourly_rate_from_env(self) -> None:
        """INSTANCE_HOURLY_USD=0.40 in env -> hourly_rate = 0.40."""
        from minivess.observability.infrastructure_timing import get_hourly_rate_usd

        with patch.dict(os.environ, {"INSTANCE_HOURLY_USD": "0.40"}):
            assert get_hourly_rate_usd() == 0.40

    def test_hourly_rate_default_zero(self) -> None:
        """No INSTANCE_HOURLY_USD in env -> hourly_rate = 0.0."""
        from minivess.observability.infrastructure_timing import get_hourly_rate_usd

        with patch.dict(os.environ, {}, clear=True):
            # Ensure INSTANCE_HOURLY_USD is not set
            os.environ.pop("INSTANCE_HOURLY_USD", None)
            assert get_hourly_rate_usd() == 0.0
