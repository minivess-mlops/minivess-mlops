"""Tests for QA Prefect flow (#277).

Covers:
- QA check functions (backend, params, ghosts, metrics)
- QA flow orchestration
- QA report generation
"""

from __future__ import annotations

from unittest.mock import MagicMock


class TestQACheckBackend:
    """Test MLflow backend consistency check."""

    def test_check_backend_server(self) -> None:
        from minivess.orchestration.flows.qa_flow import check_backend_consistency

        result = check_backend_consistency("http://mlflow:5000")
        assert result["status"] == "pass"
        assert result["backend_type"] == "server"

    def test_check_backend_local_warns(self) -> None:
        from minivess.orchestration.flows.qa_flow import check_backend_consistency

        result = check_backend_consistency("mlruns")
        assert result["status"] == "warning"
        assert result["backend_type"] == "local"


class TestQACheckParams:
    """Test required param validation check."""

    def test_check_params_all_present(self) -> None:
        from minivess.orchestration.flows.qa_flow import check_run_params

        mock_run = MagicMock()
        mock_run.data.params = {
            "loss_name": "dice_ce",
            "model_family": "dynunet",
            "batch_size": "2",
            "learning_rate": "0.001",
            "max_epochs": "100",
            "seed": "42",
        }

        result = check_run_params(mock_run)
        assert result["status"] == "pass"

    def test_check_params_missing(self) -> None:
        from minivess.orchestration.flows.qa_flow import check_run_params

        mock_run = MagicMock()
        mock_run.data.params = {"batch_size": "2"}

        result = check_run_params(mock_run)
        assert result["status"] == "fail"
        assert len(result["missing_params"]) > 0


class TestQACheckGhosts:
    """Test ghost run detection check."""

    def test_no_ghosts(self) -> None:
        from minivess.orchestration.flows.qa_flow import check_ghost_runs

        mock_client = MagicMock()
        mock_client.search_runs.return_value = []

        result = check_ghost_runs(mock_client, experiment_ids=["0"])
        assert result["status"] == "pass"
        assert result["ghost_count"] == 0


class TestQAReport:
    """Test QA report generation."""

    def test_generate_report(self) -> None:
        from minivess.orchestration.flows.qa_flow import generate_qa_report

        checks = [
            {"name": "backend", "status": "pass", "message": "OK"},
            {"name": "params", "status": "fail", "message": "Missing loss_name"},
        ]
        report = generate_qa_report(checks)
        assert "backend" in report
        assert "FAIL" in report.upper() or "fail" in report
        assert isinstance(report, str)

    def test_report_summary_counts(self) -> None:
        from minivess.orchestration.flows.qa_flow import summarize_qa_results

        checks = [
            {"name": "a", "status": "pass"},
            {"name": "b", "status": "fail"},
            {"name": "c", "status": "warning"},
            {"name": "d", "status": "pass"},
        ]
        summary = summarize_qa_results(checks)
        assert summary["total"] == 4
        assert summary["passed"] == 2
        assert summary["failed"] == 1
        assert summary["warnings"] == 1
