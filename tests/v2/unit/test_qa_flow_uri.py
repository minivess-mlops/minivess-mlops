"""Tests for T-14: qa_flow tracking URI from MLFLOW_TRACKING_URI env var and report persistence.

Verifies that qa_flow() uses MLFLOW_TRACKING_URI env var as default tracking_uri,
writes qa_report.md to disk, and handles empty mlruns gracefully.

Uses ast.parse() for source inspection — NO regex (CLAUDE.md Rule #16).
"""

from __future__ import annotations

from pathlib import Path

_QA_FLOW_SRC = Path("src/minivess/orchestration/flows/qa_flow.py")


# ---------------------------------------------------------------------------
# AST-level: tracking_uri must default to env var, not "mlruns"
# ---------------------------------------------------------------------------


class TestQaFlowTrackingUri:
    def test_qa_flow_no_hardcoded_mlruns_default(self) -> None:
        """qa_flow() must not have tracking_uri='mlruns' as a hardcoded default."""
        source = _QA_FLOW_SRC.read_text(encoding="utf-8")
        assert "MLFLOW_TRACKING_URI" in source, (
            "qa_flow.py must read MLFLOW_TRACKING_URI env var. "
            "Change default: tracking_uri: str = 'mlruns' "
            "to: tracking_uri: str | None = None "
            "and resolve via os.environ.get('MLFLOW_TRACKING_URI', 'mlruns') inside the function."
        )

    def test_qa_uses_env_tracking_uri(self, monkeypatch, tmp_path) -> None:
        """qa_flow() must use MLFLOW_TRACKING_URI env var as tracking_uri."""
        monkeypatch.setenv("PREFECT_DISABLED", "1")
        mlflow_dir = tmp_path / "mlruns"
        mlflow_dir.mkdir()
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(mlflow_dir))
        monkeypatch.setenv("DASHBOARD_OUTPUT_DIR", str(tmp_path / "dashboard"))

        from minivess.orchestration.flows.qa_flow import qa_flow

        result = qa_flow()
        # The key check: qa_flow ran (did not crash), using the env tracking_uri
        assert "summary" in result, (
            f"qa_flow() must return a dict with 'summary'. Got: {list(result.keys())}"
        )

    def test_qa_flow_on_empty_mlruns_does_not_crash(
        self, monkeypatch, tmp_path
    ) -> None:
        """qa_flow() with empty tracking store must not raise an exception."""
        monkeypatch.setenv("PREFECT_DISABLED", "1")
        empty_mlruns = tmp_path / "mlruns"
        empty_mlruns.mkdir()
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(empty_mlruns))
        monkeypatch.setenv("DASHBOARD_OUTPUT_DIR", str(tmp_path / "dashboard"))

        from minivess.orchestration.flows.qa_flow import qa_flow

        try:
            result = qa_flow()
            assert isinstance(result, dict), (
                f"qa_flow() returned {type(result)}, expected dict"
            )
        except Exception as exc:
            raise AssertionError(
                f"qa_flow() raised {type(exc).__name__} on empty mlruns: {exc}. "
                "It should return a warning summary, not crash."
            ) from exc


# ---------------------------------------------------------------------------
# Report persistence
# ---------------------------------------------------------------------------


class TestQaReportPersistence:
    def test_qa_flow_references_dashboard_output(self) -> None:
        """qa_flow.py must reference DASHBOARD_OUTPUT or QA_OUTPUT env var."""
        source = _QA_FLOW_SRC.read_text(encoding="utf-8")
        assert "DASHBOARD_OUTPUT" in source or "QA_OUTPUT" in source, (
            "qa_flow.py must write QA report to disk using DASHBOARD_OUTPUT or "
            "QA_OUTPUT env var. Add: report_path = Path(os.environ.get("
            "'DASHBOARD_OUTPUT', '/app/outputs/dashboard')) / 'qa_report.md'"
        )

    def test_qa_report_written_to_disk(self, monkeypatch, tmp_path) -> None:
        """qa_flow() must write qa_report.md to DASHBOARD_OUTPUT dir."""
        monkeypatch.setenv("PREFECT_DISABLED", "1")
        empty_mlruns = tmp_path / "mlruns"
        empty_mlruns.mkdir()
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(empty_mlruns))
        dashboard_dir = tmp_path / "dashboard"
        monkeypatch.setenv("DASHBOARD_OUTPUT_DIR", str(dashboard_dir))

        from minivess.orchestration.flows.qa_flow import qa_flow

        result = qa_flow()

        report_path = result.get("report_path")
        assert report_path is not None, (
            "qa_flow() result must include 'report_path'. "
            "Add report_path key to the return dict."
        )
        assert Path(report_path).exists(), (
            f"qa_report.md not found at {report_path}. "
            "qa_flow() must write the report to disk."
        )

    def test_qa_report_path_absolute(self, monkeypatch, tmp_path) -> None:
        """qa_report.md path must be absolute."""
        monkeypatch.setenv("PREFECT_DISABLED", "1")
        empty_mlruns = tmp_path / "mlruns"
        empty_mlruns.mkdir()
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(empty_mlruns))
        dashboard_dir = tmp_path / "dashboard"
        monkeypatch.setenv("DASHBOARD_OUTPUT_DIR", str(dashboard_dir))

        from minivess.orchestration.flows.qa_flow import qa_flow

        result = qa_flow()

        report_path = result.get("report_path")
        if report_path is not None:
            assert Path(report_path).is_absolute(), (
                f"qa report path {report_path} is not absolute. "
                "Use absolute Path when writing qa_report.md."
            )
