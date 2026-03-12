"""Tests for drift summary in Dashboard Flow (#574 T8, #606).

Verifies that collect_drift_section_task produces drift summary data
for the dashboard JSON output.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd


def _setup_mlflow_with_drift(tmp_path: Path) -> str:
    """Create an MLflow run with drift artifacts and return run_id."""
    from minivess.observability.drift import persist_drift_reports

    rng = np.random.default_rng(42)
    ref = pd.DataFrame(
        {
            "mean": rng.normal(100.0, 10.0, 50),
            "std": rng.normal(30.0, 5.0, 50),
        }
    )
    cur = ref.copy()
    cur["mean"] = cur["mean"] + 50.0

    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    mlflow.set_experiment("test_dashboard_drift")
    with mlflow.start_run() as run:
        persist_drift_reports(
            reference_features=ref,
            current_features=cur,
            tmp_dir=tmp_path / "drift_tmp",
        )
        return run.info.run_id


class TestDashboardDriftPanels:
    """Verify drift section in Dashboard Flow."""

    def test_dashboard_includes_drift_tier1_section(self, tmp_path: Path) -> None:
        """drift_tier1 key present in dashboard drift section."""
        from minivess.orchestration.flows.dashboard_flow import (
            collect_drift_section_task,
        )

        run_id = _setup_mlflow_with_drift(tmp_path)
        fn = (
            collect_drift_section_task.fn
            if hasattr(collect_drift_section_task, "fn")
            else collect_drift_section_task
        )
        section = fn(
            tracking_uri=str(tmp_path / "mlruns"),
            drift_run_id=run_id,
        )
        assert "drift_tier1" in section

    def test_dashboard_includes_drift_tier2_section(self, tmp_path: Path) -> None:
        """drift_tier2 key present when Tier 2 data exists."""
        from minivess.observability.drift import persist_drift_reports
        from minivess.orchestration.flows.dashboard_flow import (
            collect_drift_section_task,
        )

        rng = np.random.default_rng(42)
        ref = pd.DataFrame({"mean": rng.normal(0, 1, 30), "std": rng.normal(0, 1, 30)})
        cur = ref.copy()
        cur["mean"] = cur["mean"] + 50.0
        ref_emb = rng.standard_normal((30, 64)).astype(np.float32)
        cur_emb = ref_emb + 5.0

        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test_dashboard_drift2")
        with mlflow.start_run() as run:
            persist_drift_reports(
                reference_features=ref,
                current_features=cur,
                reference_embeddings=ref_emb,
                current_embeddings=cur_emb,
                tmp_dir=tmp_path / "drift_tmp",
            )
            run_id = run.info.run_id

        fn = (
            collect_drift_section_task.fn
            if hasattr(collect_drift_section_task, "fn")
            else collect_drift_section_task
        )
        section = fn(
            tracking_uri=str(tmp_path / "mlruns"),
            drift_run_id=run_id,
        )
        assert "drift_tier2" in section

    def test_dashboard_handles_missing_drift_data(self, tmp_path: Path) -> None:
        """Graceful when no drift runs exist."""
        from minivess.orchestration.flows.dashboard_flow import (
            collect_drift_section_task,
        )

        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test_empty")
        with mlflow.start_run() as run:
            run_id = run.info.run_id  # No drift artifacts

        fn = (
            collect_drift_section_task.fn
            if hasattr(collect_drift_section_task, "fn")
            else collect_drift_section_task
        )
        section = fn(
            tracking_uri=str(tmp_path / "mlruns"),
            drift_run_id=run_id,
        )
        assert section.get("drift_tier1") is None
        assert section.get("drift_tier2") is None
        assert "status" in section
