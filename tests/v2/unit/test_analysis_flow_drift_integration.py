"""Tests for embedding drift integration in Analysis Flow (#574 T7, #605).

Verifies that embedding_drift_task is a Prefect task in the analysis flow
that runs Tier 2 MMD drift detection on model embeddings.
"""

from __future__ import annotations

import ast
from pathlib import Path

import mlflow
import numpy as np


def _make_embeddings(
    *, n_samples: int = 30, dim: int = 64, seed: int = 42
) -> np.ndarray:
    """Generate random embeddings."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, dim)).astype(np.float32)


class TestAnalysisFlowDriftIntegration:
    """Verify embedding drift task in Analysis Flow."""

    def test_embedding_drift_task_exists(self) -> None:
        """embedding_drift_task is importable."""
        from minivess.orchestration.flows.analysis_flow import (
            embedding_drift_task,
        )

        assert callable(embedding_drift_task)

    def test_embedding_drift_task_has_task_decorator(self) -> None:
        """Verify @task decorator via AST inspection."""
        source_path = (
            Path(__file__).resolve().parents[3]
            / "src"
            / "minivess"
            / "orchestration"
            / "flows"
            / "analysis_flow.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "embedding_drift_task"
            ):
                decorator_names = []
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Name):
                        decorator_names.append(dec.id)
                    elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                        decorator_names.append(dec.func.id)
                assert "task" in decorator_names
                return
        raise AssertionError("embedding_drift_task not found in analysis_flow.py")  # noqa: EM101

    def test_embedding_drift_detected_on_shifted_model(self) -> None:
        """Synthetic embedding shift → drift detected."""
        from minivess.orchestration.flows.analysis_flow import (
            embedding_drift_task,
        )

        ref = _make_embeddings(seed=42)
        cur = ref + 5.0  # Large shift
        fn = (
            embedding_drift_task.fn
            if hasattr(embedding_drift_task, "fn")
            else embedding_drift_task
        )
        result = fn(reference_embeddings=ref, current_embeddings=cur, p_val_threshold=0.05)
        assert result.drift_detected is True

    def test_embedding_drift_report_saved_to_mlflow(self, tmp_path: Path) -> None:
        """Tier 2 MMD results saved as MLflow artifact."""
        from minivess.orchestration.flows.analysis_flow import (
            embedding_drift_task,
        )

        ref = _make_embeddings(seed=42)
        cur = ref + 5.0
        fn = (
            embedding_drift_task.fn
            if hasattr(embedding_drift_task, "fn")
            else embedding_drift_task
        )

        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test_emb_drift")
        with mlflow.start_run() as run:
            fn(
                reference_embeddings=ref,
                current_embeddings=cur,
                p_val_threshold=0.05,
                tmp_dir=tmp_path / "drift_tmp",
            )
            run_id = run.info.run_id

        client = mlflow.tracking.MlflowClient()
        artifacts = [a.path for a in client.list_artifacts(run_id, "drift_reports")]
        tier2_artifacts = [a for a in artifacts if "tier2" in a and a.endswith(".json")]
        assert len(tier2_artifacts) >= 1, f"No tier2 artifact in {artifacts}"
