"""Tests for drift report persistence to MLflow artifacts (#574 T5, #603).

Verifies that drift detection results (Evidently HTML, JSON summaries,
Tier 2 MMD results, metadata) are saved as MLflow artifacts under
the drift_reports/ artifact path convention.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("evidently", reason="evidently not installed")

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _make_reference_features(*, n_samples: int = 50, seed: int = 42) -> pd.DataFrame:
    """Generate reference feature DataFrame."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "mean": rng.normal(100.0, 10.0, n_samples),
            "std": rng.normal(30.0, 5.0, n_samples),
            "min": rng.normal(0.0, 2.0, n_samples),
            "max": rng.normal(255.0, 5.0, n_samples),
            "p5": rng.normal(20.0, 5.0, n_samples),
            "p95": rng.normal(230.0, 10.0, n_samples),
            "snr": rng.normal(3.5, 0.5, n_samples),
            "contrast": rng.normal(210.0, 12.0, n_samples),
            "entropy": rng.normal(7.0, 0.3, n_samples),
        }
    )


def _make_shifted_features(reference: pd.DataFrame) -> pd.DataFrame:
    """Create features with synthetic intensity drift."""
    shifted = reference.copy()
    shifted["mean"] = shifted["mean"] + 50.0
    shifted["p5"] = shifted["p5"] + 25.0
    shifted["p95"] = shifted["p95"] + 40.0
    return shifted


class TestDriftMLflowArtifacts:
    """Verify drift reports are persisted as MLflow artifacts."""

    def test_tier1_report_saved_as_mlflow_artifact(self, tmp_path: Path) -> None:
        """Evidently HTML report logged to MLflow run."""
        from minivess.observability.drift import persist_drift_reports

        ref = _make_reference_features()
        cur = _make_shifted_features(ref)

        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test_drift")
        with mlflow.start_run() as run:
            persist_drift_reports(
                reference_features=ref,
                current_features=cur,
                tmp_dir=tmp_path / "drift_tmp",
            )
            run_id = run.info.run_id

        client = mlflow.tracking.MlflowClient()
        artifacts = [a.path for a in client.list_artifacts(run_id, "drift_reports")]
        html_artifacts = [a for a in artifacts if a.endswith(".html")]
        assert len(html_artifacts) >= 1, f"No HTML artifact in {artifacts}"

    def test_tier1_json_summary_saved(self, tmp_path: Path) -> None:
        """JSON drift summary saved in artifacts."""
        from minivess.observability.drift import persist_drift_reports

        ref = _make_reference_features()
        cur = _make_shifted_features(ref)

        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test_drift")
        with mlflow.start_run() as run:
            persist_drift_reports(
                reference_features=ref,
                current_features=cur,
                tmp_dir=tmp_path / "drift_tmp",
            )
            run_id = run.info.run_id

        client = mlflow.tracking.MlflowClient()
        artifacts = [a.path for a in client.list_artifacts(run_id, "drift_reports")]
        json_artifacts = [a for a in artifacts if "tier1" in a and a.endswith(".json")]
        assert len(json_artifacts) >= 1, f"No tier1 JSON artifact in {artifacts}"

    def test_tier2_mmd_summary_saved(self, tmp_path: Path) -> None:
        """MMD results saved in drift_reports/ artifacts."""
        from minivess.observability.drift import persist_drift_reports

        ref = _make_reference_features()
        cur = _make_shifted_features(ref)
        rng = np.random.default_rng(42)
        ref_emb = rng.standard_normal((30, 64)).astype(np.float32)
        cur_emb = ref_emb + 5.0  # Shifted embeddings

        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test_drift")
        with mlflow.start_run() as run:
            persist_drift_reports(
                reference_features=ref,
                current_features=cur,
                reference_embeddings=ref_emb,
                current_embeddings=cur_emb,
                tmp_dir=tmp_path / "drift_tmp",
            )
            run_id = run.info.run_id

        client = mlflow.tracking.MlflowClient()
        artifacts = [a.path for a in client.list_artifacts(run_id, "drift_reports")]
        json_artifacts = [a for a in artifacts if "tier2" in a and a.endswith(".json")]
        assert len(json_artifacts) >= 1, f"No tier2 JSON artifact in {artifacts}"

    def test_drift_metadata_includes_timestamps(self, tmp_path: Path) -> None:
        """Metadata artifact has provenance fields."""
        from minivess.observability.drift import persist_drift_reports

        ref = _make_reference_features()
        cur = _make_shifted_features(ref)

        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test_drift")
        with mlflow.start_run() as run:
            persist_drift_reports(
                reference_features=ref,
                current_features=cur,
                tmp_dir=tmp_path / "drift_tmp",
            )
            run_id = run.info.run_id

        # Read metadata artifact from local mlruns
        client = mlflow.tracking.MlflowClient()
        artifacts = [a.path for a in client.list_artifacts(run_id, "drift_reports")]
        metadata_artifacts = [a for a in artifacts if "metadata" in a]
        assert len(metadata_artifacts) >= 1

        # Download and verify
        artifact_dir = client.download_artifacts(run_id, metadata_artifacts[0])
        metadata = json.loads(Path(artifact_dir).read_text(encoding="utf-8"))
        assert "timestamp" in metadata
        assert "evidently_version" in metadata

    def test_artifact_paths_follow_convention(self, tmp_path: Path) -> None:
        """All drift artifacts under drift_reports/ prefix."""
        from minivess.observability.drift import persist_drift_reports

        ref = _make_reference_features()
        cur = _make_shifted_features(ref)

        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test_drift")
        with mlflow.start_run() as run:
            persist_drift_reports(
                reference_features=ref,
                current_features=cur,
                tmp_dir=tmp_path / "drift_tmp",
            )
            run_id = run.info.run_id

        client = mlflow.tracking.MlflowClient()
        all_artifacts = client.list_artifacts(run_id)
        drift_artifacts = [
            a for a in all_artifacts if a.path.startswith("drift_reports")
        ]
        assert len(drift_artifacts) >= 1, "No artifacts under drift_reports/"
