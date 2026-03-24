"""E2E integration test: synthetic drift through full pipeline (#574 T9, #607).

Exercises the complete drift detection pipeline using synthetic data:
1. Generate reference features from "clean" distribution
2. Apply parametric drift (intensity shift)
3. Run Tier 1 detection → verify drift detected
4. Generate synthetic embeddings with shift
5. Run Tier 2 detection → verify drift detected
6. Verify reports saved to MLflow artifacts
7. Verify dashboard section collects drift data

Marked @pytest.mark.integration — excluded from staging tier.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

FEATURE_COLUMNS = [
    "mean",
    "std",
    "min",
    "max",
    "p5",
    "p95",
    "snr",
    "contrast",
    "entropy",
]


def _make_clean_features(*, n_samples: int = 50, seed: int = 42) -> pd.DataFrame:
    """Generate clean reference features."""
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


def _apply_intensity_drift(
    features: pd.DataFrame, *, shift: float = 60.0
) -> pd.DataFrame:
    """Apply synthetic intensity drift to majority of features."""
    drifted = features.copy()
    drifted["mean"] = drifted["mean"] + shift
    drifted["std"] = drifted["std"] + shift * 0.3
    drifted["min"] = drifted["min"] + shift * 0.5
    drifted["max"] = drifted["max"] + shift * 0.7
    drifted["p5"] = drifted["p5"] + shift * 0.4
    drifted["p95"] = drifted["p95"] + shift * 0.6
    drifted["contrast"] = drifted["contrast"] + shift * 0.3
    return drifted


@pytest.mark.integration
class TestDriftPipelineE2E:
    """Full drift detection pipeline E2E test."""

    def test_e2e_synthetic_drift_detected_tier1(self, tmp_path: Path) -> None:
        """Full Tier 1 pipeline: inject → detect → report."""
        from minivess.observability.drift import FeatureDriftDetector

        ref = _make_clean_features()
        cur = _apply_intensity_drift(ref)
        detector = FeatureDriftDetector(ref)
        result = detector.detect(cur)

        assert result.drift_detected is True
        assert result.n_drifted >= 5  # Majority of 9 features drifted
        assert result.dataset_drift_score >= 0.5

    def test_e2e_synthetic_drift_detected_tier2(self, tmp_path: Path) -> None:
        """Full Tier 2 pipeline: embedding shift → MMD detection."""
        from minivess.observability.drift import EmbeddingDriftDetector

        rng = np.random.default_rng(42)
        ref_emb = rng.standard_normal((30, 64)).astype(np.float32)
        cur_emb = ref_emb + 5.0

        detector = EmbeddingDriftDetector(ref_emb, p_val_threshold=0.05, n_permutations=200)
        result = detector.detect(cur_emb)

        assert result.drift_detected is True
        assert result.dataset_drift_score < 0.05  # p-value
        assert result.feature_scores["mmd_statistic"] > 0

    def test_e2e_drift_reports_in_mlflow(self, tmp_path: Path) -> None:
        """Both tier reports in MLflow artifact store."""
        from minivess.observability.drift import persist_drift_reports

        ref = _make_clean_features()
        cur = _apply_intensity_drift(ref)
        rng = np.random.default_rng(42)
        ref_emb = rng.standard_normal((30, 64)).astype(np.float32)
        cur_emb = ref_emb + 5.0

        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("e2e_drift")
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

        # Verify all expected artifacts
        names = [Path(a).name for a in artifacts]
        assert "tier1_evidently_report.html" in names
        assert "tier1_drift_summary.json" in names
        assert "tier2_mmd_summary.json" in names
        assert "drift_metadata.json" in names

        # Verify Tier 1 JSON content
        tier1_path = client.download_artifacts(
            run_id, "drift_reports/tier1_drift_summary.json"
        )
        tier1_data = json.loads(Path(tier1_path).read_text(encoding="utf-8"))
        assert tier1_data["drift_detected"] is True

        # Verify Tier 2 JSON content
        tier2_path = client.download_artifacts(
            run_id, "drift_reports/tier2_mmd_summary.json"
        )
        tier2_data = json.loads(Path(tier2_path).read_text(encoding="utf-8"))
        assert tier2_data["drift_detected"] is True

    def test_e2e_triage_agent_recommends_action(self, tmp_path: Path) -> None:
        """Triage agent produces valid recommendation for detected drift."""
        from minivess.orchestration.flows.data_flow import drift_detection_task

        ref = _make_clean_features()
        cur = _apply_intensity_drift(ref)
        fn = (
            drift_detection_task.fn
            if hasattr(drift_detection_task, "fn")
            else drift_detection_task
        )
        result = fn(reference_features=ref, current_features=cur)

        assert result.drift_detected is True
        assert result.triage_recommendation is not None
        assert "action" in result.triage_recommendation

    def test_e2e_no_false_positive_on_clean_data(self, tmp_path: Path) -> None:
        """No drift detected on identical distributions."""
        from minivess.observability.drift import (
            EmbeddingDriftDetector,
            FeatureDriftDetector,
        )

        # Tier 1: same distribution, different seeds
        ref = _make_clean_features(seed=42)
        cur = _make_clean_features(seed=43)
        tier1_result = FeatureDriftDetector(ref).detect(cur)
        assert tier1_result.drift_detected is False

        # Tier 2: same distribution embeddings
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)
        ref_emb = rng1.standard_normal((30, 64)).astype(np.float32)
        cur_emb = rng2.standard_normal((30, 64)).astype(np.float32)
        tier2_result = EmbeddingDriftDetector(ref_emb, p_val_threshold=0.05, n_permutations=200).detect(
            cur_emb
        )
        assert tier2_result.drift_detected is False
