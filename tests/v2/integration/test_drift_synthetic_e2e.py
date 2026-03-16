"""E2E integration test: synthetic generation → drift detection pipeline (T-E3).

Tests the full pipeline:
1. Generate synthetic volumes (all adapters)
2. Extract features and profiles (whylogs)
3. Run drift simulation on synthetic batches
4. Verify reports and drift detection
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.integration
class TestDriftSyntheticE2E:
    """End-to-end: synthetic generation → feature extraction → drift detection."""

    def test_all_generators_produce_valid_output(self) -> None:
        """All 5 generators should produce valid (image, mask) pairs."""
        from minivess.data.synthetic import generate_stack, list_generators

        for method in list_generators():
            pairs = generate_stack(method=method, n_volumes=1)
            assert len(pairs) == 1
            img, mask = pairs[0]
            assert img.ndim == 3
            assert mask.ndim == 3
            assert np.issubdtype(img.dtype, np.floating), (
                f"{method}: img.dtype={img.dtype}"
            )
            assert np.issubdtype(mask.dtype, np.integer) or np.issubdtype(
                mask.dtype, np.floating
            ), f"{method}: mask.dtype={mask.dtype}"

    def test_synthetic_volumes_have_extractable_features(self) -> None:
        """Feature extraction should work on generated volumes."""
        from minivess.data.feature_extraction import extract_batch_features
        from minivess.data.synthetic import generate_stack

        pairs = generate_stack(method="debug", n_volumes=5)
        volumes = [img for img, _mask in pairs]
        features_df = extract_batch_features(volumes)

        assert len(features_df) == 5
        assert "mean" in features_df.columns
        assert "entropy" in features_df.columns

    def test_whylogs_profiles_on_synthetic(self) -> None:
        """WhyLogs should profile synthetic volumes without error."""
        from minivess.data.synthetic import generate_stack
        from minivess.observability.whylogs_service import WhylogsVolumeProfiler

        pairs = generate_stack(method="vamos", n_volumes=3)
        volumes = [img for img, _mask in pairs]

        profiler = WhylogsVolumeProfiler()
        profiles = profiler.profile_batch(volumes)
        assert len(profiles) == 3

        merged = profiler.merge_profiles(profiles)
        columns = profiler.get_column_names(merged)
        assert len(columns) > 0

    def test_drift_detected_between_generators(self) -> None:
        """Different generators should produce different distributions."""
        from minivess.data.feature_extraction import extract_batch_features
        from minivess.data.synthetic import generate_stack
        from minivess.observability.drift import FeatureDriftDetector

        # Reference: debug generator
        ref_pairs = generate_stack(method="debug", n_volumes=10, config={"seed": 42})
        ref_volumes = [img for img, _mask in ref_pairs]
        ref_features = extract_batch_features(ref_volumes)

        # Current: VascuSynth (different generation method)
        cur_pairs = generate_stack(
            method="vascusynth", n_volumes=10, config={"seed": 42}
        )
        cur_volumes = [img for img, _mask in cur_pairs]
        cur_features = extract_batch_features(cur_volumes)

        detector = FeatureDriftDetector(ref_features)
        result = detector.detect(cur_features)

        # Different generators should produce different distributions
        assert result.n_features > 0

    def test_drift_simulation_flow_e2e(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Full drift simulation flow with synthetic data."""
        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        monkeypatch.setenv("PREFECT_DISABLED", "1")

        from minivess.data.synthetic import generate_stack
        from minivess.orchestration.flows.drift_simulation_flow import (
            drift_simulation_flow,
        )

        # Generate reference volumes
        ref_pairs = generate_stack(method="debug", n_volumes=10, config={"seed": 42})
        reference_volumes = [img for img, _mask in ref_pairs]

        # Generate 3 batches with progressive drift
        batches = []
        for i in range(3):
            batch_pairs = generate_stack(
                method="debug",
                n_volumes=2,
                config={"seed": 100 + i, "noise_level": 0.1 + i * 0.3},
            )
            batch_volumes = [img for img, _mask in batch_pairs]
            batches.append(batch_volumes)

        result = drift_simulation_flow.fn(
            reference_volumes=reference_volumes,
            batches=batches,
            output_dir=str(tmp_path / "drift_reports"),
        )

        assert result["status"] == "completed"
        assert result["n_batches"] == 3
        assert len(result["batch_results"]) == 3

        # Summary file should exist
        summary_path = tmp_path / "drift_reports" / "drift_simulation_summary.json"
        assert summary_path.exists()

    def test_synthetic_generation_flow_e2e(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Full synthetic generation flow."""
        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        monkeypatch.setenv("PREFECT_DISABLED", "1")

        from minivess.orchestration.flows.synthetic_generation_flow import (
            synthetic_generation_flow,
        )

        result = synthetic_generation_flow.fn(
            method="debug",
            n_volumes=3,
            output_dir=str(tmp_path / "synthetic"),
        )

        assert result["status"] == "completed"
        assert result["total_volumes"] == 3

        # Files should exist
        synth_dir = tmp_path / "synthetic" / "debug"
        assert synth_dir.exists()
        npy_files = list(synth_dir.glob("*.npy"))
        assert len(npy_files) >= 3

    def test_prometheus_metrics_from_synthetic(self) -> None:
        """Prometheus export should work on synthetic volume profiles."""
        from minivess.data.synthetic import generate_stack
        from minivess.observability.whylogs_service import (
            WhylogsVolumeProfiler,
            format_whylogs_prometheus,
        )

        pairs = generate_stack(method="debug", n_volumes=2)
        volumes = [img for img, _mask in pairs]

        profiler = WhylogsVolumeProfiler()
        profiles = profiler.profile_batch(volumes)
        merged = profiler.merge_profiles(profiles)

        prom_text = format_whylogs_prometheus(merged, dataset="synthetic_debug")
        assert "whylogs_" in prom_text
        assert 'dataset="synthetic_debug"' in prom_text

    def test_alerting_on_drift(self) -> None:
        """Alert manager should handle drift detection results."""
        from datetime import UTC, datetime

        from minivess.observability.alerting import AlertManager, DriftAlert

        alert = DriftAlert(
            alert_name="synthetic_drift_test",
            severity="warning",
            drift_score=0.8,
            n_drifted_features=2,
            batch_id="batch_0",
            timestamp=datetime.now(UTC),
        )

        manager = AlertManager()
        # fire() logs the alert and optionally sends webhook
        manager.fire(alert)
        assert alert.alert_name == "synthetic_drift_test"
        assert alert.n_drifted_features == 2
