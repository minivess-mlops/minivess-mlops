"""Tests for drift detection: feature extraction, Evidently tier 1, MMD tier 2."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def reference_volumes() -> list[np.ndarray]:
    """20 clean synthetic volumes with consistent statistics."""
    rng = np.random.default_rng(42)
    return [rng.random((32, 32, 8), dtype=np.float32) for _ in range(20)]


@pytest.fixture()
def drifted_volumes() -> list[np.ndarray]:
    """20 volumes with intensity drift (brightness shift)."""
    rng = np.random.default_rng(42)
    clean = [rng.random((32, 32, 8), dtype=np.float32) for _ in range(20)]
    # Apply intensity shift: multiply by 2.0 + add 0.3
    return [(v * 2.0 + 0.3).astype(np.float32) for v in clean]


# ---------------------------------------------------------------------------
# T1: Feature extraction
# ---------------------------------------------------------------------------


class TestExtractVolumeFeatures:
    """Test image feature extraction from 3D volumes."""

    def test_extract_volume_features_keys(self) -> None:
        from minivess.data.feature_extraction import extract_volume_features

        rng = np.random.default_rng(42)
        volume = rng.random((32, 32, 8), dtype=np.float32)
        features = extract_volume_features(volume)

        expected_keys = {
            "mean", "std", "min", "max", "p5", "p95",
            "snr", "contrast", "entropy",
        }
        assert expected_keys.issubset(set(features.keys()))

    def test_extract_volume_features_types(self) -> None:
        from minivess.data.feature_extraction import extract_volume_features

        rng = np.random.default_rng(42)
        volume = rng.random((32, 32, 8), dtype=np.float32)
        features = extract_volume_features(volume)

        for key, value in features.items():
            assert isinstance(value, float), f"{key} is {type(value)}, not float"

    def test_extract_volume_features_reasonable_values(self) -> None:
        """Features should have reasonable values for [0, 1] uniform data."""
        from minivess.data.feature_extraction import extract_volume_features

        rng = np.random.default_rng(42)
        volume = rng.random((32, 32, 8), dtype=np.float32)
        features = extract_volume_features(volume)

        assert 0.4 < features["mean"] < 0.6  # uniform [0,1] mean ≈ 0.5
        assert features["std"] > 0.0
        assert features["contrast"] > 0.0
        assert features["entropy"] > 0.0

    def test_extract_batch_features_shape(
        self, reference_volumes: list[np.ndarray]
    ) -> None:
        from minivess.data.feature_extraction import extract_batch_features

        df = extract_batch_features(reference_volumes)
        assert len(df) == 20
        assert "mean" in df.columns
        assert "entropy" in df.columns

    def test_features_change_under_drift(self) -> None:
        """Drifted volumes should have different feature values."""
        from minivess.data.feature_extraction import extract_volume_features

        rng = np.random.default_rng(42)
        clean = rng.random((32, 32, 8), dtype=np.float32)
        drifted = (clean * 2.0 + 0.3).astype(np.float32)

        clean_feats = extract_volume_features(clean)
        drift_feats = extract_volume_features(drifted)

        # Mean should be significantly different
        assert abs(clean_feats["mean"] - drift_feats["mean"]) > 0.2


# ---------------------------------------------------------------------------
# T2: Evidently drift detector (Tier 1)
# ---------------------------------------------------------------------------


class TestFeatureDriftDetector:
    """Test Evidently-based feature drift detection."""

    def test_no_drift_same_distribution(
        self, reference_volumes: list[np.ndarray]
    ) -> None:
        """Identical data should NOT trigger drift."""
        from minivess.data.feature_extraction import extract_batch_features
        from minivess.observability.drift import FeatureDriftDetector

        ref_features = extract_batch_features(reference_volumes)
        # Use same distribution (different seed for slight variation)
        rng = np.random.default_rng(99)
        current = [rng.random((32, 32, 8), dtype=np.float32) for _ in range(20)]
        cur_features = extract_batch_features(current)

        detector = FeatureDriftDetector(ref_features)
        result = detector.detect(cur_features)
        assert not result.drift_detected

    def test_drift_detected_shifted_features(
        self,
        reference_volumes: list[np.ndarray],
        drifted_volumes: list[np.ndarray],
    ) -> None:
        """Intensity-shifted data should trigger drift."""
        from minivess.data.feature_extraction import extract_batch_features
        from minivess.observability.drift import FeatureDriftDetector

        ref_features = extract_batch_features(reference_volumes)
        cur_features = extract_batch_features(drifted_volumes)

        detector = FeatureDriftDetector(ref_features)
        result = detector.detect(cur_features)
        assert result.drift_detected

    def test_drift_result_has_correct_fields(
        self, reference_volumes: list[np.ndarray]
    ) -> None:
        from minivess.data.feature_extraction import extract_batch_features
        from minivess.observability.drift import FeatureDriftDetector

        ref_features = extract_batch_features(reference_volumes)
        detector = FeatureDriftDetector(ref_features)
        result = detector.detect(ref_features)

        assert hasattr(result, "drift_detected")
        assert hasattr(result, "dataset_drift_score")
        assert hasattr(result, "feature_scores")
        assert hasattr(result, "drifted_features")
        assert hasattr(result, "n_features")
        assert hasattr(result, "n_drifted")

    def test_drift_result_feature_scores(
        self,
        reference_volumes: list[np.ndarray],
        drifted_volumes: list[np.ndarray],
    ) -> None:
        """Per-feature scores should be present for each feature."""
        from minivess.data.feature_extraction import extract_batch_features
        from minivess.observability.drift import FeatureDriftDetector

        ref_features = extract_batch_features(reference_volumes)
        cur_features = extract_batch_features(drifted_volumes)

        detector = FeatureDriftDetector(ref_features)
        result = detector.detect(cur_features)

        assert len(result.feature_scores) > 0
        for _feature_name, score in result.feature_scores.items():
            assert isinstance(score, float)

    def test_drift_threshold_configurable(
        self, reference_volumes: list[np.ndarray]
    ) -> None:
        """Custom threshold should change sensitivity."""
        from minivess.data.feature_extraction import extract_batch_features
        from minivess.observability.drift import FeatureDriftDetector

        ref_features = extract_batch_features(reference_volumes)
        # Very strict threshold — should NOT detect drift on same distribution
        detector_strict = FeatureDriftDetector(ref_features, threshold=0.001)
        result = detector_strict.detect(ref_features)
        # With same data, very strict threshold still shouldn't detect
        # (p-values should be high for identical distributions)
        assert not result.drift_detected

    def test_generate_html_report(
        self,
        reference_volumes: list[np.ndarray],
        drifted_volumes: list[np.ndarray],
        tmp_path: Path,
    ) -> None:
        """Should generate an HTML drift report."""
        from minivess.data.feature_extraction import extract_batch_features
        from minivess.observability.drift import FeatureDriftDetector

        ref_features = extract_batch_features(reference_volumes)
        cur_features = extract_batch_features(drifted_volumes)

        detector = FeatureDriftDetector(ref_features)
        report_path = detector.generate_html_report(
            cur_features, output_path=tmp_path / "drift_report.html"
        )
        assert report_path.exists()
        assert report_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# T3: MMD drift detector (Tier 2)
# ---------------------------------------------------------------------------


class TestEmbeddingDriftDetector:
    """Test kernel MMD embedding drift detection."""

    def test_embedding_drift_no_shift(self) -> None:
        """Same distribution → no drift."""
        from minivess.observability.drift import EmbeddingDriftDetector

        rng = np.random.default_rng(42)
        reference = rng.standard_normal((50, 64)).astype(np.float32)
        current = rng.standard_normal((50, 64)).astype(np.float32)

        detector = EmbeddingDriftDetector(reference)
        result = detector.detect(current)
        assert not result.drift_detected

    def test_embedding_drift_detected(self) -> None:
        """Shifted distribution → drift detected."""
        from minivess.observability.drift import EmbeddingDriftDetector

        rng = np.random.default_rng(42)
        reference = rng.standard_normal((50, 64)).astype(np.float32)
        # Large shift
        current = (rng.standard_normal((50, 64)) + 3.0).astype(np.float32)

        detector = EmbeddingDriftDetector(reference)
        result = detector.detect(current)
        assert result.drift_detected

    def test_mmd_detector_configurable_threshold(self) -> None:
        """Custom p-value threshold should change sensitivity."""
        from minivess.observability.drift import EmbeddingDriftDetector

        rng = np.random.default_rng(42)
        reference = rng.standard_normal((50, 64)).astype(np.float32)
        current = rng.standard_normal((50, 64)).astype(np.float32)

        # Very strict threshold should still not detect on same distribution
        detector = EmbeddingDriftDetector(reference, p_val_threshold=0.001)
        result = detector.detect(current)
        assert not result.drift_detected

    def test_mmd_result_has_pvalue(self) -> None:
        """MMD result should include p-value."""
        from minivess.observability.drift import EmbeddingDriftDetector

        rng = np.random.default_rng(42)
        reference = rng.standard_normal((50, 64)).astype(np.float32)
        current = rng.standard_normal((50, 64)).astype(np.float32)

        detector = EmbeddingDriftDetector(reference)
        result = detector.detect(current)
        assert result.dataset_drift_score >= 0.0


# ---------------------------------------------------------------------------
# T4: Dynaconf settings
# ---------------------------------------------------------------------------


class TestDriftConfig:
    """Test drift configuration in Dynaconf settings."""

    def test_settings_have_drift_config(self) -> None:
        """Drift config keys should exist in settings.toml."""
        from dynaconf import Dynaconf

        settings = Dynaconf(
            settings_files=[
                "configs/deployment/settings.toml",
            ],
            environments=True,
        )
        assert hasattr(settings, "DRIFT_DETECTION_THRESHOLD")
        assert hasattr(settings, "DRIFT_ALERT_ENABLED")


# ---------------------------------------------------------------------------
# T5: Integration — synthetic drift → detection pipeline
# ---------------------------------------------------------------------------


class TestDriftDetectionIntegration:
    """End-to-end: synthetic drift → feature extraction → detection."""

    def test_synthetic_intensity_drift_tier1(self) -> None:
        """Intensity shift should be detected by Tier 1 (Evidently)."""
        from minivess.data.drift_synthetic import DriftType, apply_drift
        from minivess.data.feature_extraction import extract_batch_features
        from minivess.observability.drift import FeatureDriftDetector

        rng = np.random.default_rng(42)
        clean = [
            torch.tensor(rng.random((1, 32, 32, 8), dtype=np.float32))
            for _ in range(20)
        ]
        drifted = [
            apply_drift(v, DriftType.INTENSITY_SHIFT, severity=0.8, seed=i)
            for i, v in enumerate(clean)
        ]

        ref_features = extract_batch_features(
            [v.numpy().squeeze() for v in clean]
        )
        cur_features = extract_batch_features(
            [v.numpy().squeeze() for v in drifted]
        )

        detector = FeatureDriftDetector(ref_features)
        result = detector.detect(cur_features)
        assert result.drift_detected
        assert result.n_drifted > 0

    def test_synthetic_noise_drift_tier1(self) -> None:
        """Noise injection should be detected by Tier 1."""
        from minivess.data.drift_synthetic import DriftType, apply_drift
        from minivess.data.feature_extraction import extract_batch_features
        from minivess.observability.drift import FeatureDriftDetector

        rng = np.random.default_rng(42)
        clean = [
            torch.tensor(rng.random((1, 32, 32, 8), dtype=np.float32))
            for _ in range(20)
        ]
        noisy = [
            apply_drift(v, DriftType.NOISE_INJECTION, severity=0.9, seed=i)
            for i, v in enumerate(clean)
        ]

        ref_features = extract_batch_features(
            [v.numpy().squeeze() for v in clean]
        )
        cur_features = extract_batch_features(
            [v.numpy().squeeze() for v in noisy]
        )

        detector = FeatureDriftDetector(ref_features)
        result = detector.detect(cur_features)
        assert result.drift_detected

    def test_clean_data_no_drift_tier1(self) -> None:
        """Clean data from same distribution should NOT trigger drift."""
        from minivess.data.feature_extraction import extract_batch_features
        from minivess.observability.drift import FeatureDriftDetector

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(99)
        ref = [rng1.random((32, 32, 8), dtype=np.float32) for _ in range(20)]
        cur = [rng2.random((32, 32, 8), dtype=np.float32) for _ in range(20)]

        ref_features = extract_batch_features(ref)
        cur_features = extract_batch_features(cur)

        detector = FeatureDriftDetector(ref_features)
        result = detector.detect(cur_features)
        assert not result.drift_detected


# ---------------------------------------------------------------------------
# T6: Topology metrics
# ---------------------------------------------------------------------------


class TestTopologyMetrics:
    """Test clDice proxy for vessel connectivity monitoring."""

    def test_cldice_proxy_returns_float(self) -> None:
        from minivess.data.feature_extraction import compute_cl_dice_proxy

        mask = np.zeros((32, 32, 8), dtype=np.float32)
        mask[10:20, 10:20, 2:6] = 1.0
        score = compute_cl_dice_proxy(mask)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_cldice_proxy_detects_topology_change(self) -> None:
        """Eroded mask should have lower connectivity score."""
        from minivess.data.drift_synthetic import DriftType, apply_drift
        from minivess.data.feature_extraction import compute_cl_dice_proxy

        # Create a binary mask with thin connected structures
        mask = np.zeros((32, 32, 8), dtype=np.float32)
        # Thin line (vessel-like)
        mask[15, 5:25, 4] = 1.0
        mask[5:25, 15, 4] = 1.0

        mask_tensor = torch.tensor(mask).unsqueeze(0)
        eroded = apply_drift(
            mask_tensor, DriftType.TOPOLOGY_CORRUPTION, severity=0.8, seed=42
        )

        score_original = compute_cl_dice_proxy(mask)
        score_eroded = compute_cl_dice_proxy(eroded.numpy().squeeze())

        # Eroded mask should have less connectivity
        assert score_eroded <= score_original
