"""Tests for numerical edge cases across modules (Issue #53 — R5.11).

Dice with no true positives, ECE with single batch, KS with identical
samples, volume_ratio edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# R5.11 T1: Dice score with no true positives
# ---------------------------------------------------------------------------


class TestDiceNoTruePositives:
    """Dice score should handle gracefully when there are no true positives."""

    def test_dice_all_background(self) -> None:
        """All-background prediction and label should not raise."""
        from minivess.pipeline.metrics import SegmentationMetrics

        metrics = SegmentationMetrics(num_classes=2)
        # Both prediction and target are all background (class 0)
        pred = torch.zeros(1, 2, 8, 8, 4)
        pred[:, 0] = 1.0  # All background probability
        target = torch.zeros(1, 8, 8, 4, dtype=torch.long)
        metrics.update(pred, target)
        result = metrics.compute()
        assert "dice" in result.values
        assert np.isfinite(result.values["dice"])

    def test_dice_all_foreground(self) -> None:
        """All-foreground prediction and label should produce dice ~1.0."""
        from minivess.pipeline.metrics import SegmentationMetrics

        metrics = SegmentationMetrics(num_classes=2)
        pred = torch.zeros(1, 2, 8, 8, 4)
        pred[:, 1] = 1.0  # All foreground
        target = torch.ones(1, 8, 8, 4, dtype=torch.long)
        metrics.update(pred, target)
        result = metrics.compute()
        assert result.values["dice"] > 0.9

    def test_dice_prediction_mismatch(self) -> None:
        """Prediction=all background, label=all foreground should produce low dice."""
        from minivess.pipeline.metrics import SegmentationMetrics

        metrics = SegmentationMetrics(num_classes=2)
        pred = torch.zeros(1, 2, 8, 8, 4)
        pred[:, 0] = 1.0  # All background
        target = torch.ones(1, 8, 8, 4, dtype=torch.long)  # All foreground
        metrics.update(pred, target)
        result = metrics.compute()
        # Dice should be 0 or very low
        assert result.values["dice"] < 0.5


# ---------------------------------------------------------------------------
# R5.11 T2: ECE with single batch (small sample)
# ---------------------------------------------------------------------------


class TestECESingleBatch:
    """ECE should handle edge cases with small sample sizes."""

    def test_ece_single_sample(self) -> None:
        """ECE with a single sample should not raise."""
        from minivess.ensemble.calibration import expected_calibration_error

        confidences = np.array([0.9])
        accuracies = np.array([1.0])
        ece, mce = expected_calibration_error(confidences, accuracies)
        assert np.isfinite(ece)
        assert np.isfinite(mce)

    def test_ece_perfect_calibration(self) -> None:
        """Perfectly calibrated predictions should have ECE near 0."""
        from minivess.ensemble.calibration import expected_calibration_error

        rng = np.random.default_rng(42)
        n = 1000
        confidences = rng.uniform(0.5, 1.0, n)
        # Each sample is correct with probability equal to its confidence
        accuracies = (rng.random(n) < confidences).astype(np.float64)
        ece, _mce = expected_calibration_error(confidences, accuracies)
        # ECE should be small for well-calibrated predictions
        assert ece < 0.15

    def test_ece_empty_bins(self) -> None:
        """ECE with all samples in one bin should still work."""
        from minivess.ensemble.calibration import expected_calibration_error

        # All confidences near 1.0 — most bins empty
        confidences = np.full(100, 0.95)
        accuracies = np.ones(100)
        ece, mce = expected_calibration_error(confidences, accuracies)
        assert np.isfinite(ece)
        assert ece < 0.1  # near-perfect calibration in one bin


# ---------------------------------------------------------------------------
# R5.11 T3: KS test with identical samples
# ---------------------------------------------------------------------------


class TestKSIdenticalSamples:
    """KS test on identical distributions should report no drift."""

    def test_ks_identical_features_no_drift(self) -> None:
        """Identical reference and current features should NOT trigger drift."""
        from minivess.observability.drift import FeatureDriftDetector

        rng = np.random.default_rng(42)
        data = rng.random((50, 3))
        df = pd.DataFrame(data, columns=["f1", "f2", "f3"])

        detector = FeatureDriftDetector(df, threshold=0.05)
        result = detector.detect(df)
        # Identical samples: p-values should be 1.0 (or very high)
        assert not result.drift_detected
        assert result.n_drifted == 0

    def test_ks_completely_shifted_detects_drift(self) -> None:
        """Completely shifted features should trigger drift."""
        from minivess.observability.drift import FeatureDriftDetector

        rng = np.random.default_rng(42)
        ref_data = rng.random((50, 3))
        cur_data = ref_data + 10.0  # massive shift
        ref_df = pd.DataFrame(ref_data, columns=["f1", "f2", "f3"])
        cur_df = pd.DataFrame(cur_data, columns=["f1", "f2", "f3"])

        detector = FeatureDriftDetector(ref_df, threshold=0.05)
        result = detector.detect(cur_df)
        assert result.drift_detected


# ---------------------------------------------------------------------------
# R5.11 T4: Volume ratio edge cases
# ---------------------------------------------------------------------------


class TestVolumeRatioEdgeCases:
    """volume_ratio should handle empty mask and full mask gracefully."""

    def test_volume_ratio_empty_mask(self) -> None:
        """Empty mask should return volume_ratio=0.0."""
        from minivess.pipeline.segmentation_qc import SegmentationQC

        qc = SegmentationQC()
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        assert qc.compute_volume_ratio(mask) == 0.0

    def test_volume_ratio_full_mask(self) -> None:
        """Full mask should return volume_ratio=1.0."""
        from minivess.pipeline.segmentation_qc import SegmentationQC

        qc = SegmentationQC()
        mask = np.ones((10, 10, 10), dtype=np.uint8)
        assert qc.compute_volume_ratio(mask) == 1.0

    def test_volume_ratio_single_voxel(self) -> None:
        """Mask with one voxel in 10x10x10 volume should return 0.001."""
        from minivess.pipeline.segmentation_qc import SegmentationQC

        qc = SegmentationQC()
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[5, 5, 5] = 1
        expected = 1.0 / 1000.0
        assert abs(qc.compute_volume_ratio(mask) - expected) < 1e-9

    def test_confidence_empty_foreground(self) -> None:
        """Confidence on empty foreground mask should return 0.0."""
        from minivess.pipeline.segmentation_qc import SegmentationQC

        qc = SegmentationQC()
        prob_map = np.full((10, 10, 10), 0.9, dtype=np.float32)
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        assert qc.compute_confidence(prob_map, mask) == 0.0
