"""Tests for segmentation quality control (Issue #13)."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# T1: QCFlag enum
# ---------------------------------------------------------------------------


class TestQCFlag:
    """Test quality control flag enum."""

    def test_enum_values(self) -> None:
        """QCFlag should have three levels."""
        from minivess.pipeline.segmentation_qc import QCFlag

        assert QCFlag.PASS == "pass"
        assert QCFlag.WARNING == "warning"
        assert QCFlag.FAIL == "fail"


# ---------------------------------------------------------------------------
# T2: QCResult
# ---------------------------------------------------------------------------


class TestQCResult:
    """Test quality control result."""

    def test_construction(self) -> None:
        """QCResult should capture QC metrics."""
        from minivess.pipeline.segmentation_qc import QCFlag, QCResult

        result = QCResult(
            flag=QCFlag.PASS,
            confidence_score=0.92,
            num_components=1,
            volume_ratio=0.05,
            reasons=[],
        )
        assert result.flag == QCFlag.PASS
        assert result.confidence_score == 0.92

    def test_fail_with_reasons(self) -> None:
        """QCResult should carry failure reasons."""
        from minivess.pipeline.segmentation_qc import QCFlag, QCResult

        result = QCResult(
            flag=QCFlag.FAIL,
            confidence_score=0.3,
            num_components=15,
            volume_ratio=0.45,
            reasons=["Too many components", "Volume ratio too high"],
        )
        assert len(result.reasons) == 2


# ---------------------------------------------------------------------------
# T3: SegmentationQC checks
# ---------------------------------------------------------------------------


class TestSegmentationQC:
    """Test individual QC checks."""

    def test_connected_components_pass(self) -> None:
        """Single connected blob should pass component check."""
        from minivess.pipeline.segmentation_qc import SegmentationQC

        qc = SegmentationQC()
        mask = np.zeros((32, 32, 16), dtype=np.uint8)
        mask[10:20, 10:20, 5:12] = 1
        n_components = qc.count_connected_components(mask)
        assert n_components == 1

    def test_connected_components_fragmented(self) -> None:
        """Multiple separate blobs should be detected."""
        from minivess.pipeline.segmentation_qc import SegmentationQC

        qc = SegmentationQC()
        mask = np.zeros((32, 32, 16), dtype=np.uint8)
        mask[2:5, 2:5, 2:5] = 1
        mask[25:28, 25:28, 12:15] = 1
        n_components = qc.count_connected_components(mask)
        assert n_components >= 2

    def test_volume_ratio(self) -> None:
        """Volume ratio should be fraction of foreground voxels."""
        from minivess.pipeline.segmentation_qc import SegmentationQC

        qc = SegmentationQC()
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[0:5, 0:5, 0:5] = 1  # 125 out of 1000
        ratio = qc.compute_volume_ratio(mask)
        assert abs(ratio - 0.125) < 1e-6

    def test_confidence_score(self) -> None:
        """Confidence score should be mean of foreground probabilities."""
        from minivess.pipeline.segmentation_qc import SegmentationQC

        qc = SegmentationQC()
        prob_map = np.full((10, 10, 10), 0.9, dtype=np.float32)
        mask = np.ones((10, 10, 10), dtype=np.uint8)
        score = qc.compute_confidence(prob_map, mask)
        assert abs(score - 0.9) < 1e-5

    def test_border_touching(self) -> None:
        """Mask touching volume edges should be flagged."""
        from minivess.pipeline.segmentation_qc import SegmentationQC

        qc = SegmentationQC()
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[0, 5, 5] = 1  # Touches first slice
        assert qc.check_border_touching(mask) is True

    def test_no_border_touching(self) -> None:
        """Mask not touching edges should not be flagged."""
        from minivess.pipeline.segmentation_qc import SegmentationQC

        qc = SegmentationQC()
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[3:7, 3:7, 3:7] = 1
        assert qc.check_border_touching(mask) is False


# ---------------------------------------------------------------------------
# T4: End-to-end evaluation
# ---------------------------------------------------------------------------


class TestEvaluateQuality:
    """Test end-to-end quality evaluation."""

    def test_good_segmentation_passes(self) -> None:
        """Clean segmentation should receive PASS."""
        from minivess.pipeline.segmentation_qc import (
            QCFlag,
            evaluate_segmentation_quality,
        )

        mask = np.zeros((32, 32, 16), dtype=np.uint8)
        mask[10:20, 10:20, 5:12] = 1
        prob_map = np.zeros((32, 32, 16), dtype=np.float32)
        prob_map[10:20, 10:20, 5:12] = 0.95

        result = evaluate_segmentation_quality(mask, prob_map)
        assert result.flag == QCFlag.PASS

    def test_empty_mask_fails(self) -> None:
        """Empty segmentation should receive FAIL."""
        from minivess.pipeline.segmentation_qc import (
            QCFlag,
            evaluate_segmentation_quality,
        )

        mask = np.zeros((32, 32, 16), dtype=np.uint8)
        prob_map = np.zeros((32, 32, 16), dtype=np.float32)

        result = evaluate_segmentation_quality(mask, prob_map)
        assert result.flag == QCFlag.FAIL

    def test_low_confidence_warns(self) -> None:
        """Low confidence should trigger WARNING or FAIL."""
        from minivess.pipeline.segmentation_qc import (
            QCFlag,
            evaluate_segmentation_quality,
        )

        mask = np.zeros((32, 32, 16), dtype=np.uint8)
        mask[10:20, 10:20, 5:12] = 1
        prob_map = np.zeros((32, 32, 16), dtype=np.float32)
        prob_map[10:20, 10:20, 5:12] = 0.4

        result = evaluate_segmentation_quality(mask, prob_map)
        assert result.flag in {QCFlag.WARNING, QCFlag.FAIL}
