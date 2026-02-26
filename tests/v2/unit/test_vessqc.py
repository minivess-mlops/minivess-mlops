"""Tests for VessQC-style uncertainty-guided annotation curation (Issue #10)."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# T1: CurationFlag and CurationReport dataclasses
# ---------------------------------------------------------------------------


class TestCurationFlag:
    """Test CurationFlag dataclass."""

    def test_creation(self) -> None:
        """CurationFlag should store flagged region info."""
        from minivess.validation.vessqc import CurationFlag

        flag = CurationFlag(
            sample_id="vol_001",
            voxel_count=100,
            mean_uncertainty=0.8,
            max_uncertainty=0.95,
            volume_fraction=0.05,
        )
        assert flag.sample_id == "vol_001"
        assert flag.voxel_count == 100
        assert flag.volume_fraction == 0.05


class TestCurationReport:
    """Test CurationReport summary dataclass."""

    def test_creation(self) -> None:
        """CurationReport should summarize flagging results."""
        from minivess.validation.vessqc import CurationFlag, CurationReport

        flags = [
            CurationFlag(
                sample_id="vol_001",
                voxel_count=100,
                mean_uncertainty=0.8,
                max_uncertainty=0.95,
                volume_fraction=0.05,
            ),
        ]
        report = CurationReport(
            flags=flags,
            total_flagged_voxels=100,
            total_voxels=2000,
            flagged_fraction=0.05,
            uncertainty_threshold=0.7,
        )
        assert len(report.flags) == 1
        assert report.flagged_fraction == 0.05


# ---------------------------------------------------------------------------
# T2: flag_uncertain_regions — core flagging function
# ---------------------------------------------------------------------------


class TestFlagUncertainRegions:
    """Test uncertainty-guided region flagging."""

    def test_explicit_threshold(self) -> None:
        """Should flag voxels above the given threshold."""
        from minivess.validation.vessqc import flag_uncertain_regions

        # (B=2, 1, D=4, H=4, W=4) uncertainty maps
        uncertainty = np.zeros((2, 1, 4, 4, 4), dtype=np.float32)
        uncertainty[0, 0, 0, 0, 0] = 0.9  # one hot voxel in sample 0
        uncertainty[1, 0, :, :, :] = 0.8  # entire sample 1 is uncertain

        report = flag_uncertain_regions(
            uncertainty,
            threshold=0.7,
            sample_ids=["vol_001", "vol_002"],
        )
        assert len(report.flags) == 2
        assert report.flags[0].sample_id == "vol_001"
        assert report.flags[0].voxel_count == 1
        assert report.flags[1].voxel_count == 64  # 4*4*4

    def test_percentile_threshold(self) -> None:
        """Should auto-compute threshold from percentile."""
        from minivess.validation.vessqc import flag_uncertain_regions

        rng = np.random.default_rng(42)
        uncertainty = rng.uniform(0.0, 1.0, (3, 1, 8, 8, 4)).astype(np.float32)

        report = flag_uncertain_regions(
            uncertainty,
            percentile=90.0,
            sample_ids=["a", "b", "c"],
        )
        # ~10% of voxels should be flagged
        assert 0.0 < report.flagged_fraction < 0.3
        assert report.uncertainty_threshold > 0.0

    def test_auto_sample_ids(self) -> None:
        """Should generate sample IDs if not provided."""
        from minivess.validation.vessqc import flag_uncertain_regions

        uncertainty = np.ones((2, 1, 4, 4, 4), dtype=np.float32) * 0.9
        report = flag_uncertain_regions(uncertainty, threshold=0.5)
        assert len(report.flags) == 2
        assert report.flags[0].sample_id == "sample_0"
        assert report.flags[1].sample_id == "sample_1"

    def test_no_flags_below_threshold(self) -> None:
        """Should produce zero flags if all uncertainty is below threshold."""
        from minivess.validation.vessqc import flag_uncertain_regions

        uncertainty = np.ones((1, 1, 4, 4, 4), dtype=np.float32) * 0.1
        report = flag_uncertain_regions(uncertainty, threshold=0.5)
        assert report.total_flagged_voxels == 0
        assert report.flagged_fraction == 0.0

    def test_volume_fraction_correct(self) -> None:
        """Volume fraction should be flagged/total per sample."""
        from minivess.validation.vessqc import flag_uncertain_regions

        uncertainty = np.zeros((1, 1, 4, 4, 4), dtype=np.float32)
        # Flag exactly half the voxels
        uncertainty[0, 0, :2, :, :] = 0.9  # 2*4*4 = 32 out of 64

        report = flag_uncertain_regions(uncertainty, threshold=0.5)
        assert report.flags[0].volume_fraction == 32 / 64


# ---------------------------------------------------------------------------
# T3: compute_error_detection_metrics
# ---------------------------------------------------------------------------


class TestErrorDetectionMetrics:
    """Test error detection recall/precision computation."""

    def test_perfect_overlap(self) -> None:
        """Perfect overlap should give recall=1.0, precision=1.0."""
        from minivess.validation.vessqc import compute_error_detection_metrics

        # Flagged mask and error mask match exactly
        flagged_mask = np.zeros((1, 4, 5, 5), dtype=bool)
        flagged_mask[0, 0, :2, :] = True  # 10 voxels

        error_mask = np.zeros((1, 4, 5, 5), dtype=bool)
        error_mask[0, 0, :2, :] = True  # same 10 voxels

        metrics = compute_error_detection_metrics(flagged_mask, error_mask)
        assert metrics["recall"] == 1.0
        assert metrics["precision"] == 1.0

    def test_no_overlap(self) -> None:
        """No overlap should give recall=0.0."""
        from minivess.validation.vessqc import compute_error_detection_metrics

        flagged_mask = np.zeros((1, 4, 4, 4), dtype=bool)
        flagged_mask[0, 0, :, :] = True  # top slice

        error_mask = np.zeros((1, 4, 4, 4), dtype=bool)
        error_mask[0, 3, :, :] = True  # bottom slice

        metrics = compute_error_detection_metrics(flagged_mask, error_mask)
        assert metrics["recall"] == 0.0

    def test_partial_overlap(self) -> None:
        """Partial overlap should give 0 < recall < 1."""
        from minivess.validation.vessqc import compute_error_detection_metrics

        flagged_mask = np.zeros((1, 4, 4, 4), dtype=bool)
        flagged_mask[0, :2, :, :] = True  # top 2 slices (32 voxels)

        error_mask = np.zeros((1, 4, 4, 4), dtype=bool)
        error_mask[0, 1:3, :, :] = True  # middle 2 slices (32 voxels)

        metrics = compute_error_detection_metrics(flagged_mask, error_mask)
        # Overlap: slice 1 (16 voxels), errors: 32 voxels → recall=0.5
        assert metrics["recall"] == 0.5
        assert 0.0 < metrics["precision"] < 1.0

    def test_no_errors_in_ground_truth(self) -> None:
        """No errors in ground truth should give recall=1.0 (vacuously)."""
        from minivess.validation.vessqc import compute_error_detection_metrics

        flagged_mask = np.ones((1, 4, 4, 4), dtype=bool)
        error_mask = np.zeros((1, 4, 4, 4), dtype=bool)

        metrics = compute_error_detection_metrics(flagged_mask, error_mask)
        assert metrics["recall"] == 1.0


# ---------------------------------------------------------------------------
# T4: rank_samples_by_uncertainty
# ---------------------------------------------------------------------------


class TestRankSamples:
    """Test sample ranking by uncertainty."""

    def test_ordering(self) -> None:
        """Samples should be ranked from highest to lowest uncertainty."""
        from minivess.validation.vessqc import rank_samples_by_uncertainty

        uncertainty = np.array(
            [
                [[[0.1, 0.1], [0.1, 0.1]]],  # sample 0: low
                [[[0.9, 0.9], [0.9, 0.9]]],  # sample 1: high
                [[[0.5, 0.5], [0.5, 0.5]]],  # sample 2: medium
            ],
            dtype=np.float32,
        ).reshape(3, 1, 2, 2, 1)

        ranked = rank_samples_by_uncertainty(
            uncertainty,
            sample_ids=["low", "high", "medium"],
        )
        assert ranked[0][0] == "high"
        assert ranked[1][0] == "medium"
        assert ranked[2][0] == "low"

    def test_top_k(self) -> None:
        """top_k should limit the number of returned samples."""
        from minivess.validation.vessqc import rank_samples_by_uncertainty

        uncertainty = (
            np.random.default_rng(42)
            .uniform(
                0.0,
                1.0,
                (5, 1, 4, 4, 2),
            )
            .astype(np.float32)
        )

        ranked = rank_samples_by_uncertainty(
            uncertainty,
            sample_ids=["a", "b", "c", "d", "e"],
            top_k=2,
        )
        assert len(ranked) == 2

    def test_auto_sample_ids(self) -> None:
        """Should generate IDs if not provided."""
        from minivess.validation.vessqc import rank_samples_by_uncertainty

        uncertainty = np.ones((3, 1, 2, 2, 1), dtype=np.float32)
        ranked = rank_samples_by_uncertainty(uncertainty)
        assert len(ranked) == 3
        assert ranked[0][0].startswith("sample_")
