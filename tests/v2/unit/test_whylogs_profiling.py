"""Tests for whylogs continuous profiling service (T-B2).

Tests that every volume is profiled, profiles are mergeable,
and Prometheus metrics are exported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

whylogs = pytest.importorskip("whylogs", reason="whylogs not installed")

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_volumes() -> list[np.ndarray]:
    """10 synthetic 3D volumes for profiling."""
    rng = np.random.default_rng(42)
    return [rng.random((32, 32, 8), dtype=np.float32) for _ in range(10)]


@pytest.fixture()
def drifted_volumes() -> list[np.ndarray]:
    """10 volumes with intensity drift for comparison."""
    rng = np.random.default_rng(42)
    clean = [rng.random((32, 32, 8), dtype=np.float32) for _ in range(10)]
    return [(v * 2.5 + 0.4).astype(np.float32) for v in clean]


# ---------------------------------------------------------------------------
# T-B2.1: Single volume profiling
# ---------------------------------------------------------------------------


class TestVolumeProfiler:
    """Test profiling of individual 3D volumes."""

    def test_profile_single_volume_returns_result_set(self) -> None:
        """profile_volume() should return a whylogs ResultSet."""
        from minivess.observability.whylogs_service import WhylogsVolumeProfiler

        rng = np.random.default_rng(42)
        volume = rng.random((32, 32, 8), dtype=np.float32)

        profiler = WhylogsVolumeProfiler()
        result = profiler.profile_volume(volume, volume_id="vol_001")
        assert result is not None

    def test_profile_captures_feature_columns(self) -> None:
        """Profile should contain expected statistical features."""
        from minivess.observability.whylogs_service import WhylogsVolumeProfiler

        rng = np.random.default_rng(42)
        volume = rng.random((32, 32, 8), dtype=np.float32)

        profiler = WhylogsVolumeProfiler()
        profile = profiler.profile_volume(volume, volume_id="vol_001")
        columns = profiler.get_column_names(profile)

        expected = {"mean", "std", "p5", "p95", "snr"}
        assert expected.issubset(set(columns)), (
            f"Missing columns: {expected - set(columns)}"
        )

    def test_profile_volume_with_metadata(self) -> None:
        """Profile should accept and store metadata tags."""
        from minivess.observability.whylogs_service import WhylogsVolumeProfiler

        rng = np.random.default_rng(42)
        volume = rng.random((32, 32, 8), dtype=np.float32)

        profiler = WhylogsVolumeProfiler()
        profile = profiler.profile_volume(
            volume,
            volume_id="vol_001",
            tags={"dataset": "minivess", "split": "train"},
        )
        assert profile is not None


# ---------------------------------------------------------------------------
# T-B2.2: Batch profiling
# ---------------------------------------------------------------------------


class TestBatchProfiling:
    """Test profiling of volume batches."""

    def test_profile_batch_returns_profiles(
        self, sample_volumes: list[np.ndarray]
    ) -> None:
        """profile_batch() should return one profile per volume."""
        from minivess.observability.whylogs_service import WhylogsVolumeProfiler

        profiler = WhylogsVolumeProfiler()
        profiles = profiler.profile_batch(sample_volumes)
        assert len(profiles) == len(sample_volumes)

    def test_profile_batch_with_ids(self, sample_volumes: list[np.ndarray]) -> None:
        """Batch profiling should accept volume IDs."""
        from minivess.observability.whylogs_service import WhylogsVolumeProfiler

        profiler = WhylogsVolumeProfiler()
        ids = [f"vol_{i:03d}" for i in range(len(sample_volumes))]
        profiles = profiler.profile_batch(sample_volumes, volume_ids=ids)
        assert len(profiles) == len(sample_volumes)


# ---------------------------------------------------------------------------
# T-B2.3: Profile merging
# ---------------------------------------------------------------------------


class TestProfileMerging:
    """Test that whylogs profiles are mergeable across batches."""

    def test_merge_two_profiles(self, sample_volumes: list[np.ndarray]) -> None:
        """Two profiles should merge into one without error."""
        from minivess.observability.whylogs_service import WhylogsVolumeProfiler

        profiler = WhylogsVolumeProfiler()
        p1 = profiler.profile_volume(sample_volumes[0], volume_id="vol_000")
        p2 = profiler.profile_volume(sample_volumes[1], volume_id="vol_001")

        merged = profiler.merge_profiles([p1, p2])
        assert merged is not None

    def test_merge_preserves_columns(self, sample_volumes: list[np.ndarray]) -> None:
        """Merged profile should have the same columns as originals."""
        from minivess.observability.whylogs_service import WhylogsVolumeProfiler

        profiler = WhylogsVolumeProfiler()
        p1 = profiler.profile_volume(sample_volumes[0], volume_id="vol_000")
        p2 = profiler.profile_volume(sample_volumes[1], volume_id="vol_001")

        cols_before = set(profiler.get_column_names(p1))
        merged = profiler.merge_profiles([p1, p2])
        cols_after = set(profiler.get_column_names(merged))

        assert cols_before == cols_after

    def test_merge_batch_profiles(self, sample_volumes: list[np.ndarray]) -> None:
        """All batch profiles should merge into a single summary."""
        from minivess.observability.whylogs_service import WhylogsVolumeProfiler

        profiler = WhylogsVolumeProfiler()
        profiles = profiler.profile_batch(sample_volumes)
        merged = profiler.merge_profiles(profiles)
        assert merged is not None


# ---------------------------------------------------------------------------
# T-B2.4: Profile persistence
# ---------------------------------------------------------------------------


class TestProfilePersistence:
    """Test saving and loading profiles."""

    def test_save_profile_to_disk(
        self, sample_volumes: list[np.ndarray], tmp_path: Path
    ) -> None:
        """Profile should be saveable to disk as a .bin file."""
        from minivess.observability.whylogs_service import WhylogsVolumeProfiler

        profiler = WhylogsVolumeProfiler()
        profile = profiler.profile_volume(sample_volumes[0], volume_id="vol_000")

        out_path = tmp_path / "profile.bin"
        profiler.save_profile(profile, out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_load_profile_from_disk(
        self, sample_volumes: list[np.ndarray], tmp_path: Path
    ) -> None:
        """Saved profile should be loadable and have same columns."""
        from minivess.observability.whylogs_service import WhylogsVolumeProfiler

        profiler = WhylogsVolumeProfiler()
        profile = profiler.profile_volume(sample_volumes[0], volume_id="vol_000")

        out_path = tmp_path / "profile.bin"
        profiler.save_profile(profile, out_path)
        loaded = profiler.load_profile(out_path)

        assert set(profiler.get_column_names(profile)) == set(
            profiler.get_column_names(loaded)
        )


# ---------------------------------------------------------------------------
# T-B2.5: Prometheus metrics export
# ---------------------------------------------------------------------------


class TestPrometheusExport:
    """Test Prometheus text exposition format export."""

    def test_format_prometheus_metrics(self, sample_volumes: list[np.ndarray]) -> None:
        """Should produce valid Prometheus text format."""
        from minivess.observability.whylogs_service import (
            WhylogsVolumeProfiler,
            format_whylogs_prometheus,
        )

        profiler = WhylogsVolumeProfiler()
        profile = profiler.profile_volume(sample_volumes[0], volume_id="vol_000")
        prom_text = format_whylogs_prometheus(profile, dataset="minivess")

        assert isinstance(prom_text, str)
        assert len(prom_text) > 0
        # Prometheus format: metric_name{labels} value
        assert "whylogs_" in prom_text
        assert "dataset=" in prom_text

    def test_prometheus_metrics_have_expected_lines(
        self, sample_volumes: list[np.ndarray]
    ) -> None:
        """Prometheus output should include distribution summary metrics."""
        from minivess.observability.whylogs_service import (
            WhylogsVolumeProfiler,
            format_whylogs_prometheus,
        )

        profiler = WhylogsVolumeProfiler()
        profile = profiler.profile_volume(sample_volumes[0], volume_id="vol_000")
        prom_text = format_whylogs_prometheus(profile, dataset="minivess")

        # Should have count and distribution metrics
        assert "whylogs_column_count" in prom_text or "whylogs_profile" in prom_text


# ---------------------------------------------------------------------------
# T-B2.6: Drift comparison via profiles
# ---------------------------------------------------------------------------


class TestProfileDriftComparison:
    """Test drift detection via whylogs profile comparison."""

    def test_compare_profiles_same_distribution(
        self, sample_volumes: list[np.ndarray]
    ) -> None:
        """Same-distribution profiles should show no significant drift."""
        from minivess.observability.whylogs_service import WhylogsVolumeProfiler

        profiler = WhylogsVolumeProfiler()
        ref_profiles = profiler.profile_batch(sample_volumes[:5])
        cur_profiles = profiler.profile_batch(sample_volumes[5:])

        ref_merged = profiler.merge_profiles(ref_profiles)
        cur_merged = profiler.merge_profiles(cur_profiles)

        comparison = profiler.compare_profiles(ref_merged, cur_merged)
        assert isinstance(comparison, dict)
        assert "drift_detected" in comparison

    def test_compare_profiles_detects_drift(
        self,
        sample_volumes: list[np.ndarray],
        drifted_volumes: list[np.ndarray],
    ) -> None:
        """Shifted distribution should be flagged as drifted."""
        from minivess.observability.whylogs_service import WhylogsVolumeProfiler

        profiler = WhylogsVolumeProfiler()
        ref_profiles = profiler.profile_batch(sample_volumes)
        cur_profiles = profiler.profile_batch(drifted_volumes)

        ref_merged = profiler.merge_profiles(ref_profiles)
        cur_merged = profiler.merge_profiles(cur_profiles)

        comparison = profiler.compare_profiles(ref_merged, cur_merged)
        assert comparison["drift_detected"] is True
