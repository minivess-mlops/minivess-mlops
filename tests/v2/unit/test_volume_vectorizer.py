"""Tests for 3D volume vectorization to 1D feature vectors.

TDD RED phase for Task T-C3 (Issue #767).
Enables tabular drift tools (NannyML, Deepchecks) on 3D volumetric data.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sample_volume() -> np.ndarray:
    """Create a simple 3D volume for testing."""
    rng = np.random.default_rng(42)
    return rng.random((32, 32, 32), dtype=np.float32)


@pytest.fixture
def sample_volumes() -> list[np.ndarray]:
    """Create a batch of 3D volumes."""
    rng = np.random.default_rng(42)
    return [rng.random((32, 32, 32), dtype=np.float32) for _ in range(5)]


class TestStatisticalVectorizer:
    """Test statistical feature vectorization (existing 9 features)."""

    def test_vectorize_single_volume(self, sample_volume: np.ndarray) -> None:
        from minivess.data.volume_vectorizer import vectorize_volume

        vector = vectorize_volume(sample_volume, strategy="statistical")
        assert isinstance(vector, np.ndarray)
        assert vector.ndim == 1
        assert len(vector) >= 9  # at least 9 statistical features

    def test_vectorize_batch(self, sample_volumes: list[np.ndarray]) -> None:
        from minivess.data.volume_vectorizer import vectorize_batch

        matrix = vectorize_batch(sample_volumes, strategy="statistical")
        assert isinstance(matrix, np.ndarray)
        assert matrix.ndim == 2
        assert matrix.shape[0] == 5  # 5 volumes
        assert matrix.shape[1] >= 9

    def test_vectorize_returns_float(self, sample_volume: np.ndarray) -> None:
        from minivess.data.volume_vectorizer import vectorize_volume

        vector = vectorize_volume(sample_volume, strategy="statistical")
        assert vector.dtype in (np.float32, np.float64)


class TestHistogramVectorizer:
    """Test histogram-based vectorization."""

    def test_histogram_vector_length(self, sample_volume: np.ndarray) -> None:
        from minivess.data.volume_vectorizer import vectorize_volume

        vector = vectorize_volume(
            sample_volume, strategy="histogram", config={"n_bins": 64}
        )
        assert len(vector) == 64

    def test_histogram_sums_to_one(self, sample_volume: np.ndarray) -> None:
        from minivess.data.volume_vectorizer import vectorize_volume

        vector = vectorize_volume(sample_volume, strategy="histogram")
        assert abs(vector.sum() - 1.0) < 1e-5  # normalized histogram


class TestFFTVectorizer:
    """Test spatial frequency (FFT) vectorization."""

    def test_fft_vector_exists(self, sample_volume: np.ndarray) -> None:
        from minivess.data.volume_vectorizer import vectorize_volume

        vector = vectorize_volume(sample_volume, strategy="fft")
        assert isinstance(vector, np.ndarray)
        assert vector.ndim == 1
        assert len(vector) > 0

    def test_fft_different_volumes_differ(self) -> None:
        from minivess.data.volume_vectorizer import vectorize_volume

        rng = np.random.default_rng(42)
        vol_smooth = np.ones((32, 32, 32), dtype=np.float32) * 0.5
        vol_noisy = rng.random((32, 32, 32), dtype=np.float32)

        v_smooth = vectorize_volume(vol_smooth, strategy="fft")
        v_noisy = vectorize_volume(vol_noisy, strategy="fft")
        assert not np.allclose(v_smooth, v_noisy)


class TestVectorizerRegistry:
    """Test the strategy registry."""

    def test_available_strategies(self) -> None:
        from minivess.data.volume_vectorizer import VECTORIZATION_STRATEGIES

        assert "statistical" in VECTORIZATION_STRATEGIES
        assert "histogram" in VECTORIZATION_STRATEGIES
        assert "fft" in VECTORIZATION_STRATEGIES

    def test_unknown_strategy_raises(self, sample_volume: np.ndarray) -> None:
        from minivess.data.volume_vectorizer import vectorize_volume

        with pytest.raises(KeyError, match="nonexistent"):
            vectorize_volume(sample_volume, strategy="nonexistent")

    def test_vectorize_batch_returns_dataframe_option(
        self, sample_volumes: list[np.ndarray]
    ) -> None:
        from minivess.data.volume_vectorizer import vectorize_batch_to_dataframe

        df = vectorize_batch_to_dataframe(sample_volumes, strategy="statistical")
        assert df.shape[0] == 5
        assert df.shape[1] >= 9
        assert list(df.columns)  # has named columns
