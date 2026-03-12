"""Tests for Tier 2 MMD drift detection with permutation p-values (#574 T3, #601).

Spec adaptation: alibi-detect requires numpy<2.0 (project needs numpy>=2.0).
Tier 2 uses scipy/sklearn-based permutation MMD instead of alibi-detect.
Tests verify statistically rigorous p-values, configurable kernels, and
correct detection of embedding drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _make_embeddings(
    *, n_samples: int = 30, dim: int = 64, seed: int = 42
) -> NDArray[np.float32]:
    """Generate random embeddings from a standard normal distribution."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, dim)).astype(np.float32)


def _make_shifted_embeddings(
    reference: NDArray[np.float32],
    *,
    shift: float = 3.0,
    seed: int = 99,
) -> NDArray[np.float32]:
    """Create embeddings with synthetic mean shift (covariate drift)."""
    rng = np.random.default_rng(seed)
    shifted = reference + shift
    # Add small noise to avoid degenerate kernel
    shifted += rng.standard_normal(shifted.shape).astype(np.float32) * 0.1
    return shifted


class TestMMDTier2:
    """Verify Tier 2 MMD drift detection with permutation p-values."""

    def test_mmd_drift_detected_on_shifted_embeddings(self) -> None:
        """Synthetic mean shift → drift detected (p < threshold)."""
        from minivess.observability.drift import EmbeddingDriftDetector

        ref = _make_embeddings()
        cur = _make_shifted_embeddings(ref, shift=5.0)
        detector = EmbeddingDriftDetector(ref, n_permutations=200)
        result = detector.detect(cur)
        assert result.drift_detected is True
        assert result.dataset_drift_score < 0.05  # p-value

    def test_mmd_no_drift_on_same_embeddings(self) -> None:
        """Same distribution → no drift (p > threshold)."""
        from minivess.observability.drift import EmbeddingDriftDetector

        ref = _make_embeddings(seed=42)
        cur = _make_embeddings(seed=43)  # Same distribution, different seed
        detector = EmbeddingDriftDetector(ref, n_permutations=200)
        result = detector.detect(cur)
        assert result.drift_detected is False
        assert result.dataset_drift_score > 0.05  # p-value

    def test_mmd_returns_p_value(self) -> None:
        """p_value field populated in result (stored as dataset_drift_score)."""
        from minivess.observability.drift import EmbeddingDriftDetector

        ref = _make_embeddings()
        cur = _make_shifted_embeddings(ref)
        detector = EmbeddingDriftDetector(ref, n_permutations=100)
        result = detector.detect(cur)
        # p-value must be between 0 and 1
        assert 0.0 <= result.dataset_drift_score <= 1.0
        # mmd_statistic should be in feature_scores
        assert "mmd_statistic" in result.feature_scores

    def test_embedding_drift_detector_configurable_kernel(self) -> None:
        """RBF vs linear kernel both work and detect drift."""
        from minivess.observability.drift import EmbeddingDriftDetector

        ref = _make_embeddings()
        cur = _make_shifted_embeddings(ref, shift=5.0)

        # RBF kernel (default)
        det_rbf = EmbeddingDriftDetector(ref, kernel="rbf", n_permutations=100)
        result_rbf = det_rbf.detect(cur)
        assert result_rbf.drift_detected is True

        # Linear kernel
        det_linear = EmbeddingDriftDetector(ref, kernel="linear", n_permutations=100)
        result_linear = det_linear.detect(cur)
        assert result_linear.drift_detected is True

    def test_mmd_statistic_increases_with_shift_magnitude(self) -> None:
        """Larger mean shift → larger MMD statistic."""
        from minivess.observability.drift import EmbeddingDriftDetector

        ref = _make_embeddings()
        cur_small = _make_shifted_embeddings(ref, shift=1.0)
        cur_large = _make_shifted_embeddings(ref, shift=10.0)

        detector = EmbeddingDriftDetector(ref, n_permutations=50)
        result_small = detector.detect(cur_small)
        result_large = detector.detect(cur_large)

        mmd_small = result_small.feature_scores["mmd_statistic"]
        mmd_large = result_large.feature_scores["mmd_statistic"]
        assert mmd_large > mmd_small
