"""Tests for Reproducibility Phase 2 (Issue #55 — R5.22, R5.23, R5.25).

R5.22: Domain randomization seed fallback to torch.initial_seed()
R5.23: VesselFM weight checksum validation
R5.25: Configurable MAPIE seed (random_state passthrough)
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# ---------------------------------------------------------------------------
# R5.22: Domain randomization seed fallback
# ---------------------------------------------------------------------------


class TestDomainRandomizationSeedFallback:
    """When seed=None, domain randomization should fall back to torch.initial_seed()."""

    def test_seed_none_uses_torch_initial_seed(self) -> None:
        """DomainRandomizationPipeline with seed=None should use torch.initial_seed()."""
        from minivess.data.domain_randomization import (
            DomainRandomizationConfig,
            DomainRandomizationPipeline,
        )

        torch.manual_seed(99)
        config = DomainRandomizationConfig(seed=None)
        pipeline = DomainRandomizationPipeline(config)

        # The pipeline should have resolved seed=None -> torch.initial_seed()
        assert pipeline.resolved_seed is not None
        assert pipeline.resolved_seed == 99

    def test_seed_none_deterministic_with_torch_seed(self) -> None:
        """Two pipelines with seed=None and same torch seed produce identical output."""
        from minivess.data.domain_randomization import (
            DomainRandomizationConfig,
            DomainRandomizationPipeline,
        )

        volume = np.random.default_rng(0).random((8, 8, 8)).astype(np.float32)
        mask = np.zeros((8, 8, 8), dtype=np.uint8)

        torch.manual_seed(42)
        config1 = DomainRandomizationConfig(seed=None)
        p1 = DomainRandomizationPipeline(config1)
        out1, _ = p1.apply(volume, mask)

        torch.manual_seed(42)
        config2 = DomainRandomizationConfig(seed=None)
        p2 = DomainRandomizationPipeline(config2)
        out2, _ = p2.apply(volume, mask)

        np.testing.assert_array_equal(out1, out2)

    def test_explicit_seed_still_works(self) -> None:
        """Explicit seed should still take priority over torch.initial_seed()."""
        from minivess.data.domain_randomization import (
            DomainRandomizationConfig,
            DomainRandomizationPipeline,
        )

        torch.manual_seed(999)
        config = DomainRandomizationConfig(seed=7)
        pipeline = DomainRandomizationPipeline(config)

        assert pipeline.resolved_seed == 7


# ---------------------------------------------------------------------------
# R5.23: VesselFM weight checksum validation
# ---------------------------------------------------------------------------


class TestVesselFMWeightChecksum:
    """VesselFM adapter should provide SHA256 checksum validation for weights."""

    def test_sha256_constant_defined(self) -> None:
        """vesselfm module should define VESSELFM_WEIGHT_SHA256."""
        from minivess.adapters.vesselfm import VESSELFM_WEIGHT_SHA256

        assert isinstance(VESSELFM_WEIGHT_SHA256, str)
        assert len(VESSELFM_WEIGHT_SHA256) == 64  # SHA256 hex digest length

    def test_verify_checksum_passes_on_match(self) -> None:
        """verify_checksum should return True when hash matches."""
        from minivess.adapters.vesselfm import verify_checksum

        data = b"test model weights content"
        expected = hashlib.sha256(data).hexdigest()
        assert verify_checksum(data, expected) is True

    def test_verify_checksum_fails_on_mismatch(self) -> None:
        """verify_checksum should return False when hash does not match."""
        from minivess.adapters.vesselfm import verify_checksum

        data = b"test model weights content"
        wrong_hash = "a" * 64
        assert verify_checksum(data, wrong_hash) is False


# ---------------------------------------------------------------------------
# R5.25: Configurable MAPIE seed
# ---------------------------------------------------------------------------


class TestConfigurableMAPIESeed:
    """LogisticRegression random_state in MAPIE should be configurable."""

    def test_default_random_state_is_42(self) -> None:
        """Default random_state should remain 42 for backward compat."""
        from minivess.ensemble.mapie_conformal import MapieConformalSegmentation

        predictor = MapieConformalSegmentation()
        assert predictor.random_state == 42

    def test_custom_random_state(self) -> None:
        """Should accept a custom random_state parameter."""
        from minivess.ensemble.mapie_conformal import MapieConformalSegmentation

        predictor = MapieConformalSegmentation(alpha=0.1, random_state=123)
        assert predictor.random_state == 123

    def test_random_state_passed_to_logistic_regression(self) -> None:
        """random_state should be forwarded to LogisticRegression during calibrate()."""
        from minivess.ensemble.mapie_conformal import MapieConformalSegmentation

        predictor = MapieConformalSegmentation(alpha=0.1, random_state=77)

        rng = np.random.default_rng(42)
        n_volumes, n_classes, d, h, w = 2, 3, 4, 4, 2
        raw = rng.random((n_volumes, n_classes, d, h, w)).astype(np.float32)
        probs = raw / raw.sum(axis=1, keepdims=True)
        labels = probs.argmax(axis=1).astype(np.int64)

        with patch(
            "minivess.ensemble.mapie_conformal.LogisticRegression"
        ) as mock_lr:
            mock_clf = MagicMock()
            mock_lr.return_value = mock_clf
            mock_clf.classes_ = np.arange(n_classes)

            # We need to mock the SplitConformalClassifier too
            with patch(
                "minivess.ensemble.mapie_conformal.SplitConformalClassifier"
            ):
                predictor.calibrate(probs, labels)

            mock_lr.assert_called_once_with(max_iter=200, random_state=77)
