"""Tests for centralized seed management (Code Review R3.1).

Validates that set_global_seed sets seeds for all relevant
frameworks (random, numpy, torch) and produces reproducible results.
"""

from __future__ import annotations

import numpy as np
import torch

# ---------------------------------------------------------------------------
# T1: set_global_seed
# ---------------------------------------------------------------------------


class TestSetGlobalSeed:
    """Test centralized seed setting."""

    def test_sets_numpy_seed(self) -> None:
        """set_global_seed should make numpy random reproducible."""
        from minivess.utils.seed import set_global_seed

        set_global_seed(42)
        a = np.random.default_rng(42).random(5)
        set_global_seed(42)
        b = np.random.default_rng(42).random(5)
        np.testing.assert_array_equal(a, b)

    def test_sets_torch_seed(self) -> None:
        """set_global_seed should make torch random reproducible."""
        from minivess.utils.seed import set_global_seed

        set_global_seed(42)
        a = torch.randn(5)
        set_global_seed(42)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_sets_python_random_seed(self) -> None:
        """set_global_seed should make Python random reproducible."""
        import random

        from minivess.utils.seed import set_global_seed

        set_global_seed(42)
        a = [random.random() for _ in range(5)]
        set_global_seed(42)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_different_seeds_give_different_results(self) -> None:
        """Different seeds should produce different random sequences."""
        from minivess.utils.seed import set_global_seed

        set_global_seed(42)
        a = torch.randn(10)
        set_global_seed(99)
        b = torch.randn(10)
        assert not torch.equal(a, b)

    def test_returns_seed_value(self) -> None:
        """set_global_seed should return the seed that was set."""
        from minivess.utils.seed import set_global_seed

        result = set_global_seed(123)
        assert result == 123


# ---------------------------------------------------------------------------
# T2: Re-exports from utils
# ---------------------------------------------------------------------------


class TestSeedReExport:
    """Test that seed utilities are re-exported from utils package."""

    def test_importable_from_utils(self) -> None:
        """set_global_seed should be importable from minivess.utils."""
        from minivess.utils import set_global_seed

        assert callable(set_global_seed)
