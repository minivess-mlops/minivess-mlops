"""Tests for comparison.py extensions (Phase 3, Task 3.1): Cliff's delta + VDA."""

from __future__ import annotations

import numpy as np

from minivess.pipeline.comparison import (
    cliffs_delta,
    interpret_cliffs_delta,
    interpret_vda,
    vargha_delaney_a,
)


class TestCliffsDelta:
    def test_identical_arrays_is_zero(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert cliffs_delta(a, a) == 0.0

    def test_perfectly_separated_is_one(self) -> None:
        a = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = cliffs_delta(a, b)
        assert d == 1.0

    def test_antisymmetric(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        assert cliffs_delta(a, b) == -cliffs_delta(b, a)

    def test_relationship_to_vda(self) -> None:
        a = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = cliffs_delta(a, b)
        vda = vargha_delaney_a(a, b)
        assert abs(vda - (d + 1) / 2) < 1e-10


class TestVarghaDelaneyA:
    def test_identical_is_half(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        assert vargha_delaney_a(a, a) == 0.5

    def test_range_zero_to_one(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 50)
        b = rng.normal(0, 1, 50)
        vda = vargha_delaney_a(a, b)
        assert 0.0 <= vda <= 1.0


class TestInterpretCliffsDelta:
    def test_negligible(self) -> None:
        assert interpret_cliffs_delta(0.1) == "negligible"

    def test_small(self) -> None:
        assert interpret_cliffs_delta(0.2) == "small"

    def test_medium(self) -> None:
        assert interpret_cliffs_delta(0.4) == "medium"

    def test_large(self) -> None:
        assert interpret_cliffs_delta(0.6) == "large"

    def test_negative_uses_abs(self) -> None:
        assert interpret_cliffs_delta(-0.1) == "negligible"


class TestInterpretVDA:
    def test_negligible_around_half(self) -> None:
        assert interpret_vda(0.5) == "negligible"

    def test_small(self) -> None:
        assert interpret_vda(0.6) == "small"

    def test_medium(self) -> None:
        assert interpret_vda(0.7) == "medium"

    def test_large(self) -> None:
        assert interpret_vda(0.8) == "large"
