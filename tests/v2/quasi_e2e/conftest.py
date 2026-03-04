"""Conftest for quasi-E2E tests — dynamic parametrization.

Discovers all implemented model×loss combinations from the capability
schema and dynamically parametrizes tests at collection time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from minivess.testing.capability_discovery import (
    build_practical_combinations,
)
from minivess.testing.quasi_e2e_runner import generate_test_ids

if TYPE_CHECKING:
    import pytest


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Dynamically parametrize quasi-E2E tests with model×loss combos."""
    if "model_loss_combo" in metafunc.fixturenames:
        combos = build_practical_combinations()
        ids = generate_test_ids(combos)
        metafunc.parametrize("model_loss_combo", combos, ids=ids)
