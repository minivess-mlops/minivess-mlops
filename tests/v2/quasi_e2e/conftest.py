"""Conftest for quasi-E2E tests — dynamic parametrization.

Discovers all implemented model×loss combinations from the capability
schema and dynamically parametrizes tests at collection time.

SAM3 TopoLoRA tests are auto-skipped when GPU VRAM < 16 GB.
"""

from __future__ import annotations

import pytest
import torch

from minivess.testing.capability_discovery import (
    build_practical_combinations,
)
from minivess.testing.quasi_e2e_runner import generate_test_ids


def _gpu_vram_gb() -> float:
    """Return total VRAM of GPU 0 in GB, or 0.0 if no CUDA GPU."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Dynamically parametrize quasi-E2E tests with model×loss combos."""
    if "model_loss_combo" in metafunc.fixturenames:
        combos = build_practical_combinations()
        ids = generate_test_ids(combos)
        metafunc.parametrize("model_loss_combo", combos, ids=ids)


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip sam3_topolora tests when GPU VRAM < 16 GB."""
    vram = _gpu_vram_gb()
    if vram >= 16.0:
        return

    skip_marker = pytest.mark.skip(
        reason=(f"SAM3 TopoLoRA requires >= 16 GB VRAM (detected {vram:.1f} GB)"),
    )
    for item in items:
        if "sam3_topolora" in item.nodeid:
            item.add_marker(skip_marker)
