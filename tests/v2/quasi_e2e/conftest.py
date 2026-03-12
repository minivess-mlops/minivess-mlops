"""Conftest for quasi-E2E tests — dynamic parametrization.

Discovers all implemented model×loss combinations from the capability
schema and dynamically parametrizes tests at collection time.

SAM3 tests are marked ``slow`` + ``model_loading`` so they're excluded from
staging/prod tiers.  SAM3 TopoLoRA is additionally skipped on <16 GB VRAM.
"""

from __future__ import annotations

import pytest
import torch

from minivess.testing.capability_discovery import (
    build_practical_combinations,
)
from minivess.testing.quasi_e2e_runner import (
    _SAM3_MODELS,
    generate_test_ids,
)


def _gpu_vram_gb() -> float:
    """Return total VRAM of GPU 0 in GB, or 0.0 if no CUDA GPU."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def _is_sam3_test(nodeid: str) -> bool:
    """Return True if the test nodeid belongs to a SAM3 model combo."""
    return any(name in nodeid for name in _SAM3_MODELS)


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
    """Mark SAM3 tests as slow + model_loading; skip sam3_topolora on low VRAM.

    SAM3 model loading (ViT-32L, 848M params from HuggingFace) takes >60 min
    on dev machines.  Marking with ``slow`` and ``model_loading`` ensures the
    staging tier (``-m "not model_loading and not slow"``) excludes them.
    """
    vram = _gpu_vram_gb()
    slow_marker = pytest.mark.slow
    model_loading_marker = pytest.mark.model_loading

    skip_topolora = pytest.mark.skip(
        reason=f"SAM3 TopoLoRA requires >= 16 GB VRAM (detected {vram:.1f} GB)",
    )

    for item in items:
        if not _is_sam3_test(item.nodeid):
            continue

        # All SAM3 combos: mark slow + model_loading
        item.add_marker(slow_marker)
        item.add_marker(model_loading_marker)

        # sam3_topolora additionally needs ≥16 GB VRAM
        if "sam3_topolora" in item.nodeid and vram < 16.0:
            item.add_marker(skip_topolora)
