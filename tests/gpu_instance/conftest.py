"""GPU instance tests — NOT part of the standard test suite.

These tests are excluded from default testpaths in pyproject.toml.
They run ONLY on GPU instances (RunPod, intranet servers) via:

    make test-gpu
    # or: uv run pytest tests/gpu_instance/ --run-gpu-heavy

SAM3 tests here require:
- Real SAM3 pretrained weights (HuggingFace facebook/sam3)
- Adequate GPU VRAM (>= 6 GB for vanilla, >= 16 GB for TopoLoRA)
- CUDA-capable GPU (CPU is too slow for forward passes)

See #564 for the dockerized GPU benchmark plan.
"""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest


@pytest.fixture(scope="session", autouse=True)
def _gpu_instance_env() -> Generator[None]:
    """Set environment for GPU instance tests."""
    os.environ.setdefault("MINIVESS_ALLOW_HOST", "1")
    yield
    os.environ.pop("MINIVESS_ALLOW_HOST", None)
