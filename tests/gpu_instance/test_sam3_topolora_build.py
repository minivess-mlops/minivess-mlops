"""SAM3 TopoLoRA model build test — GPU instance only.

Requires ≥16 GB VRAM (LoRA unfreezes encoder, needs training-level memory).
Run with: make test-gpu (on RunPod/intranet GPU, NEVER on dev machine).

Moved from tests/v2/unit/test_model_builder.py to enforce ZERO skips in
staging/prod tiers. VRAM-gated tests belong here, not in the standard suite.
"""

from __future__ import annotations

import pytest


@pytest.mark.gpu_heavy
def test_build_sam3_topolora() -> None:
    """SAM3 TopoLoRA adapter builds with LoRA layers applied."""
    from minivess.adapters.model_builder import build_adapter
    from minivess.config.model_config import ModelConfig, ModelFamily

    config = ModelConfig(
        family=ModelFamily.SAM3_TOPOLORA,
        name="topolora-test",
        in_channels=1,
        out_channels=2,
        lora_rank=2,
    )
    adapter = build_adapter(config)
    cfg = adapter.get_config()
    assert cfg.family == "sam3_topolora"
