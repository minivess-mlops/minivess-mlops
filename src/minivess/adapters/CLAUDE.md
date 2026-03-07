# SAM3 Adapters — AI Context

## VRAM Requirements (Non-Negotiable)

SAM3 ViT-32L requires **≥16 GB GPU VRAM** (A100-40GB, H100, or RTX 4090).

| Variant | Approx VRAM | Notes |
|---------|-------------|-------|
| V1 Vanilla | ~16 GB | Encoder frozen but full ViT-32L must load |
| V2 TopoLoRA | ~18 GB | LoRA unfrozen; gradient checkpointing recommended |
| V3 Hybrid | ~22 GB | ViT-32L + DynUNet-3D branches |

This requirement is enforced at runtime by `check_sam3_vram()` in
`sam3_vram_check.py`, called from `build_adapter()` before any weights load.

## No Stub, Never

`_StubSam3Encoder` has been **permanently removed** (2026-03-07).

Using the stub = training on random noise = meaningless metrics that look real.
This caused the 2026-03-02 SAM3 fuckup:
see `.claude/metalearning/2026-03-02-sam3-implementation-fuckup.md`.

The enforcement invariants are checked by 10 AST-based tests in
`tests/unit/adapters/test_no_sam3_stub.py` — they will catch stub code
if it ever reappears.

If SAM3 is not installed → **RuntimeError** with installation instructions.
If GPU < 16 GB → **RuntimeError** with hardware requirements and alternatives.

## Key Files

| File | Role |
|------|------|
| `sam3_backbone.py` | `Sam3Backbone` — wraps real SAM3 ViT-32L + FPN neck |
| `sam3_vanilla.py` | V1: frozen encoder, trainable mask decoder |
| `sam3_topolora.py` | V2: LoRA on FFN layers, `cbdice_cldice` loss |
| `sam3_hybrid.py` | V3: frozen ViT-32L + gated DynUNet-3D fusion |
| `sam3_decoder.py` | `Sam3MaskDecoder` — wraps SAM3 mask prediction head |
| `sam3_feature_cache.py` | Feature caching to reduce inference VRAM |
| `sam3_vram_check.py` | Pre-flight VRAM enforcement (≥16 GB) |
| `model_builder.py` | `build_adapter()` factory — calls `_require_sam3()` + `check_sam3_vram()` |

## Installation

```bash
# Step 1: Request model access (Meta gated model — usually instant)
# https://huggingface.co/facebook/sam3 → "Agree and access repository"

# Step 2: Authenticate
uv run huggingface-cli login

# Step 3: Install via Transformers (recommended)
uv add "transformers>=4.50"
uv run python -c "from transformers import Sam3Model; print('OK')"
```

## CI Behavior

Tests that require SAM3 weights are decorated with:

```python
_sam3_skip = pytest.mark.skipif(
    not _sam3_package_available(), reason="SAM3 not installed"
)
```

These tests are skipped in CI (where SAM3 is not installed) and run only
on machines with SAM3 + sufficient GPU VRAM.

Tests that verify stub absence (`test_no_sam3_stub.py`) and VRAM check logic
(`test_sam3_vram_check.py`) run in CI with mocked hardware — they catch
regressions without requiring real SAM3 weights.
