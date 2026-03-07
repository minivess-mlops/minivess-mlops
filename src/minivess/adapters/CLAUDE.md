# SAM3 Adapters — AI Context

## VRAM Requirements (Per-Variant)

VRAM requirements depend on **whether the SAM3 encoder is frozen** AND on
**attention implementation**. SDPA (Scaled Dot-Product Attention) is critical.

| Variant | Encoder Frozen | Attention | Min VRAM | Peak (measured) | Notes |
|---------|---------------|-----------|----------|----------------|-------|
| V1 Vanilla | Yes | SDPA | **≥4 GB** | **2.9 GB** | Verified on RTX 2070 Super (2026-03-07) |
| V1 Vanilla | Yes | Eager | ≥8 GB | **7+ GB OOM** | Eager materializes 5184×5184 attention matrices |
| V2 TopoLoRA | No | SDPA | **≥16 GB** | TBD | LoRA on all 32 ViT-32L blocks, gradients flow |
| V3 Hybrid | Yes | SDPA | **≥4 GB** | TBD | Sequential SAM→DynUNet, not simultaneous |

Gate is enforced by `check_sam3_vram(variant, mode)` in `sam3_vram_check.py`.
- `mode="inference"` → 6 GB gate → V1, V3
- `mode="training"` → 16 GB gate → V2

### Validation ROI Size ≠ Training Patch Size (Critical for Speed)

The ViT-32L encoder always resizes input to 1008×1008 regardless of spatial size.
A 64×64 patch costs the **same** encoder compute as a 512×512 patch.
But with patch=(64,64,3), a 512×512 volume creates 11×11=121 spatial windows.

**Training patches** (64,64,3): small crops for data augmentation + batch diversity.
**Validation ROI** (512,512,3): full-slice windows to minimize encoder calls.

With (512,512,3) validation: ~27 windows per 512×512×61 vol → ~4 min total validation.
With (64,64,3) validation: ~3267 windows per volume → ~6 hours total validation.

This is set in `train_flow.py` via separate `val_roi` for SAM3 models.

### SDPA is Non-Negotiable (Verified 2026-03-07)

Without SDPA, the ViT-32L encoder materializes 5184×5184 attention matrices per head
(72×72 patch grid from 1008×1008 input). With 16 heads, this is ~862 MB per layer.
**Eager attention OOMs on 8 GB GPUs.**

SDPA avoids materializing the full attention matrix, reducing encoder peak from
~7 GB to ~1.1 GB. This is set via `attn_implementation="sdpa"` in
`Sam3Model.from_pretrained()`.

```python
model = Sam3Model.from_pretrained(
    "facebook/sam3",
    attn_implementation="sdpa",  # CRITICAL for 8 GB GPUs
    torch_dtype=torch.float16,
)
```

### MONAI Dimension Order (B, C, H, W, D) — NOT (B, C, D, H, W)

**CRITICAL:** MONAI outputs volumes as `(B, C, H, W, D)` where depth is LAST.
The SAM3 adapters must use:
```python
b, c, h, w, d = images.shape
for z_idx in range(d):
    slice_2d = images[:, :, :, :, z_idx]  # (B, C, H, W)
```

Getting this wrong (treating dim 2 as depth) causes 21x more encoder calls
(64 instead of 3 with patch_size=(64,64,3)) → OOM. This was a bug found
and fixed on 2026-03-07.

### Practical on RTX 2070 Super (8 GB) — Verified 2026-03-07

| Task | Peak VRAM | Time/Epoch | Feasible? |
|------|-----------|------------|-----------|
| V1 Vanilla training step (B=4, D=3) | 2.9 GB | ~3s | ✅ Yes |
| V1 Vanilla training epoch (47 vols) | 3.5 GB | ~8 min | ✅ Yes |
| V1 Vanilla validation (512×512×61, SW, roi=512²) | 3.6 GB | ~15 s/vol | ✅ Yes |
| V2 TopoLoRA | — | — | ❌ No (OOM) |

Validation uses `val_roi_size=(512,512,3)` (full-slice), reducing windows from
~3300 to ~27 per volume. The encoder costs the same per-window (always 1008×1008),
so larger ROI is strictly better. `val_interval=10` (validate every 10 epochs).

### Multi-Environment Compute (CPU / Consumer GPU / Cloud)

| Variant | CPU (64 GB RAM) | RTX 2070 Super (8 GB) | A100-40GB |
|---------|----------------|----------------------|-----------|
| V1 Vanilla (50 ep × 3 fold) | ~30 days | **~4-6 h** (val_roi=512², val_interval=10) | ~2 h |
| V2 TopoLoRA | 470 days | OOM | **38 h** |
| V3 Hybrid | TBD | TBD | TBD |

## No Stub, Never

`_StubSam3Encoder` has been **permanently removed** (2026-03-07).

Using the stub = training on random noise = meaningless metrics that look real.
This caused the 2026-03-02 SAM3 fuckup:
see `.claude/metalearning/2026-03-02-sam3-implementation-fuckup.md`.

The enforcement invariants are checked by 10 AST-based tests in
`tests/unit/adapters/test_no_sam3_stub.py` — they will catch stub code
if it ever reappears.

If SAM3 is not installed → **RuntimeError** with installation instructions.
If GPU VRAM below mode-appropriate threshold → **RuntimeError** with hardware requirements.

## Key Files

| File | Role |
|------|------|
| `sam3_backbone.py` | `Sam3Backbone` — wraps real SAM3 ViT-32L + FPN neck; SDPA + FP16; `freeze=True` wraps encoder in `no_grad` |
| `sam3_vanilla.py` | V1: frozen encoder, lightweight Conv decoder (66K params), slice-by-slice 3D |
| `sam3_topolora.py` | V2: LoRA on FFN layers (training gate 16 GB), `cbdice_cldice` loss |
| `sam3_hybrid.py` | V3: frozen ViT-32L + gated DynUNet-3D fusion |
| `sam3_decoder.py` | `Sam3MaskDecoder` — native SAM3 head or lightweight Conv decoder (HF path) |
| `sam3_feature_cache.py` | Feature caching for slice-by-slice 3D inference (reduces VRAM) |
| `sam3_vram_check.py` | Pre-flight VRAM enforcement (`mode="inference"` ≥6 GB, `mode="training"` ≥16 GB) |
| `model_builder.py` | `build_adapter()` factory — calls `_require_sam3(config, encoder_frozen=bool)` |

## Installation

```bash
# Step 1: Request model access (Meta gated model — usually instant)
# https://huggingface.co/facebook/sam3 → "Agree and access repository"

# Step 2: Authenticate
uv run huggingface-cli login

# Step 3: Install via Transformers (includes SAM3 as of v5.2.0)
uv add "transformers>=5.2.0" torchvision
uv run python -c "from transformers import Sam3Model; print('OK')"
```

**Note:** `torchvision` is required — `modeling_sam3.py` imports it at the top.
It is listed as a runtime dependency in `pyproject.toml`.

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
