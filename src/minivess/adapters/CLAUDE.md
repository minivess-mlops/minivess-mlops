# SAM3 Adapters — AI Context

## VRAM Requirements (Per-Variant)

VRAM requirements depend on **whether the SAM3 encoder is frozen**.
When frozen, `torch.no_grad()` is applied → no activation graph built for 848M params.

| Variant | Encoder Frozen | Task | Min VRAM | Notes |
|---------|---------------|------|----------|-------|
| V1 Vanilla | ✅ Yes | Inference-level gate | **≥6 GB** | Only lightweight decoder trains; ~5-7 GB peak FP16 |
| V2 TopoLoRA | ❌ No | Training gate | **≥16 GB** | LoRA on all 32 ViT-32L blocks, gradients flow |
| V3 Hybrid | ✅ Yes | Inference-level gate | **≥6 GB** | Sequential execution: SAM frozen (5-7 GB peak) then DynUNet (~2.8 GB) |

Gate is enforced by `check_sam3_vram(variant, mode)` in `sam3_vram_check.py`.
- `mode="inference"` → 6 GB gate → V1, V3
- `mode="training"` → 16 GB gate → V2

### What "Frozen Encoder" Means for VRAM

```python
# In Sam3Backbone.extract_features() when self._frozen=True:
with torch.no_grad():
    out = self.encoder(x)  # no activation graph stored for 848M params
```

The 848M-param ViT-32L encoder pass uses ~4-6 GB BF16 for activation values, but
since `torch.no_grad()` prevents gradient tracking, no backward graph is allocated.
This means V1 Vanilla with frozen encoder has similar VRAM to pure inference.

### Practical on RTX 2070 Super (8 GB)

| Task | Feasible? |
|------|-----------|
| V1 Vanilla frozen encoder training | ✅ Likely (5-7 GB expected) — **VERIFY EMPIRICALLY** |
| V3 Hybrid (SAM frozen + DynUNet-3D) | ✅ Likely (5-7 GB peak) — sequential SAM→DynUNet, not simultaneous |
| V2 TopoLoRA (LoRA on unfrozen encoder) | ❌ No — requires A100-40GB |

Sources for VRAM numbers: GH Issues #235, #200, #307 at facebookresearch/sam3;
debuggercafe.com SAM3 benchmarks. **Do not state VRAM numbers without citing a source.**

### Dtype Selection (AMP handles this automatically)

PyTorch AMP (`torch.cuda.amp.autocast()`) automatically picks the optimal dtype per GPU.
**No hardcoded dtypes in adapter code.** Dtype is config-driven or auto-detected.

| GPU Architecture | Compute Cap | AMP autocast dtype | Note |
|-----------------|-------------|-------------------|------|
| Turing (RTX 2070 Super) | 7.5 | FP16 (auto) | BF16 exists but 6.7x slower (software emulation) |
| Ampere (RTX 3090, A100) | 8.0+ | BF16 or FP16 (auto) | Both native |

Only relevant if someone explicitly forces a dtype in config — don't force BF16 on Turing.

### Multi-Environment Compute (CPU / Consumer GPU / Cloud)

| Variant | CPU (64 GB RAM) | RTX 2070 Super (8 GB) | A100-40GB |
|---------|----------------|----------------------|-----------|
| V1 Vanilla (cached) | 75 h (3.2 days) | **2 h** | <1 h |
| V1 Vanilla (no cache) | 157 days | 2.6 days | ~6 h |
| V2 TopoLoRA | 470 days | OOM | **38 h** |
| V3 Hybrid (cached) | 76 h (3.2 days) | **13-25 h** | ~4 h |

**Feature caching** is the critical optimization for V1/V3: extract frozen SAM3
features once to disk (8 GB/fold), then train decoder on cached tensors.
This makes CPU training feasible for a long-weekend job.

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
| `sam3_backbone.py` | `Sam3Backbone` — wraps real SAM3 ViT-32L + FPN neck; `freeze=True` wraps encoder in `no_grad` |
| `sam3_vanilla.py` | V1: frozen encoder (inference gate 6 GB), trainable mask decoder |
| `sam3_topolora.py` | V2: LoRA on FFN layers (training gate 16 GB), `cbdice_cldice` loss |
| `sam3_hybrid.py` | V3: frozen ViT-32L + gated DynUNet-3D fusion (inference gate 6 GB) |
| `sam3_decoder.py` | `Sam3MaskDecoder` — wraps SAM3 mask prediction head |
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
