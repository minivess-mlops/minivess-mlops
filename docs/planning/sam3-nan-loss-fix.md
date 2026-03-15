# SAM3 val_loss=NaN Root Cause Analysis & Fix Plan

**Issue**: [#715](https://github.com/petteriTeikari/minivess-mlops/issues/715)
**Date**: 2026-03-15
**Status**: Investigation
**Models affected**: sam3_hybrid (confirmed), sam3_vanilla (untested)

## Symptom

sam3_hybrid training on RunPod RTX 4090 produces valid training metrics
(`loss=0.795`, `dice=0.377`) but **val_loss=NaN** on every validation epoch.
This persists even with `val_interval=1` (every epoch), disproving hypothesis H1
(sentinel skip).

```
# RTX 4090 run (T5.1, 2026-03-15)
Training: loss=0.795, dice=0.377  ← valid
Validation: val_loss=NaN          ← broken
```

## Upstream Context: SAM3 IABCEMdetr Loss Bug

SAM3's own training code has a **known NaN bug** in the `IABCEMdetr` loss function.
While MinIVess does NOT use `IABCEMdetr` (we use `dice_ce`/`cbdice_cldice`), the
upstream findings inform our numerical stability analysis.

### Root Cause (SAM3 upstream)

**Source**: [facebookresearch/sam3#289](https://github.com/facebookresearch/sam3/issues/289#issuecomment-3717103683)
(yhy258, 2026-01-07), [facebookresearch/sam3#440](https://github.com/facebookresearch/sam3/issues/440)

The `IABCEMdetr` loss function has `presence_gamma=0.0` as the Python `__init__` default:

```python
class IABCEMdetr(LossWithWeights):
    def __init__(self, ..., presence_gamma=0.0, ...):  # <-- BUG: should be 2.0
```

When `gamma=0`, the Triton sigmoid focal loss backward pass computes:

```python
# _inner_focal_loss_bwd in sigmoid_focal_loss.py
tmp = pow(1 - p_t, gamma - 1)     # gamma=0 → pow(1-p_t, -1) = 1/(1-p_t)
# When p_t → 1.0 (correct prediction): tmp → ∞
d_mod_factor = -gamma * d_pt * tmp  # = 0 * finite * ∞ = NaN (IEEE 754)
```

**IEEE 754**: `0 × ∞ = NaN`. This is a mathematical certainty, not a statistical fluke.

**Fix**: Set `presence_gamma: 2.0` in loss config. The official SAM3 example YAMLs
set `gamma: 2` for the main classification loss but **fail to set `presence_gamma`**,
leaving it at the dangerous default of 0.0.

### Why This Matters for MinIVess

We don't use `IABCEMdetr`, but the same class of numerical instability — **FP16
overflow/NaN in the frozen SAM3 encoder during validation** — may be our root cause.
The SAM3 ViT-32L encoder runs in FP16 (for VRAM efficiency), and validation uses
sliding window inference with different input distributions than training patches.

## Multi-Hypothesis Decision Matrix

| ID | Hypothesis | Likelihood | Evidence | Test | Fix Complexity |
|----|-----------|------------|----------|------|----------------|
| H1 | val_interval sentinel skip | **REJECTED** | NaN persists with val_interval=1 on RTX 4090 | Tested T5.1 | N/A |
| H2 | FP16 overflow in encoder during validation | **HIGH** | Encoder runs `.half()`, val uses larger ROI (512,512,3) vs train patch (32,32,3) | Add `torch.isnan` check on encoder output | Low |
| H3 | Sliding window edge patches produce degenerate inputs | MEDIUM | `sliding_window_inference` can produce small edge patches with near-zero/constant values | Log patch stats before encoder | Low |
| H4 | AMP autocast + FP16 encoder interaction | MEDIUM | Training uses `autocast()`, encoder is already FP16; double-casting? | Test with `autocast(enabled=False)` during val | Low |
| H5 | Decoder NaN from FP16→FP32 cast of extreme values | MEDIUM | FP16 max = 65504; encoder features could exceed this for some patches | Clamp encoder output before FP32 cast | Low |
| H6 | Loss function on empty/degenerate predictions | LOW | `dice_ce` has `smooth_nr`/`smooth_dr` terms but edge cases exist | Log prediction stats before loss | Low |
| H7 | MONAI `sliding_window_inference` overlap blending NaN | LOW | Overlap regions averaged; NaN in any window poisons the entire output | Test with `overlap=0.0` | Low |

### Hypothesis H2 Deep Dive (Most Likely)

The SAM3 backbone runs in FP16 for VRAM efficiency:

```python
# sam3_backbone.py
if self._frozen:
    x = x.half()  # Cast to FP16
    with torch.no_grad():
        out = self.encoder(x)
# Caller casts back to FP32
```

**Training path**: Small patches `(B=1, C=1, 32, 32, 3)` → FP16 encoder → FP32 decoder.
Works fine because patches are small and well-conditioned.

**Validation path**: Larger ROI `(512, 512, 3)` via `sliding_window_inference` → FP16
encoder receives overlapping windows → some windows may have extreme intensity values →
FP16 overflow (max 65504) → NaN features → NaN loss.

**Key difference**: Validation uses `sliding_window_inference` which:
1. Extracts overlapping windows from the full volume
2. Each window passes through the encoder independently
3. Results are stitched back together with overlap averaging
4. If ANY window produces NaN, the entire output is poisoned

### Hypothesis H3 Deep Dive

`sliding_window_inference` at the boundaries can produce patches that are:
- Partially zero-padded (if `mode="constant"`)
- Very small (if the volume doesn't divide evenly)
- Constant-valued (if the boundary region has no signal)

The SAM3 ViT-32L encoder may not handle these degenerate inputs gracefully in FP16.

## Proposed Fix Strategy

### Phase 1: Diagnostic (T5.4-D1) — Isolate the NaN Source

Add targeted NaN detection in the validation forward pass:

```python
# In Sam3Backbone.forward() — temporary diagnostic
features = self.encoder(x.half())
if torch.isnan(features).any():
    nan_count = torch.isnan(features).sum().item()
    total = features.numel()
    logger.warning(
        "NaN in SAM3 encoder output: %d/%d elements (%.1f%%)",
        nan_count, total, 100 * nan_count / total,
    )
```

### Phase 2: Fix (T5.4-F1) — FP16 Output Guard

If H2 is confirmed, add a NaN-safe FP16→FP32 conversion:

```python
# sam3_backbone.py — production fix
features = self.encoder(x.half())
features = features.float()  # FP16 → FP32
# Replace NaN with zero (safe for downstream; zero features = no activation)
if torch.isnan(features).any():
    features = torch.nan_to_num(features, nan=0.0)
    logger.warning("NaN in SAM3 encoder output — replaced with zeros")
```

### Phase 3: Structural Fix (T5.4-F2) — FP16 Clamp Before Overflow

Prevent NaN at the source by clamping FP16 values to safe range:

```python
# sam3_backbone.py — structural fix
x_fp16 = x.half()
# FP16 safe range: [-65504, 65504]. Clamp to 99% to prevent overflow in LayerNorm/GELU.
x_fp16 = x_fp16.clamp(-60000, 60000)
features = self.encoder(x_fp16)
```

### Phase 4: Validation Pipeline Hardening

```python
# trainer.py validation loop — add NaN guard
val_loss = loss_fn(val_pred, val_target)
if not torch.isfinite(val_loss):
    logger.warning(
        "Non-finite val_loss detected (epoch %d). "
        "Logging NaN but continuing training.",
        epoch,
    )
    # Log NaN explicitly — don't skip the metric
    # This preserves the NaN signal in MLflow for debugging
```

## Decision: Which Fix First?

**Recommended order**:

1. **Phase 1 (diagnostic)** — 30 min. Run sam3_hybrid with NaN logging on RunPod.
   Confirms whether NaN originates in encoder (H2) or downstream (H6/H7).

2. **Phase 2 (nan_to_num guard)** — 15 min. If H2 confirmed, add `nan_to_num` guard.
   This is safe and non-invasive. Zero features produce zero decoder output, which
   produces valid (if uninformative) loss values.

3. **Phase 4 (val pipeline hardening)** — 30 min. Even if Phase 2 fixes the immediate
   NaN, the validation loop should handle non-finite losses gracefully (log NaN, don't
   crash). Training should never be killed by a validation-only NaN.

4. **Phase 3 (FP16 clamp)** — only if Phase 2 shows systematic overflow, not sporadic
   edge patches. Clamping is more aggressive and could affect model quality.

## References

- [yhy258 (2026). "IABCEMdetr presence_gamma=0 causes NaN." *GitHub facebookresearch/sam3#289*.](https://github.com/facebookresearch/sam3/issues/289#issuecomment-3717103683)
- [WuJunxu (2026). "Loss is nan." *GitHub facebookresearch/sam3#440*.](https://github.com/facebookresearch/sam3/issues/440)
- [Ravi et al. (2025). "SAM 3: Segment Anything in Images, Videos, and 3D." *arXiv:2511.16719*.](https://arxiv.org/abs/2511.16719)
- [IEEE 754-2019. "Standard for Floating-Point Arithmetic." *IEEE*.](https://standards.ieee.org/ieee/754/6210/)
- MinIVess Issue [#715](https://github.com/petteriTeikari/minivess-mlops/issues/715) — sam3_hybrid val_loss=NaN tracking
- MinIVess Issue [#710](https://github.com/petteriTeikari/minivess-mlops/issues/710) — Original sam3_hybrid investigation

## SAM3 Config Reference (from upstream YAML)

The screenshot from the user shows SAM3's IABCEMdetr config:

```yaml
# SAM3 default loss config (NOT used by MinIVess)
- _target_: sam3.train.loss.loss_fns.IABCEMdetr
  weak_loss: False
  weight_dict:
    loss_ce: 20.0
    presence_loss: 20.0
  pos_weight: 10.0
  alpha: 0.25
  gamma: 2            # main focal gamma (OK)
  use_presence: True
  pos_focal: false
  # presence_gamma: NOT SET → defaults to 0.0 → NaN!
  # FIX: add presence_gamma: 2.0
```

**Note**: MinIVess uses its own loss registry (`dice_ce`, `cbdice_cldice`) — not
`IABCEMdetr`. The upstream NaN bug is informative for understanding FP16/numerical
stability but is not the direct cause of our val_loss=NaN.
