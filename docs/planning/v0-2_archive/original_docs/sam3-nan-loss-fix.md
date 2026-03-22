# SAM3 val_loss=NaN Root Cause Analysis & Fix Plan

**Issue**: [#715](https://github.com/petteriTeikari/minivess-mlops/issues/715)
**Date**: 2026-03-15
**Status**: H1 CONFIRMED — sentinel NaN. AMP isolation in progress.
**Models affected**: sam3_hybrid (confirmed), sam3_vanilla (untested)

## Symptom

sam3_hybrid training on RunPod RTX 4090 produces valid training metrics
(`loss=0.661`, `dice=0.471`) but **val_loss=NaN** logged to MLflow.

### Critical Discovery (2026-03-15, Ralph Loop Run 2)

The val_loss=NaN was a **sentinel value, not an actual NaN from validation**:

1. `debug: true` in config → `max_epochs=1` (code forced override, ignoring config's `max_epochs: 5`)
2. `sam3_hybrid + debug` → `val_interval = max_epochs + 1 = 6` (skip validation sentinel)
3. Since `val_interval(6) > max_epochs(1)` → `_skip_all_val = True`
4. `val_result = EpochResult(loss=float("nan"))` — **default placeholder, not real NaN**
5. Placeholder logged to MLflow as `val_loss: nan`

**We never actually ran validation on sam3_hybrid** — not locally (OOM on 8 GB GPU),
not on cloud (debug flag accidentally skipped it). The NaN was a phantom from the
skip sentinel, not from sliding_window_inference or AMP.

**Fix applied**: train_flow.py now reads `val_interval` and `max_epochs` from config
when explicitly set, only falling back to model-based heuristics when not specified.

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
| H1 | val_interval sentinel skip | **CONFIRMED** | `debug: true` forced `max_epochs=1`, `val_interval=6` → `_skip_all_val=True` → sentinel NaN logged. Validation never ran. | Fixed: config overrides code heuristics | Done |
| H2 | FP16 overflow in encoder during validation | MEDIUM | Encoder NaN guard did NOT fire on RTX 4090 run (encoder output was clean). Still possible with different input distributions. | Rerun with actual validation enabled | Low |
| H2b | AMP autocast + 3D DynUNet NaN | **HIGH** | MONAI maintainers confirm "AMP does not support very well with 3D operations" ([MONAI#4243](https://github.com/Project-MONAI/MONAI/discussions/4243)). sam3_hybrid fuses DynUNet-3D output under AMP. | Test with `mixed_precision: false` first | Low |
| H3 | Sliding window edge patches produce degenerate inputs | MEDIUM | `sliding_window_inference` can produce small edge patches with near-zero/constant values | Log patch stats before encoder | Low |
| H4 | AMP autocast + FP16 encoder interaction | MEDIUM | Training uses `autocast()`, encoder is already FP16; double-casting? | Test with `autocast(enabled=False)` during val | Low |
| H5 | Decoder NaN from FP16→FP32 cast of extreme values | MEDIUM | FP16 max = 65504; encoder features could exceed this for some patches | Clamp encoder output before FP32 cast | Low |
| H6 | Loss function on empty/degenerate predictions | LOW | `dice_ce` has `smooth_nr`/`smooth_dr` terms but edge cases exist | Log prediction stats before loss | Low |
| H7 | MONAI `sliding_window_inference` overlap blending NaN | LOW | Overlap regions averaged; NaN in any window poisons the entire output | Test with `overlap=0.0` | Low |
| H8 | FP16 vs BF16 encoder dtype mismatch | **MEDIUM-HIGH** | SAM3 HF model card uses `torch.bfloat16` in all examples. Our encoder uses `torch.float16`. If pretrained with BF16 activations, FP16 range (65504) may overflow. Same pattern as mT5 BF16→FP16 NaN. | Switch `sam3_backbone.py` to `torch.bfloat16` on Ampere+ GPUs | Low |

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

## Proposed Fix Strategy (Updated 2026-03-15)

### Phase 0: Config Fix — DONE

**Problem**: `debug: true` forced `max_epochs=1` and skipped validation entirely.
**Fix**: `train_flow.py` now reads `val_interval` and `max_epochs` from experiment config
when explicitly set, only falling back to model-based heuristics otherwise. Also reads
`mixed_precision` from config.

### Phase 1: Diagnostic — NaN Guard Instrumentation (DONE)

NaN detection added at three levels:
1. `sam3_backbone.py:extract_features()` — NaN guard after encoder output
2. `sam3_hybrid.py:forward()` — NaN guards after SAM features (FP16→FP32), DynUNet logits,
   and fused logits (post-gate)
3. `trainer.py:validate_epoch()` — NaN warning on non-finite avg_loss

### Phase 2: AMP Isolation (Run 3, IN PROGRESS)

**Test**: `mixed_precision: false` in `smoke_sam3_hybrid_cloud.yaml`.
If val_loss is finite → AMP (H2b) is the root cause.
If val_loss is still NaN → issue is elsewhere (fusion, loss, data).

### Phase 3: Systematic Dtype Test Matrix

If Phase 2 confirms AMP as the cause, run systematic dtype tests to pinpoint exactly
which component fails under AMP:

| Test | Encoder dtype | DynUNet dtype | AMP | Expected | Purpose |
|------|---------------|---------------|-----|----------|---------|
| T1 | FP16 (frozen) | FP32 | OFF | Finite | Baseline — no AMP |
| T2 | FP16 (frozen) | FP32 | ON | NaN? | Current config — AMP interaction |
| T3 | FP16 (frozen) | FP16 | OFF | NaN? | Is FP16 DynUNet the issue? |
| T4 | BF16 (frozen) | FP32 | OFF | Finite? | BF16 encoder (no overflow) |
| T5 | BF16 (frozen) | FP32 | ON | Finite? | BF16 encoder + AMP DynUNet |
| T6 | FP16 (frozen) | FP32 | ON (val OFF) | Finite? | AMP for train only |
| T7 | FP32 (frozen) | FP32 | OFF | Finite | Full FP32 (gold standard) |

**Key dtype questions**:
- **FP16 encoder → autocast context**: Does autocast try to cast FP16 inputs to FP16 again
  (no-op) or does it interact with the trainable FP32 modules receiving the output?
- **Instance norm in FP16**: DynUNet uses `norm_name="instance"`. Instance norm with
  batch_size=1 has high variance-to-mean ratio which can overflow in FP16.
- **BF16 vs FP16**: BF16 has same exponent range as FP32 (no overflow at 65504), but lower
  mantissa precision. SAM3 encoder's LayerNorm/softmax may benefit from BF16 range.

### Phase 4: Validation Pipeline Hardening (DONE)

```python
# trainer.py validate_epoch() — NaN guard
if not math.isfinite(avg_loss):
    logger.warning("Non-finite val_loss detected...")
```

### Phase 5: Structural Fixes (conditional on Phase 2-3 results)

If AMP is confirmed as the root cause:

**Option A: Disable AMP for validation only**
```python
# trainer.py — separate AMP policy for train vs val
with autocast(device_type=..., enabled=self.config.mixed_precision and self.training):
```

**Option B: Disable AMP for sam3_hybrid entirely**
```yaml
# smoke_sam3_hybrid_cloud.yaml
mixed_precision: false  # MONAI/PyTorch AMP + 3D known issue
```

**Option C: Switch frozen encoder to BF16** (if GPU supports it)
```python
# sam3_backbone.py — BF16 instead of FP16 for encoder
if torch.cuda.is_bf16_supported():
    x = x.to(torch.bfloat16)
else:
    x = x.half()
```

## Audit Trail: Run Log

| Run | Date | Config Changes | Result | Diagnosis |
|-----|------|---------------|--------|-----------|
| 1 | 2026-03-15 01:59 | Original (debug=true) | val_loss=nan, train_loss=0.661 | Sentinel NaN — validation never ran (H1 CONFIRMED) |
| 2 | 2026-03-15 02:14 | H1 fix: respect config val_interval/max_epochs, AMP OFF | **ALL FINITE**: ep0 val_loss=0.7264 train_loss=0.7534 val_dice=0.300; ep1 val_loss=0.7259 train_loss=0.7278 val_dice=0.301. Loss decreasing. 22 min/epoch, RTX 4090, 6150 MiB. 2/5 epochs done, 3 remaining. | **H1 CONFIRMED** — sentinel NaN. Zero NaN in 2 epochs of training+validation without AMP. |
| 3 | (planned) | AMP ON if Run 2 succeeds | (planned) | Phase 3 — confirm AMP as root cause |
| 4 | (planned) | BF16 encoder if Run 3 fails | (planned) | Phase 3 — dtype isolation |

## Decision: Current Fix Order

1. **Phase 0 (config fix)** — DONE. Config values respected.
2. **Phase 2 (AMP isolation)** — IN PROGRESS. Run with `mixed_precision: false`.
3. **Phase 3 (dtype test matrix)** — if Phase 2 succeeds, systematically re-enable AMP
   with different dtype configurations.
4. **Phase 5 (structural fix)** — permanent fix based on Phase 2-3 findings.

## AMP + 3D Operations: Community Evidence

### MONAI Community

**[Project-MONAI/MONAI#4243](https://github.com/Project-MONAI/MONAI/discussions/4243)** — MONAI
maintainer (nourmagde00 reply): *"AMP does not support very well with 3D operations. Sometimes
turning on AMP would return NaN loss values, especially for SegResNet."* Recommended fix: disable
AMP or switch architecture. This is directly relevant since sam3_hybrid uses DynUNet-3D under AMP.

**[Project-MONAI/MONAI#2637](https://github.com/Project-MONAI/MONAI/discussions/2637)** — NaN
traced to input data containing NaN after normalization of border crops: *"NaNs errors only
triggered when amp was on"* but *"input already contained NaNs"* from *"normalization with a
division by zero"* at volume borders. Fix: `SignalFillEmptyd` transform to cast NaN inputs to 0.
Also: learning rate `1e-3` was too high — `1e-5` recommended. Redundant `Spacingd` + `Resized`
caused double-resampling, increasing errors.

**[Project-MONAI/tutorials#1637](https://github.com/Project-MONAI/tutorials/discussions/1637)** —
Explains sliding window inference rationale but no NaN-specific content.

### PyTorch Community

**[PyTorch Discuss: "Loss is calculated as NaN when using AMP autocast"](https://discuss.pytorch.org/t/loss-is-calculated-as-nan-when-using-amp-autocast/186272)** —
Binary segmentation U-Net with AMP: NaN loss at the first batch, before gradient updates.
Debugging approach: "narrow down where the first invalid value is created by checking the loss,
calculation as well as the intermediates of your forward pass." NaN occurred during forward pass
(inside autocast), not during backpropagation.

**[PyTorch Discuss: "FP16 gives NaN loss when using pre-trained model"](https://discuss.pytorch.org/t/fp16-gives-nan-loss-when-using-pre-trained-model/94133)** —
Critical for our case (frozen pre-trained SAM3 encoder + trainable decoder):
- **Overflow in intermediate activations** when fine-tuning well-trained models (not training from scratch)
- **Deep supervision**: some branches produce `-inf`/NaN while others remain valid → aggregated loss is NaN
- **Batch normalization** with very low standard deviation values can overflow in FP16
- Fix: register **forward hooks** on each module to identify which layer first produces invalid values

**[PyTorch Discuss: "Why my train and validation loss is as NaN"](https://discuss.pytorch.org/t/why-my-train-and-validation-loss-is-as-nan/206578)** —
Custom activation functions passing negative values to `torch.log()` produce NaN. Fix: clip inputs
before log operations. Also: validate that new training data doesn't contain NaN/Inf values.

### Pretrained Model + Fine-Tuning NaN Patterns

**[PyTorch Discuss: "Loss NaN when resuming from a pretrained model"](https://discuss.pytorch.org/t/loss-nan-when-resuming-from-a-pretrained-model/92234)** —
NaN appears intermittently when fine-tuning pretrained models. LogSoftmax backward triggers
the error. Debugging: check `abs().max()` on all parameters and use forward hooks to find
first NaN layer. Issue is recurring and often unresolved — suggests environmental factors.

**[MLX Examples #620: "NaN loss during LoRA fine-tuning"](https://github.com/ml-explore/mlx-examples/issues/620)** —
LoRA fine-tuning of quantized Mistral 7B: loss spikes then goes NaN. Root cause: **FP16
overflow** during training. Fix: quantize with FP32 instead of FP16, or reduce LoRA
rank/alpha. Directly analogous to our frozen FP16 encoder scenario.

**[HuggingFace Discuss: "Training loss 0.0, validation loss NaN"](https://discuss.huggingface.co/t/training-loss-0-0-validation-loss-nan/27950)** —
mT5 fine-tuning: NaN val_loss. Root cause: **T5 was pretrained on TPU with BF16, but user
trained with FP16**. FP16 range is [-65504, 65504] vs BF16's FP32-equivalent range. Fix:
disable FP16 and use BF16 or FP32. **Directly relevant** — SAM3 may have been pretrained
with BF16 (HuggingFace examples use `torch.bfloat16`).

### SAM3 Official Dtype Recommendation (CRITICAL — Lost Institutional Knowledge)

> **FUCKUP ALERT**: BF16 support was one of the original reasons for pursuing cloud GPUs!
> The RTX 2070 Super (Turing) does NOT support BF16. Cloud GPUs (Ampere+: A100, RTX 3090,
> RTX 4090) DO support BF16. Yet `sam3_backbone.py` was written with `torch.float16` for
> the 2070 Super and **never updated when we moved to cloud**. Classic knowledge amnesia.
> See: `.claude/metalearning/2026-03-15-sam3-bf16-fp16-fuckup.md`

**[HuggingFace: facebook/sam3 Model Card](https://huggingface.co/facebook/sam3)** —
SAM3's official HuggingFace examples consistently use **`torch.bfloat16`**, not `torch.float16`:
```python
model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
```
**Our `sam3_backbone.py:197` uses `torch.float16`** — this dtype mismatch with the pretrained
weights could cause overflow (FP16 max = 65504 vs BF16's FP32-equivalent range of 3.4e38).
If SAM3 was pretrained with BF16 activations, running inference in FP16 may truncate values
beyond FP16 range → Inf → NaN. Same pattern as mT5 BF16→FP16 NaN.

**Fix**: Auto-detect GPU capability and use BF16 when available:
```python
# sam3_backbone.py — should be:
if torch.cuda.is_bf16_supported():
    torch_dtype = torch.bfloat16  # Ampere+ (A100, RTX 3090/4090)
else:
    torch_dtype = torch.float16   # Turing fallback (RTX 2070 Super)
```

This is Phase 3 test T4/T5 in the dtype matrix.

No SAM3-specific NaN discussions found on HuggingFace (167 discussions, mostly access requests).

### General NaN Debugging (PyTorch Best Practices)

**[Leyaa.ai: "How to fix NaN loss in PyTorch"](https://leyaa.ai/codefly/learn/pytorch/qna/how-to-fix-nan-loss-pytorch)** —
Standard checklist: data validation (`torch.isnan/isinf`), epsilon before log/division,
gradient clipping (`clip_grad_norm_`), monitor loss early, stop on first NaN.

### Synthesis: Root Cause Likelihood for sam3_hybrid

Given the community evidence, the most likely NaN sources for sam3_hybrid validation (in order):

1. **H1 — Sentinel NaN (CONFIRMED)**: val_interval skip sentinel logged `float("nan")`
   as placeholder. Validation never actually ran. Fixed by reading config values directly.

2. **AMP + 3D DynUNet (H2b, HIGH)**: MONAI maintainers explicitly warn about this.
   sam3_hybrid's DynUNet-3D runs under `autocast()` with instance norm + residual blocks.
   Instance norm with small batch sizes can produce NaN in FP16.

3. **FP16 vs BF16 encoder dtype (NEW, MEDIUM-HIGH)**: SAM3 HuggingFace examples use
   `torch.bfloat16`, but our `sam3_backbone.py` uses `torch.float16`. If SAM3 was
   pretrained with BF16 activations, FP16 inference may overflow (range 65504 vs 3.4e38).
   Same pattern as mT5 BF16→FP16 NaN (HuggingFace discuss).

4. **Frozen encoder + AMP interaction**: Pre-trained FP16 encoder outputs fed into
   trainable FP32 modules under AMP autocast → potential dtype mismatches or double-casts.

5. **Border patches from sliding_window_inference (H3)**: Normalization of near-zero
   border regions → NaN input → poisoned output. Could be exacerbated by AMP.

### Diagnostic Strategy (Updated)

1. **Run 3 (current)**: `mixed_precision: false` — test WITHOUT AMP. If val_loss is finite,
   AMP is the root cause. If still NaN, the issue is elsewhere.
2. **If AMP is the cause**: Add `autocast(enabled=False)` specifically for validation (keep
   AMP for training where it's working). Or disable AMP entirely for sam3_hybrid.
3. **If NOT AMP**: Add forward hooks to sam3_hybrid's DynUNet layers to find the first NaN
   source.

## References

- [yhy258 (2026). "IABCEMdetr presence_gamma=0 causes NaN." *GitHub facebookresearch/sam3#289*.](https://github.com/facebookresearch/sam3/issues/289#issuecomment-3717103683)
- [WuJunxu (2026). "Loss is nan." *GitHub facebookresearch/sam3#440*.](https://github.com/facebookresearch/sam3/issues/440)
- [Ravi et al. (2025). "SAM 3: Segment Anything in Images, Videos, and 3D." *arXiv:2511.16719*.](https://arxiv.org/abs/2511.16719)
- [IEEE 754-2019. "Standard for Floating-Point Arithmetic." *IEEE*.](https://standards.ieee.org/ieee/754/6210/)
- MinIVess Issue [#715](https://github.com/petteriTeikari/minivess-mlops/issues/715) — sam3_hybrid val_loss=NaN tracking
- MinIVess Issue [#710](https://github.com/petteriTeikari/minivess-mlops/issues/710) — Original sam3_hybrid investigation
- [MONAI Discussion #4243. "AMP + 3D operations NaN." *Project-MONAI/MONAI*.](https://github.com/Project-MONAI/MONAI/discussions/4243)
- [MONAI Discussion #2637. "NaN errors with AMP and border crops." *Project-MONAI/MONAI*.](https://github.com/Project-MONAI/MONAI/discussions/2637)
- [MONAI Tutorials Discussion #1637. "sliding_window_inference." *Project-MONAI/tutorials*.](https://github.com/Project-MONAI/tutorials/discussions/1637)
- [PyTorch Discuss. "Loss is calculated as NaN when using AMP autocast."](https://discuss.pytorch.org/t/loss-is-calculated-as-nan-when-using-amp-autocast/186272)
- [PyTorch Discuss. "FP16 gives NaN loss when using pre-trained model."](https://discuss.pytorch.org/t/fp16-gives-nan-loss-when-using-pre-trained-model/94133)
- [PyTorch Discuss. "Why my train and validation loss is as NaN."](https://discuss.pytorch.org/t/why-my-train-and-validation-loss-is-as-nan/206578)
- [PyTorch Discuss. "Loss NaN when resuming from a pretrained model."](https://discuss.pytorch.org/t/loss-nan-when-resuming-from-a-pretrained-model/92234)
- [MLX Examples #620. "NaN loss during LoRA fine-tuning." *GitHub ml-explore/mlx-examples*.](https://github.com/ml-explore/mlx-examples/issues/620)
- [HuggingFace Discuss. "Training loss 0.0, validation loss NaN." (mT5 BF16/FP16 mismatch).](https://discuss.huggingface.co/t/training-loss-0-0-validation-loss-nan/27950)
- [HuggingFace Model Card. "facebook/sam3" — official BF16 dtype recommendation.](https://huggingface.co/facebook/sam3)
- [Leyaa.ai. "How to fix NaN loss in PyTorch."](https://leyaa.ai/codefly/learn/pytorch/qna/how-to-fix-nan-loss-pytorch)

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
