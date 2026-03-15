# AMP Validation NaN — 3D Operations + Autocast = NaN

**Date:** 2026-03-15
**Severity:** Critical — wasted 3 GCP runs + 1 full BF16 refactor before finding root cause
**Category:** Mixed precision + 3D operations incompatibility

## What Happened

sam3_hybrid val_loss=NaN on BOTH T4 (FP16) and L4 (BF16). Training loss was
always finite. BF16 auto-detect was implemented as a fix but it wasn't sufficient.

## Root Cause

**AMP autocast + MONAI sliding_window_inference + 3D convolutions produces NaN**
during validation, regardless of encoder dtype (FP16 or BF16).

MONAI maintainers acknowledge: "AMP does not support very well with 3D operations"
(Project-MONAI/MONAI#4243).

The specific failure path:
1. validate_epoch() wraps forward pass in `autocast(enabled=True)`
2. sliding_window_inference creates overlapping 3D windows
3. 3D convolutions in DynUNet run in reduced precision via autocast
4. Overlap accumulation/blending produces intermediate values that overflow
5. Loss receives NaN logits → val_loss=NaN

Training works because:
- Training uses small patches (64×64×3) — no sliding window
- Forward + backward + gradient scaling keeps values in range
- Smaller spatial extent = less accumulation risk

## Evidence

| Run | GPU | Encoder dtype | AMP | val_loss | Result |
|-----|-----|---------------|-----|----------|--------|
| GCP T4 | T4 (Turing) | FP16 | ON | NaN | BF16 hypothesis formed |
| GCP L4 | L4 (Ada) | BF16 | ON | NaN | BF16 NOT sufficient |
| RunPod 4090 | RTX 4090 | FP16 | OFF | 0.725 | AMP OFF = finite |

## Fix

TrainingConfig.mixed_precision_val = False (default). Validation runs in FP32.
Training keeps AMP ON for speed. This is the standard MONAI recommendation.

```python
# trainer.py validate_epoch()
autocast(device_type=..., enabled=self.config.mixed_precision_val)  # False

# trainer.py train_epoch()
autocast(device_type=..., enabled=self.config.mixed_precision)       # True
```

## Lesson

Don't assume dtype is the root cause when NaN appears. Check the full compute path:
- Encoder dtype (FP16 vs BF16) — necessary but not sufficient
- AMP autocast scope — the actual root cause here
- Spatial operations (sliding window, overlap) — amplify numerical errors
- 3D vs 2D — 3D has more intermediate values and accumulation
