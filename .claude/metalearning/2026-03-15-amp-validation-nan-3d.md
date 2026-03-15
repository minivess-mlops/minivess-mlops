# AMP Validation NaN — 3D Operations + Autocast (Preventive, Not Confirmed)

**Date:** 2026-03-15
**Severity:** Medium — preventive mitigation based on community evidence
**Category:** Mixed precision + 3D operations incompatibility
**Status:** PREVENTIVE FIX — not directly observed in our runs (see correction below)

## Correction (2026-03-15)

The original version of this doc claimed AMP was a **confirmed** root cause based
on GCP T4 and L4 runs showing NaN. This was incorrect — all GCP runs used
`smoke_sam3_hybrid.yaml` which has `val_interval: 3 > max_epochs: 2`, triggering the
validation sentinel skip. **Validation never actually ran on any GCP run.** The NaN
was the sentinel placeholder, not an AMP-induced numerical failure.

The AMP + 3D NaN risk is real (documented by MONAI maintainers) but was not observed
in our experiments. The mitigation (`mixed_precision_val=False`) is preventive.

See: [wrong-config-chasing-phantoms.md](2026-03-15-wrong-config-chasing-phantoms.md)

## Community Evidence (Real, Not Our Observation)

MONAI maintainers acknowledge: "AMP does not support very well with 3D operations"
([Project-MONAI/MONAI#4243](https://github.com/Project-MONAI/MONAI/discussions/4243)).

The theoretical failure path:
1. validate_epoch() wraps forward pass in `autocast(enabled=True)`
2. sliding_window_inference creates overlapping 3D windows
3. 3D convolutions in DynUNet run in reduced precision via autocast
4. Overlap accumulation can produce intermediate values that overflow
5. Loss receives NaN logits → val_loss=NaN

## Our Evidence (Corrected)

| Run | GPU | Config | Validation ran? | val_loss | Interpretation |
|-----|-----|--------|----------------|----------|----------------|
| GCP T4 | T4 | smoke_sam3_hybrid | **NO** (sentinel) | NaN | NOT AMP-related |
| GCP L4 | L4 | smoke_sam3_hybrid | **NO** (sentinel) | NaN | NOT AMP-related |
| RunPod 4090 | RTX 4090 | smoke_sam3_hybrid_cloud | **YES** | 0.725 | Finite (AMP OFF) |

**Run 2 is the only run with actual validation.** It used AMP OFF, so it does not
test whether AMP causes NaN. A future run with AMP ON + correct config (Run 9) is
needed to test the AMP hypothesis.

## Preventive Fix (Applied)

```python
# trainer.py validate_epoch()
autocast(device_type=..., enabled=self.config.mixed_precision_val)  # False by default
```

`TrainingConfig.mixed_precision_val = False` — validation runs in FP32 by default.
This is the standard MONAI recommendation for 3D operations.

## Lesson

Don't present community-reported failure modes as confirmed root causes without
direct experimental evidence. The MONAI AMP+3D risk is real and the mitigation is
valuable, but it was never the cause of the NaN we observed.
