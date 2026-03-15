# SAM3 Hybrid val_loss=NaN: Final Incident Report

**Issue**: [#715](https://github.com/petteriTeikari/minivess-mlops/issues/715)
**Date**: 2026-03-15
**Severity**: P1-high (blocked cloud model validation for ~10 hours)
**Status**: RESOLVED — Root cause confirmed and verified. val_loss=0.6786 (finite) on GCP L4.

> **Important context**: Issue #715 was initially filed as "real validation bug, not
> sentinel issue" based on early RunPod evidence (H1 rejected). The GCP investigation
> in this session revealed that the GCP-specific NaN WAS a sentinel — a different
> config path (`smoke_sam3_hybrid.yaml` vs `smoke_sam3_hybrid_cloud.yaml`) triggered
> the same H1 mechanism on GCP that had been fixed for RunPod. The original RunPod
> NaN (Run 1, `debug: true` forcing `max_epochs=1`) and the GCP NaN (Runs 3-7,
> `val_interval: 3 > max_epochs: 2`) are two instances of the same sentinel class
> but with different trigger paths.

## 1. Executive Summary

sam3_hybrid training on cloud GPUs reported `val_loss=NaN` in MLflow across 7 runs
on RunPod RTX 4090, GCP T4, and GCP L4. Eight hypotheses were investigated. The root
cause for ALL observed NaN was **the validation sentinel** — validation was never
actually executed in any run that showed NaN.

- **Run 1** (RunPod): `debug: true` forced `max_epochs=1`, sentinel computed `val_interval=6 > 1`
- **Runs 3-7** (GCP): SkyPilot YAML selected `smoke_sam3_hybrid.yaml` which has `val_interval: 3 > max_epochs: 2`
- **Run 2** (RunPod, correct config): `smoke_sam3_hybrid_cloud.yaml` with `val_interval=1`, AMP OFF → **val_loss=0.7264 (FINITE)**

Two additional latent risks were identified and mitigated preventively:
- **FP16→BF16 dtype**: SAM3 encoder loaded in FP16 despite official BF16 examples ([Ref 8](#references))
- **AMP+3D NaN**: MONAI documents that AMP can produce NaN with 3D operations ([Ref 3](#references))

**Neither H2b (AMP) nor H8 (BF16) was tested with actual validation enabled on GCP.**
Their mitigations (BF16 auto-detect, AMP-val-off) are preventive, not confirmed fixes.
Run 8 (pending) will test the correct config on GCP L4.

## 2. Root Cause: Config Sentinel — Two Trigger Paths

### 2.1 Mechanism

The training flow uses a val_interval sentinel to skip validation on memory-constrained
GPUs. When `val_interval > max_epochs`, validation is skipped entirely. A placeholder
is logged instead:

```python
# trainer.py — the sentinel logic
_skip_all_val = val_interval > self.config.max_epochs
# ...
if run_val:
    val_result = self.validate_epoch(...)
else:
    # Reuse prior val result, or NaN sentinel if no prior validation exists
    val_result = _last_val_result or EpochResult(loss=float("nan"))
```

### 2.2 Two Trigger Paths

| Trigger | Config | val_interval | max_epochs | Sentinel? |
|---------|--------|-------------|-----------|-----------|
| **Run 1** (RunPod) | smoke_sam3_hybrid + `debug: true` | 6 (code heuristic: max_epochs+1) | 1 (debug forced) | **YES** (6 > 1) |
| **Runs 3-7** (GCP) | smoke_sam3_hybrid.yaml | **3** (hardcoded in YAML) | **2** (YAML) | **YES** (3 > 2) |
| **Run 2** (RunPod) | smoke_sam3_hybrid_cloud.yaml | **1** | **5** | **NO** → finite val_loss |

The GCP smoke test selected the config via `EXPERIMENT="smoke_${MODEL_FAMILY}"` →
`smoke_sam3_hybrid` (local config, validation skipped), not `smoke_sam3_hybrid_cloud`
(cloud config, validation enabled).

### 2.3 Confirmation

Run 2 on RunPod RTX 4090 (2026-03-15 02:14 UTC) used `smoke_sam3_hybrid_cloud.yaml`
with `val_interval=1` and `mixed_precision=false`. Result over 2 completed epochs:
- ep0: val_loss=0.7264, train_loss=0.7534, val_dice=0.300
- ep1: val_loss=0.7259, train_loss=0.7278, val_dice=0.301
- VRAM: 6150 MiB on RTX 4090 (24 GB), 22 min/epoch

### 2.4 Fix

```bash
# deployment/skypilot/smoke_test_gcp.yaml — prefer cloud config when available
CLOUD_CONFIG="configs/experiment/smoke_${MODEL_FAMILY}_cloud.yaml"
if [ -f "${CLOUD_CONFIG}" ]; then
  export EXPERIMENT="smoke_${MODEL_FAMILY}_cloud"
else
  export EXPERIMENT="smoke_${MODEL_FAMILY}"
fi
```

## 3. Multi-Hypothesis Investigation

Eight hypotheses were evaluated. **Only H1 was directly tested and confirmed.**
H2b and H8 are mitigated preventively based on community evidence, but were never
observed as actual failure modes because validation never ran on the GCP runs.

| ID | Hypothesis | Prior | Outcome | Evidence |
|----|-----------|-------|---------|----------|
| **H1** | val_interval sentinel skip | Medium | **CONFIRMED** | Run 2: val_loss=0.7264 with correct config |
| H2 | FP16 encoder overflow during validation | Medium | **Untested** | No GCP run actually ran validation |
| **H2b** | AMP autocast + 3D DynUNet NaN | High | **Preventively mitigated** | [MONAI#4243](https://github.com/Project-MONAI/MONAI/discussions/4243) community evidence; not directly tested |
| H3 | Sliding window edge patches | Medium | Untested | Would require patch-level logging |
| H4 | AMP + FP16 encoder double-cast | Medium | Superseded by H2b | Same compute path |
| H5 | Decoder NaN from FP16→FP32 extreme values | Medium | Untested | NaN guard handles defensively |
| H6 | Loss function degenerate inputs | Low | Unlikely | `dice_ce` has smooth terms |
| H7 | MONAI overlap blending NaN propagation | Low | Untested | Would require overlap=0.0 test |
| **H8** | FP16 vs BF16 encoder dtype mismatch | Medium-High | **Preventively mitigated** | HF model card uses `bfloat16`; auto-detect implemented |

### 3.1 Upstream: SAM3 IABCEMdetr Loss Bug (Informative)

SAM3's own training code has a known NaN bug in the `IABCEMdetr` loss function
([facebookresearch/sam3#289](https://github.com/facebookresearch/sam3/issues/289#issuecomment-3717103683)):
`presence_gamma=0.0` (default) causes IEEE 754 `0 × ∞ = NaN` in the Triton sigmoid
focal loss backward pass. MinIVess uses `dice_ce`/`cbdice_cldice` (unaffected),
but the numerical instability class informed the hypothesis generation.

### 3.2 Community Evidence: AMP + 3D NaN

MONAI maintainers document AMP incompatibility with 3D operations:

> *"AMP does not support very well with 3D operations. Sometimes turning on AMP
> would return NaN loss values, especially for SegResNet."*
> — [Project-MONAI/MONAI#4243](https://github.com/Project-MONAI/MONAI/discussions/4243)

Additional community reports:
- [MONAI#2637](https://github.com/Project-MONAI/MONAI/discussions/2637): NaN traced to border crop normalization + AMP
- [PyTorch Discuss](https://discuss.pytorch.org/t/loss-is-calculated-as-nan-when-using-amp-autocast/186272): AMP NaN in binary segmentation U-Net
- [PyTorch Discuss](https://discuss.pytorch.org/t/fp16-gives-nan-loss-when-using-pre-trained-model/94133): FP16 overflow when fine-tuning pretrained models

**Note**: These are community-reported failure modes, not observed in our runs.
They motivated preventive mitigations (Section 5).

### 3.3 BF16 vs FP16 Numerical Ranges

| Property | FP16 (float16) | BF16 (bfloat16) | FP32 (float32) |
|----------|---------------|----------------|----------------|
| Exponent bits | 5 | 8 | 8 |
| Mantissa bits | 10 | 7 | 23 |
| Max value | 65504 | ~3.4 × 10³⁸ | ~3.4 × 10³⁸ |
| Overflow risk | HIGH (ViT LayerNorm/softmax) | LOW (FP32 range) | None |

SAM3's HuggingFace model card uses `torch.bfloat16` in all examples. Our code
used `torch.float16` — written for RTX 2070 Super (Turing, no BF16 support).
This was institutional knowledge lost during the transition to cloud ([Ref 8](#references)).

## 4. GPU Architecture and BF16 Compatibility

BF16 support requires NVIDIA Ampere architecture (compute capability ≥ 8.0) or later.

| GPU | Architecture | Compute Cap. | BF16 | SAM3 Validation |
|-----|-------------|-------------|------|-----------------|
| T4 | Turing (2018) | 7.5 | **NO** | **BANNED** — no BF16 fallback |
| RTX 2070 Super | Turing (2019) | 7.5 | **NO** | Dev only (NaN guard) |
| RTX 3090 | Ampere (2020) | 8.6 | YES | OK |
| A100-40GB | Ampere (2020) | 8.0 | YES | OK |
| **L4** | **Ada Lovelace (2023)** | **8.9** | **YES** | **Default GCP choice** |
| RTX 4090 | Ada Lovelace (2022) | 8.9 | YES | OK |
| H100 | Hopper (2023) | 9.0 | YES | OK |

**T4 is banned for SAM3 models** — enforced in `smoke_test_gcp.yaml`
(`accelerators: {L4: 1, A100: 1}`) and `CLAUDE.md`. Note: the ban is a preventive
measure based on the FP16 overflow theoretical risk (H2/H8). It was NOT directly
triggered by observed NaN on T4 (those NaN were from the sentinel, not FP16 overflow).
See: [Ref 9](#references).

### 4.1 FinOps: Cost-Effectiveness per Job

GCP spot pricing varies by region and availability. Observed prices during this
investigation (2026-03-15, SkyPilot catalog):

| GPU | Region | Spot $/hr | FP16 TFLOPS | Per-job cost (10 min) |
|-----|--------|-----------|-------------|----------------------|
| T4 | me-west1-b | $0.16 | 65 | $0.027 |
| L4 | asia-northeast3-a | $0.19 | 121 | $0.017 |
| A100 | me-west1-a | $1.39 | 312 | $0.049 |

L4 is 1.86× faster than T4 and 37% cheaper per job due to higher throughput.
See full analysis in [skypilot-and-finops-complete-report.md](skypilot-and-finops-complete-report.md).

## 5. Preventive Mitigations Implemented

These address latent risks (H2b, H8) that were NOT the observed root cause but
are likely to manifest when validation actually runs with AMP enabled on future runs.

### 5.1 BF16 Auto-Detection (sam3_backbone.py)

```python
self._encoder_dtype = (
    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)
```

### 5.2 Separate AMP Policy (trainer.py)

```python
# Training: AMP ON (speed)
autocast(device_type=..., enabled=self.config.mixed_precision)       # True

# Validation: AMP OFF (correctness)
autocast(device_type=..., enabled=self.config.mixed_precision_val)   # False (default)
```

### 5.3 NaN Guard (sam3_backbone.py, sam3_hybrid.py, trainer.py)

Three-level NaN detection as safety net.

### 5.4 Config Fix (smoke_test_gcp.yaml)

Cloud launches prefer `_cloud.yaml` experiment configs (validation enabled) over
local configs (validation skipped).

## 6. Lessons Learned

1. **Verify the simplest hypothesis first.** H1 (wrong config) was the root cause.
   A 30-second diff between the two YAML files would have found it immediately.

2. **Encode constraints in code, not documentation.** BF16 auto-detection makes the
   hardware constraint self-enforcing.

3. **Sentinel values are technical debt.** Using `float("nan")` as a "validation
   skipped" marker makes it indistinguishable from genuine numerical failures.
   A dedicated sentinel would have prevented hours of debugging.

4. **Log the experiment config name to MLflow.** If `EXPERIMENT=smoke_sam3_hybrid`
   had been visible in the MLflow UI, the mismatch would have been obvious.

5. **Don't assume a fix propagated to all execution paths.** The H1 fix (respect
   config values) was correct for Run 2 but irrelevant when the SkyPilot YAML
   selected a different config file entirely.

## 7. Verification: All Models on GCP L4 (Debug Mode)

| Model | GPU | val_loss | val_dice | train_loss | Status |
|-------|-----|---------|---------|-----------|--------|
| **sam3_vanilla** | T4 (me-west1) | **0.632** | 0.493 | 0.627 | PASS |
| **dynunet** | T4 (me-west1) | **0.749** | 0.313 | 0.783 | PASS |
| **sam3_hybrid** | **L4 (asia-ne3)** | **0.679** | 0.338 | 0.742 | **PASS** (Run 8) |
| **vesselfm** | L4 (asia-ne3) | — | — | — | FAILED (SSH error, not code) |

3/4 models verified with finite validation metrics on GCP spot. VesselFM failure
was a transient SSH/network error (spot preemption or connectivity), not a code
issue — the Z=32 patch fix is in the Docker image but the VM lost connectivity
before training started.

## 8. Open Items

- **VesselFM**: Retry on GCP L4 (transient network error, not code bug)
- **Run 9** (future): Test AMP ON with correct config → validates H2b preventive fix
- **Log experiment config**: Add `EXPERIMENT` as MLflow param in train_flow.py

## References

1. [facebookresearch/sam3#289](https://github.com/facebookresearch/sam3/issues/289#issuecomment-3717103683) — SAM3 IABCEMdetr presence_gamma=0 NaN bug (yhy258, 2026-01-07)
2. [facebookresearch/sam3#440](https://github.com/facebookresearch/sam3/issues/440) — Related SAM3 NaN discussion
3. [Project-MONAI/MONAI#4243](https://github.com/Project-MONAI/MONAI/discussions/4243) — AMP + 3D operations NaN (maintainer confirmation)
4. [Project-MONAI/MONAI#2637](https://github.com/Project-MONAI/MONAI/discussions/2637) — NaN from border crop normalization + AMP
5. [PyTorch Discuss: AMP autocast NaN](https://discuss.pytorch.org/t/loss-is-calculated-as-nan-when-using-amp-autocast/186272) — Binary segmentation U-Net AMP NaN
6. [PyTorch Discuss: FP16 pretrained NaN](https://discuss.pytorch.org/t/fp16-gives-nan-loss-when-using-pre-trained-model/94133) — FP16 overflow with pretrained models
7. [HuggingFace Discuss: mT5 BF16/FP16](https://discuss.huggingface.co/t/training-loss-0-0-validation-loss-nan/27950) — BF16→FP16 NaN pattern
8. [metalearning: sam3-bf16-fp16-fuckup](../../.claude/metalearning/2026-03-15-sam3-bf16-fp16-fuckup.md) — Institutional knowledge loss analysis
9. [metalearning: t4-turing-fp16-nan-ban](../../.claude/metalearning/2026-03-15-t4-turing-fp16-nan-ban.md) — T4 ban (preventive, based on theoretical FP16 risk)
10. [metalearning: amp-validation-nan-3d](../../.claude/metalearning/2026-03-15-amp-validation-nan-3d.md) — AMP + 3D validation NaN (preventive, based on community evidence)
11. [metalearning: wrong-config-chasing-phantoms](../../.claude/metalearning/2026-03-15-wrong-config-chasing-phantoms.md) — Metafailure analysis: why it took so long
12. [FinOps report: T4 vs L4](skypilot-and-finops-complete-report.md) — GPU cost-performance comparison

---

## Appendix A: Experiment Run Log

| Run | Date (UTC) | GPU | Config | val_interval | max_epochs | Validation ran? | val_loss | train_loss |
|-----|-----------|-----|--------|-------------|-----------|----------------|---------|-----------|
| 1 | 2026-03-15 01:59 | RTX 4090 (RunPod) | smoke_sam3_hybrid (debug=true) | 6 | 1 | **NO** (sentinel) | NaN | 0.661 |
| 2 | 2026-03-15 02:14 | RTX 4090 (RunPod) | **smoke_sam3_hybrid_cloud** | **1** | **5** | **YES** | **0.7264** | 0.7534 |
| 3 | ~03:00 | T4 (GCP me-west1) | smoke_sam3_hybrid | 3 | 2 | **NO** (sentinel) | NaN | — |
| 4 | ~03:20 | T4 (GCP me-west1) | smoke_sam3_hybrid | 3 | 2 | **NO** (sentinel) | NaN | 0.783 |
| 5 | ~07:40 | L4 (GCP asia-ne3) | smoke_sam3_hybrid | 3 | 2 | **NO** (sentinel) | NaN | 0.904 |
| 6 | ~09:20 | L4 (GCP asia-ne3) | smoke_sam3_hybrid | 3 | 2 | **NO** (sentinel) | NaN | 0.904 |
| 7 | ~10:00 | L4 (GCP asia-ne3) | smoke_sam3_hybrid | 3 | 2 | **NO** (sentinel) | NaN | — |
| 8 | 2026-03-15 14:50 | **L4 (GCP asia-ne3)** | **smoke_sam3_hybrid_cloud** | **1** | **5** | **YES** | **0.6786** | 0.7424 |

**Key observation**: Run 2 (RunPod) and Run 8 (GCP L4) are the only runs where
validation actually executed. Both produced finite val_loss. All other runs used
configs that trigger the sentinel skip.

## Appendix B: Key Code Paths

### Config selection (smoke_test_gcp.yaml) — THE ROOT CAUSE
```bash
# BEFORE fix: always used local config (validation skipped for sam3_hybrid)
export EXPERIMENT="smoke_${MODEL_FAMILY}"

# AFTER fix: prefers cloud config when available (validation enabled)
CLOUD_CONFIG="configs/experiment/smoke_${MODEL_FAMILY}_cloud.yaml"
if [ -f "${CLOUD_CONFIG}" ]; then
  export EXPERIMENT="smoke_${MODEL_FAMILY}_cloud"
fi
```

### Sentinel NaN origin (trainer.py)
```python
_skip_all_val = val_interval > self.config.max_epochs
# ...
if run_val:
    val_result = self.validate_epoch(...)
else:
    val_result = _last_val_result or EpochResult(loss=float("nan"))
```

### BF16 auto-detection (sam3_backbone.py) — preventive
```python
self._encoder_dtype = (
    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)
```
