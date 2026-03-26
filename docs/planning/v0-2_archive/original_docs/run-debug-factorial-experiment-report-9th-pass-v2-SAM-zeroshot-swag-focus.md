# 9th Pass v2 Debug Factorial — SAM3 + Zero-Shot + SWAG Validation

**Date**: 2026-03-25
**Branch**: `test/run-debug-9th-pass-report`
**Config**: `configs/factorial/debug.yaml`
**Docker image**: `base:latest` rebuilt 2026-03-25T17:45 with autocast fix + HF repo fix
**Prior**: 9th pass v1 — zero SAM3 SUCCEEDED (OOM in pre_training_checks FP32)

## 1. Executive Summary

**Objective**: Validate the two P0 fixes that blocked all SAM3 training across 9 passes:
1. `check_gradient_flow()` now uses `autocast()` (was FP32, OOMed SAM3)
2. SAM3 HF repo `facebook/sam3-hiera-large` → `facebook/sam3` (was 404)

**Critical question**: Will SAM3 TopoLoRA pass pre-training checks and start
the actual training loop for the first time ever?

**Secondary**: Validate SAM3 Hybrid, zero-shot baselines, SWAG artifact upload.

## 2. Status Matrix

| # | Condition | Status | Duration | Notes |
|---|-----------|--------|----------|-------|
| **SAM3 TopoLoRA (8 conditions — KEY TEST)** |
| — | sam3_topolora-cbdice_cldice-calibtrue-f0 | PENDING | — | |
| — | sam3_topolora-cbdice_cldice-calibfalse-f0 | PENDING | — | |
| — | sam3_topolora-dice_ce-calibtrue-f0 | PENDING | — | |
| — | sam3_topolora-dice_ce-calibfalse-f0 | PENDING | — | |
| — | sam3_topolora-dice_ce_cldice-calibtrue-f0 | PENDING | — | |
| — | sam3_topolora-dice_ce_cldice-calibfalse-f0 | PENDING | — | |
| — | sam3_topolora-bce_dice_05cldice-calibtrue-f0 | PENDING | — | |
| — | sam3_topolora-bce_dice_05cldice-calibfalse-f0 | PENDING | — | |
| **SAM3 Hybrid (8 conditions — FIRST EVER)** |
| — | All 8 SAM3 Hybrid conditions | PENDING | — | Never tested before |
| **Zero-shot baselines (2 conditions — FIRST EVER)** |
| — | sam3_vanilla-zeroshot-minivess-f0 | PENDING | — | |
| — | vesselfm-zeroshot-deepvess-f0 | PENDING | — | |
| **DynUNet + MambaVesselNet — SKIP via --resume** |
| — | 16 conditions | SKIPPED | — | Already SUCCEEDED in 8th pass |

## 3. Timeline

| Time (UTC) | Event |
|------------|-------|
| 2026-03-25 T17:43 | Docker image rebuilt with autocast fix + HF repo fix |
| 2026-03-25 T17:45 | Docker image pushed to GAR, freshness gate passes |
| 2026-03-25 T17:46 | 9th pass v2 launched (PID 2935050) with --resume |
| | *Updates will be appended as jobs progress* |

## 4. Validation Checklist

- [ ] SAM3 TopoLoRA passes pre-training checks (autocast prevents FP32 OOM)
- [ ] SAM3 TopoLoRA completes Epoch 1/2 (training loop actually runs)
- [ ] SAM3 TopoLoRA reaches SUCCEEDED (full 2-epoch debug run)
- [ ] SAM3 Hybrid passes pre-training checks
- [ ] SAM3 Hybrid reaches SUCCEEDED
- [ ] sam3_vanilla zero-shot produces non-zero metrics
- [ ] vesselfm zero-shot produces non-zero metrics
- [ ] MLflow checkpoint artifact uploaded (no HTTP 413)
- [ ] SWAG post-training completes for at least 1 condition
- [ ] `sky jobs logs` shows actual training output (NOT just setup time)

## 5. Key Metrics to Capture (for manuscript)

| Metric | Expected | Actual |
|--------|----------|--------|
| SAM3 TopoLoRA VRAM (BS=1, AMP) | ~13 GB | — |
| SAM3 TopoLoRA epoch time | ~14 min | — |
| SAM3 Hybrid VRAM (BS=1, AMP) | ~7.2 GB | — |
| DynUNet epoch time (reference) | ~4.5 min | 4.5 min (8th pass) |
| MambaVesselNet epoch time (reference) | ~4 min | 4 min (8th pass) |

## 6. Observations

### O1: SAM3 TopoLoRA STILL OOMs — now in LoRA forward pass WITH autocast

Job 73 (`sam3_topolora-dice_ce-calibtrue-f0`) OOMed at `sam3_topolora.py:112`
inside `check_gradient_flow()` even WITH autocast enabled. The traceback shows:

```
sam3_topolora.py line 112: result = original_out + lora_out
torch.OutOfMemoryError: Tried to allocate 376 MiB.
GPU 0: 21.96 GiB total, 295 MiB free. 21.67 GiB in use.
```

**The autocast fix IS being used** (logs show `mixed_precision=mixed_precision`
in the call). But SAM3 TopoLoRA at BS=1 in AMP still uses 21.67 GiB of L4's
21.96 GiB — leaving only 295 MiB free. The LoRA adapter's additional forward
pass pushes it over the edge.

**Root cause**: SAM3 TopoLoRA + LoRA adapters is simply TOO LARGE for L4 (24 GB)
even at BS=1 with AMP. The model VRAM estimate of ~13 GB was wrong — actual is
~22 GB (the LoRA adapters add significant overhead for both forward and backward
passes because they duplicate intermediate activations).

**Options**:
1. Skip pre-training checks for SAM3 (skip `check_gradient_flow` — it requires
   backward pass which doubles VRAM vs inference-only training with checkpointing)
2. Use gradient checkpointing for SAM3 (reduces activation memory)
3. Use A100 (80 GB) for SAM3 — requires yaml_contract.yaml change + user auth
4. Reduce patch_size further for SAM3 TopoLoRA

### O2: The "13 GB VRAM estimate" was wrong

The model profile `configs/model_profiles/sam3_topolora.yaml` says:
```yaml
vram:
  training_gb: 13.0
  per_batch_size:
    1: 13.0
```

This estimate was from before LoRA adapters were accounted for. The actual VRAM
with LoRA forward+backward is ~22 GB. The profile needs updating.

### O3: The 8th pass VRAM report was misleading

The 8th pass report said "SAM3 TopoLoRA at BS=2: 21.92 GB → OOM". It then
estimated "BS=1: ~13 GB". This linear extrapolation was wrong because VRAM
doesn't scale linearly with batch size — the LoRA adapter activations and
gradient buffers are a constant overhead independent of batch size.

## 7. Fixes Applied in This Docker Image

| Fix | File | What |
|-----|------|------|
| autocast in pre-training checks | `pre_training_checks.py` | All 4 check functions now use `autocast()` for AMP |
| FP32 upcast in gradient flow | `pre_training_checks.py` | `output.float()**2` prevents FP16 underflow |
| mixed_precision threading | `train_flow.py` | `training_config.mixed_precision` passed to checks |
| SAM3 HF repo name | 3 SkyPilot YAMLs | `facebook/sam3-hiera-large` → `facebook/sam3` |
| SAM3 HF filename | 3 SkyPilot YAMLs | `sam3_hiera_large.pt` → `sam3.pt` |
