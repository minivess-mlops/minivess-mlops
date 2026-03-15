# T4 (Turing) Banned — FP16-Only GPU Produces NaN in SAM3 Models

**Date:** 2026-03-15
**Severity:** Critical — wasted 3 GCP spot launches + debugging time
**Category:** Hardware selection failure

## What Happened

Launched sam3_hybrid smoke test on GCP T4 spot (me-west1-b, $0.16/hr). Training
completed but **val_loss=NaN** — every validation epoch produced non-finite values.

## Root Cause

T4 is a **Turing architecture** GPU. Turing does NOT support BF16 (bfloat16).
SAM3's ViT-32L encoder has 648M parameters loaded in half precision. On T4, this
means FP16 (float16), which has **max representable value = 65504**.

During validation with sliding_window_inference (ROI 512×512×3), encoder internal
operations (LayerNorm, softmax, GELU) overflow FP16 range → Inf → NaN propagation.

**BF16 has the same exponent range as FP32** (max ~3.4e38), so these overflows
simply don't happen on Ampere+ GPUs (L4, A100, RTX 3090/4090).

## The Misleading Signal

T4 spot at $0.16/hr LOOKS cheapest. But:
- T4 produces NaN for SAM3 models → **100% wasted spend**
- L4 at $0.19/hr is 1.86x faster AND produces correct results
- Per-job cost on L4 is actually 37% lower than T4

**Comparing hourly rates without considering compute capability and correctness
is comparing apples to bananas.** A GPU that produces NaN has infinite cost per
useful result.

## What Should Have Happened

1. Check GPU architecture before selecting: Turing = no BF16 = banned for SAM3
2. Default to L4 (Ampere, BF16 support) — cheapest correct GPU
3. Only fall back to T4 for models that don't use half-precision encoders (DynUNet)

## Fix Applied

1. `sam3_backbone.py`: Auto-detect BF16 via `torch.cuda.is_bf16_supported()`
   - Ampere+ (L4, A100, RTX 3090+) → `torch.bfloat16`
   - Turing (T4, RTX 2070) → `torch.float16` (with NaN guard as safety net)
2. `smoke_test_gcp.yaml`: Removed T4 from accelerators — `{L4: 1, A100: 1}`
3. `CLAUDE.md`: Added T4 ban to Cloud GPU Strategy + "What AI Must NEVER Do"
4. Tests: AST verification that no hardcoded `torch.float16` in from_pretrained

## GPU Architecture Reference

| GPU | Architecture | BF16 Support | SAM3 Compatible |
|-----|-------------|-------------|-----------------|
| T4 | Turing (2018) | NO | NO — FP16 overflow → NaN |
| RTX 2070 Super | Turing (2019) | NO | Training only (NaN guard), no cloud validation |
| L4 | Ada Lovelace (2023) | YES | YES — default GCP choice |
| RTX 3090 | Ampere (2020) | YES | YES |
| RTX 4090 | Ada Lovelace (2022) | YES | YES |
| A100 | Ampere (2020) | YES | YES |
| H100 | Hopper (2023) | YES | YES |

## Rule

**NEVER use T4 or any Turing GPU for SAM3 models in SkyPilot YAMLs.**
Default GCP accelerator: `{L4: 1, A100: 1}`. T4 only as explicit opt-in for
non-SAM3 models (DynUNet, SegResNet).
