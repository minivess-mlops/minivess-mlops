# Metalearning: TopoLoRA-SAM Paper Implementation Fidelity (2026-03-21)

## What Happened

Issue #879 was created as "P2: Implement LoRAConv2d for Conv2d layers in SAM3 encoder."
This was WRONG. The TopoLoRA-SAM paper (Khazem et al., 2025) does NOT apply LoRA to
Conv2d layers. LoRA targets FFN layers ONLY (mlp.lin1, mlp.lin2).

Claude invented "LoRAConv2d" — a technique that doesn't exist in the paper — instead
of reading the actual paper and reference implementation (SegLab repo).

## Root Cause: Implementing Own Stuff Instead of Following the Literature

When Glitch #9 crashed on Conv2d LoRA during the GCP debug run, Claude:
1. Fixed the crash by skipping Conv2d layers (correct)
2. Created issue #879 proposing "LoRAConv2d" (WRONG)
3. Never read the paper to verify whether Conv2d LoRA was intended
4. Never checked the SegLab reference implementation

This violates:
- **Rule #10**: Verify models beyond knowledge cutoff — ALWAYS web-search
- **Rule #12**: Never confabulate — "LoRAConv2d" was fabricated, not from the paper
- **Rule #3**: Library-first — the SegLab repo IS the reference implementation

## The REAL Gap

The paper has a **Spatial Adapter** (Conv_DW 3×3 + Conv 1×1 + BN + GELU + residual)
between the encoder and decoder. This ~66K trainable component refines spatial features
after the frozen encoder. Our implementation is MISSING this entirely.

Additionally, the paper's exact architecture has:
- LoRA ONLY on FFN layers (our implementation is CORRECT on this)
- Frozen encoder (CORRECT)
- Trainable decoder (CORRECT)
- Loss: BCE + Dice + 0.5*clDice (we use cbdice_cldice — similar but not identical)

## The Anti-Pattern: "Inventing" vs "Implementing"

This repo is a **PLATFORM** (Nature Protocols). We implement published methods
EXACTLY as described. We do NOT invent new techniques. When a model crashes,
the fix is to match the paper, not to invent a new adaptation method.

**Correct response to Glitch #9:**
1. Read the paper: https://arxiv.org/html/2601.02273v1
2. Read the code: https://github.com/salimkhazem/Seglab
3. Discover: paper doesn't LoRA Conv2d
4. Discover: paper HAS Spatial Adapter that we're missing
5. Fix: implement Spatial Adapter, keep Conv2d frozen

**What Claude did instead:**
1. Skip Conv2d LoRA (correct half)
2. Propose "LoRAConv2d" as a NEW technique (wrong half)
3. Never read the paper or reference code

## Rule for Future Sessions

**Before implementing ANY model architecture or technique:**
1. FIND the original paper — get the arxiv URL
2. FIND the reference implementation — get the GitHub repo
3. READ BOTH before writing any code
4. COMPARE our implementation against the reference line-by-line
5. Document ALL deviations with explicit justification
6. NEVER invent new techniques — implement what's published
7. If the paper doesn't exist yet → ASK the user before implementing

The phrase "based on" a paper is BANNED. Either we implement the paper
EXACTLY (with documented adaptations) or we don't implement it at all.

## References

- Paper: https://arxiv.org/html/2601.02273v1
- Code: https://github.com/salimkhazem/Seglab
- Issue #879 (relabeled from LoRAConv2d to Spatial Adapter)
