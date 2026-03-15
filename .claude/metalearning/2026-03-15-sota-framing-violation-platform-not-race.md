# FAILURE: Wrote "ARBOR will surpass MambaVesselNet" — SOTA Race Framing in a Platform Paper

**Date**: 2026-03-15
**Severity**: CRITICAL — Violated the single most fundamental constraint of the entire paper
**Location**: Commit message `2a16d04`, cover letter §3.2, cover letter §16.2
**Source of truth violated**: `docs/planning/repo-to-manuscript-prompt.md`, verbatim user prompt:

> "we do not have results in traditional sense as we are not looking to show what is SOTA in
> vasculature segmentation, but rather to show a system that facilitates the development of
> next-gen vasculature SOTA segmentations."

---

## What Happened

When adding the MambaVesselNet citation (Chen et al. 2024) to the cover letter and bibliography,
the commit message contained:

> "added MambaVesselNet as the literature baseline ARBOR will surpass"

The word "surpass" frames MambaVesselNet as a competitor ARBOR is trying to beat on metrics.
This is the exact OPPOSITE of what the paper claims.

Additionally, cover letter §16.2 contained:

> "Hypothesis (for R3b): Mamba will show better topology preservation (higher clDice)
> at comparable DSC to DynUNet"

This predicts which model wins — also SOTA race framing.

---

## Root Cause: Confusing Two Different Roles for Prior Art

The failure stems from confusing two fundamentally different relationships to prior work:

| Role | Correct for ARBOR? | Example phrasing |
|------|-------------------|-----------------|
| **Baseline to beat** (SOTA race) | ❌ NEVER | "ARBOR surpasses / outperforms / beats X" |
| **Literature context showing the domain exists** | ✅ YES | "Chen et al. (2024) demonstrate Mamba is viable for cerebrovascular — ARBOR provides the reproducible platform to benchmark it" |
| **Motivation for platform design choice** | ✅ YES | "MambaVesselNet exists but has no MLOps pipeline — ARBOR fills this gap" |

MambaVesselNet is cited because:
1. It shows Mamba architecture is **scientifically motivated** for cerebrovasculature (motivates including Mamba family in the platform comparison)
2. It has **no reproducible pipeline** — our platform fills the infrastructure gap
3. It shows the **domain exists** without solving the MLOps problem

MambaVesselNet is NOT cited as a competition target. DSC numbers from Chen et al. are not something ARBOR tries to beat.

---

## The Core Paper Framing (read this before every writing task)

**ARBOR is a PLATFORM paper, not a model paper.**

The model comparison (DynUNet vs Mamba vs SAM3 vs VesselFM) exists to prove ONE THING:
**the ModelAdapter ABC works across architecturally diverse model families.**

The correct mental model:
```
WRONG: ARBOR vs MambaVesselNet → who gets better clDice?
RIGHT: ARBOR includes Mamba family → platform works across CNN + SSM + Foundation Models
```

The models are INTERCHANGEABLE DEMONSTRATIONS of platform capability.
They are not candidates in a performance competition.

**The platform is the hero. The models are supporting cast.**

---

## How to Correctly Frame Prior Art in Writing Tasks

When citing MambaVesselNet or any segmentation paper in Introduction/Discussion:

✅ CORRECT:
> "Chen et al. (2024) demonstrate that Mamba state-space models achieve competitive
> performance on cerebrovascular segmentation, motivating their inclusion as one of
> four model families in ARBOR's generalizability demonstration."

✅ CORRECT:
> "ARBOR provides the missing reproducible infrastructure to benchmark Mamba-based
> segmentation alongside CNN and foundation model architectures."

✅ CORRECT:
> "The platform's model-agnostic design is demonstrated across four architecturally
> diverse families: DynUNet (CNN), Mamba (SSM), SAM3 (foundation model), and
> VesselFM (domain-specific foundation model)."

❌ NEVER:
> "ARBOR surpasses / outperforms / beats / improves upon MambaVesselNet"

❌ NEVER:
> "Mamba will show better topology preservation than DynUNet" (performance prediction)

❌ NEVER:
> "Our cbdice_cldice result (+8.9% clDice) sets a new state of the art for..."

---

## Correct Framing for the +8.9% clDice Result

The DynUNet loss ablation result IS a quantitative result, but frame it as:

✅ "The platform enables rigorous multi-fold topology-aware loss ablation; using
cbdice_cldice yields +8.9% clDice at −5.3% DSC versus dice+CE, informing
the platform's default loss selection."

NOT as: "We achieve SOTA clDice of 0.906 on MiniVess."

The result validates the PLATFORM CAPABILITY (running a 4-loss × 3-fold × 100-epoch
ablation and producing bootstrap CIs automatically), not a SOTA claim.

---

## Affected Files to Audit Before Next Commit

The following locations contain potentially wrong framing that must be reviewed:

1. `docs/planning/cover-letter-to-sci-llm-writer-for-knowledge-graph.md` §16.2 Hook A
   - Remove: "Hypothesis: Mamba will show better topology preservation than DynUNet"
   - Replace with: "The platform enables systematic comparison of SSM vs CNN architectures"

2. Any future manuscript .tex sections referencing model comparison results
   - Always frame as platform demonstration, not model competition

3. Commit messages touching model comparison work
   - BANNED phrase: "surpass", "outperform", "beat", "better than" applied to our results vs prior work
   - ALLOWED: "demonstrates", "enables comparison of", "includes", "supports"

---

## Rule to Apply Going Forward

Before writing ANY sentence that involves a model performance number or comparison:

**Ask**: "Am I claiming ARBOR produces better numbers than prior work, OR am I claiming
the platform successfully runs the experiment?"

If the answer is "better numbers than prior work" → REWRITE. That is not our claim.
If the answer is "platform successfully runs the experiment" → CORRECT framing.

The paper's reproducibility proof is:
- 73/73 artifact checks PASS (pipeline works)
- 35+ artifacts in 8.26 seconds (platform is fast)
- 4 model families integrated (platform is model-agnostic)

NOT:
- "We get higher clDice than Chen et al."
- "ARBOR achieves state-of-the-art DSC on MiniVess"
