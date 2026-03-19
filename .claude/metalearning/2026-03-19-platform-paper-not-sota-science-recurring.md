# Metalearning: Platform Paper, NOT SOTA Science — Recurring Misunderstanding

**Date:** 2026-03-19
**Severity:** P0 — fundamental misunderstanding of project purpose
**Trigger:** Claude framed external test evaluation as "KEY FINDING" when it's
just standard practice. User corrected: "we are a MLOps paper, not publishing
SOTA science."

---

## What Happened

Claude wrote in the test plan XML:
> "The generalization gap (trainval vs test) is a KEY FINDING of the paper."

User correction (verbatim):
> "is not strictly key finding, but an obligatory normal metric that is possibly
> reported in typical experiments, and as we are a MLOps paper, we don't strictly
> speaking care that we only have this tiny external test set! We only care about
> the mechanisms how people can use multiple external test sets for subgroup analysis
> and get that aggregate, and automatically quantify the differences between the
> train and test set!"

## Root Cause

### RC1: Same Pattern as 2026-03-15 SOTA Framing Violation
`metalearning/2026-03-15-sota-framing-violation-platform-not-race.md` documented
the EXACT same failure: "ARBOR surpasses MambaVesselNet" → wrong frame.
The platform is the hero, models are supporting cast.

### RC2: Claude defaults to "scientific results" framing
Claude's training data is dominated by ML papers that ARE about SOTA results.
When Claude sees "external test set" + "paper," it automatically frames the
evaluation as a key scientific finding rather than a platform demonstration.

### RC3: The distinction is subtle but critical
- **SOTA paper**: "Our model achieves X% generalization gap → breakthrough!"
- **Platform paper**: "Our platform automates test/deepvess/{metric} logging,
  subgroup analysis, aggregate computation, and train/test comparison. Any
  researcher can add their own test datasets via YAML config."

## What This Repo IS About

**MinIVess is a MODEL-AGNOSTIC MLOps PLATFORM.**

The paper demonstrates:
1. **Mechanism**: How the platform handles external test sets automatically
2. **Extensibility**: Adding test/newdataset/{metric} requires zero code changes
3. **Automation**: train/test differences computed and visualized automatically
4. **Reproducibility**: The same pipeline runs on any lab's data

The paper does NOT claim:
- SOTA segmentation results on DeepVess
- That the generalization gap is a novel scientific finding
- That 7 test volumes constitute a rigorous external validation study

## The Correct Framing

**External test evaluation in this paper serves to demonstrate that:**
1. The platform CAN evaluate on arbitrary external datasets
2. The `test/{dataset}/{metric}` prefix convention enables subgroup analysis
3. The aggregate metric computation works automatically
4. The biostatistics module computes train/test splits separately
5. Researchers can replicate this with THEIR test datasets

**NOT to demonstrate that:**
- DynUNet generalizes well to DeepVess
- The generalization gap is small/large
- The external test results are the "true" metric

## Rule

**NEVER frame scientific results as the main contribution of this platform paper.**

The contribution is the PLATFORM CAPABILITY, not the numbers. When writing about
external test evaluation:
- WRONG: "The generalization gap is a KEY FINDING"
- RIGHT: "The platform automates external test evaluation with extensible
  test/{dataset}/{metric} prefix convention and split-aware biostatistics"

## Related

- `2026-03-15-sota-framing-violation-platform-not-race.md` — same pattern
- `2026-03-14-poor-repo-vision-understanding.md` — same root cause (not understanding WHY)
- CLAUDE.md TOP-1: "Flexible MLOps as MONAI Ecosystem Extension"
