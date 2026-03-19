# Metalearning: VesselFM is Zero-Shot ONLY — Never Fine-Tuned

**Date:** 2026-03-19
**Severity:** P0 — incorrect experiment design persisted in KG
**Trigger:** User corrected: "we are not fucking finetuning vesselfm for anything!"

---

## What Happened

1. `paper_model_comparison.yaml` line 116 had `training_strategy: zero_shot_AND_finetuned`
2. `factorial-design-demo-experiment-plan.md` (Phase 3) described: "Zero-shot + fine-tuned on DeepVess/TubeNet"
3. User never authorized VesselFM fine-tuning — only zero-shot evaluation
4. The incorrect info persisted in the KG across multiple sessions

## Root Cause

### RC1: Confabulation During Planning
When writing the factorial experiment plan (2026-03-17), Claude likely inferred
"zero_shot_AND_finetuned" from VesselFM's paper description (Wittmann et al. 2024
mentions fine-tuning capability). But the user only requested zero-shot evaluation.

### RC2: KG Was Treated as Authoritative Without User Verification
The `zero_shot_AND_finetuned` value was written to the KG during the factorial
planning session. Once in the KG, it was treated as a resolved decision — but
the user never explicitly approved the fine-tuning component.

### RC3: "Can Do" ≠ "Should Do"
VesselFM CAN be fine-tuned. The paper describes this capability. But the user's
experiment design uses it as a ZERO-SHOT BASELINE to show domain gap between
pre-trained vessel FM and the project's specific microvasculature data.

Fine-tuning VesselFM would defeat this purpose — it would no longer be a
zero-shot baseline comparison.

## The Correct VesselFM Design

**VesselFM in the paper:**
- **Training strategy**: ZERO-SHOT ONLY (no fine-tuning)
- **Dataset**: DeepVess + TubeNet (NOT MiniVess — data leakage)
- **Paper section**: R3c (external evaluation)
- **Purpose**: Show domain gap — pretrained FM on general vessels vs. our specific 2PM microvasculature
- **Expected result**: Low DSC, demonstrating why task-specific models (DynUNet, Mamba++, SAM3) are needed

## KG Fix Applied

Changed `paper_model_comparison.yaml`:
```yaml
# BEFORE (wrong):
training_strategy: zero_shot_AND_finetuned

# AFTER (correct):
training_strategy: zero_shot_only
```

Also fixed `paper_factorial.yaml` zero-shot baselines section.

## Behavioral Rule

**VesselFM = ZERO-SHOT ONLY. Never fine-tune. Never propose fine-tuning.**

When a model's paper describes fine-tuning capability, check the USER'S experiment
design, not the model's paper. "Can fine-tune" ≠ "should fine-tune in this project."

## Related

- `factorial-design-demo-experiment-plan.md` Phase 3 — also needs correction
- `paper_model_comparison.yaml` — KG source of truth, now fixed
