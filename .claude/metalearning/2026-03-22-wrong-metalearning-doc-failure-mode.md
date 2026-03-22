# Metalearning: Writing WRONG Metalearning Docs — Meta-Failure Mode

**Date**: 2026-03-22
**Severity**: P0-CRITICAL — the correction mechanism itself produced a wrong correction
**Parent**: `2026-03-20-full-factorial-is-not-24-cells.md`

## What Happened

Claude Code wrote a metalearning doc titled "Post-Training Is NOT a GPU Factor" that
was WRONG. Post-training IS a GPU factor — it runs in the SAME SkyPilot job as training
via the parent flow merger (sub-flow 1: training, sub-flow 2: post-training). The
metalearning doc claimed post-training was CPU-only, which contradicts the ENTIRE
PURPOSE of the flow merger that was just implemented.

The wrong metalearning doc was deleted within the same session after user correction.

## Why This Is Catastrophic

Metalearning docs are designed to PREVENT recurring mistakes. A WRONG metalearning doc
causes FUTURE sessions to make the SAME mistake, but now with FALSE CONFIDENCE because
"the metalearning doc says so." This is worse than having no metalearning at all:
- No doc → next session might get it right or wrong (50/50)
- Wrong doc → next session will get it wrong with high confidence (>90%)

## The Three-Position Oscillation

In a SINGLE session, Claude Code held THREE contradictory positions:

1. **First implementation (WRONG)**: Added post_training_method as a separate SkyPilot
   job loop dimension → doubles GPU cost from 72 to 144 jobs
2. **"Correction" metalearning (ALSO WRONG)**: Wrote doc saying post-training is NOT
   a GPU factor, runs on local CPU only → removes post-training from GPU entirely
3. **Actual truth**: Post-training runs IN THE SAME SkyPilot job as training.
   The parent flow iterates over methods internally. 72 GPU jobs total.
   "none" comes for free from training sub-flow. SWAG runs in post-training sub-flow.

## Root Cause

Claude Code made the metalearning error because:
1. **Panicked overcorrection**: User said "wrong with factorial runs," Claude overcorrected
   by claiming post-training is CPU-only (the opposite extreme)
2. **Did not re-read the flow merger plan**: The plan at
   `docs/planning/training-and-post-training-into-two-subflows-under-one-flow.md`
   explicitly shows post-training in the same SkyPilot job
3. **Cherry-picked from the old metalearning**: The 2026-03-20 doc says "Local CPU:
   Layer B post-training" which Claude took as gospel without checking if it was
   still accurate after the flow merger was designed
4. **Did not ASK before writing**: Should have asked the user to clarify before
   committing a metalearning doc that contradicts the flow merger architecture

## Prevention Rules

1. **NEVER write metalearning docs in a panic**. When the user corrects you, STOP.
   Read ALL relevant docs. ASK the user. THEN write the metalearning.
2. **Cross-reference before committing**: A metalearning doc that contradicts an
   existing plan document (e.g., the flow merger plan) is automatically suspect.
   Check for contradictions with at least 3 related docs before writing.
3. **Metalearning docs about architecture decisions MUST cite the source plan**.
   If no plan exists, that's a red flag — the decision may not have been made yet.
4. **The user is always right about their own design**. If the user says "post-training
   IS a GPU plan," don't write a doc saying it isn't.

## Correct Understanding (Ground Truth)

- Post-training methods: **only "none" and "swag"** (decided in previous session)
- Execution model: same SkyPilot job, parent flow iterates internally
- "none" is free from training sub-flow result
- SWAG runs in post-training sub-flow (same GPU, same DataLoaders)
- GPU job count: 24 cells x 3 folds = 72 jobs (UNCHANGED by post-training)
- Debug = production with ONLY: 1 fold, 2 epochs, half data

## Occurrence

8th factorial misunderstanding. 1st meta-failure (wrong metalearning doc).
