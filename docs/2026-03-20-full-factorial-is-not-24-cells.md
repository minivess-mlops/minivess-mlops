# Metalearning: Full Factorial Is NOT 24 Cells — Critical Persistent Misunderstanding

**Date**: 2026-03-20
**Severity**: P0-CRITICAL — 6th occurrence of the same misunderstanding
**Root Cause**: Claude Code repeatedly confuses the TRAINING GRID (4×3×2=24)
with the FULL FACTORIAL DESIGN which includes post-training, recalibration,
and ensemble factors.

## The Actual Full Factorial (from pre-gcp-master-plan.xml line 16)

```
4 models × 3 losses × 2 aux_calib × 3 post-training × 2 recalibration × 5 ensemble
= 4 × 3 × 2 × 3 × 2 × 5 = 720 conditions
```

**This is explicitly documented in** `docs/planning/pre-gcp-master-plan.xml` line 16.
It has been there since 2026-03-17. Claude Code has read this file multiple times
but NEVER internalized this line.

## What Claude Keeps Saying (WRONG)

"The full factorial is 4×3×2 = 24 conditions × 3 folds = 72 runs"

This is the TRAINING GRID ONLY. The full factorial multiplies by:
- 3 post-training methods (none, SWA, multi-SWA)
- 2 recalibration methods (none, temperature scaling)
- 5 ensemble strategies (none, per_loss_single_best, all_loss_single_best,
  per_loss_all_best, all_loss_all_best)

## Why This Keeps Happening

1. `paper_factorial.yaml` defines only the training factors (4×3×2)
2. Claude reads this file and concludes "this IS the full factorial"
3. Post-training, recalibration, and ensemble factors are defined in
   DIFFERENT config files, not in the same YAML
4. Claude never traces the cross-file factor composition
5. The synthesis document (intermedia-plan-synthesis-pre-debug-run.md)
   was written by Claude and perpetuates this error

## The Fix

1. The factorial YAML must define ALL factors across ALL layers — not just
   training factors. This is why the user wants "All in one YAML."
2. The synthesis document must be corrected to show the REAL full factorial
3. Named configs: smoke_test.yaml (1×1×1), debug.yaml (4×3×2×2×2×4),
   paper_full.yaml (4×3×2×3×2×5)

## Prevention Rule

**Before writing "the full factorial is X×Y×Z":**
1. Read `docs/planning/pre-gcp-master-plan.xml` line 16
2. Count ALL factors across ALL layers (training + post-training + ensemble)
3. If your count is < 100 conditions, you're probably wrong

## Occurrences

1. 2026-03-17: Initial factorial plan said 4×3 = 12 (missing aux_calib)
2. 2026-03-19: QA round 1 — asked "2-way or 3-way ANOVA?" (should have been N-way)
3. 2026-03-20: Synthesis doc says "24 conditions" as full factorial
4. 2026-03-20: XML plans describe "24 training conditions" as complete design
5. 2026-03-20: AskUserQuestion presents 4×3×2 as "full" design option
6. 2026-03-20: User explodes — "How many times I have to override this"

## Cost Clarification (User Q&A 2026-03-20)

**User**: "You hopefully also realize that not all 720 conditions need to be run
on train Flow (i.e. on GCP with Skypilot), as 4 models x 3 losses x 2 aux_calib
are only done there, so our financial cost is not as ridiculous as one would guess
from that."

**Correct cost breakdown:**
- **GCP (GPU, expensive)**: ONLY Layer A training: 24 conditions × 3 folds = 72 runs (~$65)
- **Local CPU (free)**: Layer B post-training: 72 × 3 methods × 2 recalib = 432 variants
- **Local CPU (free)**: Layer C ensemble: 432 × 5 strategies = 2160 evaluations
- **Local CPU (free)**: Layer D biostatistics: analytical choices applied to all above

The 720-condition factorial is the TOTAL across ALL layers, but the FINANCIAL cost
is dominated by the 72 GPU training runs only. Post-training, analysis, and ensemble
evaluations run on the researcher's local CPU/laptop at zero cloud cost.

## Low-Confidence Resolution Q&A (Verbatim, 2026-03-20)

**Q**: "For the recalibration factor (#5): the KG has 4 calibration methods at 0.25
posterior each (complete tie). Which methods should be factorial levels for the publication?"
**A**: "{none, temp_scaling} only" — 2 levels, keeps factorial at 720/384.

**Q**: "Multi-SWA produces N independent sub-models. Should each sub-model be a SEPARATE
MLflow run, or ONE run with an ensemble artifact?"
**A**: "N separate MLflow runs" — each sub-model individually trackable in biostatistics.

**Q**: "Model soups: should we sweep interpolation alpha coefficients or just use optimal?"
**A**: "Optimal-only (both debug+full)" — model soup is binary factor (with/without).

**Q**: "The tag mismatch (B3): fix NOW or rely on fallback?"
**A**: "Fix NOW (5-min change)" — turned out code was already correct; added guard test.

## Factorial Design Alignment Q&A (Verbatim, 2026-03-20)

**Q**: "The full factorial from pre-gcp-master-plan.xml is: 4 models × 3 losses × 2 aux_calib
× 3 post-training × 2 recalibration × 5 ensemble = 720 conditions. For the DEBUG run (named
config), should we reduce some of these factors or keep ALL 6 dimensions but with fewer levels
per factor?"
**A**: "All 6 dimensions, fewer levels" — 4×3×2×2×2×4 = 384 conditions × 1 fold

**Q**: "Should post-training methods, recalibration, and ensemble strategies be defined under
separate sections in the same YAML, or flat alongside training factors?"
**A**: "Sectioned by layer (Recommended)" — factors.training, factors.post_training, factors.analysis

**Q**: "For the biostatistics ANOVA: single 6-way ANOVA or layered analyses?"
**A**: "Layered ANOVAs (Recommended)" — core 3-way → extended 5-way → ensemble analysis separately

**Q**: "Should the synthesis doc and XML plans be updated NOW?"
**A**: "Update docs NOW"

**Q**: "Are you sure that the concept of a factorial design is non-ambiguous?"
**A** (paraphrased): "We are in no way having THE factorial design, but two example cases:
1) minimum, 2) and full with the full being on our publication gate. The config could be
composable for the n different factorial designs that the users want to define."

## Cross-References

- `docs/planning/pre-gcp-master-plan.xml` line 16 — THE source of truth
- `docs/planning/factorial-design-demo-experiment-plan.md` — Q4 budget $150+
- `.claude/metalearning/2026-03-20-factorial-design-context-amnesia.md`
- `.claude/metalearning/2026-03-20-increasing-context-amnesia-confusing-docs.md`
