# Context Compounding Failure — Prevention Plan

**Date**: 2026-03-22
**Issue**: #906 (P0-CRITICAL)
**Status**: IN PROGRESS

---

## The Problem

Claude Code has failed to maintain consistent understanding of the factorial design
across 8+ sessions. The same wrong conclusions are re-derived every session:

| Wrong Statement | Correct Statement | Occurrences |
|----------------|-------------------|-------------|
| "Full factorial = 24 cells" | Full factorial = 4×3×2×2×2×5 = 480 conditions across layers | 6+ |
| "Post-training is CPU-only" | Post-training runs in same flow as training (execution location = user's choice) | 2 |
| "Layer A (GPU, cloud)" | Factorial configs define WHAT, not WHERE — execution location is NOT hardcoded | 9+ |
| "Debug should skip X" | Debug = production minus epochs/data/folds ONLY | 8+ |
| "Post-training methods: 3-4 options" | Only "none" and "swag" (decided) | 3+ |
| Wrote WRONG metalearning doc | Metalearning doc contradicted flow merger plan | 1 |

**Worst case**: On 2026-03-22, Claude wrote a metalearning doc that was itself wrong.
The correction mechanism produced a wrong correction, which would have poisoned
future sessions.

---

## Ground Truth (Authoritative Decisions)

### Factorial Design (Full)

Source: `docs/planning/pre-gcp-master-plan.xml` line 16

```
Layer A (Training, cloud GPU):
  4 models × 3 losses × 2 aux_calib = 24 cells × 3 folds = 72 GPU jobs

Layer B (Post-training, same GPU job):
  Post-training methods: {none, swag}
  "none" is free from training sub-flow result
  "swag" runs in post-training sub-flow (same GPU, same DataLoaders)

Layer C (Analysis + Ensemble, local CPU):
  Recalibration: {none, temperature_scaling}
  Ensemble: {none, per_loss_single_best, all_loss_single_best, per_loss_all_best, all_loss_all_best}

Layer D (Biostatistics, local CPU):
  Analytical choices (alpha, ROPE, etc.)
```

### Post-Training Methods (Decided)

**Only 2 methods: "none" and "swag"**

- Decided in session 2026-03-21 (3rd pass debug validation)
- "none" = identity (training result is final)
- "swag" = SWAG posterior approximation (Maddox et al. 2019)
- Checkpoint averaging and subsampled ensemble are NOT factorial levels
- DO NOT offer additional methods without user authorization

### SkyPilot Job Structure

```
One SkyPilot job per (model × loss × aux_calib × fold):
┌─────────────────────────────────────────────────┐
│ @flow("training-flow") ← PARENT                │
│ ├─ Sub-flow 1: training (produces "none" cell)  │
│ ├─ Sub-flow 2: SWAG (produces "swag" cell)      │
│ └─ Returns: both cells to MLflow                │
└─────────────────────────────────────────────────┘

Total GPU jobs: 24 cells × 3 folds = 72 (UNCHANGED by post-training)
POST_TRAINING_METHODS env var: comma-separated list → parent iterates internally
```

### Debug = Production (Rule 27, ABSOLUTE)

Three differences ONLY:
1. **1 fold** (not 3)
2. **2 epochs** (not 50)
3. **Half data** (~23 train / ~12 val)

NOTHING else changes. Post-training included. All flows included. All factors included.

---

## Root Causes

### 1. Context Amnesia Across Sessions

Each session starts fresh. Claude re-reads `paper_factorial.yaml` (which only has
training factors) and concludes "this IS the full factorial." The metalearning docs
and MEMORY.md should prevent this, but Claude doesn't reliably cross-reference them.

### 2. Wrong Source of Truth Files

The factorial design is scattered across multiple files:
- `pre-gcp-master-plan.xml` line 16 (full factorial dimensions)
- `paper_factorial.yaml` (training factors only)
- `debug_factorial.yaml` (debug overrides only)
- `default.yaml` (post-training plugin config)
- Multiple metalearning docs (some now wrong)

No SINGLE file defines the complete factorial with all layers.

### 3. Metalearning Docs Are Unreliable

The 2026-03-22 incident proved that metalearning docs can themselves be wrong. A wrong
metalearning doc is WORSE than no doc — it gives false confidence to future sessions.

### 4. AskUserQuestion as Amnesia Crutch

Claude uses AskUserQuestion to re-ask decisions that have ALREADY been made, instead
of reading existing docs. This wastes user time and signals that Claude hasn't
internalized prior decisions.

---

## Prevention Plan

### P1: Single Authoritative Factorial Config (DONE — 2026-03-22)

Consolidated to `configs/factorial/` with layered structure. No execution location in headers.

- `configs/factorial/paper_full.yaml` — 480-condition production factorial
- `configs/factorial/debug.yaml` — same factors, reduced epochs/data/folds
- `configs/factorial/smoke_test.yaml` — 1-condition pipeline validation
- `configs/factorial/smoke_local.yaml` — 8-condition local validation
- Legacy configs (`configs/experiment/`, `configs/hpo/`) marked DEPRECATED.
- `run_factorial.sh` updated to parse layered structure (`factors.training.*`).

### P2: KG Update (Immediate)

Update `knowledge-graph/decisions/L3-technology/hpo_engine.yaml` to:
- Remove "24-cell factorial config" language
- Reference full factorial (training + post-training + analysis layers)
- Document that only "none" and "swag" are factorial post-training levels

### P3: Metalearning Doc Audit (This Session)

Review ALL metalearning docs for contradictions with the ground truth above.
Flag or delete any doc that says:
- "Full factorial is 24 cells"
- "Post-training is CPU-only"
- "Debug should skip X"
- Any doc offering >2 post-training methods

### P4: CLAUDE.md Factorial Guard Rule (This Session)

Add to CLAUDE.md:
```
## Factorial Design (NON-NEGOTIABLE)
Post-training methods: ONLY "none" and "swag". Already decided. DO NOT offer more.
Debug = production minus {epochs, data, folds}. DO NOT ask "should debug include X?"
run_factorial.sh launches Layer A ONLY. Post-training runs in same GPU job.
```

### P5: Pre-Session Checklist

Every session that touches factorial/training/post-training MUST:
1. Read `pre-gcp-master-plan.xml` line 16
2. Read `2026-03-20-full-factorial-is-not-24-cells.md`
3. Read `2026-03-22-wrong-metalearning-doc-failure-mode.md`
4. Verify any new metalearning doc against the flow merger plan
5. NEVER ask "should debug include X?" (Rule 27)

---

## Cross-References

- Issue #906 (this plan's tracking issue)
- `docs/planning/pre-gcp-master-plan.xml` line 16
- `.claude/metalearning/2026-03-20-full-factorial-is-not-24-cells.md`
- `.claude/metalearning/2026-03-22-wrong-metalearning-doc-failure-mode.md`
- `.claude/metalearning/2026-03-22-debug-equals-production-8th-violation.md`
- `.claude/metalearning/2026-03-19-debug-run-is-full-production-no-shortcuts.md`
- `docs/planning/training-and-post-training-into-two-subflows-under-one-flow.md`
