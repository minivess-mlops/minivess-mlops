# Metalearning: Factorial Design Context Amnesia — REPEATED Failure

**Date**: 2026-03-20
**Severity**: P0 — identical failure pattern to 2026-03-19 QA round 1
**Root Cause**: Claude Code asks questions about the factorial design that are
ALREADY ANSWERED in the knowledge graph, despite being explicitly told to read
the KG line-by-line before asking questions.

## What Happened

After reading the entire knowledge graph and creating 3 XML plans, Claude Code
STILL asked the user: "Should ensemble permutations be a NEW FACTOR in the
factorial (4×3×2×4=96)? Or separate?"

This is the EXACT SAME failure pattern from 2026-03-19 QA round 1, where Claude
asked "ANOVA design: 3-way factorial or 2-way + separate AuxCalib?" — a question
that was already answered in the KG.

## The Knowledge Graph CLEARLY States

From `knowledge-graph/decisions/L2-architecture/ensemble_strategy.yaml`:
- Winner: `heterogeneous_multi_model` (posterior=0.80)
- 4 ensemble strategies are DEFINED

From `configs/hpo/paper_factorial.yaml`:
- The factorial design is 4 models × 3 losses × 2 calibrations = 24 cells
- Ensemble strategies are applied BY THE ANALYSIS FLOW on TOP of training runs
- They are NOT a training factor — they are an analysis factor

From the user's original prompt (saved verbatim):
- "different ensembling permutations (that are obviously factors now for
  Biostatistics flow)"

The user literally said "obviously factors" — the answer was IN THE PROMPT.

## Why This Keeps Happening

1. **Shallow KG reading**: Claude reads the files but doesn't INTERNALIZE the
   relationships between them. Reading navigator → domains → decisions is not
   enough if the connections aren't traced.

2. **Default to asking**: When Claude is uncertain, it defaults to asking the
   user instead of reasoning from the KG. The KG exists PRECISELY to answer
   these questions without bothering the user.

3. **Context window management**: After reading 75+ decision nodes and 10+ source
   files, Claude's effective comprehension degrades. The factorial design details
   are lost in the sea of information.

4. **Pattern matching failure**: Claude sees "factorial" and thinks "training
   factors" but doesn't connect that Analysis Flow creates ADDITIONAL factorial
   conditions from ensemble permutations.

## Prevention Rule

**BEFORE asking ANY question about the factorial design:**
1. Re-read `configs/hpo/paper_factorial.yaml` — what are the TRAINING factors?
2. Re-read `knowledge-graph/decisions/L2-architecture/ensemble_strategy.yaml` — what are the ANALYSIS factors?
3. Check: is the answer already in the KG or the user's prompt?
4. If yes → DO NOT ASK. State the answer and proceed.
5. If genuinely ambiguous → ask, but explain EXACTLY what the KG says and why it's ambiguous.

## Cross-References

- `.claude/metalearning/2026-03-19-context-amnesia-bad-questions.md` (if exists)
- `docs/planning/biostatistic-flow-debug-double-check.xml` — appendix QA round 1
- Memory: `feedback_read_kg_before_questions.md`
