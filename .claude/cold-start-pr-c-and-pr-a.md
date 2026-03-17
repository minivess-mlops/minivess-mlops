# Cold-Start: PR-C + PR-A Sequential Execution (TDD)

**Created**: 2026-03-17
**Estimated context**: ~80-90% of 1M window for both PRs
**Checkpoint gate**: If PR-C exceeds 8 inner iterations, STOP and save PR-A for next session.

## Paste this to resume:

```
Execute two XML plans sequentially using the self-learning-iterative-coder skill.

## SESSION PLAN

### Phase 1: PR-C (Post-Training Factorial)
- Branch: feat/post-training-factorial (ALREADY EXISTS, 1 commit ahead of main)
- Plan: docs/planning/pr-c-post-training-factorial-plan.xml
- Tasks: T1-T7 (7 tasks, issues #801-#807)
- CRITICAL: Fetch github.com/cai4cai/Average-Calibration-Losses code FIRST (T1)

### Phase 2: PR-A (Biostatistics Gaps) — ONLY if PR-C completes in <8 iterations
- Branch: feat/biostatistics-factorial-gaps (create from main AFTER merging PR-C)
- Plan: docs/planning/pr-a-biostatistics-gaps-plan.xml
- Tasks: T1-T9 (9 tasks, issues #808-#816)

### Between PRs:
1. Create PR for PR-C, merge via squash
2. Pull main
3. Create new branch feat/biostatistics-factorial-gaps
4. Start PR-A execution

## WHAT'S ALREADY DONE (do NOT redo):
- Branch feat/post-training-factorial exists with pre-existing test fixes
- 4 pre-existing test failures already fixed (issue #820 on branch)
- Known pre-existing failure: Issue #800 (KG level values) — IGNORE
- All task issues created (#801-#816)
- License changed to Apache-2.0

## KEY CONSTRAINTS:
- TDD: RED (tests first) → GREEN (implement) → VERIFY (make test-staging) → FIX → CHECKPOINT → CONVERGE
- Port mL1-ACE from cai4cai repo with CC BY 4.0 attribution
- OpenLineage: Use Marquez + PostgreSQL in docker-compose (NOT local JSON only!)
- Docker IS the execution model. NEVER propose "simpler" local alternatives.
- uv ONLY (--all-extras). No pip/conda. No import re.
- from __future__ import annotations at top of every Python file
- encoding='utf-8' for file ops, pathlib.Path() for paths
- Close GitHub issues (#801-#816) as tasks complete

## ISSUE TRACKING:
When completing each task:
1. git commit with "Closes #NNN" in message
2. gh issue close NNN --reason completed

PR-C tasks → issues:
  T1 (#801): hL1-ACE loss function
  T2 (#802): Compound loss integration
  T3 (#803): Factorial post-training config
  T4 (#804): Factorial execution loop
  T5 (#805): Hydra config for aux_calib
  T6 (#806): Checkpoint size tracking
  T7 (#807): OpenLineage emission

PR-A tasks → issues:
  T1 (#808): Two-way ANOVA
  T2 (#809): Interaction plots
  T3 (#810): ANOVA LaTeX table
  T4 (#811): Calibration metrics
  T5 (#812): Riley instability
  T6 (#813): Cost appendix table
  T7 (#814): TRIPOD compliance test
  T8 (#815): Integration test
  T9 (#816): OpenLineage emission

## VERIFICATION COMMANDS:
make test-staging          # Staging tier (<3 min, no model loading)
uv run ruff check src/ tests/   # Lint
uv run mypy src/           # Type check

## START:
1. git checkout feat/post-training-factorial
2. Read docs/planning/pr-c-post-training-factorial-plan.xml
3. Fetch https://github.com/cai4cai/Average-Calibration-Losses (read their loss code)
4. Begin T1 RED phase: write test_hl1ace_loss.py
```
