# Metalearning: Increasing Context Amnesia from Confusing Documentation

**Date**: 2026-03-20
**Severity**: P0 — blocking productive work, user losing trust
**Root Cause**: Multiple shallow planning docs + no definitive synthesis = Claude
reads many files superficially instead of one authoritative reference deeply.

## What Happened

Over 6 sessions (2026-03-07 through 2026-03-20), the documentation grew organically:
- intermedia-plan-synthesis.md (v1, 2026-03-07)
- intermedia-plan-synthesis-v2.md (v2, 2026-03-08)
- cold-start-prompt-session-continuation-2026-03-20.md
- cold-start-prompt-pre-debug-qa-verification.md
- 4 XML plans (biostatistics, post-training, analysis, local integration)
- 10+ metalearning docs
- 12+ planning docs

Each doc captures a different aspect at a different time. No single doc has the
complete picture. When Claude reads them, it gets a fragmented understanding:
- v1 mentions 9 flows, v2 mentions 12 flows, the factorial XML mentions 5
- Factorial factors are described differently in each document
- Decisions marked "open" in v1 are "resolved" in v2 but Claude doesn't track this

## The Failure Pattern

1. Claude reads 40+ files in agents (breadth-first, not depth-first)
2. Extracts "key points" from each (lossy compression)
3. Synthesizes from compressed summaries (compounding information loss)
4. The synthesis has CONTRADICTIONS that Claude doesn't detect
5. When implementing, Claude asks questions that reveal the contradictions
6. User must re-explain what should have been obvious from the docs
7. User loses trust, session productivity drops

## Why This Keeps Getting Worse

- Each session adds MORE planning docs (more surface area to read)
- No doc is ever DELETED (only appended to)
- Claude's "line-by-line reading" is actually "scan-for-keywords-and-summarize"
- The 1M context window doesn't help because the information is SCATTERED
  across many files, not concentrated in one authoritative reference

## The Fix

1. **Create ONE definitive synthesis** (`intermedia-plan-synthesis-pre-debug-run.md`)
   that supersedes all previous synthesis docs for the factorial experiment scope
2. **Include the COMPLETE factor table** — every factor, every level, every layer
3. **Include the COMPLETE flow pipeline** — exact tag names, experiment names
4. **Mark all prior synthesis docs as `status: reference`** (not `active`)
5. **Future sessions: READ THE SYNTHESIS FIRST** before reading individual docs

## Prevention Rule

**Before starting ANY implementation work in a new session:**
1. Read `intermedia-plan-synthesis-pre-debug-run.md` FIRST (the ONE definitive doc)
2. Read CLAUDE.md rules
3. Read the specific XML plan for the current task
4. Do NOT read 40 files in parallel agents — read 3 authoritative files deeply

**Quality over quantity.** Reading 5 files deeply beats reading 50 files shallowly.

## Cross-References

- `.claude/metalearning/2026-03-20-factorial-design-context-amnesia.md`
- `.claude/metalearning/2026-03-20-hardcoded-significance-level-antipattern.md`
- `docs/planning/intermedia-plan-synthesis-pre-debug-run.md` (the fix)
