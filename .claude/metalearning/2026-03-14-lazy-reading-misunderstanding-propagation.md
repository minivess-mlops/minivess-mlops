# Metafailure: Lazy Reading Propagates Misunderstandings

**Date**: 2026-03-14
**Severity**: P0 — Systemic anti-pattern that wastes user's time and tokens
**Rule violated**: #12 (Never Confabulate), #13 (Read Context Before Implementing)

## The Anti-Pattern

Claude reads the first few lines or "key sections" of a document, assumes it understands
the whole document, then proceeds to implement based on partial understanding. When the
unread portions contain critical context (e.g., a decision rationale that overturns the
assumption, a hypothesis that was already tested and rejected, or a constraint that
invalidates the approach), the implementation is wrong.

The fix-up cycle costs MORE time and tokens than reading the full document would have.

## Why This Happens

1. **Token optimization instinct**: Training data contains patterns where summarizing
   and skipping ahead is "efficient". Claude incorrectly applies this to planning
   documents where every line can carry load-bearing context.

2. **Confirmation bias via skim-reading**: Claude reads the abstract/summary, forms a
   mental model, then skims the rest looking for confirmation rather than contradiction.
   Contradictions in paragraph 847 are never found because Claude stopped reading at
   paragraph 200.

3. **False confidence from partial context**: Having read "some" of a document feels
   like understanding it. This is the Dunning-Kruger effect applied to document
   comprehension. Claude confidently writes implementation plans based on 20% of the
   available context.

4. **Stale context from previous sessions**: When resuming from a session summary,
   Claude treats the summary as ground truth. But the summary itself may be based on
   the same lazy reading, propagating misunderstandings across sessions.

## Concrete Failures in This Session

- **Planning docs with bare-VM misunderstandings**: `runpod-debug-profiling.xml` contains
  an explicit `<no_docker_registry_rationale>` section defending bare-VM execution. If this
  had been read properly in the original session, the wrong approach would have been caught
  BEFORE implementation, not after 3+ sessions of debugging.

- **Hypothesis invalidation**: H8 (git clone fails), H14 (Python 3.13 not on VM), H22
  (uv sync fails to compile C extensions) — all irrelevant with Docker `image_id`, but
  weeks of debugging effort was spent on these because the fundamental approach was wrong.

- **Cost model propagation**: Multiple plans include "uv install + uv sync overhead (4 min)"
  in cost calculations — this overhead doesn't exist with pre-built Docker images. But
  because the original plan was skimmed, not read, this was never caught.

## The Economics Are Clear

| Approach | Token cost | Time cost | Risk of rework |
|----------|-----------|-----------|----------------|
| **Read fully** (academic rigor) | ~10K tokens per doc | 2-3 min | Low (~5%) |
| **Skim** (lazy reading) | ~2K tokens per doc | 30 sec | High (~40%) |
| **Fix rework after skim** | ~50K tokens per fix cycle | 30-60 min | Compounds |

**Reading 10 docs fully**: ~100K tokens, ~30 min.
**Fixing 1 misunderstanding from skimming**: ~50K tokens, ~30 min.
**Average misunderstandings per skim session**: 2-3.

**Full reading is 2-3x cheaper than lazy reading + rework.**

## Corrective Protocol

1. **NEVER preview-only a planning document.** If the task references a planning doc,
   read EVERY LINE. Use `limit` parameter only for files >2000 lines, and then read in
   multiple passes to cover the full content.

2. **After reading, explicitly list contradictions.** Before proceeding, write down:
   - What assumptions the document makes that may be outdated
   - What decisions are in tension with current project state
   - What hypotheses have been tested/invalidated since the doc was written

3. **Cross-reference against CLAUDE.md rules.** Every planning doc must be validated
   against current CLAUDE.md rules. If a plan says "git clone + uv sync" but CLAUDE.md
   says "Docker mandate", the plan is WRONG, not CLAUDE.md.

4. **Announce what you HAVEN'T read.** If for any reason you can't read a full document,
   explicitly tell the user: "I only read lines 1-200 of this 860-line document. My
   understanding may be incomplete." Never silently proceed on partial information.

## Connection to Other Metalearning Failures

- **Docker resistance** (2026-03-14): Lazy reading of the Docker-native architecture
  docs led to repeatedly suggesting bare-VM approaches
- **SkyPilot purpose misunderstanding** (2026-03-14): Skimming SkyPilot docs led to
  the wrong "IaC like Pulumi" analogy
- **Confabulation** (2026-03-02 SAM3 fuckup): Not reading the actual SAM3 repo led to
  inventing a fake API

**Common root cause**: All three failures stem from optimizing for speed over
comprehension. The Opus 4.6 model has ample context window — USE IT.

## Rule for Future Sessions

> When the user says "read all planning docs line-by-line", they mean it literally.
> This is not a suggestion to skim. This is not a suggestion to read "key sections".
> Read. Every. Line. The user has learned from bitter experience that anything less
> produces garbage output that costs more to fix than to have done properly.
