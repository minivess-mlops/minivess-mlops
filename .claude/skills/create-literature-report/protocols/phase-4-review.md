# Phase 4: REVIEW — Iterated Council Convergence

**Time budget**: 10 minutes per iteration, max `max_iterations`
**Exit criteria**: verdict >= quality_target OR max_iterations reached

## When to Skip Phase 4

Skip if ALL of these are true:
- `quality_target` is `MAJOR_REVISION` (lowest bar)
- Phase 3 found 0 hallucinated citations
- Report has ≥ target_paper_count verified papers

## Step 4.1: Launch Reviewer Agents

Spawn 2 reviewer agents in parallel (background):

1. **Domain Expert** — uses `prompts/review-domain-expert.md`
   - Checks factual accuracy, coverage, balance, contextualization
   - Output: YAML with per-section scores

2. **Novelty Assessor** — uses `prompts/review-novelty-assessor.md`
   - Scores Markov novelty per section ("So What?" test)
   - Detects anti-patterns (annotated bibliography, citation spam)
   - Output: YAML with per-section scores + anti-pattern count

## Step 4.2: Aggregate Verdicts (L2 Synthesis)

### Aggregation Rule

```
aggregate_verdict = min(domain_verdict, novelty_verdict)
```

If either reviewer says FAIL, aggregate = FAIL (must fix before proceeding).
If both say PASS, aggregate = PASS.
Otherwise, aggregate = WEAK (= MINOR_REVISION equivalent).

### Verdict Mapping

| Aggregate | Maps to |
|-----------|---------|
| PASS | ACCEPT |
| WEAK | MINOR_REVISION |
| FAIL | MAJOR_REVISION |

### Conflict Resolution

If reviewers disagree by >1 level (e.g., PASS vs FAIL on same section):
- The LOWER score wins for safety
- Log the disagreement in state as `review_conflicts`

## Step 4.3: Check Convergence

```
if aggregate_verdict >= quality_target:
    → CONVERGED. Proceed to Phase 5.
elif iteration >= max_iterations:
    → DONE with warning. Proceed to Phase 5 with caveat.
elif iteration > 1 AND aggregate_verdict == previous_verdict
     AND issues are identical:
    → CEILING_REACHED. No progress possible. Proceed with warning.
else:
    → Fix issues and run iteration N+1.
```

## Step 4.4: Apply Fixes (if not converged)

For each issue flagged by reviewers:
1. FACTUAL_ERROR → Fix the claim or remove it
2. COVERAGE_GAP → Add missing paper (requires Phase 3 re-verify for new citation)
3. ANTI_PATTERN → Rewrite the flagged paragraph
4. WEAK_SECTION → Strengthen cross-domain synthesis

After fixes, re-run Phase 3 VERIFY on any NEW citations only.

## Step 4.5: Update State

```json
{
  "phase": "REVIEW",
  "substep": "iteration_1_complete",
  "iteration": 1,
  "quality_verdict": "MINOR_REVISION",
  "review_history": [
    {
      "iteration": 1,
      "domain_verdict": "PASS",
      "novelty_verdict": "WEAK",
      "aggregate": "WEAK",
      "anti_pattern_count": 2,
      "issues_fixed": 3,
      "new_citations_added": 1
    }
  ]
}
```

## Step 4.6: CHECKPOINT

Git commit: `docs: review iteration {N} — verdict: {verdict}`

## FORCE_STOP Triggers

- 3 consecutive iterations with no verdict improvement → CEILING_REACHED
- Reviewer agent fails to produce structured output → Retry once, then skip
- Total Phase 4 time > 30 minutes → STOP, proceed with current quality
