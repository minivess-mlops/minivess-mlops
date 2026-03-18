# TDD Skill Upgrade Plan: v2.1.0 → v3.0.0

**Date**: 2026-03-18
**Issue**: #855
**Branch**: fix/fda-readiness-improvement-xml
**Status**: Planning (awaiting web research completion)

## User Requirements (Verbatim)

> Two recurring issues keep happening with the TDD skill:
> 1. Silent dismissal of failures (BUG-1, metalearning 2026-03-07)
> 2. Whac-a-mole serial fixing (BUG-2, metalearning 2026-03-18)
>
> Make sure Skills 2.0 and agentic skill evals are introduced.
> Up to date with latest Skills best practices from Anthropic.
> Optimize with multiple reviewer agent iterations until converging.
> Good for production-grade code with Ralph Loop for infrastructure monitoring.
> Quality code over quick slop.
> More tokens upfront rather than fixing after.
> Write "scientific production-grade code" principle to CLAUDE.md and KG.

## Root Cause Plan Reference

`docs/planning/root-cause-bug-fixing-plan.xml` Phase 2 defines:
- `protocols/failure-triage.md` (GATHER→CATEGORIZE→PLAN→FIX→VERIFY)
- SKILL.md Rules #9 (zero tolerance) + #10 (triage before fixing)
- verify-phase.md FAILURE GATE
- fix-phase.md PRE-CHECK
- ACTIVATION-CHECKLIST.md non-green baseline

## Upgrade Scope: 8 Work Items

### WI-1: Implement Phase 2 of root-cause-bug-fixing-plan.xml (BUG-1 + BUG-2)
- New: `protocols/failure-triage.md`
- Update: SKILL.md (Rules #9, #10, anti-patterns)
- Update: `protocols/verify-phase.md` (FAILURE GATE)
- Update: `protocols/fix-phase.md` (PRE-CHECK)
- Update: `ACTIVATION-CHECKLIST.md` (non-green baseline)

### WI-2: "Scientific Production-Grade Code" Principle
- Add to CLAUDE.md as a new overarching principle or rule
- Add KG decision node: `code_quality_philosophy`
- Principle: "Spend more tokens upfront on thorough implementation rather than
  cutting corners and fixing later. The cost of sloppy initial code is always
  higher than the cost of careful initial code."
- Concrete mandates: read before edit, understand before implement, test before
  ship, verify before claim

### WI-3: Agentic Skill Evals (Skills 2.0 Pattern)
- Extend `evals/test_tdd_skill.py` from 13 to 25+ tests
- Add behavioral evals (not just structural):
  - Eval: "Given 5 failures, does the skill invoke failure-triage?"
  - Eval: "Given a pre-existing failure, does the skill create an issue?"
  - Eval: "Given a plan with 3 tasks, does state track correctly?"
- Define eval runner that can be invoked via `/tdd-eval`

### WI-4: Ralph Loop Integration
- Review Ralph Loop v1.0.0 for compatibility with TDD skill v3.0.0
- Ensure Ralph Loop's DIAGNOSE phase uses the same failure-triage protocol
- Add cross-skill reference: TDD skill can invoke Ralph Loop for cloud failures
- Update Ralph Loop if needed for consistency

### WI-5: SKILL.md v3.0.0 Structural Upgrade
- YAML front matter (consistent with create-literature-report v2.0.0)
- Version bump to 3.0.0
- Add "Tokens Upfront" philosophy section
- Add progress banner format
- Reference new protocols

### WI-6: "Tokens Upfront" Philosophy in Skill
- New section: "Quality Over Speed"
- Principle: Read all relevant files BEFORE writing code
- Principle: Understand the full test surface BEFORE implementing
- Principle: Spend 30% of task time reading, 70% implementing (not 5%/95%)
- Anti-pattern: "I'll just try this and see if it works" without reading context

### WI-7: Update CLAUDE.md
- New rule or principle about scientific production-grade code
- Reference to TDD skill v3.0.0
- "Tokens upfront" principle codified

### WI-8: KG Decision Node
- `knowledge-graph/decisions/L2-architecture/code_quality_philosophy.yaml`
- Options: production-grade (resolved), move-fast-break-things (rejected)
- Evidence: metalearning docs on BUG-1, BUG-2, time wasted

## Execution Strategy

1. Implement WI-1 first (directly from the XML plan)
2. Then WI-2 + WI-7 + WI-8 (CLAUDE.md + KG updates)
3. Then WI-3 (evals)
4. Then WI-4 + WI-5 + WI-6 (structural upgrades)
5. Run reviewer agents for quality convergence
6. Final verification: all evals pass

## Web Research Findings (Complete)

### Skills 2.0 — Key Features (Feb-Mar 2026)

1. **Formal eval framework**: 4-agent architecture (Executor, Grader, Comparator, Analyzer)
   with `evals/evals.json` test cases, parallel with-skill/baseline runs, HTML report viewer
   - Source: https://claude.com/blog/improving-skill-creator-test-measure-and-refine-agent-skills

2. **Trigger optimization loop**: `run_loop.py` optimizes `description` frontmatter with
   60/40 train/holdout split, 5 cycles. 5/6 public skills improved activation.

3. **Hooks for quality enforcement**:
   - `Stop` hook (type: "agent"): run test suite before allowing completion
   - `PostToolUse` hook (matcher: "Edit|Write"): auto-run tests after every code edit
   - `UserPromptSubmit` hook: force skill evaluation (25% → 90%+ activation)
   - Source: https://code.claude.com/docs/en/hooks-guide

4. **Multi-agent TDD architecture** (AlexOp pattern): Separate subagent contexts for
   test-writer and implementer prevents cross-contamination of test/impl knowledge

5. **Progressive disclosure budget**: Metadata (~100 words) always in context,
   SKILL.md body (<500 lines) when triggered, bundled resources unlimited

6. **Anthropic skill-creator philosophy**: "Explain WHY not just MUST. Generalize from
   feedback, never overfit. Keep prompt lean. Look for repeated work."

### Actionable Improvements (Priority Order)

| # | Improvement | Impact | Effort | Source |
|---|------------|--------|--------|--------|
| 1 | Add `evals/evals.json` with 10+ test cases | HIGH | MEDIUM | Skills 2.0 |
| 2 | Add `UserPromptSubmit` hook for activation | HIGH | LOW | HackerNoon governance |
| 3 | Add `Stop` hook enforcing test pass before completion | HIGH | LOW | Hooks guide |
| 4 | Implement failure-triage.md (BUG-1 + BUG-2) | HIGH | LOW | root-cause-plan.xml |
| 5 | "Tokens upfront" philosophy section | MEDIUM | LOW | User requirement |
| 6 | Multi-agent RED/GREEN isolation | MEDIUM | HIGH | AlexOp TDD pattern |
| 7 | Trigger description optimization | MEDIUM | LOW | Skills 2.0 |
| 8 | PostToolUse hook for auto-test | MEDIUM | LOW | Hooks guide |
| 9 | YAML front matter standardization | LOW | LOW | Skills spec |
| 10 | Ralph Loop cross-integration | LOW | MEDIUM | Ralph docs |
