# Skills Upgrade Plan — March 2026

**Date**: 2026-03-19 (v2 — post-review iteration)
**Branch**: `fix/skills-upgrade`
**Status**: PLANNING (not yet implemented)
**Reviews**: 3 parallel reviewer agents (Architecture, Pragmatist, Self-Improvement Specialist)

---

## 1. Executive Summary

Restructure all 15 skills in `.claude/skills/` to align with:

1. **Anthropic's official guide** ("The Complete Guide to Building Skills for Claude", 2025)
2. **SkillNet ontology** (zjunlp/SkillNet, arXiv:2603.04448v1) — typed inter-skill relations
3. **JJ Englert's folder pattern** (Ultrathink v2) — orchestrator + instructions + examples + eval

The upgrade transforms flat/monolithic skills into a modular folder architecture with
progressive disclosure, explicit inter-skill relations, and built-in quality evaluation.

---

## 2. Sources of Truth

| Source | Key Contribution |
|--------|-----------------|
| Anthropic Guide (2025) | Three-level progressive disclosure, frontmatter spec, folder structure, 5 workflow patterns |
| SkillNet (arXiv:2603.04448v1) | Three-layer ontology, 4 typed relations, metadata-first routing, quality dimensions |
| Ultrathink v2 pattern | SKILL.md as orchestrator (no rules), instructions/ for rules, examples/ for learning, eval/ for quality |
| Current repo audit | 15 skills, 106 files, 2505 SKILL.md lines; 7 monolithic, 8 folder-based |

---

## 3. Current State Audit

### 3.1 Skill Inventory

| # | Skill | Files | Lines | Pattern | Maturity |
|---|-------|-------|-------|---------|----------|
| 1 | create-literature-report | 12 | 191 | Folder (protocols+prompts+state) | 5/5 |
| 2 | fetch-docs | 1 | 63 | Monolithic | 3/5 |
| 3 | issue-creator | 13 | 179 | Folder (protocols+templates+evals) | 5/5 |
| 4 | kg-sync | 3 | 196 | Folder (SKILL.md+evals) | 4/5 |
| 5 | knowledge-reviewer | 3 | 202 | Folder (SKILL.md+evals) | 4/5 |
| 6 | openspec-apply-change | 1 | 156 | Monolithic | 4/5 |
| 7 | openspec-archive-change | 1 | 114 | Monolithic | 4/5 |
| 8 | openspec-explore | 1 | 288 | Monolithic (narrative) | 5/5 |
| 9 | openspec-propose | 1 | 110 | Monolithic | 4/5 |
| 10 | overnight-runner | 2 | 204 | Folder (SKILL.md+template) | 5/5 |
| 11 | planning-backlog | 1 | 139 | Monolithic | 3/5 |
| 12 | prd-update | 10 | 151 | Folder (protocols+templates) | 4/5 |
| 13 | ralph-loop | 1 | 211 | Monolithic | 5/5 |
| 14 | self-learning-iterative-coder | 17 | 194 | Folder (full pattern) | 5/5 |
| 15 | sync-roadmap | 1 | 107 | Monolithic | 3/5 |

### 3.2 Existing Patterns

**Pattern A — Full folder (5 skills)**: create-literature-report, issue-creator, prd-update,
overnight-runner, self-learning-iterative-coder. Uses `protocols/`, `prompts/`, `templates/`,
`state/`, `evals/` subdirectories. Gold standard = self-learning-iterative-coder (17 files).

**Pattern B — Partial folder (3 skills)**: kg-sync, knowledge-reviewer (SKILL.md + evals only).
Missing protocol decomposition.

**Pattern C — Monolithic (7 skills)**: fetch-docs, openspec-apply/archive/explore/propose,
planning-backlog, ralph-loop, sync-roadmap. Single SKILL.md file.

---

## 4. Target Architecture

### 4.1 Canonical Folder Structure (Anthropic + Ultrathink + SkillNet)

```
skill-name/
  SKILL.md                          # Orchestrator ONLY — workflow steps + file pointers
  instructions/                     # Domain rules, constraints, anti-patterns
    core-rules.md                   # Invariant rules for this skill
    anti-patterns.md                # What NOT to do (documented failures)
    [domain-specific].md            # Additional rule sets
  protocols/                        # Step-by-step procedures (one per phase)
    phase-1-*.md
    phase-2-*.md
    ...
  prompts/                          # Agent/reviewer prompt templates
    agent-*.md
  examples/                         # Good/bad examples for in-context learning
    good/
      example-1.md
    bad/
      anti-example-1.md
  eval/                             # Quality gates
    checklist.md                    # Pass/fail tests (run after every execution)
    advisory-board.md               # AI reviewer personas (optional, for complex skills)
  templates/                        # Output format templates
    *.md / *.yaml / *.sh
  state/                            # State schemas for crash recovery (if stateful)
    schema.json
    example.json
  references/                       # External docs loaded on demand (Anthropic pattern)
    api-guide.md
    ...
```

### 4.2 SKILL.md as Pure Orchestrator

Per the Ultrathink pattern, SKILL.md contains **zero rules** — only:
1. YAML frontmatter (name, description with triggers, metadata with relations)
2. Workflow steps pointing to files in subdirectories
3. Phase transitions and decision points

Rules, examples, and evaluation criteria live in their own files and are loaded
only when the relevant phase is active (progressive disclosure).

### 4.3 Enhanced Frontmatter (SkillNet-Informed)

```yaml
---
name: skill-name
description: >
  What it does. Use when [trigger phrases].
  Do NOT use for [negative triggers].
metadata:
  version: "2.0.0"
  category: mlops          # SkillNet Layer 1: taxonomy category
  tags: [training, docker, testing, ...]  # Fine-grained tags
  relations:               # SkillNet Layer 2: typed edges
    compose_with: [skill-a, skill-b]     # Output feeds into these skills
    depend_on: [skill-c]                 # Cannot run without these
    similar_to: [skill-d]               # Functionally equivalent alternatives
    belong_to: [parent-skill]           # Sub-component of larger workflow
---
```

### 4.4 Skill Taxonomy (SkillNet Layer 1)

| Category | Skills | Tags |
|----------|--------|------|
| **Development** | self-learning-iterative-coder, issue-creator | tdd, testing, git, github |
| **Research** | create-literature-report, fetch-docs | literature, citations, web-search |
| **Knowledge** | kg-sync, knowledge-reviewer, prd-update | knowledge-graph, prd, validation |
| **Orchestration** | openspec-apply/archive/explore/propose | spec-driven, planning, workflow |
| **Operations** | overnight-runner, ralph-loop, sync-roadmap | skypilot, monitoring, automation |
| **Planning** | planning-backlog | github-projects, backlog, prioritization |

### 4.5 Skill Relation Graph (SkillNet Layer 2)

```
self-learning-iterative-coder
  ├── compose_with → issue-creator (FORCE_STOP triggers issue creation)
  ├── compose_with → ralph-loop (RED→GREEN cycle on cloud failures)
  └── depend_on → fetch-docs (beyond-cutoff library verification)

create-literature-report
  ├── compose_with → prd-update (paper findings → PRD decision updates)
  ├── compose_with → kg-sync (citations → bibliography.yaml)
  └── depend_on → fetch-docs (web search for papers)

overnight-runner
  ├── compose_with → ralph-loop (child session monitoring)
  ├── compose_with → self-learning-iterative-coder (plan execution)
  └── compose_with → issue-creator (failure → issue creation)

openspec-propose
  ├── compose_with → openspec-apply-change (proposal → implementation)
  └── compose_with → openspec-archive-change (implementation → archive)

openspec-explore
  └── compose_with → openspec-propose (discovery → proposal)

kg-sync
  ├── depend_on → knowledge-reviewer (validation before sync)
  └── compose_with → prd-update (KG changes → PRD updates)

planning-backlog
  ├── compose_with → issue-creator (backlog items → GitHub issues)
  └── compose_with → sync-roadmap (issues → timeline sync)

ralph-loop
  ├── belong_to → overnight-runner (child executor)
  └── compose_with → issue-creator (unrecoverable failure → issue)
```

---

## 5. Upgrade Strategy Per Skill

### 5.1 Upgrade Tiers

**Tier 1 — Full Restructure** (monolithic >150 lines with mixed concerns):
- ralph-loop (211 lines, 14 failure patterns mixed with workflow)
- openspec-explore (288 lines, narrative stance mixed with procedures)
- planning-backlog (139 lines, field metadata mixed with operations)
- sync-roadmap (107 lines, field metadata mixed with operations)

**Tier 2 — Decompose Existing Folders** (folder-based but SKILL.md still monolithic):
- kg-sync (196 lines in SKILL.md, 7 steps should be protocols)
- knowledge-reviewer (202 lines in SKILL.md, 4 agents should be protocols)

**Tier 3 — Enhance Existing Folders** (already well-structured, add missing pieces):
- create-literature-report (add examples/, eval/checklist.md, instructions/)
- issue-creator (add examples/, enhance eval/)
- prd-update (add examples/, eval/checklist.md)
- overnight-runner (add instructions/, eval/checklist.md)
- self-learning-iterative-coder (add examples/good/, examples/bad/)

**Tier 4 — Frontmatter + Relations Only** (appropriately simple/delegation):
- fetch-docs (63 lines — appropriate monolith, just enhance frontmatter)
- openspec-apply-change (delegation — enhance frontmatter + add eval/checklist.md)
- openspec-archive-change (delegation — enhance frontmatter + add eval/checklist.md)
- openspec-propose (delegation — enhance frontmatter + add eval/checklist.md)

### 5.2 Detailed Plan Per Skill

#### Skill 1: `self-learning-iterative-coder` (Tier 3 — Enhance)

**Current**: 17 files, full protocols+prompts+state+evals. Gold standard.
**Changes**:
- [ ] Add `instructions/core-rules.md` — extract 11 critical rules from SKILL.md
- [ ] Add `instructions/anti-patterns.md` — extract 11 anti-patterns from SKILL.md
- [ ] Add `examples/good/tdd-session-minivess.md` — annotated real session excerpt
- [ ] Add `examples/bad/whac-a-mole.md` — annotated failure mode
- [ ] Add `eval/checklist.md` — 9 pass/fail tests from existing rules
- [ ] Refactor SKILL.md to orchestrator-only (point to instructions/ and examples/)
- [ ] Add SkillNet relations to frontmatter
- [ ] Update description with trigger phrases and negative triggers

#### Skill 2: `create-literature-report` (Tier 3 — Enhance)

**Current**: 12 files, protocols+prompts+state. Very mature.
**Changes**:
- [ ] Add `instructions/writing-rules.md` — extract Markov novelty rules from SKILL.md
- [ ] Add `instructions/anti-patterns.md` — extract BANNED behaviors
- [ ] Add `examples/good/` — annotated excerpt from a shipped report
- [ ] Add `examples/bad/memory-based-citations.md` — annotated failure
- [ ] Add `eval/checklist.md` — hallucination rate, URL validity, citation format
- [ ] Refactor SKILL.md to orchestrator-only
- [ ] Add SkillNet relations to frontmatter

#### Skill 3: `issue-creator` (Tier 3 — Enhance)

**Current**: 13 files, protocols+templates+evals.
**Changes**:
- [ ] Add `instructions/metadata-rules.md` — extract YAML metadata encoding rules
- [ ] Add `instructions/validation-gates.md` — extract 7 hard gates
- [ ] Add `examples/good/well-formed-issue.md` — annotated real issue
- [ ] Add `examples/bad/missing-metadata.md` — common failure
- [ ] Add `eval/checklist.md` — formalize validation gates as pass/fail
- [ ] Refactor SKILL.md to orchestrator-only
- [ ] Add SkillNet relations to frontmatter

#### Skill 4: `kg-sync` (Tier 2 — Decompose)

**Current**: 3 files (SKILL.md 196 lines + evals). 7 steps all inline.
**Changes**:
- [ ] Split into `protocols/scan-code.md`, `scan-experiments.md`, `propagate.md`,
      `stamp.md`, `staleness.md`, `generate.md`, `validate.md`, `export.md`
- [ ] Add `instructions/invariants.md` — extract idempotency rules, orphan detection
- [ ] Add `instructions/anti-patterns.md` — timestamps in .tex, manual DuckDB
- [ ] Add `eval/checklist.md` — staleness detection, orphan check, pdflatex compile
- [ ] Refactor SKILL.md to orchestrator-only
- [ ] Add SkillNet relations to frontmatter

#### Skill 5: `knowledge-reviewer` (Tier 2 — Decompose)

**Current**: 3 files (SKILL.md 202 lines + evals). 4 agents inline.
**Changes**:
- [ ] Split into `protocols/link-checker.md`, `prd-auditor.md`, `legacy-detector.md`,
      `staleness-scanner.md`
- [ ] Add `instructions/severity-rules.md` — ERROR/WARN/INFO definitions
- [ ] Add `instructions/quick-mode.md` — pre-commit fast path rules
- [ ] Add `eval/checklist.md` — 40+ checks as structured pass/fail
- [ ] Refactor SKILL.md to orchestrator-only
- [ ] Add SkillNet relations to frontmatter

#### Skill 6: `ralph-loop` (Tier 1 — Full Restructure)

**Current**: 1 file (211 lines). 14 failure patterns + workflow + cost tracking mixed.
**Changes**:
- [ ] Create `instructions/failure-patterns.md` — all 14 categories with auto-fix logic
- [ ] Create `instructions/cost-tracking.md` — JSONL event format, hourly rates
- [ ] Create `instructions/anti-patterns.md` — regex ban, infinite retry
- [ ] Create `protocols/preflight.md`, `launch.md`, `monitor.md`, `diagnose.md`,
      `fix-relaunch.md`, `report.md`
- [ ] Create `eval/checklist.md` — max 3 retries, cost logged, diagnosis correct
- [ ] Create `templates/cost-event.jsonl` — event schema
- [ ] Refactor SKILL.md to orchestrator-only
- [ ] Add SkillNet relations to frontmatter

#### Skill 7: `openspec-explore` (Tier 1 — Full Restructure)

**Current**: 1 file (288 lines). Largest monolithic skill. Narrative stance + procedures.
**Changes**:
- [ ] Create `instructions/thinking-partner-stance.md` — philosophical rules
- [ ] Create `instructions/anti-patterns.md` — force structure, rush conclusions
- [ ] Create `references/entry-points.md` — 4 activation modes with ASCII diagrams
- [ ] Create `references/visualization-patterns.md` — ASCII diagram templates
- [ ] Create `eval/checklist.md` — did we stay exploratory? did we capture insights?
- [ ] Refactor SKILL.md to orchestrator-only
- [ ] Add SkillNet relations to frontmatter

**Note**: This skill is intentionally narrative/philosophical. Splitting must preserve
coherence — test reading flow after restructure.

#### Skill 8: `prd-update` (Tier 3 — Enhance)

**Current**: 10 files, protocols+templates. Already well-structured.
**Changes**:
- [ ] Add `instructions/citation-rules.md` — extract citation preservation invariants
- [ ] Add `instructions/anti-patterns.md` — deleting refs, skipping sub-citations
- [ ] Add `examples/good/decision-node-with-citations.yaml` — annotated example
- [ ] Add `eval/checklist.md` — 7-point reviewer checklist as pass/fail
- [ ] Refactor SKILL.md to orchestrator-only
- [ ] Add SkillNet relations to frontmatter

#### Skill 9: `overnight-runner` (Tier 3 — Enhance)

**Current**: 2 files (SKILL.md + template). Well-designed but minimal structure.
**Changes**:
- [ ] Create `instructions/observability-rules.md` — heartbeat, stall detection
- [ ] Create `instructions/anti-patterns.md` — long prompts, missing SKIP_TO
- [ ] Create `protocols/validate.md`, `launch.md`, `monitor.md`, `diagnose.md`, `report.md`
- [ ] Add `eval/checklist.md` — heartbeat frequency, stall detection, cost logged
- [ ] Refactor SKILL.md to orchestrator-only
- [ ] Add SkillNet relations to frontmatter

#### Skill 10: `planning-backlog` (Tier 1 — Full Restructure)

**Current**: 1 file (139 lines). Field metadata mixed with operations.
**Changes**:
- [ ] Create `references/github-project-fields.md` — field IDs, option IDs, labels
- [ ] Create `instructions/priority-rules.md` — P0/P1/P2 assignment guidelines
- [ ] Create `instructions/prd-connection.md` — how issues link to PRD decisions
- [ ] Create `protocols/sprint-planning.md`, `create-issue.md`, `progress-review.md`,
      `reprioritize.md`
- [ ] Add `eval/checklist.md` — priority assigned, PRD linked, project board updated
- [ ] Refactor SKILL.md to orchestrator-only
- [ ] Add SkillNet relations to frontmatter

#### Skill 11: `sync-roadmap` (Tier 1 — Full Restructure)

**Current**: 1 file (107 lines). Field metadata mixed with sync logic.
**Changes**:
- [ ] Create `references/github-project-fields.md` — field IDs (shared with planning-backlog?)
- [ ] Create `instructions/timeline-rules.md` — date assignment logic
- [ ] Create `instructions/size-heuristics.md` — XS/S/M/L/XL mapping
- [ ] Create `protocols/sync-all.md`, `sync-single.md`, `sync-recently-closed.md`
- [ ] Add `eval/checklist.md` — all fields populated, dates correct, sizes assigned
- [ ] Refactor SKILL.md to orchestrator-only
- [ ] Add SkillNet relations to frontmatter

#### Skill 12: `fetch-docs` (Tier 4 — Frontmatter Only)

**Current**: 1 file (63 lines). Simple delegation to chub CLI.
**Changes**:
- [ ] Enhance description with trigger phrases and negative triggers
- [ ] Add SkillNet relations to frontmatter
- [ ] Add `eval/checklist.md` — max 3 chub calls, local checked first
- [ ] Keep monolithic (63 lines is appropriate)

#### Skills 13-15: `openspec-apply-change`, `openspec-archive-change`, `openspec-propose` (Tier 4)

**Current**: 1 file each (110-156 lines). Delegation to openspec CLI.
**Changes** (same for all three):
- [ ] Enhance description with trigger phrases and negative triggers
- [ ] Add SkillNet relations to frontmatter (compose_with edges between them)
- [ ] Add `eval/checklist.md` — task completion tracked, artifacts validated
- [ ] Keep monolithic (delegation is the correct pattern here)

---

## 6. Implementation Plan

### Phase 1: Groundwork (this PR)

1. Create this plan document
2. Define the canonical folder structure
3. Define the SkillNet relation graph
4. Define the skill taxonomy

### Phase 2: Gold Standard Upgrade (1 skill)

Upgrade `self-learning-iterative-coder` first as the reference implementation.
This is already the most mature skill — upgrading it establishes the exact pattern
that all other skills will follow.

**Deliverable**: Upgraded skill with orchestrator SKILL.md, instructions/, examples/, eval/.
**Validation**: Run a real TDD session with the upgraded skill and verify no drift.

### Phase 3: Tier 1 — Full Restructures (4 skills)

Upgrade ralph-loop, openspec-explore, planning-backlog, sync-roadmap.
These are the skills with the most to gain from restructuring.

**Deliverable**: 4 skills upgraded to folder pattern.
**Validation**: Manual trigger testing for each skill.

### Phase 4: Tier 2 — Decompose (2 skills)

Upgrade kg-sync, knowledge-reviewer.
Split monolithic SKILL.md into protocol files.

**Deliverable**: 2 skills with protocol decomposition.
**Validation**: Run knowledge-reviewer in full mode, verify all 4 agents work.

### Phase 5: Tier 3 — Enhance (5 skills)

Upgrade create-literature-report, issue-creator, prd-update, overnight-runner.
Add missing instructions/, examples/, eval/ to already-structured skills.

**Deliverable**: 5 skills with enhanced structure.
**Validation**: Verify progressive disclosure works (SKILL.md loads only, details on demand).

### Phase 6: Tier 4 — Frontmatter + Relations (4 skills)

Upgrade fetch-docs, openspec-apply/archive/propose.
Add SkillNet relations and enhanced descriptions. Minimal structural changes.

**Deliverable**: 4 skills with enhanced frontmatter.
**Validation**: Verify trigger accuracy with test queries.

### Phase 7: Skill Network Manifest

Create `skills/SKILL-NETWORK.yaml` — the SkillNet Layer 2 relation graph as a
machine-readable file. This enables future tooling to:
- Visualize skill dependencies
- Auto-suggest related skills
- Detect redundancy
- Plan multi-skill workflows

### Phase 8: Quality Audit

Run all eval/checklist.md files across all 15 skills. Document results.
Create a quality scorecard per the 5 SkillNet dimensions:
Safety, Completeness, Executability, Maintainability, Cost-awareness.

---

## 7. Key Design Decisions

### 7.1 SKILL.md as Orchestrator vs. Container

**Decision**: SKILL.md = pure orchestrator (no rules, no examples, no evaluation criteria).

**Why**: When rules live in SKILL.md alongside workflow steps, Claude tries to hold
everything at once. Rules compete with each other. Examples get buried. The longer the
file, the worse the output. (Source: Ultrathink v2 restructuring results.)

**Exception**: Skills under 80 lines (fetch-docs, openspec-*) can remain monolithic.
The overhead of a folder structure exceeds the benefit for simple delegation skills.

### 7.2 Progressive Disclosure Mapping

| Level | What Loads | When | Source |
|-------|-----------|------|--------|
| L1 (frontmatter) | Name + description + relations | Always (system prompt) | Anthropic guide |
| L2 (SKILL.md body) | Workflow steps + file pointers | When skill triggered | Anthropic guide |
| L3 (linked files) | Instructions, protocols, examples, eval | When specific phase active | Anthropic guide |

### 7.3 When NOT to Split

Not every skill benefits from folder decomposition. Keep monolithic when:
- Skill is pure delegation (openspec-apply/archive/propose, fetch-docs)
- Skill is under 80 lines
- Skill is a philosophical stance, not a procedure (openspec-explore — but this one
  is 288 lines, so we split it anyway for navigation; test reading flow after)

### 7.4 Shared References

`planning-backlog` and `sync-roadmap` both reference GitHub Project field IDs.
**Decision**: Each skill maintains its own copy. Reason: skills must be self-contained
and portable. A shared reference creates a hidden dependency that breaks if either
skill is moved or distributed independently.

### 7.5 SkillNet Relations in Frontmatter

**Decision**: Add `metadata.relations` to every SKILL.md frontmatter, even if empty.

**Why**: Explicit declaration of inter-skill relationships enables future tooling
and helps Claude understand which skills to suggest after completing one. The
relation graph in Section 4.5 is the source of truth; frontmatter is the distributed
copy for progressive disclosure (each skill knows its own edges).

---

## 8. Quality Evaluation Framework (SkillNet-Informed)

Each skill will be scored on 5 dimensions after upgrade:

| Dimension | Definition | Measurement |
|-----------|-----------|-------------|
| **Safety** | Does the skill ever perform destructive operations? | Audit for git push --force, rm -rf, --no-verify |
| **Completeness** | Are all prerequisites, dependencies, and edge cases defined? | eval/checklist.md pass rate |
| **Executability** | Can the skill run successfully in the current environment? | Manual trigger test + functional test |
| **Maintainability** | Can the skill be updated without breaking dependents? | File count, concern separation score |
| **Cost-awareness** | How many tokens/API calls does the skill consume? | Token count comparison: before vs. after upgrade |

---

## 9. Success Criteria

- [ ] All 15 skills have SkillNet relations in frontmatter
- [ ] All 15 skills have enhanced descriptions with trigger + negative trigger phrases
- [ ] All skills >80 lines follow the folder pattern (orchestrator SKILL.md + subdirs)
- [ ] All skills have eval/checklist.md with pass/fail tests
- [ ] SKILL-NETWORK.yaml documents the complete relation graph
- [ ] No regression: every skill triggers correctly on existing use cases
- [ ] Progressive disclosure verified: SKILL.md under 5000 words for all skills

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Splitting narrative skills (openspec-explore) loses coherence | Claude drifts on tone/stance | Test reading flow after split; if coherence drops, merge back |
| Over-engineering simple delegation skills | Maintenance overhead | Tier 4 skills get frontmatter only, no folder restructure |
| SkillNet relations become stale | Wrong skill suggestions | SKILL-NETWORK.yaml is version-controlled; review on each skill change |
| Token increase from progressive disclosure | Slower skill loading | Benchmark token usage before/after; keep SKILL.md under 5000 words |
| Breaking existing skill invocations | User workflow disruption | Test all trigger phrases before merging each phase |

---

---

## 11. Review Synthesis (3 Parallel Reviewer Agents)

### 11.1 Architecture Reviewer — Key Findings

**TOP 3 GAPS exposed by autoresearch pattern:**

1. **No iterative mutation loop.** The plan treats skills as write-once artifacts.
   Restructure → validate once → move on. Autoresearch says: evaluation is the
   inner loop, not a final step. Run N times, score, mutate, re-score, converge.

2. **eval/checklist.md is designed for human review, not automated scoring.**
   The plan never specifies WHO evaluates the checklist (human, pytest, LLM-as-judge).
   Autoresearch requires the LLM to be BOTH executor AND evaluator. No LLM-as-judge
   layer exists in the plan.

3. **No baseline measurement.** Without before/after behavioral data, the plan
   cannot prove restructuring improved anything. Success criteria (Section 9) are
   structural ("all skills have eval/checklist.md") not behavioral ("Skill X pass
   rate improved from 72% to 96%").

**TOP 3 STRENGTHS to preserve:**
- Tiered upgrade strategy (correct triage: don't over-engineer simple skills)
- SkillNet relation graph (real compositional intelligence between skills)
- Progressive disclosure mapping (L1 frontmatter → L2 body → L3 linked files)

**Anti-patterns flagged:**
- "Pure orchestrator" rule too dogmatic — cross-cutting rules (e.g., TDD's 11 rules
  that apply to ALL phases) should STAY in SKILL.md, not be hidden behind a file pointer
- 40+ checks for knowledge-reviewer is specification, not evaluation. Max 6-10 per skill.
- No triggering test suite — each skill should ship with eval/trigger-tests.yaml
- Shared references between planning-backlog and sync-roadmap should be shared, not duplicated
  (portability argument is specious since these skills never leave this repo)

### 11.2 Pragmatist Reviewer — Key Findings

**Effort estimate:** 7-9 sessions, ~2.5 hours user review. Shippable in 2-3 days.

**Critical verdict: Plan is over-scoped.**
> "Only 2 skills (ralph-loop, self-learning-iterative-coder) have genuine structural
> problems that cause real failures. The other 13 need metadata (frontmatter + eval
> checklist) and nothing more."

**Recommended cuts:**
- Drop Phase 7 (SKILL-NETWORK.yaml) — premature at 15 skills
- Drop Phase 8 (quality audit scorecard) — replaced by per-skill validation
- Drop openspec-explore restructure — monolithic narrative works, splitting loses coherence
- Drop planning-backlog/sync-roadmap protocol decomposition — not worth it
- Drop kg-sync/knowledge-reviewer protocol decomposition — not worth it
- Drop examples/good/ and examples/bad/ for ALL skills — add opportunistically when drift occurs

**Proposed 3-tier model (replaces 4 tiers):**

| Tier | Skills | Action |
|------|--------|--------|
| **A: Full Restructure** | ralph-loop, self-learning-iterative-coder | Extract rules→instructions/, add examples/, eval/ |
| **B: Add Eval + Frontmatter** | 8 skills (create-lit-report through sync-roadmap) | SkillNet frontmatter + eval/checklist.md |
| **C: Frontmatter Only** | 5 skills (fetch-docs, openspec-explore, openspec-apply/archive/propose) | SkillNet frontmatter only |

**Shippable in 3 sessions:**
- Session 1: Restructure self-learning-iterative-coder + batch add frontmatter to all 15
- Session 2: Restructure ralph-loop + add eval/checklist.md to Tier B skills
- Session 3: Remaining eval/checklist.md + regression test run

### 11.3 Self-Improvement Specialist — Key Findings

**Autoresearch feasibility per skill:**

| Autoresearchable? | Skills | Count |
|-------------------|--------|-------|
| **YES** | issue-creator, planning-backlog, sync-roadmap | 3 |
| **PARTIAL** | create-literature-report, prd-update, openspec-propose/apply/archive | 5 |
| **NO** | self-learning-iterative-coder, kg-sync, knowledge-reviewer, ralph-loop, openspec-explore, overnight-runner, fetch-docs (too trivial) | 7 |

**Why most skills CANNOT be autoresearched:**
- **Code-producing skills** (self-learning-iterative-coder): compiler/tests already evaluate;
  fitness landscape is non-stationary (every plan is different)
- **Cloud-interaction skills** (ralph-loop, overnight-runner): each run costs real money,
  takes 10-60 min, needs real infrastructure
- **Intentionally unstructured skills** (openspec-explore): defining "success" would destroy
  the skill's purpose
- **Script-orchestrating skills** (kg-sync, knowledge-reviewer): quality = script correctness,
  not prompt quality

**Self-eval bias risks:**
1. Goodhart's Law — skill optimizes for detectable criteria, ignoring semantic quality
2. Self-consistency bias — same model writes and evaluates, inheriting same blind spots
3. Convergence to verbosity — adding rules monotonically increases prompt length

**Critical design rule:** Structural criteria (YAML present, file exists, API returns expected)
MUST use Python parsers, NEVER LLM self-eval. LLM-eval only for natural language understanding
criteria, weighted at 0.5x.

**Recommendation for code-producing skills:** Population-level metrics collected over 20+
real sessions (iterations-to-convergence, FORCE_STOP rate, regression rate), not per-run
autoresearch loops.

---

## 12. Revised Plan (Post-Review)

Based on the three reviews, the plan is revised as follows:

### 12.1 What Changed

| Original | Revised | Reason |
|----------|---------|--------|
| 4 upgrade tiers | 3 tiers (A/B/C) | Tier 2 merged into B (Pragmatist: not worth separate phase) |
| 8 sequential phases | 3 sessions | Pragmatist: shippable plan, not shelfware |
| Pure orchestrator SKILL.md for all | Cross-cutting rules stay in SKILL.md | Architecture: hiding rules behind pointers = less likely to be followed |
| eval/checklist.md (static, human) | Split: structural (pytest) + behavioral (binary criteria) | Architecture + Self-Improvement: two eval layers needed |
| SKILL-NETWORK.yaml (Phase 7) | Relations in frontmatter only | Pragmatist: premature at 15 skills |
| Quality audit scorecard (Phase 8) | Per-skill baseline + validation | Architecture: baselines before restructuring |
| No autoresearch | Phase 4 (optional): autoresearch for 3 eligible skills | Self-Improvement: only 3 skills qualify |
| examples/good/ examples/bad/ for all | Add opportunistically when drift occurs | Pragmatist: most labor-intensive, least measurable |
| openspec-explore restructure | Leave monolithic | Pragmatist: narrative coherence > folder structure |
| planning-backlog, sync-roadmap protocol split | Frontmatter + eval only | Pragmatist: each "protocol" would be 5-10 lines |
| Shared references rejected (DRY violation) | Shared `references/github-project-fields.md` | Architecture: these skills never leave this repo |

### 12.2 Revised Implementation Plan

#### Session 1: Foundation (~1 hour)

**Step 1a**: Baseline measurement. Run each of the 3 autoresearchable skills
(issue-creator, planning-backlog, sync-roadmap) once against a test input.
Record current behavior as `eval/baseline.md`.

**Step 1b**: Restructure self-learning-iterative-coder.
- Extract 11 critical rules to `instructions/core-rules.md`
- Extract 11 anti-patterns to `instructions/anti-patterns.md`
- Keep cross-cutting rules in SKILL.md (do NOT make it a pure pointer file)
- Add `eval/checklist.md` with max 8 binary criteria
- Add SkillNet relations to frontmatter

**Step 1c**: Batch add SkillNet frontmatter to ALL 15 skills.
- Add `metadata.category`, `metadata.tags`, `metadata.relations` to every SKILL.md
- Enhance `description` with trigger phrases and negative triggers
- 10 minutes total, batch operation

#### Session 2: High-ROI Restructure (~1 hour)

**Step 2a**: Restructure ralph-loop.
- Extract 14 failure patterns to `instructions/failure-patterns.md`
- Extract cost tracking format to `instructions/cost-tracking.md`
- Create `protocols/` for the 6 workflow steps
- Add `eval/checklist.md` with max 6 binary criteria
- Cross-cutting rules (max 3 retries, no regex) stay in SKILL.md

**Step 2b**: Add eval/checklist.md to Tier B skills.
- create-literature-report: 5 criteria (hallucination rate, URL validity, citation format, cross-domain synthesis, Markov novelty)
- issue-creator: 5 criteria (YAML metadata, domain valid, priority label, citations linked, no duplicate)
- prd-update: 5 criteria (probability sum, no citation loss, DAG acyclic, author-year format, bibliography entry exists)
- overnight-runner: 4 criteria (heartbeat frequency, stall detection, cost logged, child success rate)
- kg-sync: 4 criteria (pdflatex compiles, idempotency, orphan detection, schema validation)
- knowledge-reviewer: 4 criteria (ERROR→fix/issue, WARN→non-blocking, pre-commit quick mode, exit code correct)
- planning-backlog: 4 criteria (priority assigned, PRD linked, project board updated, labels valid)
- sync-roadmap: 4 criteria (dates set, size assigned, estimate matches size, no overwrite)

**Step 2c**: Create shared `references/github-project-fields.md` for planning-backlog + sync-roadmap.

#### Session 3: Validation + Polish (~30 min)

**Step 3a**: Trigger testing. For each skill, verify:
- Triggers on 3 obvious prompts
- Does NOT trigger on 2 unrelated prompts
- Document in `eval/trigger-tests.md`

**Step 3b**: Run self-learning-iterative-coder on a small real TDD task. Verify no drift.

**Step 3c**: Run knowledge-reviewer in full mode. Verify all 4 agents work with upgraded frontmatter.

#### Session 4 (Optional): Autoresearch Loop

For the 3 strongly autoresearchable skills (issue-creator, planning-backlog, sync-roadmap):
- Define `eval/autoresearch-config.yaml` with test inputs + binary criteria
- Run each skill 5 times against test inputs
- Score against structural criteria (Python parsers, NOT LLM self-eval)
- If pass rate <90%, mutate ONE prompt element and re-run
- Max 10 mutations per skill
- Record baseline → final scores in `eval/scores.json`

### 12.3 What Gets Deferred (Not Cut — Deferred)

| Item | Trigger to Revisit |
|------|-------------------|
| SKILL-NETWORK.yaml manifest | When skill count exceeds 30 |
| examples/good/ and examples/bad/ | When a specific skill demonstrates voice/quality drift |
| openspec-explore restructure | When it exceeds 400 lines |
| Protocol decomposition for kg-sync, knowledge-reviewer | When individual steps need independent updates |
| Autoresearch meta-skill | After Session 4 validates the approach on 3 skills |
| Population-level telemetry for code-producing skills | After 20+ real TDD sessions collected |

### 12.4 Revised Success Criteria

- [ ] All 15 skills have SkillNet relations in frontmatter
- [ ] All 15 skills have enhanced descriptions with trigger + negative trigger phrases
- [ ] 2 skills (ralph-loop, self-learning-iterative-coder) restructured with instructions/
- [ ] All 10 non-trivial skills have eval/checklist.md with max 8 binary criteria
- [ ] eval/checklist.md criteria split: structural (Python-parseable) vs behavioral (LLM-judge)
- [ ] Trigger testing documented for all skills (3 positive + 2 negative prompts)
- [ ] No regression: every skill triggers correctly on existing use cases
- [ ] Baseline behavioral scores recorded for 3 autoresearchable skills

---

## 13. New Skill: `factorial-monitor` (Separate Effort)

### 13.1 Why a New Skill (Not an Upgrade)

The reviewers are unanimous: **Ralph Loop is a single-job lifecycle manager. Factorial
monitoring is a multi-job orchestrator.** These are different levels of abstraction.

| Concern | Ralph Loop | factorial-monitor |
|---------|-----------|-------------------|
| Scope | One SkyPilot job | N concurrent SkyPilot jobs |
| Diagnosis | Per-job log analysis | Cross-job aggregation by root cause |
| Fix strategy | Auto-fix + retry (3 max) | Batch fix + selective re-launch (2 max) |
| State tracking | JSONL event log | Factorial manifest (job → condition mapping) |
| Relationship | Inner loop | Outer loop (composes WITH Ralph) |

Ralph Loop's diagnostic functions (`analyze_logs()`) are **reused** by factorial-monitor.
The TDD skill's failure triage protocol (GATHER → CATEGORIZE → PLAN → FIX → VERIFY)
provides the **philosophy**, adapted from local tests to cloud jobs.

### 13.2 Five Non-Negotiable Rules (Anti-Pattern Prevention)

These rules prevent every known anti-pattern from the metalearning history:

**Rule F1: WAIT-FOR-TERMINAL — No Diagnosis Until All Jobs Are Done**

No error diagnosis, no code fixes, no re-launches until ALL factorial jobs have
reached terminal state (`SUCCEEDED`/`FAILED`/`FAILED_SETUP`/`CANCELLED`).

The ONLY permitted intervention during execution is `sky jobs cancel <id>` for
a job provably wasting money (infinite loop, disk full). "This job looks like
it will fail" is NOT grounds for cancellation — let it fail and capture the
full error.

**Kill-switch exception**: If 3+ jobs fail with the IDENTICAL error within 5
minutes AND remaining running jobs haven't passed the failure point → cancel
remaining jobs with the same configuration, begin batch diagnosis. Jobs with
different configurations continue.

Prevents: Panic Fixing (Anti-Pattern 3), premature whac-a-mole (Anti-Pattern 2).

**Rule F2: AGGREGATE-BEFORE-FIX — Batch Error Analysis Is Mandatory**

After all jobs reach terminal state, collect ALL failure logs and categorize
by root cause BEFORE writing any fix.

Output format:
```json
{
  "root_cause_id": {
    "description": "DVC_NO_GIT — .dvc not initialized in container",
    "affected_jobs": [3, 7, 11],
    "fix_strategy": "Add dvc init to Docker entrypoint",
    "affected_files": ["deployment/Dockerfile.train"],
    "confidence": "high"
  }
}
```

BANNED: Fixing the first failure you see. Re-launching without a written root
cause. "Probably transient" without evidence (spot preemption exit code or
identical job succeeded on zero-code-change retry).

Prevents: Silent Dismissal (Anti-Pattern 1), Whac-a-Mole (Anti-Pattern 2).

**Rule F3: REBUILD-BEFORE-RELAUNCH — Full Pipeline On Every Fix Cycle**

Every fix-relaunch cycle MUST execute this exact sequence:
1. Code fix with tests
2. `make test-staging` passes
3. Docker image rebuild + push with new tag
4. Update SkyPilot YAML to reference new image tag/digest
5. Re-launch ONLY the failed jobs

Skipping any step is BANNED. Re-launching with a stale Docker image is the
most expensive possible mistake — it wastes the full GPU cost of every
re-launched job.

Prevents: Docker Image Staleness (Anti-Pattern 9).

**Rule F4: MAX-TWO-CYCLES — Hard Stop After Two Fix-Relaunch Iterations**

A factorial run permits at most 2 fix-relaunch cycles (initial launch + 2
retries). If jobs still fail after the second cycle, STOP and present a full
diagnostic report to the user.

The report must include: all root causes found, all fixes attempted, cost
incurred so far, and a recommendation (redesign experiment, fix upstream
dependency, or authorize additional cycles with budget cap).

Prevents: Fix-Relaunch Infinite Loop (Anti-Pattern 6), Cost Overrun (Anti-Pattern 8).

**Rule F5: FACTORIAL-MANIFEST — Single Source of Truth for All Jobs**

Before launch, create `factorial_manifest.json` that records: experiment_id,
all factorial factors and their levels, the full job list with parameters,
and expected outputs.

During execution, update with: job_id, status, start/end time, cost,
MLflow run_id, failure category, relaunch_batch (0=original, 1=first retry,
2=second retry). This manifest is the SOLE authority on experiment state.
Re-launched jobs are linked to the original entry, not added as new entries.

Prevents: Partial Factorial Amnesia (Anti-Pattern 5).

### 13.3 Workflow

```
User: /factorial-monitor --config configs/experiment/debug_factorial.yaml

1. LAUNCH PHASE
   - Calls run_factorial.sh (or assumes already launched)
   - Records job_id → condition mapping in factorial_manifest.json
   - READ-ONLY from this point: no code changes while jobs run

2. MONITOR PHASE (polling loop, 60s interval)
   - sky jobs queue → parse all job statuses
   - Print live status table: | condition | status | duration |
   - For each newly-terminal failure: ralph_monitor.analyze_logs()
   - Continue until ALL jobs are terminal (Rule F1)

3. DIAGNOSE PHASE (all jobs terminal)
   - Group failures by root cause category (Rule F2)
   - Present single aggregated report with reviewer agents
   - If 0 failures → skip to REPORT

4. FIX PHASE (with reviewer agents)
   - For EACH root cause: reviewer agents plan batch fix strategy
   - If code fix needed → TDD loop (compose_with: self-learning-iterative-coder)
   - If config fix → edit YAML/env directly
   - make test-staging → Docker rebuild → push (Rule F3)
   - Commit all fixes in ONE batch commit

5. RELAUNCH PHASE (max 2 cycles, Rule F4)
   - Generate filtered re-launch command (only failed conditions)
   - Execute and return to MONITOR PHASE
   - Update manifest with relaunch_batch number

6. REPORT (when all SUCCEEDED or retry budget exhausted)
   - Summary: X succeeded, Y failed (root causes: A, B, C)
   - Cost: total $ across all cycles
   - If unrecoverable failures remain → compose_with: issue-creator
   - Save to outputs/factorial_run_<experiment_id>.jsonl
```

### 13.4 SkillNet Relations

```yaml
metadata:
  category: operations
  tags: [skypilot, monitoring, factorial, cloud, batch]
  relations:
    compose_with:
      - ralph-loop                      # Per-job diagnosis via analyze_logs()
      - self-learning-iterative-coder   # TDD loop when fix requires code changes
      - issue-creator                   # Unrecoverable failures become GitHub issues
    depend_on:
      - ralph-loop                      # Cannot diagnose without failure pattern library
    similar_to: []
    belong_to:
      - overnight-runner                # Factorial runs are a type of batch execution
```

### 13.5 Binary Eval Criteria

1. **All-jobs-tracked**: Every condition in the factorial grid has a corresponding
   job_id in the manifest. YES/NO.
2. **Failures-aggregated-not-serial**: Multiple failures → ONE aggregated report
   grouped by root cause (not separate per-job reports). YES/NO.
3. **No-silent-dismiss**: Every failed job resulted in (a) fixed + relaunched,
   (b) GitHub issue created, or (c) explicitly reported to user. YES/NO.
4. **Selective-relaunch**: Re-launch command launches ONLY failed conditions,
   not the entire grid. YES/NO.
5. **Cost-logged**: Total estimated cost recorded in manifest. YES/NO.

### 13.6 Implementation Plan

**Separate feature branch** (`feat/factorial-monitor`), **not** part of the
skills-upgrade work. Created "born upgraded" using the canonical folder structure
from Section 4.1:

```
factorial-monitor/
  SKILL.md                    # Orchestrator: 5-step workflow + file pointers
  instructions/
    rules.md                  # Rules F1-F5 (non-negotiable)
    anti-patterns.md          # 9 anti-patterns from reviewer analysis
  protocols/
    launch.md                 # Launch + manifest creation
    monitor.md                # Polling loop + status table
    diagnose.md               # Aggregation + categorization
    fix.md                    # Reviewer-backed batch fix strategy
    relaunch.md               # Selective re-launch protocol
    report.md                 # Final report generation
  eval/
    checklist.md              # 5 binary criteria above
  templates/
    factorial-manifest.json   # Schema for experiment state tracking
```

**MVP artifacts** (minimum for first debug run):
1. `factorial-monitor/SKILL.md` (~100 lines, orchestrator)
2. `factorial-monitor/instructions/rules.md` (Rules F1-F5)
3. `scripts/monitor_factorial.py` (~150 lines, polling + aggregation)
4. `--relaunch-failed` mode added to `run_factorial.sh`

---

## Appendix A: User's Original Prompt (Verbatim)

> While I have full prod tests running on another session, could we branch from main fix/skills-upgrade which will make sure that all Skills are optimized for the latest guide from Anthropic on how to best structure Skills, see e.g. /home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/agentic-development/resources/anthropic-2025-building-skills-claude.md (https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf) https://x.com/JJEnglert/status/2034329261960475086 : "I just finished restructuring all my skills based on
> @AnthropicAI
>  's latest recommendations for how to build them.
>
> Here's what's different, why it matters, and how you can do it too.
>
> Our #1 AI newsletter (https://tenex.co/ultrathink) skill used to be one long file. Voice rules, examples, subject line logic — all crammed into one document.
>
> It worked pretty well. But Claude would drift every now and again. It would nail the tone in one section and lose it in another. And every time I wanted to fix one thing, I risked breaking something else.
>
> The problem isn't Claude. It's how you feed it information.
>
> When everything lives in one file, Claude tries to hold it all at once. Rules compete with each other. Examples get buried. The longer the file, the worse the output.
>
> The fix: break one file into a folder of specialized files.
>
> 1. SKILL.md is the boss. It doesn't contain any rules itself — it just tells Claude which files to read and when. Like a playbook.
>
> 2. instructions/ holds the actual rules. One file for voice. One for subject lines. One for section-specific guidance. They never compete because Claude only loads what it needs for the current step.
>
> 3. examples/ is where Claude learns what good and bad look like. Good examples from real shipped work. Bad examples showing 12 common AI writing patterns to avoid. Claude reads these right before writing so the voice is fresh.
>
> 4. eval/ is the quality check. After every draft, two things run automatically:
>
> 5. A checklist with 9 pass/fail tests
>
> 6. An advisory board — 3 AI personas (Exec, Builder, Lurker) review the draft in parallel and give feedback
>
> The workflow it runs:
>
> - Load the rules
> - Gather inputs
> - Read relevant examples
> - Write the draft
> - Run the checklist + eval (9 tests)
> - Run the advisory board (3 reviewers in parallel)
> - Revise based on feedback
> - Save and queue for human preview / review
>
> Every step loads only what it needs, when it needs it.
>
> Want to restructure your own skills? Paste this into Claude Code:
>
> --
>
> I want to restructure my Claude Code skill files. Right now my skills are single files that try to do everything. I want to break them into a folder system like this:
>
> SKILL.md — the orchestrator that tells Claude which files to read and when
>
> instructions/ — one file per set of rules (voice, formatting, section guides)
>
> examples/good/ — annotated examples of great output
>
> examples/bad/ — anti-patterns to avoid
>
> eval/checklist.md — pass/fail tests that run after every draft
>
> eval/advisory-board.md — AI reviewer personas that evaluate drafts in parallel
>
> templates/ — output format templates
>
> Phase 1: Read my existing skill files and identify every distinct concern (voice rules, formatting, examples, evaluation criteria, templates). Show me the audit before building anything.
>
> Phase 2: Create the folder structure and move each concern into its own file.
>
> Phase 3: Build SKILL.md as the orchestrator — it should contain no rules, just the step-by-step workflow pointing to the right files.
>
> Phase 4: Build the eval layer with a checklist and 2-3 reviewer personas.
>
> Phase 5: Run the skill on a real task and verify everything works.
>
> Start with Phase 1."  for every Skill that we have now in this repo: /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/skills/create-literature-report
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/skills/fetch-docs
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/skills/issue-creator
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/skills/kg-sync
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/skills/knowledge-reviewer
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/skills/openspec-apply-change
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/skills/openspec-archive-change
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/skills/openspec-explore
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/skills/openspec-propose
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/skills/overnight-runner
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/skills/planning-backlog
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/skills/prd-update
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/skills/ralph-loop
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/skills/self-learning-iterative-coder
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/skills/sync-roadmap! See also this recent paper on Skills as a graph for some further ideas: "SkillNet is the first paper I've seen that treats agent skills as a network, a three-layer ontology that turns isolated skill files into a structured, composable network.
>
> Externalizing knowledge into files isn't enough. You also need to know how those files relate to each other.
>
> Layer 1 is a Skill Taxonomy. Ten top-level categories (Development, AIGC, Research, Science, Business, Testing, Productivity, Security, Lifestyle, Other), each broken into fine-grained tags: frontend, python, llm, physics, biology, plotting, debugging. This is the semantic skeleton. It answers "what domain does this skill belong to?"
>
> Layer 2 is the Skill Relation Graph. This is where SkillNet diverges from other skill repositories. Tags from Layer 1 get instantiated into specific skill entities (Matplotlib, Playwright, kegg-database, gget). Then four typed relations define how skills connect:
> > similar_to: two skills do the same thing. Matplotlib and Seaborn both plot. Enables redundancy detection.
> > belong_to: a skill is a sub-component of a larger workflow. Captures hierarchy and abstraction.
> > compose_with: two skills chain together. One's output feeds the other's input. This is the relation that enables automatic workflow generation.
> > depend_on: a skill can't run without a prerequisite. Enables safe execution by resolving the dependency graph before running anything.
>
> These four relations form a directed, typed multi-relational graph. Nodes are skills, edges are typed relationships. And the graph is dynamic. As new skills enter the system, LLMs infer relations from their metadata.
>
> Layer 3 is the Skill Package Library. Individual skills bundled into deployable packages. A data-science-visualization package contains Matplotlib, Seaborn, Plotly, GeoPandas with their relations pre-configured. You install a package, you get a coherent set of skills that already know how to compose with each other.
>
> This is a good example of what comes after a flat package manager.
>
> The paper also (you can test here http://skillnet.openkg.cn) has a science case on a real research workflow: identifying disease-associated genes and candidate therapeutic targets from large-scale biological data.
>
> Without encoded relations, the agent figures out the research pipeline from scratch every time. With them, it receives a pre-structured execution plan. The agent still reasons about which genes to focus on and which pathways to investigate. But the pipeline architecture is given.
>
> So the skill metadata is actually doing routing work too. The metadata encodes the judgment a domain expert would make when choosing between tools.
>
> I also like this framing from the paper: Skills are how memory becomes executable and workflows become flexible.
>
> While the network effect and layered architecture is actually useful today, they also acknowledge this: "Low-frequency or highly tacit abilities are difficult to capture, particularly when they resist explicit linguistic description."
>
> From my short research career, I'd say the hardest parts are hypothesis generation, experimental design judgment, and interpreting ambiguous results etc.
>
> SkillNet handles the structured pipeline well; fetch data → analyze → validate → report. It doesn't handle the creative work where a scientist's (not just in science but in any white-collar field) intuition drives what's worth investigating in the first place.
>
> Skills encode "how to run the analysis." They don't encode "what's worth analyzing." That gap is where domain expertise still sits." from https://github.com/zjunlp/SkillNet https://arxiv.org/html/2603.04448v1 . Plan with multiple iterations with iteration agents to /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/skills/skills-upgrade-plan-march2026.md on how to upgrade every Skills to be optimally structured according to the current best knowledge!

---

## Appendix B: Source References

### B.1 Anthropic — "The Complete Guide to Building Skills for Claude" (2025)

**URL**: https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf
**Local copy**: `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/agentic-development/resources/anthropic-2025-building-skills-claude.md`

**Key takeaways used in this plan**:
- Three-level progressive disclosure (frontmatter → SKILL.md body → linked files)
- File structure: SKILL.md + scripts/ + references/ + assets/
- Frontmatter: name (kebab-case), description (what + when/triggers), metadata
- Keep SKILL.md under 5,000 words; move details to references/
- 5 workflow patterns: Sequential, Multi-MCP coordination, Iterative refinement,
  Context-aware tool selection, Domain-specific intelligence
- Testing: triggering tests, functional tests, performance comparison
- Description must include BOTH what the skill does AND when to use it (trigger conditions)
- Negative triggers ("Do NOT use for X") to prevent over-triggering

### B.2 SkillNet — "Create, Evaluate, and Connect AI Skills"

**Paper**: arXiv:2603.04448v1 (Feb 2026)
**URLs**: https://github.com/zjunlp/SkillNet, https://arxiv.org/html/2603.04448v1
**Demo**: http://skillnet.openkg.cn

**Key takeaways used in this plan**:
- Three-layer ontology: Skill Taxonomy → Skill Relation Graph → Skill Package Library
- Four typed relations: similar_to, belong_to, compose_with, depend_on
- Metadata-first routing: scan lightweight descriptions first, load full instructions on match
- Five-dimensional quality evaluation: Safety, Completeness, Executability, Maintainability, Cost-awareness
- Skill granularity should favor atomic units connected via composition
- "Skills are how memory becomes executable and workflows become flexible"
- Limitation acknowledged: tacit knowledge (hypothesis generation, experimental design judgment) resists capture

### B.3 Ultrathink v2 Pattern (JJ Englert)

**URL**: https://x.com/JJEnglert/status/2034329261960475086
**Newsletter**: https://tenex.co/ultrathink

**Key takeaways used in this plan**:
- SKILL.md = orchestrator only (no rules, just file pointers + workflow steps)
- instructions/ for domain rules (one file per concern, rules never compete)
- examples/good/ and examples/bad/ for in-context learning (read right before writing)
- eval/checklist.md for pass/fail tests after every execution
- eval/advisory-board.md for parallel AI reviewer personas
- "When everything lives in one file, Claude tries to hold it all at once. Rules compete."
- "Every step loads only what it needs, when it needs it."

### B.4 Autoresearch Skill (Ole Lehmann / Karpathy Method)

**URL**: https://github.com/olelehmann100kMRR/autoresearch-skill
**Thread**: https://x.com/itsolelehmann/status/2033919415771713715
**Origin**: Andrej Karpathy's "autoresearch" method applied to Claude Code skills

**Architecture**: Pure prompt-driven specification (SKILL.md 330 lines + eval-guide.md 122 lines).
No code, no framework. The LLM is both executor AND evaluator.

**The loop**:
1. Gather: target skill path, 3-5 test inputs, 3-6 binary eval criteria
2. Read the skill fully (SKILL.md + references/)
3. Build eval suite: binary yes/no questions with pass/fail conditions
4. Establish baseline (Experiment #0): run skill AS-IS N times, score, backup as SKILL.md.baseline
5. Experiment loop (autonomous):
   - Analyze which evals fail most
   - Form hypothesis: pick ONE thing to change
   - Make the change (one targeted mutation)
   - Run N times, score all outputs
   - Score improved → KEEP. Same or worse → DISCARD (revert).
   - Log in results.tsv + results.json
   - Repeat until 95%+ pass rate for 3 consecutive experiments
6. Deliver: changelog, improved SKILL.md, scores, baseline backup

**Good mutations**: Add specific rule for common failure, reword ambiguous instruction,
add anti-pattern, move buried instruction higher, add/improve examples.
**Bad mutations**: Rewriting entire skill, adding 10 rules at once, vague instructions.

**Dashboard**: Self-contained HTML with Chart.js, auto-refreshes from results.json every 10s.
Score progression chart, per-eval breakdown, experiment table with green/red/blue bars.

**Eval design rules** (from eval-guide.md):
- Every eval MUST be binary yes/no (never scales)
- 3-6 criteria is the sweet spot (>6 = skill games them)
- 3-question test: (1) would two agents agree? (2) could skill game it? (3) does user care?

**Key takeaways used in this plan**:
- Self-improving skills are possible but only for skills with measurable, deterministic outputs
- Binary eval criteria are strictly superior to subjective checklists
- One mutation at a time prevents confounding
- Baseline measurement before ANY changes is mandatory
- Only 3 of 15 MLOps skills qualify for fully autonomous autoresearch

### B.5 User's Second Prompt — Autoresearch (Verbatim)

> Double-check on your plan and self-reflect on this plan if this makes sense, as self-improving Skills? https://x.com/itsolelehmann/status/2033919415771713715 "Your Claude skills probably fail 30% of the time and you don't even notice.
> I built a method that auto-improves any skill on autopilot, and in this article I'm going to show you exactly how to run it yourself.
> You kick it off, and the agent tests and refines the skill over and over without you touching anything.
> My landing page copy skill went from passing its quality checks 56% of the time to 92%. With zero manual work at all.
> The agent just kept testing and tightening the prompt on its own.
> Here's the method and the exact skill I built so you can run it yourself:
> P.S. If you want more AI  workflows like this one delivered to your inbox every week, join 34k readers getting them free: aisolo.beehiiv.com/subscribe
> Where this comes from
> Andrej Karpathy (co-founder of OpenAI, former head of AI at Tesla, guy who coined "vibe coding") released a method called autoresearch.
> The idea is simple: instead of you manually improving something, you let an AI agent do it for you in a loop.
> It tries a small change. Checks if the result got better. Keeps it if it did, throws it out if it didn't.
> Then it does it again. And again.
> He used it for machine learning code. But the method works on anything you can measure and improve.
> Including the skills you've built in Claude.
> I took his method and turned it into a skill that works in both Claude Code and Cowork. I just run it on any other skill in my setup.
> I say "run autoresearch on my landing page skill" and it handles the whole thing.
> How one loop auto-improves your skills
> Think of it like this.
> You have a recipe that turns out great 7 out of 10 times. The other 3 times, something's off. Maybe the sauce is bland, maybe the seasoning is wrong.
> Instead of rewriting the whole recipe from scratch, you change one ingredient. You cook it 10 times with that change.
> Did it get better? Keep the change.
> Did it get worse? Put the old ingredient back.
> Then you change the next thing. Cook 10 more times. Better or worse? Keep or revert.
> After 50 rounds of this, your recipe works 9.5 out of 10 times.
> That's exactly what autoresearch does to your skills.
> The "recipe" is your skill prompt.
> The "cooking" is running the skill.
> The "tasting" is scoring the output.
> The only thing you need to provide is the scoring criteria.
> The checklist that tells the agent exactly what 'good' means
> You give the agent a simple checklist of what "good" looks like. That's your only job in this whole process.
> You do it with a simple checklist of yes/no questions.
> Each question checks one specific thing about the output. Pass or fail. That's it.
> The agent uses this checklist to score every output, and those scores tell it whether its changes are helping or hurting.
> Think of it like a teacher grading a paper with a checklist.
> But instead of "rate the writing quality 1-10" (which is vague and different every time), each item on the checklist is a clear yes or no:
> Did the student include a thesis statement? Yes or no.
> Is every source cited? Yes or no.
> Is it under 5 pages? Yes or no.
> You can grade 100 papers with that checklist and get consistent results every time.
> Same idea here. For a landing page copy skill, your checklist might look like:
> "Does the headline include a specific number or result?" (catches vague headlines like "Grow Your Business")
> "Is the copy free of buzzwords like 'revolutionary,' 'synergy,' 'cutting-edge,' 'next-level'?"
> "Does the CTA use a specific verb phrase?" (catches weak CTAs like "Learn More" or "Click Here")
> "Does the first line call out a specific pain point?" (catches generic openers like "In today's fast-paced world...")
> "Is the total copy under 150 words?" (catches bloated pages that lose the reader)
> You don't need to figure these out on your own. When you start the autoresearch, the agent walks you through it.
> It asks what good looks like, helps you turn your vibes into specific yes/no questions, and even offers to pull from existing style guides if you have them.
> 3-6 questions is the sweet spot. More than that and the skill starts gaming the checklist (like a student who memorizes the answers without understanding the material).
> Here's how to run it
> Step 1: Download the skill. Grab it here. Drop it into your skills folder in Claude Code or Cowork.
> Step 2: Pick a skill to improve. Say "run autoresearch on my [skill name] skill." Pick the one that annoys you most. The one where you get a great output half the time and garbage the other half.
> Step 3: The agent asks you 3 things. Which skill to optimize. What test inputs to use (like "write landing page copy for an AI productivity tool"). And what your checklist questions are.
> Step 4: It runs your skill and shows you your starting score. This is the baseline. My landing page skill started at 56%. Vague headlines, buzzword soup, weak CTAs. More than half the checks were failing.
> Step 5: It opens a live dashboard in your browser. Score chart going up over time. Pass/fail breakdown for each checklist question. A log of every change it tried. Auto-refreshes every 10 seconds.
> Step 6: Walk away. The agent enters the loop. Analyzes what's failing. Makes one small change to the skill prompt. Tests again. Keeps the change if the score goes up, undoes it if it goes down.
> Then does it again. And again. It keeps going autonomously until you stop it or it hits 95%+ three times in a row.
> You can watch the dashboard or walk away entirely. It runs without you. And it saves the improved version as a separate file, so your original skill stays untouched.
> What happened to my landing page skill
> I ran it on my landing page copy skill. Here's what came back:
> 56% → 92%. 4 rounds of changes. 3 kept, 1 undone.
> Here's what the agent actually changed in my skill prompt:
> Added a specific rule for the most common failure: "Your headline must include a specific number or result. Never use vague promises like 'Transform Your Business.'"
> Added a banned buzzwords list: "NEVER use: revolutionary, cutting-edge, synergy, next-level, game-changing, leverage, unlock, transform."
> Added a worked example of a strong landing page section with the pain point opener and CTA highlighted, so the skill could see what good looks like instead of guessing.
> Tried a tighter word count, undid it because the copy got too thin and the CTA suffered. (The system catches changes that seem like improvements in isolation but hurt the overall output.)
> When it was done, I got:
> The improved skill, saved separately (the original stays untouched in case you want to revert)
> A results log showing every round's score
> A changelog explaining every change that was tried, why the agent tried it, and whether it helped
> A backup of my original skill in case I ever want to go back
> That changelog is probably the most valuable piece. It's a complete record of what works and what doesn't for that specific skill.
> When smarter models come out down the road, you hand them that changelog and they pick up right where the last agent left off.
> This works on way more than skills
> The method works on anything you can score.
> Website speed: One person ran this on page load time. Changed one thing, measured the speed, kept or reverted. Went from 1100ms to 67ms in 67 rounds.
> Cold outreach: Define your checklist: "Does it mention the prospect's company? Is it under 75 words? Does it end with a specific question?" Let the agent run 50 variations.
> Newsletter intros: "Does the opener include a personal detail?" and "Is it free of cliche phrases?" Let the agent tighten your writing on autopilot.
> Any prompt you use repeatedly
> If you can score it, you can autoresearch it.
> Go run it
> Pick your worst-performing skill. Start the autoresearch. Come back to something that actually works." -> https://github.com/olelehmann100kMRR/autoresearch-skill . Reflect on this with reviewer agents then!

### B.6 User's Third Prompt — Factorial Monitor (Verbatim)

> And make sure that the two Skills mentioned here [...] are suited for these types
> of jobs or do we need to create a new meta-Skill now /monitor-skypilot-run that knows
> how to use the Ralph Loop for infrastructure monitoring and self-learning TDD Skill
> when running the experiments with Skypilot [...] So that Claude Code knows what to do
> when there is a bug. As in it should never silently kick can down the road
> (.claude/metalearning/2026-03-07-silent-existing-failures.md) and Claude should never
> do panic-fixing (.claude/metalearning/2026-03-18-whac-a-mole-serial-failure-fixing.md)
> but rather let the tests aggregate all the errors and address them in batch by creating
> a plan with reviewer agents on how to address ALL the emerged issues as a whole leading
> to better quality fixes and shorter debugging times!

### B.7 Current Repo Skill Audit (2026-03-19)

**Scope**: All 15 skills in `.claude/skills/` (+ 1 planned: factorial-monitor)
**Totals**: 106 files, 2,505 SKILL.md lines
**Distribution**: 7 monolithic (single SKILL.md), 8 folder-based
**Gold standard**: self-learning-iterative-coder (17 files, v3.0.0)
**Largest monolith**: openspec-explore (288 lines)
**Smallest skill**: fetch-docs (63 lines, 1 file)
