# Context Management & Plan Creation Upgrade — Comprehensive Plan

**Date**: 2026-03-22
**Issue**: #906 (P0-CRITICAL — context compounding failure)
**Scope**: Entire `.claude/` infrastructure + planning SOP

---

## Problem Statement

MinIVess MLOps has built a sophisticated 6-layer knowledge architecture (205 files,
90 metalearning docs, 17 skills, 75+ KG decision nodes). Despite this investment,
**knowledge does not compound across sessions**. The same mistakes recur 6-8 times
because context loading is ad-hoc, metalearning is read-only, and decisions are
scattered across files with no automated enforcement.

The cost: hours of user frustration, wrong implementations rebuilt from scratch,
and — worst case — wrong metalearning docs that POISON future sessions.

---

## Current State Assessment

### What We Have (Strengths)
| Component | Count | Quality | Automation |
|-----------|-------|---------|------------|
| Constitutional rules (L0) | 5 + CLAUDE.md (30 rules) | Excellent | None |
| Memory topic files (L1) | 44 | Good | Manual GC |
| Navigator + domains (L2) | 11 domains | Good | Manual routing |
| KG decision nodes (L3) | 75+ Bayesian nodes | Excellent | Manual PRD updates |
| OpenSpec specs (L4) | Emerging | New | Skill-driven |
| Metalearning docs | 90 failure patterns | Comprehensive | Zero queryability |
| Production skills | 17 skills | Production-grade | Manual activation |

### What's Missing (Gaps)
1. **No automated context loading** — Claude must manually read KG/metalearning each session
2. **No searchable metalearning** — 90 docs in flat filesystem, no index, no retrieval
3. **No contradiction detection** — metalearning can contradict CLAUDE.md or other docs
4. **No code structural graph** — blind to import chains, blast radius, test coverage
5. **No planning SOP** — plan creation is ad-hoc, no mandatory checklist
6. **No decision deduplication** — same decision in MEMORY.md, metalearning, KG, CLAUDE.md
7. **AskUserQuestion abuse** — re-asks decided questions instead of reading docs

---

## Architecture: 3 Upgrade Pillars

### Pillar 1: Structural Code Graph (code-review-graph MCP)

**What**: Install `code-review-graph` as an MCP server for Tree-sitter-based
structural analysis of the codebase.

**Value**:
- **Blast radius analysis** — When changing a file, trace all callers, importers,
  and tests affected (2-hop BFS through import/call graph)
- **Token efficiency** — 6.8x fewer tokens on reviews (read 15 files not 2900)
- **Test coverage mapping** — `TESTED_BY` edges show which tests cover which functions
- **Complexity hotspots** — Find oversized functions needing refactoring
- **Incremental updates** — PostToolUse hooks keep graph current during sessions
- **Semantic search** — Natural language queries for code discovery

**Limitation**: Static analysis only — blind to Hydra config dispatch, registry patterns,
and YAML-to-code wiring. This is significant for MinIVess's config-driven architecture.

**Mitigation**: Supplement with a custom config-aware layer (Pillar 2).

**Reference**: https://github.com/tirth8205/code-review-graph (MIT, MCP-compatible,
v1.8.4, 14 languages, SQLite storage)

**Tasks**:
- [ ] Install code-review-graph: `pip install code-review-graph && code-review-graph install`
- [ ] Initial build: `code-review-graph build` (expect ~30-60s for 2900 files)
- [ ] Configure PostToolUse hooks for auto-update
- [ ] Test MCP tools from Claude Code session
- [ ] Evaluate token reduction on a real code review task
- [ ] Document in `.claude/rules/` as standard tool

### Pillar 2: Knowledge Graph Automation (Config-Aware Layer)

**What**: Extend the existing 6-layer KG with automated context loading, searchable
metalearning, and config-to-code edge tracing.

#### 2A: Metalearning Search Index

Currently 90 docs in a flat directory. Need an index for retrieval.

**Approach**: DuckDB full-text search over metalearning docs (we already have DuckDB).

```sql
-- Metalearning index (rebuilt on session start)
CREATE TABLE metalearning_index AS
SELECT
    filename,
    content,
    -- Extract YAML frontmatter fields
    regexp_extract(content, 'Severity: (.+)', 1) as severity,
    regexp_extract(content, 'Date: (.+)', 1) as date,
    -- Full-text search
    fts_main_metalearning_index.match_bm25(content, ?) as relevance
FROM read_text('.claude/metalearning/*.md');
```

**Tasks**:
- [ ] Create `scripts/build_metalearning_index.py` using DuckDB FTS
- [ ] Add pre-session hook that loads top-5 relevant docs based on task keywords
- [ ] Add `/search-metalearning` skill for ad-hoc queries
- [ ] Deduplicate docs that say the same thing (90 → target ~50)

#### 2B: Config-to-Code Edge Graph

code-review-graph is blind to YAML→Python dispatch. We need a supplementary graph.

**Approach**: Parse Hydra config groups and map them to Python entry points.

```
configs/model/dynunet.yaml → ModelFamily("dynunet") → build_adapter() → DynUNetAdapter
configs/post_training/swag.yaml → method: "swag" → SWAGPlugin
configs/experiment/debug_factorial.yaml → factors → run_factorial.sh conditions
```

**Tasks**:
- [ ] Create `scripts/build_config_graph.py` using `yaml.safe_load()` + `ast.parse()`
- [ ] Map each config group to its Python consumer
- [ ] Store in same SQLite as code-review-graph (or separate DuckDB)
- [ ] Expose via MCP tool: `get_config_impact(config_file)` → affected Python files

#### 2C: Decision Deduplication & Single-Source Registry

Currently decisions live in 4 places: CLAUDE.md, MEMORY.md, metalearning, KG.
Need a single authoritative registry with links.

**Approach**: Create `knowledge-graph/decisions/registry.yaml` — every decided question
gets ONE entry with status + source-of-truth file path.

```yaml
decisions:
  post_training_methods:
    question: "What are the factorial post-training method levels?"
    answer: "none, swag"
    decided: 2026-03-21
    source: ".claude/metalearning/2026-03-22-wrong-metalearning-doc-failure-mode.md"
    referenced_by:
      - configs/post_training/none.yaml
      - configs/post_training/swag.yaml
      - docs/planning/training-and-post-training-into-two-subflows-under-one-flow.md
    DO_NOT_RE_ASK: true

  debug_scope:
    question: "What differs between debug and production runs?"
    answer: "Only 3 things: 1 fold (not 3), 2 epochs (not 50), half data"
    decided: 2026-03-19
    source: "CLAUDE.md Rule 27"
    referenced_by:
      - configs/experiment/debug_factorial.yaml
      - .claude/metalearning/2026-03-19-debug-run-is-full-production-no-shortcuts.md
    DO_NOT_RE_ASK: true
```

**Tasks**:
- [ ] Create decision registry YAML with all "already decided" items
- [ ] Add pre-AskUserQuestion hook: check registry before asking
- [ ] Add `DO_NOT_RE_ASK` flag for decisions with user-confirmed answers
- [ ] Audit all 44 MEMORY.md topic files against registry for deduplication

### Pillar 3: Planning SOP (Mandatory Process)

**What**: Formalize plan creation as a Standard Operating Procedure with mandatory
steps, not ad-hoc question dumps.

#### 3A: Pre-Planning Context Load (Mandatory)

Before ANY plan is created, the following MUST be loaded IN ORDER:

```
Step 1: Read knowledge-graph/navigator.yaml
        → Route to relevant domain(s)

Step 2: Read relevant domain YAML(s)
        → Load resolved decisions (posterior >= 0.80)

Step 3: Search metalearning for task keywords
        → Load top-5 relevant failure patterns

Step 4: Check decision registry
        → Identify questions that are ALREADY DECIDED

Step 5: Read MEMORY.md
        → Check for prior session decisions on this topic

Step 6: Read the source-of-truth document
        → e.g., pre-gcp-master-plan.xml for factorial
```

**Implementation**: Create `/plan-context-load` skill that automates steps 1-6
and presents a summary before any planning begins.

#### 3B: Interactive Questionnaire Protocol

When the user manually invokes plan creation:

1. **State what I THINK** — Present current understanding from context load
2. **Highlight uncertainties** — Only ask about things NOT in decision registry
3. **Max 4 questions per round** — Use AskUserQuestion, never wall-of-text
4. **NEVER re-ask decided questions** — Check registry first
5. **Show provenance** — "I found this in metalearning doc X, is it still current?"

#### 3C: Post-Planning Validation

After plan is written:

1. **Contradiction check** — Cross-reference against CLAUDE.md, metalearning, KG
2. **Decision capture** — Record any NEW decisions in registry
3. **Metalearning update** — If plan changes prior understanding, update docs
4. **NEVER write metalearning in panic** — Wait for user confirmation first

#### 3D: Plan File Standards

All planning documents must include:

```markdown
---
source_of_truth: <path to authoritative doc>
decisions_referenced: [list of registry keys]
contradictions_checked: [list of docs cross-referenced]
context_loaded:
  - navigator.yaml
  - domains/training.yaml
  - metalearning/2026-03-20-full-factorial-is-not-24-cells.md
---
```

**Tasks**:
- [ ] Create `/plan-context-load` skill
- [ ] Create `.claude/rules/planning-sop.md` with mandatory process
- [ ] Add planning frontmatter template
- [ ] Create decision registry with all known "DO_NOT_RE_ASK" items
- [ ] Add PostToolUse hook for AskUserQuestion: check registry first

---

## Implementation Phases

### Phase 0: Quick Wins — COMPLETE
- [x] Write wrong-metalearning-doc metalearning
- [x] Write debug-equals-production metalearning
- [x] Create P0 issue (#906)
- [x] Write context-compounding-and-learning-repo-plan.md
- [x] Create decision registry YAML (10 decided questions)
- [x] Create `.claude/rules/planning-sop.md`

### Phase 1: Code Graph Foundation — COMPLETE
- [x] Install code-review-graph MCP server (v1.8.4)
- [x] Initial build: 1105 files, 12729 nodes, 85399 edges
- [x] Configure PostToolUse hooks (auto-update on Write/Edit)
- [x] Graph DB verified: SQLite with 4 tables (nodes, edges, metadata, sqlite_sequence)

### Phase 2: Metalearning Search — COMPLETE
- [x] Build DuckDB FTS index over 90 metalearning docs (`scripts/build_metalearning_index.py`)
- [x] Create `/search-metalearning` skill
- [ ] Deduplicate redundant metalearning docs (90 → ~50) — deferred, editorial work
- [x] Index auto-rebuilds when no query given

### Phase 3: Config Graph Extension — COMPLETE
- [x] Build config-to-code edge tracer (`scripts/build_config_graph.py`)
- [x] Map all Hydra config groups: 97 YAML files, 624 edges to Python consumers
- [x] Store in DuckDB at `.claude/config_graph.duckdb`

### Phase 4: Planning SOP Enforcement — COMPLETE
- [x] Create `/plan-context-load` skill (6-step mandatory process)
- [x] Add PreToolUse hook for AskUserQuestion: registry check reminder
- [x] Create plan file frontmatter template (`.claude/templates/plan-frontmatter.md`)
- [ ] Test full SOP on a real planning task — deferred to next planning session

### Phase 5: Analytics & Dashboards — COMPLETE
- [x] Metalearning violation frequency dashboard (`scripts/context_analytics.py --violations`)
- [x] Memory churn metrics (`scripts/context_analytics.py --memory`)
- [x] Decision registry coverage report (`scripts/context_analytics.py --registry`)
- [x] Config graph coverage report (`scripts/context_analytics.py --config`)

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Factorial misunderstandings per session | 1-3 | 0 |
| "Already decided" questions re-asked | 2-4 per session | 0 |
| Token usage per code review | ~13,000 | ~2,000 (6.8x reduction) |
| Metalearning docs searchable | 0% (flat files) | 100% (FTS indexed) |
| Decisions in registry | 0 | 50+ |
| Context loading before planning | Ad-hoc | 100% mandatory |
| Wrong metalearning docs written | 1 (this session) | 0 |

---

## Cross-References

- Issue #906 (context compounding failure)
- `docs/planning/context-compounding-and-learning-repo-plan.md` (prevention plan)
- `.claude/metalearning/2026-03-22-wrong-metalearning-doc-failure-mode.md`
- `.claude/metalearning/2026-03-22-debug-equals-production-8th-violation.md`
- `knowledge-graph/navigator.yaml` (domain routing)
- https://github.com/tirth8205/code-review-graph (structural code graph MCP)

---

## Appendix A: User's Seed Description (Verbatim)

> Let me now address your question about improving the plan creation harness. This is a great meta-question —
> planning how to plan better.
>
> The core problem is clear from this session: I keep re-deriving wrong conclusions because context loading
> is ad-hoc rather than systematic. Here's what I think a proper SOP should look like:
>
> Plan Creation SOP (Progressive Disclosure)
>
> 1. Mandatory context loading — Before ANY plan is created, load in order:
>   - knowledge-graph/navigator.yaml → route to relevant domains
>   - Relevant metalearning docs (search .claude/metalearning/ for keywords)
>   - MEMORY.md → check for prior decisions on this topic
>   - The source-of-truth document (e.g., pre-gcp-master-plan.xml for factorial)
> 2. Interactive questionnaire (always) — When you invoke plan creation (not auto-pilot):
>   - Present what I THINK the current state is (from context loading)
>   - Ask you to confirm or correct before proceeding
>   - Max 4 questions per round, specific to decisions that haven't been made yet
>   - NEVER re-ask decisions documented in metalearning/MEMORY.md
> 3. Contradiction detection — Before writing any planning artifact:
>   - Cross-check against at least 3 existing docs
>   - If the plan contradicts a metalearning doc, STOP and flag
>   - If the plan contradicts CLAUDE.md rules, STOP and flag
> 4. Metalearning checkpoint — After plan is written:
>   - Check if any NEW decisions were made that should be persisted
>   - Update existing metalearning if the plan changes prior decisions
>   - NEVER write metalearning in a panic/overcorrection

## Appendix B: code-review-graph Reference

**Repository**: https://github.com/tirth8205/code-review-graph
**Architecture**: Tree-sitter AST → SQLite graph (nodes + edges) → BFS blast radius
**Integration**: MCP server (stdio), 8 tools, 3 skills, PostToolUse hooks
**Key benefit**: 6.8x token reduction on code reviews
**Key limitation**: Static analysis only — blind to YAML/config-driven dispatch
**Mitigation**: Supplement with config-to-code edge graph (Pillar 2, Phase 3)

LinkedIn post reference: "If you're using Claude Code on a large project, read this.
There's a tool that just went open source that solves the biggest pain point nobody
talks about..." — highlighting code-review-graph as the missing context layer for
AI coding tools. The insight: "Models aren't the bottleneck anymore. Context is."
