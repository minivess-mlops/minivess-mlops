---
title: "Claude Harness Improvement Plan: KG + CLAUDE.md + Skills Evals"
date: 2026-03-16
status: reviewed-v1
branch: fix/claude-harness
triggered_by: "Unauthorized AWS S3 migration (context amnesia, 2026-03-16)"
evidence:
  - "arXiv:2602.11988 (Gloaguen et al. 2026) — ETH Zurich AGENTS.md evaluation"
  - "arXiv:2511.12884 (Chatlatanagulchai et al. 2025) — Agent READMEs empirical study"
  - "arXiv:2601.20404 (Lulla et al. 2026) — AGENTS.md efficiency impact"
  - "github.com/Fission-AI/OpenSpec — Spec-driven development framework"
  - "sci-llm-writer/.claude/skills/ — Eval patterns from manuscript repo"
---

# Claude Harness Improvement Plan

## Part 1: Intent Summary — All Verbatim User Prompts

The following table indexes every verbatim user prompt preserved in `docs/planning/`.
These represent the **ground-truth intent** for this repository — every plan, report,
and implementation is a derivative of these instructions.

### Prompt Index

| # | File | Date | Intent (1-line) |
|---|------|------|-----------------|
| P1 | `modernize-minivess-mlops-plan-prompt.md` | 2026-02-23 | Refactor v0.1-alpha to production MLOps, MONAI-first, Nature Protocols paper, agentic/FMOps |
| P2 | `experiment-planning-and-metrics-prompt.md` | 2026-02-25 | DynUNet training pipeline, compound losses, clDice/cbDice, 3-fold CV, ensembling, MLflow |
| P3 | `agentic-architecture-self-reflection-for-sdd-and-beyond-paper-as-agentic-PROMPT.md` | 2026-03-01 | Second-pass: paper novelty = agentic dev with Claude Code, SDD + probabilistic PRD |
| P4 | `advanced-segmentation-double-check-prompt.md` | 2026-03-04 | Review 40+ SOTA papers: segmentation, UQ, foundation models, Mamba, synthetic data |
| P5 | `e2e-testing-user-prompt.md` | 2026-03-10 | Design E2E testing: all flows, inter-flow contracts, 24-point interactive Q&A |
| P6 | `final-methods-quasi-e2e-testing-prompt.md` | 2026-03-10 | Dynamic model/loss/metric discovery, conditional DAG schema, combinatorial reduction |
| P7 | `prompt-574-synthetic-data-drift-detection.md` | 2026-03-12 | Synthetic data, drift detection, data quality pipelines, agentic science workflows |
| P8 | `profiler-benchmarking-user-prompt.md` | 2026-03-13 | PyTorch profiling for GPU/CPU, MLflow logging, default ON, 17 reference links |
| P9 | `s3-mounting-testing-user-prompt.md` | 2026-03-13 | Pulumi-based S3 provisioning, multi-cloud abstraction, access tests |
| P10 | `repo-to-manuscript-prompt.md` | 2026-03-15 | Manuscript scaffold, KG bridge, OpenSpec alignment, intent expression principle |

### Cross-Reference: Prompt → Plan → Implementation

| Prompt | Plans Generated | Key Implementation |
|--------|----------------|-------------------|
| P1 | `modernize-minivess-mlops-plan.md` | Full v2 rewrite, ModelAdapter ABC, Prefect flows |
| P2 | `dynunet-ablation-plan.md`, experiment configs | `dynunet_loss_variation_v2` experiment, 4 losses × 3 folds |
| P3 | `repo-to-manuscript.md` | PRD system, 52 decision nodes, knowledge graph |
| P4 | `advanced-segmentation-double-check.md` | SAM3 adapter, VesselFM adapter, Mamba adapter |
| P5 | `e2e-testing-phase-*.xml` (3 phases) | `PipelineTriggerChain`, 73 validation checks |
| P6 | `final-methods-quasi-e2e-testing-plan.xml` | `capability_discovery.py`, `quasi_e2e_runner.py` |
| P7 | `synthetic-data-drift-detection-plan.xml` | Evidently, whylogs integration (partial) |
| P8 | `profiler-benchmarking-plan.xml` | Profiler module (partial) |
| P9 | `s3-mounting-testing-plan.xml` | Pulumi DVC bucket (archived with UpCloud) |
| P10 | `knowledge-management-upgrade.md` | KG navigator, domains, manuscript scaffold |

### Intent Themes (Synthesized from P1–P10)

1. **Production MLOps for biomedical imaging** — not a toy repo, Nature Protocols-grade
2. **MONAI ecosystem extension** — adapt 3rd-party models, never fork MONAI
3. **Agentic development as paper novelty** — the process IS the contribution
4. **Zero manual work** — everything automatic, one-command reproducibility
5. **Heterogeneous lab support** — any cloud, any GPU, config-only changes
6. **Scientific rigor** — compound losses, topology metrics, proper CV, UQ
7. **Spec-driven development** — PRD as evidence base, decisions as YAML nodes

---

## Part 2: Diagnosis — Why the Harness Failed

### The AWS S3 Incident (2026-03-16)

Claude made a unilateral decision to migrate DVC to AWS S3 instead of GCS. Root causes:

| Failure | Evidence | Layer Missing (OpenSpec) |
|---------|----------|------------------------|
| Session summary treated as authorization | Executed 35-file change from continuation summary | **Intent** — no user authorization |
| Cloud architecture invisible to navigator | GCP project/region/GCS buckets only in `.env.example` | **Context** — not in KG |
| No guardrail against adding cloud providers | CLAUDE.md said "zero hardcoding" but never said "two providers only" | **Constraints** — missing |
| Root CLAUDE.md too long (700+ lines) | At Phase 3 "breaking point" — agent skims, misses critical rules | All layers degraded |

### Empirical Evidence: The CLAUDE.md Breaking Point

From [Gloaguen et al. (2026). "Evaluating AGENTS.md." *arXiv:2602.11988*](https://arxiv.org/abs/2602.11988):

- **LLM-generated context files reduce success by 2%** and increase cost by 20%
- **Developer-written files**: marginal +4% success at 19% higher cost
- **Agents spend 14-22% extra reasoning tokens** processing context files
- **Highest-value content**: tooling constraints ("use uv, not pip")
- **Counterproductive content**: repository overviews, directory listings

From [Lulla et al. (2026). "Impact of AGENTS.md on Efficiency." *arXiv:2601.20404*](https://arxiv.org/abs/2601.20404):

- **29% faster runtime**, **17% fewer tokens** with context files
- But measured **efficiency only**, not correctness
- **Key insight**: context files reduce exploration waste → directed execution

From [Chatlatanagulchai et al. (2025). "Agent READMEs." *arXiv:2511.12884*](https://arxiv.org/abs/2511.12884):

- **69.9% include implementation details**, only **14.5% include security/performance constraints**
- Context files evolve like **configuration code** — frequent small additions
- Context files are "complex, difficult-to-read artifacts" — readability matters

### Synthesis: What the Evidence Says

1. **Less is more.** The root CLAUDE.md at 700+ lines is past the breaking point.
   Agents that process it spend 20%+ more tokens and may perform worse.
2. **Constraints > descriptions.** "NEVER use AWS S3" would have prevented the incident.
   "Two providers: RunPod + GCP" is a constraint. Directory listings are waste.
3. **Progressive disclosure works.** Loading context on demand (navigator → domain)
   aligns with the ETH Zurich finding that broad context hurts.
4. **Evals catch drift.** sci-llm-writer's 67 documented failures show that Skills
   degrade without evals. Our Skills have zero evals.

---

## Part 3: The Plan

### Architecture: Four Layers of Progressive Disclosure

```
┌─────────────────────────────────────────────┐
│ Layer 0: .claude/rules/*.md                 │  ← NEW: path-scoped rules auto-loaded
│   two-providers-only, data-on-gcs, etc.     │     Fire when touching matching paths
├─────────────────────────────────────────────┤
│ Layer 1: Root CLAUDE.md (~250 lines)        │  ← SHRINK: constraints-only, no overviews
│   Critical rules, tooling, 2-provider arch  │     Reference domain files, not duplicate
├─────────────────────────────────────────────┤
│ Layer 2: Domain CLAUDE.md (11 files)        │  ← EXISTS: folder-level experts
│   deployment/, adapters/, config/, tests/   │     Load when working in that directory
├─────────────────────────────────────────────┤
│ Layer 3: KG Navigator → Domain YAML         │  ← IMPROVED: cloud domain added today
│   9 domains, keyword routing, invariants    │     Agent reads navigator → loads domain
├─────────────────────────────────────────────┤
│ Layer 4: Planning docs + metalearning       │  ← EXISTS: deep context loaded on demand
│   docs/planning/, .claude/metalearning/     │     Only loaded when referenced
└─────────────────────────────────────────────┘
```

### Task Breakdown

#### T0: Intent Capture (P0 — this document)
- [x] Extract all 10 verbatim user prompts from `docs/planning/`
- [x] Create cross-reference table (prompt → plan → implementation)
- [x] Synthesize 7 intent themes
- [ ] Create `knowledge-graph/intent-summary.yaml` with structured intent data
- [ ] Add intent themes to navigator.yaml as top-level guidance

#### T1: Root CLAUDE.md Modularization (P0 — the breaking point)

**Goal**: Shrink root CLAUDE.md from ~700 lines to ~250 lines (66% reduction).

> **Reviewer note**: 150 lines is mathematically infeasible. The 22 Critical Rules
> alone need ~120 compressed lines. Minimum viable content (constraints + commands +
> test tiers) totals ~220 lines. Target ~250 to allow minimal rationale on
> highest-stakes rules. Moving constraints to domain files recreates the amnesia
> the plan is trying to fix — constraints MUST stay in root.

**What stays in root** (constraints-only, per ETH Zurich evidence):
- Overarching principles (TOP-1, TOP-2) — ~15 lines each
- Two-Provider Architecture — ~20 lines (the exact constraint that failed)
- Critical Rules (#1-#22) — keep as negative instructions, compress rationale
- Quick Commands (~10 lines)
- Test Tiers table (~12 lines)
- Merged "NEVER Do" list (combine two duplicate sections into one)

**What moves to domain CLAUDE.md files**:
- `deployment/CLAUDE.md` ← Docker-per-flow, cloud GPU strategy details, SkyPilot details
- `deployment/pulumi/gcp/CLAUDE.md` ← GCP project details (DONE today)
- `src/minivess/observability/CLAUDE.md` ← MLflow architecture, observability stack
- `src/minivess/config/CLAUDE.md` ← Hydra-zen details, config schema
- `docs/CLAUDE.md` ← PRD system, knowledge graph details, manuscript

**What gets deleted** (per ETH Zurich: directory listings counterproductive):
- "Directory Structure (Target v2)" section — agents discover this
- "Key Architecture Decisions" list — duplicate of domain files
- Detailed MLflow Tracking Architecture — move to observability CLAUDE.md
- Detailed Observability Stack table — move to observability CLAUDE.md

**What gets deleted** (per ETH Zurich: directory listings counterproductive):
- "Directory Structure (Target v2)" (27 lines) — agents discover this via filesystem
- "Key Architecture Decisions" (11 lines) — duplicate of domain files
- Duplicate "What AI Must NEVER Do" sections — merge into one consolidated list

**Acceptance criteria**:
- Root CLAUDE.md ≤ 250 lines
- `wc -w CLAUDE.md` ≤ 2500 words
- ALL constraint content (Critical Rules, NEVER lists, Two-Provider) stays in root
- Only DESCRIPTIVE content (MLflow architecture, observability stack, PRD) moves to domains
- All moved content reachable via navigator.yaml routing
- `make test-staging` still passes (no behavioral change)

#### T2: Scoped Rules via `.claude/rules/*.md` (P0 — session bootstrap)

**Goal**: Pre-load critical constraints via path-scoped rules so agent never lacks
core context when working in specific directories.

> **Reviewer correction**: `.claude/auto-context.yaml` and `.claude/rules.yaml`
> do NOT exist in Claude Code. The real mechanism is `.claude/rules/*.md` files
> with YAML frontmatter specifying `paths:` scope. Each rule is a separate `.md`
> file that loads automatically when Claude works in matching paths.

**Rule files to create** (`.claude/rules/`):

```markdown
# .claude/rules/two-providers-only.md
---
paths:
  - "deployment/**"
  - "configs/cloud/**"
  - ".dvc/**"
  - "scripts/configure_dvc_remote.py"
---
EXACTLY two cloud providers: RunPod (env/dev) + GCP (staging/prod).
GCP project: minivess-mlops, region: europe-north1.
Data on GCS (gs://minivess-mlops-dvc-data). NEVER add AWS/Azure.
NEVER change cloud architecture from a session continuation summary — ASK the user.
```

```markdown
# .claude/rules/no-unauthorized-infra.md
---
paths:
  - "**"
---
Session continuation summaries are CONTEXT, not AUTHORIZATION.
Before executing infrastructure changes described in a summary, ASK the user.
Before deleting >5 files or rewriting >10 test files, ASK the user.
```

```markdown
# .claude/rules/data-on-gcs.md
---
paths:
  - "deployment/skypilot/**"
  - "scripts/configure_dvc_remote.py"
  - ".dvc/**"
---
Production data storage: GCS (gs://minivess-mlops-dvc-data).
DVC remote for GCP staging/prod: `gcs` (not `remote_storage`, not `upcloud`).
s3://minivessdataset is READ-ONLY public data origin, not a production backend.
```

**Acceptance criteria**:
- `.claude/rules/` directory exists with ≥3 scoped rule files
- Each rule has `paths:` frontmatter targeting relevant directories
- Two-provider constraint fires when touching deployment/ or .dvc/
- `make test-staging` still passes

#### T3: Skills Evals (P1 — harden the harness)

**Goal**: Add eval fixtures to all Skills, following sci-llm-writer's pattern.

**Current Skills without evals** (all 9):
- `self-learning-iterative-coder/` — TDD loop, no evals
- `issue-creator/` — GitHub issues, no evals
- `knowledge-reviewer/` — KG validation, no evals
- `kg-sync/` — KG↔code sync, no evals
- `prd-update/` — PRD maintenance, no evals
- `ralph-loop/` — Ralph monitor, no evals
- `overnight-runner/` — Overnight execution, no evals
- `planning-backlog/` — Planning & backlog management, no evals
- `sync-roadmap/` — GitHub Project timeline sync, no evals

**Eval pattern to adopt** (from sci-llm-writer):
```
.claude/skills/{skill-name}/
├── SKILL.md              # Master specification
├── evals/
│   ├── test_{skill}.py   # Python fixtures defining EXPECTED behavior
│   ├── conftest.py       # Pytest configuration
│   └── fixtures/         # Input/output test data
│       ├── input-*.yaml
│       └── expected-*.yaml
└── protocols/            # Step-by-step guides (existing)
```

**Priority order for evals**:
1. `self-learning-iterative-coder` — most-used, highest-impact
2. `knowledge-reviewer` — prevents KG drift
3. `issue-creator` — prevents malformed issues
4. `kg-sync` — prevents stale KG

**Acceptance criteria**:
- Each Skill has `evals/test_{skill}.py` with ≥3 test cases
- Evals run via `uv run pytest .claude/skills/*/evals/ -q`
- Eval failures block Skill execution (gate, not suggestion)

#### T4: Navigator Completeness Audit (P1)

**Goal**: Every knowledge domain routes to all relevant files. No orphan docs.

**Current gaps** (identified today):
- ~~No cloud domain~~ — FIXED (cloud.yaml created)
- 6 GCP planning docs not indexed — FIXED (added to cloud.yaml)
- `infrastructure.yaml` had stale `not_started` statuses — FIXED

**Remaining audit**:
- Verify ALL `docs/planning/*.md` files are indexed in some domain's `planning_docs`
- Verify ALL `docs/planning/*.xml` files are reachable from domain routing
- Verify ALL `.claude/metalearning/*.md` files are referenced in domain metalearning sections
- Create missing domain files if needed (e.g., `testing.yaml` has no decisions)

**Acceptance criteria**:
- `scripts/review_knowledge_links.py` reports 0 errors, ≤10 warnings
- Every doc in `docs/planning/` is indexed in at least one domain YAML
- Navigator keyword routing covers all major terms in the repo

#### T5: OpenSpec Installation + SDD Integration (P0 — MANDATORY)

**Goal**: Install OpenSpec and integrate Spec-Driven Development (SDD) into the
existing KG + CLAUDE.md system. This is a scientific novelty for the manuscript —
demonstrating SDD with agentic development in an academic MLOps platform.

> OpenSpec is MANDATORY per user directive. It is NOT optional or future work.
> The evaluation of SDD as a development methodology is a key contribution
> to the NEUROVEX manuscript.

**What is OpenSpec**: A spec-driven development framework (30.8k stars, MIT, v1.2.0)
by Fission AI. It adds a per-change specification layer between the human and the
AI coding agent. Three phases: Propose → Apply → Archive.

**Why it matters for the manuscript**: SDD is an emerging practice (2025-2026) with
Thoughtworks Technology Radar recognition but limited academic evaluation. Only one
arXiv paper exists ([Piskala, 2026. arXiv:2602.00180](https://arxiv.org/abs/2602.00180)).
Demonstrating SDD in a real academic biomedical MLOps platform is novel.

**Evidence**:
- [Piskala (2026). "Spec-Driven Development." *arXiv:2602.00180*](https://arxiv.org/abs/2602.00180) — three rigor levels, "error reductions up to 50%"
- [Hashrocket comparison](https://hashrocket.com/blog/posts/openspec-vs-spec-kit) — 250 lines vs 800 (Spec Kit), 327/328 tests passing
- [Darren Taylor](https://darrenonthe.net/2026/01/01/open-spec-a-lighter-approach-to-specification-driven-development/) — 54-file, 5,409-line feature in one session
- [Thoughtworks](https://www.thoughtworks.com/en-us/insights/blog/agile-engineering-practices/spec-driven-development-unpacking-2025-new-engineering-practices) — SDD as "key 2025 engineering practice"
- [Fowler/Boeckeler](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html) — SDD tool evaluation

**SDD rigor levels** (Piskala 2026):
1. **Spec-first** — specs guide initial dev, may drift post-implementation
2. **Spec-anchored** — living documentation, automated tests enforce alignment (our target)
3. **Spec-as-source** — humans edit only specs, machines generate all code (future)

**How OpenSpec maps to our existing system**:

| OpenSpec Component | Our Existing Equivalent | Integration |
|-------------------|------------------------|-------------|
| `openspec/project.md` | Root CLAUDE.md | Keep both — CLAUDE.md = constraints, project.md = context |
| `openspec/AGENTS.md` | Root CLAUDE.md | Let OpenSpec generate/maintain its own AGENTS.md |
| `openspec/specs/` | `knowledge-graph/domains/*.yaml` | Specs supplement KG, not replace it |
| `openspec/changes/<name>/proposal.md` | `docs/planning/*.md` plans | New features go through OpenSpec propose |
| `openspec/changes/<name>/tasks.md` | XML execution plans | OpenSpec tasks replace XML plans for new work |
| `openspec/archive/` | Git history + completed plans | Archive provides institutional memory |

**Installation steps**:
```bash
npm install -g @fission-ai/openspec@latest  # Requires Node.js 20.19+
cd /path/to/minivess-mlops
openspec init                                # Select Claude Code, generates openspec/
```

**What `openspec init` creates**:
```
openspec/
├── project.md          # Global context (tech stack, architecture)
├── AGENTS.md           # AI behavioral guidelines (auto-maintained)
├── specs/              # Living system specification (source of truth)
│   └── (empty initially — populated during first propose/archive cycle)
├── changes/            # Active feature proposals
│   └── (empty initially)
└── archive/            # Completed features with timestamps
    └── (empty initially)
```

**Integration with KG navigator**:
```yaml
# Addition to navigator.yaml
openspec:
  navigator: openspec/project.md
  covers: [active changes, system specs, archived features, SDD workflow]
  note: "Per-change specifications. Use /opsx:propose for new features."
```

**Workflow for new features** (replaces XML plan creation):
```
1. User describes feature intent
2. /opsx:propose "feature name" → generates proposal.md, specs/, design.md, tasks.md
3. Review with user → iterate on spec
4. /opsx:apply → implement tasks (uses self-learning-iterative-coder Skill for TDD)
5. /opsx:archive → merge delta specs into openspec/specs/, archive change
```

**What stays unchanged**:
- Root CLAUDE.md (constraints) — OpenSpec does not replace this
- KG navigator + domain YAMLs — these are decision-tracking, not feature specs
- `.claude/rules/*.md` — path-scoped rules independent of OpenSpec
- `.claude/skills/` — OpenSpec adds slash commands, Skills remain
- `docs/planning/` — existing plans preserved, new work uses OpenSpec

**What changes**:
- New features use `/opsx:propose` instead of manually writing XML plans
- `openspec/specs/` becomes the living system specification
- `openspec/archive/` provides audit trail for the manuscript
- AGENTS.md is auto-maintained by OpenSpec (separate from CLAUDE.md)

**Context Hub integration** (T7):
- Custom chub registry (`docs/chub-registry/`) provides upstream library docs
- OpenSpec specs reference chub content IDs for API-level detail
- The fetch-docs Skill provides library context during `/opsx:apply`

**Manuscript relevance**:
- Document the SDD evaluation in the Methods section
- Compare spec-anchored (our approach) vs spec-first (no enforcement)
- Report metrics: spec-to-code alignment, feature completion rate, error rate
- Cross-reference `openspec/archive/` as evidence of the SDD workflow

**Acceptance criteria**:
- `openspec/` directory exists with `project.md`, `AGENTS.md`, `specs/`
- `openspec init` completed successfully with Claude Code target
- Navigator.yaml updated with openspec routing
- At least one feature goes through the full propose → apply → archive cycle
- `openspec/project.md` references the KG navigator for domain routing

#### T6: Memory Architecture Formalization (P2)

**Goal**: Implement the 4-tier memory architecture (from user's reference diagrams).

> **Reviewer note**: This diagram is from the user's screenshots, not from any
> cited paper. The McMillan reference covers information architecture for SQL
> agents, not memory tiers. Attribution corrected.

```
Surface (context window) ─── 200K tokens, 1 session
    ↓ pattern noticed repeatedly
Shallow (MEMORY.md) ─── episodic, bridges sessions
    ↓ proves durable across sessions
Deep (CLAUDE.md + Skills) ─── semantic + procedural
    ↓ only via model retraining
Bedrock (model weights) ─── all projects
```

**Current gaps**:
- No formal promotion protocol (shallow → deep)
- MEMORY.md has 147 lines — approaching 200-line truncation limit
- No garbage collection for stale memories
- No periodic review trigger

**Acceptance criteria**:
- MEMORY.md stays under 100 lines (index only, no inline content)
- Memory promotion protocol documented in `knowledge-reviewer` Skill
- Stale memory detection added to `review_staleness.py`

---

## Part 4: Priority and Dependencies

```
T0 (intent capture)         ← THIS DOCUMENT (done)
  │
  ├── T1 (CLAUDE.md shrink) ← P0, most impactful, no deps
  │     │
  │     └── T2 (scoped rules) ← P0, depends on T1
  │
  ├── T5 (OpenSpec install) ← P0, MANDATORY, independent
  │
  ├── T3 (Skills evals) ← P1, independent
  │
  ├── T4 (navigator audit) ← P1, independent
  │
  ├── T7 (Context Hub) ← P1, after T5
  │
  └── T6 (memory formalization) ← P2, after T1-T5 stable
```

**Estimated effort**: T1 (2h), T2 (30min), T5 (1h install + first propose cycle),
T3 (4h), T4 (3-4h), T7 (2h), T6 (future)

> **Reviewer note**: T4 is much larger than initially scoped. 195 of 208
> planning docs (94%) and 40 of 49 metalearning docs (82%) are unindexed.
> Full indexing is 3-4h; alternatively scope to "index only actively-referenced
> docs" for 1h.

---

#### T7: Context Hub Integration (P1)

**Goal**: Install Context Hub for upstream library doc access. Context7 rejected
(cloud-only, no custom docs, no offline). See `docs/planning/context7-vs-context-hub.md`.

**Installation**: `npm install -g @aisuite/chub`

**Custom registry** for our stack:
```
docs/chub-registry/
  monai/losses/DOC.md           # MONAI losses + our custom CbDiceClDiceLoss
  monai/transforms/DOC.md       # MONAI transforms we use
  skypilot/yaml/DOC.md          # SkyPilot YAML patterns with Docker image_id
  minivess/adapters/DOC.md      # ModelAdapter ABC, SAM3 VRAM tables
```

**Skill**: `.claude/skills/fetch-docs/SKILL.md` — instructs agent to use `chub get`
before implementing with any upstream library. Demand-invoked only (not auto-triggered).

**Acceptance criteria**:
- `chub` CLI installed and available
- Custom registry built via `chub build docs/chub-registry/`
- At least MONAI losses and SkyPilot YAML docs authored
- Skill file exists and follows the two-gate verification protocol (fetch + web-search)

---

## References

- [Gloaguen et al. (2026). "Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?" *arXiv:2602.11988*](https://arxiv.org/abs/2602.11988)
- [Chatlatanagulchai et al. (2025). "Agent READMEs: An Empirical Study of Context Files for Agentic Coding." *arXiv:2511.12884*](https://arxiv.org/abs/2511.12884)
- [Lulla et al. (2026). "On the Impact of AGENTS.md Files on the Efficiency of AI Coding Agents." *arXiv:2601.20404*](https://arxiv.org/abs/2601.20404)
- [Piskala (2026). "Spec-Driven Development: From Code to Contract in the Age of AI Coding Assistants." *arXiv:2602.00180*](https://arxiv.org/abs/2602.00180)
- [OpenSpec. Fission-AI. Spec-driven development framework.](https://github.com/Fission-AI/OpenSpec)
- [Context Hub. Andrew Ng / aisuite. Curated API documentation for coding agents.](https://github.com/andrewyng/context-hub)
- [Thoughtworks (2025). "SDD: Unpacking 2025's New Engineering Practices."](https://www.thoughtworks.com/en-us/insights/blog/agile-engineering-practices/spec-driven-development-unpacking-2025-new-engineering-practices)
- [Fowler & Boeckeler (2026). "SDD: Three tools evaluated."](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html)
- [McMillan (2026). "Information Architecture for AI Agent Context." *arXiv:2602.05447*](https://arxiv.org/abs/2602.05447)
