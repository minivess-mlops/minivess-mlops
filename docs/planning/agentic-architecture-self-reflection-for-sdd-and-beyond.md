# Agentic Architecture Self-Reflection: SDD Frameworks, Context Engineering, and an Actionable Upgrade Plan for MinIVess MLOps

**Date:** 2026-03-01
**Branch:** `feat/agentic-architecture`
**Status:** Planning document — actionable recommendations for portfolio-grade agentic transition

---

## Abstract

This report evaluates the emerging Spec-Driven Development (SDD) landscape and multi-agent
orchestration patterns through the lens of a working biomedical segmentation MLOps platform
(MinIVess MLOps, 2,282 tests, 18 loss functions, 5 Prefect flows). We synthesize findings
from two recent empirical papers — Vasilopoulos (2026) on three-tier codified context
infrastructure and Gloaguen et al. (2026, ETH Zurich) on the effectiveness of repository
context files — alongside a comparative analysis of five SDD frameworks (SpecKit, OpenSpec,
BMAD, Kiro, Tessl) and the slide material from an in-preparation course on agentic
development (Teikari, 2026). We map this evidence against the current MinIVess architecture
to produce a concrete, phased upgrade plan that demonstrates portfolio-grade agentic
engineering for academic and professional audiences.

---

## 1. Motivation and Scope

### 1.1 The Portfolio Imperative

MinIVess MLOps serves a dual purpose: (1) a research platform for 3D vascular segmentation,
and (2) a portfolio artifact demonstrating production ML engineering. The repository already
implements Prefect 3.x orchestration, MLflow experiment tracking, LangGraph agent graphs,
conformal prediction UQ, 18 loss functions with a classification and warning system, and a
hierarchical Bayesian PRD with 70 decision nodes. The question is not whether to adopt
agentic practices — several are already in place — but how to systematically elevate the
repository from its current maturity level (L2, Agent-Executed) toward L3 (Agent-Monitored)
on the agentic MLOps maturity model defined in our slide material.

### 1.2 Cross-Project Coherence

A companion project, `music-attribution-scaffold` (1,311 tests, PydanticAI chat agent,
FastMCP permission server, 85-node PRD), shares significant architectural DNA with
MinIVess: the same TDD skill (`self-learning-iterative-coder`), the same `uv`-only mandate,
the same Pydantic boundary-object pattern, and the same probabilistic PRD system. Divergences
exist in orchestration (Prefect vs. custom DAG runner), agent framework (LangGraph vs.
PydanticAI), and MCP usage (planned vs. production). The agentic upgrade plan should
strengthen cross-project coherence where the shared patterns are strong, and deliberately
diverge where domain requirements differ.

### 1.3 Document Organization

- **Section 2**: SDD framework comparative analysis (SpecKit, OpenSpec, BMAD, Kiro, Tessl)
- **Section 3**: Empirical evidence — context files and codified context
- **Section 4**: Current MinIVess agentic architecture audit
- **Section 5**: Maturity model assessment (L0–L5)
- **Section 6**: Seven-layer agentic R&D stack mapping
- **Section 7**: Actionable upgrade plan (phased)
- **Section 8**: SDD framework recommendation for MinIVess
- **Section 9**: Open questions and risks

---

## 2. SDD Framework Comparative Analysis

### 2.1 Taxonomy: Three Paradigms

Fowler and Boeckeler (2025) define three paradigms for the specification–code lifecycle:

| Paradigm | Spec Fate | Code Fate | Best For |
|----------|-----------|-----------|----------|
| **Spec-First** | Discarded after implementation | Maintained | Prototyping, throwaway MVPs |
| **Spec-Anchored** | Maintained alongside code | Maintained | Production, long-lived systems |
| **Spec-as-Source** | Primary artifact | Regenerated on demand | Config, schemas, narrow domains |

All five SDD frameworks currently operate at the spec-first level. Tessl alone aspires to
spec-as-source, but its 1:1 spec-to-code mapping and non-deterministic generation remain
limitations (Bockeler, 2025). The spec-anchored paradigm — where specification and code
co-evolve — is the sweet spot for production systems but has no framework support. In
practice, CLAUDE.md files are the closest existing artifact to spec-anchored development.

### 2.2 Framework Comparison

| Dimension | SpecKit | OpenSpec | BMAD | Kiro | Tessl |
|-----------|---------|---------|------|------|-------|
| **Language** | Python (Typer CLI) | TypeScript (Node) | Markdown agents | VS Code fork (AWS) | CLI + MCP |
| **SDD Level** | Spec-first | Spec-first | Spec-first | Spec-first | Spec-as-source (aspirational) |
| **Agent Support** | 17+ agents | 20+ agents | Claude, Copilot, Gemini | Kiro IDE only | Agent-agnostic via MCP |
| **Phase Count** | 5 linear gates | 4 iterative | 2 phases, 9 agents | 3 files | Versioned deps |
| **Files per Spec** | ~8 | Flexible | 12+ agent files | 3 | 1:1 spec-to-code |
| **Memory System** | Constitution file | Lightweight | Agent personas | Steering files (4 modes) | Spec Registry (10K+ pkgs) |
| **Token Cost** | Moderate | Low | Very high (~230M tok/week) | Low–moderate | Low (2× cost-efficient) |
| **Setup Time** | ~30 min | ~5 min | Hours | Quick (IDE built-in) | Quick (CLI) |
| **Vendor Lock-in** | None (BYOA) | None (BYOA) | None (BYOA) | AWS (Kiro IDE) | None (MCP-based) |

**Sources:** Mysore (2026), Bockeler (2025), Martin Fowler articles, official documentation.

### 2.3 Speed–Verbosity Tradeoff

The fundamental tension in SDD is the speed–verbosity tradeoff. Piskala's (2026) golden
rule — "use the minimum level of specification rigor that removes ambiguity for your
context" — implies that specification investment should scale with the cost of failure:

- **Weekend prototype**: OpenSpec (~5 min setup, ~250 lines)
- **Team feature**: SpecKit (~30 min, ~800 lines) or Kiro (~5 min, moderate)
- **Enterprise system**: BMAD (hours, ~3,000+ lines)
- **Paradigm shift**: Tessl (spec = code, different model entirely)

BMAD's token overhead (~31,667 tokens per workflow run, ~230M tokens/week on large
projects) makes it impractical for academic projects with compute budgets. Kiro's
problem-size mismatch (expanding a small bug into 4 user stories with 16 acceptance
criteria) suggests overspecification for incremental work.

### 2.4 The Compliance Gap

Even the best specification cannot guarantee faithful implementation. Piskala (2026)
identifies five failure modes:

1. **False confidence** — the spec itself encodes incorrect assumptions
2. **Spec drift** — code evolves without corresponding spec updates
3. **Over-specification** — agent follows the letter but misses the spirit
4. **Under-specification** — agent fills gaps with training-data priors
5. **Hallucinated compliance** — agent reports adherence but code diverges

Bockeler (2025) found agents that duplicated existing classes despite SpecKit's research
phase documenting them. The compliance gap is structural — mitigable but not eliminable.

**Three mitigations:**
- **Hook enforcement** — CI/CD gates that check spec compliance programmatically
- **Self-spec pattern** — agent drafts spec for human review (~50% error reduction)
- **Human-in-the-loop** — mandatory review gates at critical junctures

### 2.5 Brownfield SDD

The largest unaddressed segment in SDD is brownfield adoption (Krishnan, 2026). MinIVess
is a brownfield project — a clean rewrite from v0.1-alpha, but with substantial existing
infrastructure (2,282 tests, 5 flows, 18 losses). The delta-spec approach (OpenSpec)
addresses this: specify only what changes, not the entire system.

Our CLAUDE.md already functions as a Level 2 specification (persistent project-wide
override). Delta specs for individual features (Level 3) would add surgical specification
rigor without requiring retroactive specification of the entire codebase.

---

## 3. Empirical Evidence: Context Files and Codified Context

### 3.1 The AGENTbench Study (Gloaguen et al., 2026)

The ETH Zurich study provides the first rigorous empirical evaluation of context file
effectiveness, testing Claude Code (Sonnet 4.5), Codex (GPT-5.2, GPT-5.1 mini), and
Qwen Code across 138 real GitHub issues.

**Key findings:**

| Condition | Performance (AGENTbench) | Cost Impact |
|-----------|-------------------------|-------------|
| No context file (baseline) | Reference | Reference |
| LLM-generated context file | −2% | +20–23% |
| Developer-written context file | +4% | +14–19% |

**The Exploration Paradox:** Context files make agents explore more (more testing, more
file traversal, more repository-specific tooling) but do not help agents arrive at the
correct answer. The file-discovery curves are "essentially identical" across conditions.

**Redundancy is the mechanism:** When existing documentation is stripped from repos, LLM-
generated context files *improve* performance by 2.7%. The harmful effect comes from
redundant content that duplicates READMEs and documentation. Human-written files help
because they encode *non-obvious conventions* — exactly what cannot be discovered by
reading the code.

**Implications for MinIVess:** Our CLAUDE.md is human-written and encodes non-obvious
conventions (parameter naming prefixes, default loss function, Prefect requirement, TDD
mandate). This is precisely the category that helps. However, the 20% cost overhead means
we should keep it lean — conventions, not codebase overviews.

### 3.2 The Codified Context Study (Vasilopoulos, 2026)

Vasilopoulos documents a three-tier context infrastructure built during 283 sessions of
developing a 108,256-line C# distributed system:

| Tier | Role | Size | Loading |
|------|------|------|---------|
| **T1: Constitution** (Hot Memory) | Always-loaded project law | ~660 lines, 1 file | Every session |
| **T2: Domain Agents** (Warm Memory) | Specialized expertise | ~9,300 lines, 19 agents | Per task (trigger tables) |
| **T3: Knowledge Base** (Cold Memory) | On-demand specifications | ~16,250 lines, 34 docs | MCP retrieval (1,478 calls) |

**Knowledge-to-code ratio:** 24.2% — nearly one line of context infrastructure for every
four lines of application code. Over 50% of agent specification content is *domain
knowledge* (codebase facts, formulas, failure modes), not behavioral instructions.

**Practitioner guidelines:**
- **G1**: A basic constitution does heavy lifting from day one
- **G2**: Let the planner gather context (run planning agent before implementation)
- **G3**: Route automatically or forget constantly (trigger tables in constitution)
- **G4**: If you explained it twice, write it down
- **G5**: When in doubt, create an agent and restart the session
- **G6**: Stale specs mislead — agents trust documentation absolutely

**Maintenance overhead:** ~1–2 hours/week for 54 files across 26,200 lines.

### 3.3 Reconciling the Two Studies

These papers appear contradictory but address fundamentally different scenarios:

| Aspect | AGENTbench (Gloaguen) | Codified Context (Vasilopoulos) |
|--------|----------------------|-------------------------------|
| Context file | Single flat file (~641 words) | Three-tier system (~26,200 lines) |
| Task scope | Isolated GitHub issue resolution | Sustained multi-session development |
| Loading strategy | Always-on (everything in every session) | Tiered (hot/warm/cold) |
| Authorship | LLM-generated vs. human-written | Human-authored, reactively evolved |
| Content type | Codebase overviews + conventions | Domain knowledge + failure modes |

**The synthesis:** Flat context files with codebase overviews are largely redundant with
what agents can discover by reading code (Gloaguen). The value emerges in persistent domain
knowledge that cannot be inferred from code (Vasilopoulos). Tiered loading prevents
context-window pollution while ensuring critical knowledge is always available.

**For MinIVess, the lesson is clear:**
1. Keep CLAUDE.md lean — conventions and non-obvious rules only (Tier 1)
2. Encode domain knowledge in specialized skill files (Tier 2)
3. Serve detailed specifications via MCP or on-demand loading (Tier 3)

---

## 4. Current MinIVess Agentic Architecture Audit

### 4.1 Agent Infrastructure

| Component | Status | Files | Key Observation |
|-----------|--------|-------|-----------------|
| LangGraph training graph | Implemented | `agents/graph.py` (129 lines) | Deterministic-first: NO LLM reasoning in training |
| LLM provider (LiteLLM) | Implemented | `agents/llm.py` (68 lines) | Provider-agnostic (Anthropic, OpenAI, Ollama) |
| Comparison agent | Implemented | `agents/comparison.py` (100 lines) | LLM for narrative summaries only |
| Braintrust eval suites | Scaffolded | `agents/evaluation.py` (66 lines) | Suite builders exist, no active submission |
| Langfuse tracing | Scaffolded | `agents/tracing.py` (66 lines) | Graceful degradation if unavailable |
| DiLLS diagnostics | Implemented | `observability/agent_diagnostics.py` (216 lines) | Three-tier interaction/session/aggregate |

**Design philosophy:** Deterministic-first with LLM for summarization. This is correct —
training is deterministic; narrative interpretation is where LLMs add value.

### 4.2 Orchestration

| Flow | Status | Type |
|------|--------|------|
| Flow 1: Data Engineering | Not implemented | Core |
| Flow 2: Model Training | Not implemented | Core |
| Flow 3: Model Analysis | Implemented (49 KB) | Core |
| Flow 4: Deployment | Not implemented | Core |
| Flow 5: Dashboard & Reporting | Implemented (6 KB) | Best-effort |

The `_prefect_compat.py` compatibility layer enables graceful degradation when Prefect is
unavailable (`PREFECT_DISABLED=1`), which is essential for CI runners.

### 4.3 Context Engineering

| Layer | MinIVess Current | Vasilopoulos Equivalent |
|-------|-----------------|------------------------|
| Tier 1 (Hot Memory) | CLAUDE.md (~400 lines) | Constitution (~660 lines) |
| Tier 2 (Warm Memory) | 3 skills (TDD, PRD, backlog) | 19 domain agents (~9,300 lines) |
| Tier 3 (Cold Memory) | PRD decisions (52 YAML nodes) | 34 docs + MCP server (~16,250 lines) |
| Memory persistence | `.claude/projects/.../memory/MEMORY.md` | Session memory files |

**Gap analysis:** MinIVess has a strong Tier 1 (CLAUDE.md encodes non-obvious conventions),
a nascent Tier 2 (3 skills vs. 19 agents), and a strong but disconnected Tier 3 (52 PRD
decisions, but no MCP retrieval). The critical missing piece is the *routing mechanism* —
trigger tables that automatically invoke the right skill for the right file modification.

### 4.4 Cross-Project Comparison

| Pattern | MinIVess MLOps | music-attribution-scaffold |
|---------|---------------|---------------------------|
| Orchestration | Prefect 3.x (required) | Custom DAG runner |
| Agent framework | LangGraph | PydanticAI |
| MCP usage | Planned | Production (consent infrastructure) |
| Context structure | Monolithic CLAUDE.md | Layered `.claude/` with rules/ |
| Skills | 3 (TDD, PRD, backlog) | 8 (TDD, commit, PR, figures, frontend...) |
| PRD nodes | 70 | 85 |
| Boundary objects | MLflow artifacts | Pydantic models between DAG stages |

music-attribution-scaffold has more mature context engineering (layered `.claude/` with
rules, golden-paths, and domain-specific context loading triggers) and more skills (8 vs. 3).
MinIVess has more mature ML infrastructure (Prefect flows, MLflow tracking, conformal UQ).

---

## 5. Maturity Model Assessment

### 5.1 The Six-Level Agentic MLOps Maturity Model

Extending Google's and Microsoft's MLOps maturity models (L0–L2) with three agent-specific
levels:

| Level | Name | Description | Infrastructure |
|-------|------|-------------|---------------|
| L0 | Manual | Human does everything | Basic compute |
| L1 | Agent-Assisted | Code generation, debugging | Claude Code + IDE |
| **L2** | **Agent-Executed** | Full pipeline execution | DVC + Prefect + guardrails |
| L3 | Agent-Monitored | Self-monitoring execution | Eval suite + observability |
| L4 | Agent-Coordinated | Multi-agent coordination | Agent SDK + multi-agent infra |
| L5 | Agent-Directed | Hypothesis generation + execution | Full governance (aspirational) |

### 5.2 MinIVess Current Assessment: L2 (Agent-Executed)

| L2 Criterion | MinIVess Status |
|-------------|----------------|
| Full pipeline execution | Partial (Flows 3 + 5 implemented, Flows 1/2/4 missing) |
| DVC integration | Referenced in plans, not yet integrated |
| Prefect orchestration | Implemented with compat layer |
| Guardrails | Pre-commit hooks, TDD mandate, library-first rule |
| Reproducibility | Seeded random states, MLflow tracking, CheckpointManager |

**Why not L3:** L3 requires active self-monitoring — an eval suite that runs automatically
on agent outputs and flags regressions. MinIVess has Braintrust eval *scaffolding* but no
active eval loop. Langfuse tracing is scaffolded but not instrumented in flows.

### 5.3 Target: L3 (Agent-Monitored) with L4 Demonstrations

The near-term target is L3, with selective L4 demonstrations for portfolio purposes:

| L3 Requirement | Implementation Path |
|---------------|-------------------|
| Active eval suite | Braintrust submission loop + pytest domain evals |
| Agentic observability | Langfuse tracing in Prefect flows |
| Regression detection | Continuous eval gates (scientific + technical + compliance) |
| Cost monitoring | LiteLLM proxy + dashboards |

| L4 Demonstration | Portfolio Value |
|-----------------|----------------|
| Agent teams for experiment sweeps | Shows multi-agent coordination |
| MCP server for MLflow queries | Shows protocol-native infrastructure |
| Self-spec for new features | Shows SDD in practice |

---

## 6. Seven-Layer Agentic R&D Stack Mapping

Our slide material defines a seven-layer architecture for agentic R&D systems. Here we map
each layer to MinIVess's current state and planned state:

| Layer | Role | Current State | Planned State |
|-------|------|--------------|---------------|
| **1. Intent** | Scientist + domain expertise | Researcher defines experiments via YAML | Unchanged (scientific decisions stay with researcher) |
| **2. Specification** | CLAUDE.md + experiment specs | CLAUDE.md + Hydra-zen configs | + Delta specs for new features + trigger tables |
| **3. Orchestration** | Agent coordination | LangGraph training graph + Prefect flows | + Agent teams + MCP routing |
| **4. Execution** | Pipeline runtime | Prefect + Docker + scripts | + DVC data versioning |
| **5. Data** | Storage and query | MLflow (filesystem) + DuckDB | + PostgreSQL/pgvector for RAG |
| **6. Monitoring** | Observability | MLflow metrics + Evidently drift | + Langfuse traces + Braintrust evals |
| **7. Governance** | Audit + compliance | Pre-commit hooks + AIDEV-IMMUTABLE | + OpenLineage lineage + approval gates |

The "build boundary" identified in our R&D figure plans (fig-rnd-36) is between layers 4
and 5 — most teams build execution infrastructure but skip evaluation, observability, and
governance. MinIVess already has elements of all seven layers, but layers 5–7 need
strengthening.

---

## 7. Actionable Upgrade Plan

### Phase 0: Context Engineering Refinement (1–2 days)

**Rationale:** AGENTbench shows context files add 20% cost regardless of quality. Vasilopoulos
shows tiered loading reduces this. Start by optimizing what we already have.

**Tasks:**
1. **Audit CLAUDE.md for redundancy** — remove any content that duplicates information
   discoverable from the code itself (e.g., directory listings that `tree` can produce).
   Keep only non-obvious conventions, rules, and decisions.
2. **Add trigger tables** — routing rules in CLAUDE.md that automatically invoke the right
   skill based on file patterns (e.g., modifying `tests/` triggers TDD skill, modifying
   `docs/prd/` triggers PRD skill).
3. **Extract domain knowledge into Tier 2 skills** — create 2–3 new skills:
   - `experiment-runner` skill: encodes experiment YAML schema, loss function catalogue,
     metric naming conventions
   - `topology-metrics` skill: encodes the 8-metric paper framework, loss classification
     system, novel loss debugging protocol
4. **Create `.claude/rules/` directory** — following music-attribution-scaffold pattern,
   with separate rule files for: project context, source of truth hierarchy, testing
   conventions.

### Phase 1: SDD Adoption — Delta Specs (1 week)

**Rationale:** Brownfield SDD via delta specs adds surgical specification rigor without
retroactive respecification. SpecKit's constitution concept aligns with our existing
CLAUDE.md.

**Tasks:**
1. **Evaluate SpecKit integration** — run `specify init . --ai claude --here --force
   --no-git` to assess compatibility with existing project structure. Assess whether the
   constitution concept adds value beyond our CLAUDE.md.
2. **Write delta spec for next feature** — choose one upcoming feature (e.g., Flow 1: Data
   Engineering) and write a full SDD specification: intent, acceptance criteria, technical
   constraints, context files, out of scope.
3. **Implement self-spec pattern** — before implementing any feature, have the agent draft
   a specification for human review. Track whether this reduces error rates (~50% expected
   per Piskala, 2026).
4. **Add spec-compliance hook** — pre-commit hook that checks whether modified files have
   a corresponding spec, warning if not.

### Phase 2: Observability Activation (1 week)

**Rationale:** L3 requires active monitoring. We have Langfuse and Braintrust scaffolding
but no active instrumentation.

**Tasks:**
1. **Instrument Prefect flows with Langfuse** — add trace decorators to analysis_flow and
   dashboard_flow. Each flow execution should produce a structured trace showing the
   decision sequence.
2. **Activate Braintrust eval submission** — connect the existing eval suite builders to
   actual submission. Define 3 eval tasks: segmentation quality (Dice threshold), topology
   preservation (clDice threshold), and cost efficiency (tokens per experiment).
3. **Add LiteLLM cost tracking** — instrument all LLM calls through the LiteLLM proxy with
   cost logging to MLflow.
4. **Create the three-gate CI pipeline** — technical (tests pass), scientific (metrics
   above baseline), compliance (guardrails respected).

### Phase 3: MCP Server for MLflow (1 week)

**Rationale:** MCP bridges "agent that writes Python" to "agent that operates scientific
infrastructure" (fig-rnd-32). Our MLflow data is queried via DuckDB but not accessible to
agents via protocol.

**Tasks:**
1. **Build `mlflow-mcp-server`** — FastMCP server exposing:
   - `list_experiments()` — enumerate MLflow experiments
   - `query_runs(experiment, filters)` — SQL-like run queries via DuckDB
   - `get_run_metrics(run_id)` — retrieve all metrics for a run
   - `compare_runs(run_ids, metrics)` — pairwise comparison with bootstrap CIs
   - `get_champion(category)` — retrieve current champion model per category
2. **Register as Claude Code MCP server** — add to `.claude/settings.json`
3. **Write integration tests** — verify MCP queries match direct DuckDB queries

### Phase 4: Agent Teams Demonstration (1 week)

**Rationale:** Portfolio-grade demonstration of multi-agent coordination. CooperBench shows
~50% degradation for coordinating agents, so we must choose use cases carefully — parallel
independent tasks, not sequential dependent ones.

**Tasks:**
1. **Experiment sweep via agent teams** — use Claude Code agent teams (experimental flag)
   to run parallel loss function experiments. Each teammate handles one loss configuration.
   This is embarrassingly parallel — no inter-agent coordination needed.
2. **Research + implementation split** — one agent researches existing implementations
   (library-first rule), another implements with TDD. Evaluate whether this produces better
   library-first compliance than single-agent workflow.
3. **Document findings** — record actual token costs, wall-clock time, and quality metrics.
   Compare against single-agent baselines.

### Phase 5: Cross-Project Alignment (ongoing)

**Tasks:**
1. **Publish shared TDD skill** — extract `self-learning-iterative-coder` as a standalone
   package installable across both repos.
2. **Harmonize `.claude/` structure** — adopt music-attribution-scaffold's layered pattern
   (rules/, golden-paths.md) in MinIVess.
3. **Shared PRD tooling** — extract PRD validation scripts as a shared library.

---

## 8. SDD Framework Recommendation

### 8.1 For MinIVess MLOps: Lightweight SDD (Not Full Framework)

After analyzing all five frameworks, our recommendation is **not to adopt any SDD framework
wholesale**. The reasoning:

1. **MinIVess is brownfield** — SpecKit and BMAD assume greenfield projects. The delta-spec
   pattern is what we need, and that requires only markdown files + discipline, not a CLI
   framework.
2. **Token budget matters** — BMAD's 230M tokens/week is incompatible with academic compute
   budgets. Even SpecKit's 8 files per spec adds review overhead.
3. **CLAUDE.md already works** — AGENTbench shows human-written context files provide +4%
   improvement. Our CLAUDE.md is human-written, convention-focused, and actively maintained.
4. **Vendor lock-in is unacceptable** — Kiro locks to AWS; we need multi-environment
   compatibility (laptop, on-prem, cloud, CI).

### 8.2 What We Should Adopt

Instead of a framework, adopt **SDD practices incrementally:**

| Practice | Source Framework | Implementation |
|----------|----------------|----------------|
| Delta specs for new features | OpenSpec (brownfield pattern) | Markdown files in `specs/` |
| Constitution concept | SpecKit | Already have (CLAUDE.md) |
| Trigger tables | Vasilopoulos G3 | Add to CLAUDE.md |
| Self-spec pattern | Piskala (2026) | Agent drafts spec → human reviews |
| Three-gate CI | Course material (fig-rnd-29) | Technical + scientific + compliance |
| Out-of-scope section | SDD spec template (fig-fundamentals-28) | In every delta spec |

### 8.3 Why Not Tessl?

Tessl's spec-as-source aspiration is intellectually compelling but premature for our use
case. The 1:1 spec-to-code mapping is too rigid for ML pipelines where a single training
script may depend on dozens of configuration files. The non-determinism problem (Bockeler,
2025: identical specs producing varying outputs) is unacceptable for reproducible
experiments. Revisit when Tessl exits private beta and addresses multi-file mapping.

---

## 9. Open Questions and Risks

### 9.1 Open Questions

1. **Spec drift detection** — no standard metric exists for specification-to-code alignment
   (analogous to code coverage for tests). How do we measure whether our delta specs remain
   accurate as code evolves?
2. **Agent team reliability** — CooperBench shows ~50% degradation for coordinating agents.
   Our Phase 4 plan mitigates this by choosing embarrassingly-parallel tasks, but what is
   the minimum useful coordination pattern for ML experiment sweeps?
3. **Context window economics** — at 20% overhead per AGENTbench, is the quality improvement
   from CLAUDE.md worth the cost? Should we A/B test with and without?
4. **MCP server proliferation** — how many MCP servers can an agent usefully manage before
   tool selection itself becomes a bottleneck? mcp.science provides 12; we plan to add 1.
   At what count does the overhead dominate?
5. **Tier 3 retrieval quality** — Vasilopoulos uses a simple keyword MCP server. Would
   semantic retrieval (pgvector) over our PRD decisions provide better context selection?

### 9.2 Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SDD overhead exceeds value | Medium | High | Start with one delta spec; measure before scaling |
| Agent teams produce merge conflicts | High | Medium | Only assign independent files; use file locking |
| Langfuse self-hosting complexity | Medium | Medium | Docker Compose; graceful degradation already works |
| CLAUDE.md grows past effective size | Medium | High | Enforce 400-line limit; extract to Tier 2/3 |
| Spec drift undetected | High | Medium | Quarterly spec review; pre-commit spec-check hook |

---

## 10. Connection to the Agentic Development Slide Deck

This report and the MinIVess repository serve as working demonstrations of concepts from
the in-preparation course (Teikari, 2026):

| Slide Concept | MinIVess Demonstration |
|--------------|----------------------|
| Agentic MLOps maturity (L0–L5) | L2 → L3 transition documented with metrics |
| Seven-layer R&D stack | All 7 layers mapped with current + planned state |
| SDD three mitigation levels | Level 2 (CLAUDE.md) → Level 3 (delta specs) |
| Brownfield delta specs | Working example on 2,282-test codebase |
| Three-gate CI (fig-rnd-29) | Technical + scientific + compliance gates |
| MCP for scientific infrastructure (fig-rnd-32) | mlflow-mcp-server |
| Agent teams (fig-rnd-38) | Experiment sweep via parallel agent teams |
| Agentic observability (fig-rnd-33) | Langfuse traces in Prefect flows |
| Evaluation hierarchy (fig-rnd-28) | Unit → integration → pipeline → behavioral → system |
| Context engineering (Vasilopoulos tiers) | Three-tier CLAUDE.md + skills + PRD |
| The compliance gap (Piskala) | Self-spec pattern + hook enforcement |
| Harness > model (Schmid) | Deterministic-first LangGraph + TDD skill |

### MinIVess as the "Agentic Transition" Figure (fig-rnd-26)

The generated figure `fig-rnd-26-minivess-agentic-transition.png` shows the before/after
of MinIVess's agentic transition:

| BEFORE (existing) | AFTER (augmented) |
|---|---|
| MLflow manual | MLflow + agent logging |
| Shell scripts | Prefect + agent triggers |
| DVC manual | DVC + agent checkpoints |
| Manual comparison | Agent-evaluated, approved |
| Post-hoc README | Agent-generated, reviewed |

The key message: **augmentation, not replacement**. Existing tools gain agent interfaces.
The human gate remains for all critical decisions.

---

## References

- Bockeler, B. (2025). Exploring generative AI: Understanding spec-driven development. *Martin Fowler's Blog*.
- Fowler, M. and Boeckeler, B. (2025). Three paradigms of spec-driven development. *martinfowler.com*.
- Gloaguen, T., Mundler, N., Muller, M., Raychev, V., and Vechev, M. (2026). Evaluating AGENTS.md: Are repository-level context files helpful for coding agents? *ETH Zurich SRI Lab*. arXiv:2602.11988.
- Krishnan, S. (2026). Enterprise gap analysis of SDD frameworks. *Tech report*.
- Lulla, M. et al. (2026). Context files reduce agent runtime. *AI Engineering report*.
- Mysore, S. (2026). SpecKit vs BMAD comparison. *SDD framework benchmark*.
- Ong, K. and Vikati, J. (2026). Claude Code tool choices: 2,430 responses across 20 technology categories. *Empirical study*.
- Osmani, A. (2026). The lethal trifecta: Speed, non-determinism, and cost pressure in agentic development. *Google Chrome team report*.
- Piskala, M. (2026). Three SDD mitigation levels and five pitfalls taxonomy. *SDD report*.
- Vasilopoulos, A. (2026). Codified context: Infrastructure for AI agents in a complex codebase. arXiv:2602.20478.

---

*Generated on feat/agentic-architecture branch. This document is a living specification
that should be updated as phases are completed.*
