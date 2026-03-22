---
theme: harness
doc_count: 33
last_synthesized: "2026-03-22"
implementation_health: partial
kg_domains: [infrastructure, architecture]
---

# Theme: Harness -- Cold-Starts, Agent Harness, Context Management

This theme covers the agentic development infrastructure: cold-start prompts for session
bootstrapping, Claude harness improvement, context compounding failure prevention,
plan synthesis documents, knowledge management upgrades, SDD/OpenSpec architecture,
the probabilistic PRD system, and the intent-preservation layer that maintains ground-truth
user instructions across sessions.

---

## Key Scientific Insights

### 1. Context Compounding Is the #1 Systemic Failure

Claude Code failed to maintain consistent understanding of the factorial design across 8+
sessions. The same wrong conclusions were re-derived every session: "full factorial = 24
cells" (correct: 480+), "post-training is CPU-only" (execution location is user's choice),
"debug should skip X" (debug = production minus epochs/data/folds ONLY). The worst case:
Claude wrote a metalearning doc that was itself wrong -- the correction mechanism produced
a wrong correction that would have poisoned future sessions.

The root cause is LLM context amnesia compounded by derivative documents that drift from
ground truth. The fix requires: (a) authoritative decision registry, (b) code-review-graph
MCP server for structural queries, (c) metalearning docs reviewed before each session.

**Source:** `context-compounding-and-learning-repo-plan.md`

### 2. Session Summaries Are Context, Not Authorization

A critical safety finding: Claude treated session continuation summaries as implicit
authorization for infrastructure changes (including an unauthorized AWS S3 migration).
The rule: summaries describe what happened; they do not grant permission to execute.
Before infrastructure changes, ASK the user.

**Source:** `.claude/rules/no-unauthorized-infra.md`, metalearning `2026-03-16-unauthorized-aws-s3-architecture-migration.md`

### 3. Intent Summary Is the Ground-Truth Index

The intent-summary.md document indexes every verbatim user prompt preserved in
`docs/planning/`. These are the primary sources -- all plans and implementations
are secondary derivatives. When a plan contradicts the user's original prompt, the
prompt is authoritative. 11+ prompts indexed covering the full arc from v0.1-alpha
modernization through agentic architecture to debug factorial runs.

**Source:** `intent-summary.md`, `claude-harness-improvement-plan.md`

### 4. Cold-Start Prompts Enable Deterministic Session Bootstrapping

Cold-start prompts are self-contained documents that allow a new Claude Code session to
resume work from a precise state. Each prompt specifies: mandatory reading, branch,
context, what was accomplished, what remains, and pre-requisites. 8 cold-start prompts
span the entire debug factorial lifecycle from pre-debug QA through 4th pass re-launch.

The pattern: `claude -p "Read and execute the plan at: docs/planning/cold-start-*.md"`.
This makes Claude Code sessions reproducible and recoverable -- any researcher (or future
Claude session) can pick up exactly where the previous session left off.

**Source:** All `cold-start-prompt-*.md` documents

### 5. Three Plan Synthesis Versions Track Architecture Evolution

Three synthesis documents capture the project's architectural evolution:
- **v1 (2026-03-07):** Identified two P0 blockers -- standalone scripts (#367) and Docker volume mounts (#369)
- **v2 (2026-03-08):** Docker volumes FIXED, Hydra-zen composition gap DISCOVERED (train_flow.py bypasses Hydra entirely)
- **Pre-debug-run (2026-03-20):** Complete factorial specification -- 6-factor design across 3 compute layers

Each synthesis reads 40-50+ source files and produces a single coherent state document.
The pre-debug-run synthesis supersedes v1/v2 for factorial scope but v1/v2 remain valid
for infrastructure decisions.

**Source:** `intermedia-plan-synthesis.md`, `intermedia-plan-synthesis-v2.md`, `intermedia-plan-synthesis-pre-debug-run.md`

### 6. Agentic Architecture: The Paper IS the Process

The key reframing for the Nature Protocols paper: the main novelty contribution is the
agentic development process itself (building research infrastructure with Claude Code +
SDD), not just the system produced. The probabilistic PRD translated to SDD approach
could be an abstraction that allows other researchers to create their own systems with
slight modifications. Three recent papers provide evidence:
- Gloaguen et al. (2026, ETH Zurich): AGENTS.md/CLAUDE.md effectiveness
- Chatlatanagulchai et al. (2025): Agent READMEs empirical study
- Lulla et al. (2026): AGENTS.md efficiency impact

**Source:** `agentic-architecture-self-reflection-for-sdd-and-beyond.md`, `agentic-architecture-self-reflection-for-sdd-and-beyond-paper-as-agentic-PROMPT.md`

### 7. Biomedical Agentic AI: L2->L3 Autonomy Transition

The 64-paper research report maps agentic AI in biomedicine against the L0-L5 data agent
autonomy taxonomy (Luo et al., 2026). Current MinIVess sits at L2 (tool-assisted,
human-orchestrated). The L2->L3 transition (autonomous pipeline orchestration) is the
critical unsolved leap. Three convergences matter: deterministic-agentic boundary,
regulatory-agentic alignment (FDA/IEC 62304), and platform extensibility.

**Source:** `biomedical-agentic-ai-research-report.md`, `data-science-agents-report.md`

### 8. Knowledge Management: Five Disconnected Systems

The project operated five knowledge systems that evolved independently: CLAUDE.md +
MEMORY.md (Layer 0-1), probabilistic PRD blueprint (52 nodes, never materialized as YAML),
planning docs (118 files, no status index), metalearning docs, and code. The upgrade
plan introduced: navigator.yaml (domain routing), 65 decision nodes across 11 domains,
and OpenSpec for spec-driven development.

**Source:** `knowledge-management-upgrade.md`, `hierarchical-prd-planning.md`

### 9. Data Science Agent Autonomy Taxonomy (L0-L5)

Two companion papers (Luo et al. 2026 SIGMOD, Zhu et al. 2025) establish the definitive
autonomy framework modeled on SAE J3016 driving automation. Most production systems
(Databricks, Snowflake Cortex) sit at L2-L3. Bottlenecks toward L3: limited pipeline
orchestration beyond predefined operators, incomplete lifecycle coverage, deficient
strategic reasoning.

**Source:** `data-science-agents-report.md`

### 10. Failure Metalearning as Durable Knowledge

94 metalearning failure docs exist. Error frequency ACCELERATED from 2.0/day to 6.7/day.
Each metalearning doc captures: what went wrong, root cause, prevention rule, and affected
files. These docs are the primary mechanism for preventing repeated failures across sessions.
The `failure-metalearning-001-training-launch.md` documents the first failure: training
launched without multi-metric config, wasting ~40 minutes of GPU time.

**Source:** `failure-metalearning-001-training-launch.md`, `avoid-silent-existing-failures-no-need-to-act-on.md`

---

## Architectural Decisions Made

| Decision | Outcome | Source Doc | KG Node |
|----------|---------|-----------|---------|
| SDD framework | OpenSpec (not SpecKit, BMAD, Kiro, Tessl) | agentic-architecture-self-reflection.md | -- |
| Context docs tool | Context Hub (not Context7) | context7-vs-context-hub.md | -- |
| PRD structure | 52->65 decision nodes, Bayesian network semantics | hierarchical-prd-planning.md | -- |
| KG navigation | navigator.yaml -> domain routing -> on-demand load | knowledge-management-upgrade.md | -- |
| Session bootstrap | Cold-start prompt pattern (self-contained .md) | cold-start-prompt-*.md | -- |
| Intent preservation | Verbatim user prompt index (intent-summary.md) | intent-summary.md | -- |
| Plan synthesis | Three-version evolution tracking | intermedia-plan-synthesis*.md | -- |
| Authorization model | Session summaries = context, NOT authorization | .claude/rules/no-unauthorized-infra.md | -- |
| Error persistence | Metalearning docs in .claude/metalearning/ | failure-metalearning-001.md | -- |
| GitHub management | Projects migration from ad-hoc issues | github-projects-migration-and-cleaning-plan.md | -- |

---

## Implementation Status

| Document | Type | Status | Key Deliverable |
|----------|------|--------|-----------------|
| STATUS.yaml | document | active | Auto-generated plan doc index |
| agentic-architecture-self-reflection-for-sdd-and-beyond-paper-as-agentic-PROMPT.md | prompt | reference | User prompt: paper as agentic process |
| agentic-architecture-self-reflection-for-sdd-and-beyond-paper-as-agentic.md | document | reference | 2nd-pass agentic architecture with paper angle |
| agentic-architecture-self-reflection-for-sdd-and-beyond.md | document | reference | SDD frameworks + upgrade plan |
| agentic-rag-iac-angle-plan.md | plan | planned | Agentic RAG + IaC capabilities |
| avoid-silent-existing-failures-no-need-to-act-on.md | document | active | Silent failure elimination plan |
| biomedical-agentic-ai-research-report.md | research_report | reference | 64-paper agentic AI survey |
| claude-harness-improvement-plan.md | plan | partial | KG + CLAUDE.md + Skills evals |
| cold-start-prompt-3rd-pass-debug-run.md | cold_start | executed | 3rd pass SWAG+calibration |
| cold-start-prompt-4th-pass-relaunch.md | cold_start | active | 4th pass re-launch after fixes |
| cold-start-prompt-branch-split-and-local-debug.md | cold_start | executed | Branch split + local 3-flow |
| cold-start-prompt-continue-pre-debug-qa.md | cold_start | executed | Continue pre-debug QA |
| cold-start-prompt-full-audit-and-relaunch.md | cold_start | active | Full audit after trust breakdown |
| cold-start-prompt-pre-debug-qa-verification.md | cold_start | executed | Pre-debug QA verification |
| cold-start-prompt-session-continuation-2026-03-20.md | cold_start | executed | Session continuation from 1st pass |
| cold-start-prompt-swag-calibration-implementation.md | cold_start | executed | SWAG+calibration implementation |
| context-compounding-and-learning-repo-plan.md | plan | active | Context compounding prevention (Issue #906) |
| context7-vs-context-hub.md | document | decided | Context Hub selected over Context7 |
| data-science-agents-report.md | research_report | reference | L0-L5 agent autonomy + 11-paper survey |
| failure-metalearning-001-training-launch.md | document | reference | First failure report |
| github-projects-migration-and-cleaning-plan.md | plan | partial | GitHub projects cleanup |
| hierarchical-prd-planning.md | plan | implemented | 52-node probabilistic PRD blueprint |
| intent-summary.md | document | active | Ground-truth prompt index |
| intermedia-plan-synthesis-pre-debug-run.md | plan | active | Pre-debug factorial specification |
| intermedia-plan-synthesis-v2.md | plan | reference | v2 synthesis (Docker fixed, Hydra gap found) |
| intermedia-plan-synthesis.md | plan | reference | v1 synthesis (2 P0 blockers) |
| issue-340-update-body.md | document | reference | Issue 340 analysis hierarchy |
| knowledge-management-upgrade.md | document | implemented | Navigator + PRD materialization |
| pr3-kg-tooling-refresh-plan.md | plan | partial | KG node YAML + architecture doc |
| prd-update-plan.md | plan | partial | Phase 8 paper ingestion (83 papers) |
| remaining-issue-data-driven-plan.md | plan | partial | 29 open issues triage |
| sdd.pptx | document | reference | SDD presentation |
| three-pr-planning-finops-timing-data-quality-kg-tooling-prompts.md | plan | partial | 3-PR planning session prompts |

---

## Cross-References

- **Evaluation theme:** Cold-start prompts bootstrap factorial run sessions
- **Infrastructure theme:** Vision enforcement, STOP protocol, Docker gates
- **Cloud theme:** Ralph monitor skill, SkyPilot monitoring
- **Models theme:** Model lineup decisions tracked in KG
- **KG domains:** `infrastructure.yaml`, `architecture.yaml`
- **Key metalearning:** `2026-03-16-unauthorized-aws-s3-architecture-migration.md`, `2026-03-16-kg-exists-read-it-first.md`, `2026-03-07-silent-existing-failures.md`
- **Skills:** `factorial-monitor`, `self-learning-iterative-coder`, `ralph-loop`, `prd-update`, `plan-context-load`

---

## Constituent Documents

1. `STATUS.yaml`
2. `agentic-architecture-self-reflection-for-sdd-and-beyond-paper-as-agentic-PROMPT.md`
3. `agentic-architecture-self-reflection-for-sdd-and-beyond-paper-as-agentic.md`
4. `agentic-architecture-self-reflection-for-sdd-and-beyond.md`
5. `agentic-rag-iac-angle-plan.md`
6. `avoid-silent-existing-failures-no-need-to-act-on.md`
7. `biomedical-agentic-ai-research-report.md`
8. `claude-harness-improvement-plan.md`
9. `cold-start-prompt-3rd-pass-debug-run.md`
10. `cold-start-prompt-4th-pass-relaunch.md`
11. `cold-start-prompt-branch-split-and-local-debug.md`
12. `cold-start-prompt-continue-pre-debug-qa.md`
13. `cold-start-prompt-full-audit-and-relaunch.md`
14. `cold-start-prompt-pre-debug-qa-verification.md`
15. `cold-start-prompt-session-continuation-2026-03-20.md`
16. `cold-start-prompt-swag-calibration-implementation.md`
17. `context-compounding-and-learning-repo-plan.md`
18. `context7-vs-context-hub.md`
19. `data-science-agents-report.md`
20. `failure-metalearning-001-training-launch.md`
21. `github-projects-migration-and-cleaning-plan.md`
22. `hierarchical-prd-planning.md`
23. `intent-summary.md`
24. `intermedia-plan-synthesis-pre-debug-run.md`
25. `intermedia-plan-synthesis-v2.md`
26. `intermedia-plan-synthesis.md`
27. `issue-340-update-body.md`
28. `knowledge-management-upgrade.md`
29. `pr3-kg-tooling-refresh-plan.md`
30. `prd-update-plan.md`
31. `remaining-issue-data-driven-plan.md`
32. `sdd.pptx`
33. `three-pr-planning-finops-timing-data-quality-kg-tooling-prompts.md`
