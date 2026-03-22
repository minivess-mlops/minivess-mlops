# Cold-Start Prompts for Pre-GCP Housekeeping PRs

**Plan**: `docs/planning/pre-full-gcp-housekeeping-and-qa.xml`
**Date**: 2026-03-18 (updated 2026-03-19)
**Total**: 6 PRs closing 12 issues

## Execution Strategy

```
                  ┌─── PR-1 (HPO grid + Docker pull)  ─── 2-3 days ───┐
Day 1 ───────────►├─── PR-2 (Compliance audit trail)  ─── 3-4 days ───├──► Day 4-5
(Group A parallel)└─── PR-3 (Observability + KG fix)  ─── 1-2 days ───┘      │
                                                                              ▼
                  ┌─── PR-4 (Agentic dashboard)        ─── 5-7 days ───┐
Day 5 ───────────►│                                                     ├──► Day 10-12
(Group B parallel)└─── PR-5 (KG-enrichment agent)      ─── 3-4 days ───┘      │
                                                                              ▼
Day 12 ──────────►     PR-6 (Research agents)           ─── 7-10 days ──────► Day 20
(Sequential)

AFTER PR-1 merges: Launch GCP factorial (~$98, ~1.5 days wall-clock)
PRs 4-6 can execute WHILE the factorial runs on GCP.
```

### Context Budget per PR

| PR | Read | Write | Fits 1M context? | Sessions needed |
|----|------|-------|-------------------|-----------------|
| PR-1 | ~3,000 lines | ~200 lines | Yes, easily | 1 |
| PR-2 | ~2,500 lines | ~300 lines | Yes, easily | 1 |
| PR-3 | ~500 lines | ~100 lines | Yes, easily | 1 |
| PR-4 | ~4,000 lines | ~600 lines | Yes | 1-2 |
| PR-5 | ~2,000 lines | ~350 lines | Yes | 1 |
| PR-6 | ~5,000 lines | ~1,000 lines | Yes, but large | 1-2 |

**All PRs fit in a single 1M context session.** PR-4 and PR-6 are the largest but
still well within limits. No PR requires session splitting.

### Group A Execution Results (2026-03-19)

| PR | Wall Time | Tests Pass | Skip | New Tests | PR # | Tokens | Tool Uses |
|----|-----------|------------|------|-----------|------|--------|-----------|
| PR-1 | **25 min** | 5115 | 37 | 35 | [#865](https://github.com/petteriTeikari/minivess-mlops/pull/865) | 170K | 148 |
| PR-2 | **52 min** | 5070 | 125 | 24 | [#866](https://github.com/petteriTeikari/minivess-mlops/pull/866) | 179K | 176 |
| PR-3 | **54 min** | 5159 | 14 | 17 | [#867](https://github.com/petteriTeikari/minivess-mlops/pull/867) | 122K | 125 |
| **Total** | **54 min** (parallel) | — | — | **76** | — | **471K** | **449** |

**Strategy**: 3 parallel worktree agents in single Claude Code session.
All 3 ran simultaneously; effective wall time = longest agent (54 min).

### Group B Execution Results (2026-03-19)

| PR | Wall Time | Tests Pass | Skip | New Tests | PR # | Tokens | Tool Uses |
|----|-----------|------------|------|-----------|------|--------|-----------|
| PR-4 | **48 min** | 5245 | 5 | 24 | [#869](https://github.com/petteriTeikari/minivess-mlops/pull/869) | 153K | 140 |
| PR-5 | **46 min** | 5111 | 36 | 14 | [#868](https://github.com/petteriTeikari/minivess-mlops/pull/868) | 123K | 110 |
| **Total** | **48 min** (parallel) | — | — | **38** | — | **276K** | **250** |

**Strategy**: 2 parallel worktree agents in single Claude Code session.
Both ran simultaneously; effective wall time = longest agent (48 min).

### Cumulative (Groups A + B)

| Metric | Value |
|--------|-------|
| PRs completed | 5 of 6 |
| Total wall time | **102 min** (54 + 48, sequential groups) |
| New tests added | **114** |
| Total tokens | **747K** |
| Total tool uses | **699** |

### Which PRs Need ralph-loop?

| PR | Ralph-loop? | Why |
|----|-------------|-----|
| PR-1 | YES | SkyPilot YAML + Pulumi IaC changes |
| PR-2 | NO | Code-only, mock-tested |
| PR-3 | NO | Stubs only, no infrastructure |
| PR-4 | YES | New docker-compose service (CopilotKit) |
| PR-5 | NO | Agent code, no infrastructure |
| PR-6 | YES | New Prefect flows (0b, 3.5, 5b) |

---

## PR-1: HPO Grid Partitioning + Docker Pull Optimization

**Issues**: #857 (partial), #751
**Branch**: `feat/hpo-grid-docker-pull`
**Effort**: S-M (2-3 days)
**Parallel with**: PR-2, PR-3

### Cold-Start Prompt

```
Execute PR-1 from the pre-GCP housekeeping plan.

PLAN: Read docs/planning/pre-full-gcp-housekeeping-and-qa.xml — PR id="1"
RESEARCH: Read docs/planning/hpo-implementation-background-research-report.md — Sections 1, 7, 8 (Phase 1 only), 9

CONTEXT TO READ FIRST (Rule #11 — tokens upfront):
1. src/minivess/optimization/hpo_engine.py — AllocationStrategy enum, HPOEngine class
2. src/minivess/orchestration/flows/hpo_flow.py — existing hpo_flow, PARALLEL NotImplementedError
3. src/minivess/orchestration/flows/train_flow.py — understand how training is triggered
4. configs/hpo/dynunet_grid.yaml — existing grid config format
5. configs/hpo/dynunet_example.yaml — existing Bayesian HPO config
6. deployment/skypilot/smoke_test_gpu.yaml — SkyPilot YAML template
7. deployment/pulumi/gcp/__main__.py — existing Pulumi IaC
8. knowledge-graph/decisions/L3-technology/hpo_engine.yaml — current KG state

WHAT TO BUILD (9 tasks from XML plan):
T1.1: Test — PARALLEL allocation returns disjoint trial partitions
T1.2: Implement grid partitioning in hpo_flow.py (modular arithmetic, Rule #17 compliant)
T1.3: Test — factorial config defines 24 cells (4 models × 3 losses × 2 aux_calib)
T1.4: Create paper_factorial.yaml + SkyPilot hpo_grid_worker.yaml
T1.5: MLflow provenance tags (grid_config_hash, git_sha, docker_image_digest)
T1.6: Test — GAR remote repository config valid
T1.7: Add GAR remote repo cache + SkyPilot MOUNT_CACHED
T1.8: Update KG hpo_engine node
T1.9: make test-staging — zero failures

GUARDRAILS:
- Grid worker uses Prefect flow module, NOT standalone script (Rule #17)
- SkyPilot uses image_id: docker:... (bare VM BANNED)
- No Optuna ask-tell or PostgreSQL distributed trials (deferred to #859)
- GAR remote repo in europe-north1
- The paper factorial has 24 cells (4×3×2), NOT 128. Do NOT add LR/batch as factors.
- 72 training runs total (24 cells × 3 folds). GCP cost: ~$65 (training only on L4 spot)
- Post-training, evaluation, biostatistics run LOCALLY on RTX 2070 Super (8 GB fits all models for inference)

Use /tdd-iterate (self-learning-iterative-coder skill) for TDD execution.
Branch: feat/hpo-grid-docker-pull
After all tests pass: git push, create PR targeting main.
```

---

## PR-2: Compliance & Regulatory (FDA Audit Trail)

**Issues**: #799, #821
**Branch**: `feat/compliance-audit-trail`
**Effort**: M (3-4 days)
**Parallel with**: PR-1, PR-3

### Cold-Start Prompt

```
Execute PR-2 from the pre-GCP housekeeping plan.

PLAN: Read docs/planning/pre-full-gcp-housekeeping-and-qa.xml — PR id="2"

CONTEXT TO READ FIRST:
1. src/minivess/compliance/ — all files (audit.py, lineage.py if exists)
2. src/minivess/observability/lineage.py — existing OpenLineage emission code
3. src/minivess/orchestration/flows/train_flow.py — where to wire log_data_access()
4. tests/v2/unit/test_flow_lineage_wiring.py — existing lineage tests
5. tests/v2/unit/test_sbom_generation.py — existing SBOM tests
6. deployment/docker-compose.yml — Marquez service placeholder
7. .env.example — existing env vars
8. docs/planning/openlineage-marquez-iec62304-report.md — compliance context

WHAT TO BUILD (7 tasks):
T2.1: Test — train flow calls log_data_access(dataset_name: str, file_paths: list[str])
T2.2: Wire log_data_access() into train flow (NOT split names — actual API signature)
T2.3: Test — ModelCard generates valid HuggingFace YAML
T2.4: Implement ModelCard generator (reads MLflow run metadata)
T2.5: Test — LineageEmitter sends valid OpenLineage v2 events
T2.6: Configure HttpTransport + add MARQUEZ_URL to .env.example
T2.7: make test-staging — zero failures

GUARDRAILS:
- log_data_access() takes (dataset_name: str, file_paths: list[str]) — NOT split names
- Marquez emission is MOCK-TESTED only (full deployment deferred to #860)
- MARQUEZ_URL in .env.example (single-source config, Rule #22)
- OpenLineage Phase 1 (emit-only) is already DONE — this PR adds Phase 2 transport config

Use /tdd-iterate for TDD execution.
Branch: feat/compliance-audit-trail
```

---

## PR-3: Observability Stubs + KG Fix (Quick Win)

**Issues**: #841, #848, #843
**Branch**: `feat/observability-stubs`
**Effort**: S (1-2 days)
**Parallel with**: PR-1, PR-2

### Cold-Start Prompt

```
Execute PR-3 from the pre-GCP housekeeping plan.

PLAN: Read docs/planning/pre-full-gcp-housekeeping-and-qa.xml — PR id="3"

CONTEXT TO READ FIRST:
1. src/minivess/observability/ — existing observability module
2. knowledge-graph/decisions/L2-architecture/agent_architecture.yaml — STALE: says LangGraph
3. knowledge-graph/domains/operations.yaml — add CISO Assistant note
4. .env.example — add SENTRY_DSN and POSTHOG_KEY
5. pyproject.toml — check if sentry-sdk, posthog are already optional deps

WHAT TO BUILD (6 tasks):
T3.1: Test — Sentry/PostHog init when env vars set, no-op when empty
T3.2: Implement stubs in src/minivess/observability/monitoring.py + .env.example vars
T3.3: Fix KG agent_architecture.yaml — LangGraph (0.45) → Pydantic AI (0.90)
T3.4: Add CISO Assistant Community note to compliance_posture KG node
T3.5: Test — updated KG YAML files parse correctly
T3.6: make test-staging — zero failures

GUARDRAILS:
- Sentry/PostHog are OPTIONAL deps (lazy import, no ImportError in CI)
- Both DISABLED by default (empty env vars = no-op)
- KG update for #848 is a data fix: the codebase already uses Pydantic AI, the KG node is stale
- This is the quickest PR — aim for 1 day

Use /tdd-iterate for TDD execution.
Branch: feat/observability-stubs
```

---

## PR-4: Agentic Dashboard (CopilotKit + AG-UI)

**Issues**: #840
**Branch**: `feat/agentic-dashboard`
**Effort**: L (5-7 days)
**Depends on**: PR-2 (OpenLineage), PR-3 (observability stubs)
**Timing**: Can run while GCP factorial executes

### Cold-Start Prompt

```
Execute PR-4 from the pre-GCP housekeeping plan.

PLAN: Read docs/planning/pre-full-gcp-housekeeping-and-qa.xml — PR id="4"
RESEARCH: Read docs/planning/biomedical-agentic-ai-research-report.md — Sections on CopilotKit, AG-UI

CONTEXT TO READ FIRST:
1. src/minivess/orchestration/flows/dashboard_flow.py — existing dashboard flow
2. src/minivess/agents/ — existing agent modules (agent_interface.py, agent_factory.py)
3. deployment/docker-compose.yml — where to add CopilotKit service
4. deployment/docker-compose.flows.yml — dashboard service config
5. knowledge-graph/decisions/L2-architecture/agent_architecture.yaml — now says Pydantic AI (after PR-3)
6. docs/planning/bentoml-and-ui-demo-plan.md — existing UI plans

WHAT TO BUILD (6 tasks):
T4.1: Test — CopilotKit docker-compose service config valid
T4.2: Add CopilotKit server to docker-compose + Dockerfile
T4.3: Test — AG-UI adapter translates WebMCP messages
T4.4: Implement AG-UI protocol adapter
T4.5: Pydantic AI dashboard agent + Flow 5 wiring (MLflow/DuckDB query tools)
T4.6: make test-staging — zero failures

GUARDRAILS:
- CopilotKit is a Docker service, not standalone Node.js
- Dashboard agent uses Pydantic AI (NOT LangGraph) — PR-3 already fixed the KG
- Lineage display is read-only (no writes to Marquez from dashboard)
- All new services go in deployment/docker-compose.yml on minivess-network

Use /tdd-iterate for TDD execution.
Branch: feat/agentic-dashboard
```

---

## PR-5: KG-Enrichment Agent

**Issues**: #849
**Branch**: `feat/kg-enrichment-agent`
**Effort**: M (3-4 days)
**Depends on**: PR-3 (telemetry hooks)
**Timing**: Can run while GCP factorial executes

### Cold-Start Prompt

```
Execute PR-5 from the pre-GCP housekeeping plan.

PLAN: Read docs/planning/pre-full-gcp-housekeeping-and-qa.xml — PR id="5"

CONTEXT TO READ FIRST:
1. knowledge-graph/navigator.yaml — KG structure and routing
2. knowledge-graph/_network.yaml — all decision nodes
3. knowledge-graph/decisions/ — sample decision YAML files (read 2-3)
4. src/minivess/agents/ — existing agent scaffolding
5. .claude/skills/create-literature-report/ — existing literature skill (pattern to follow)
6. docs/planning/knowledge-management-upgrade.md — KG enrichment vision

WHAT TO BUILD (6 tasks):
T5.1: Test — PubMed search returns structured metadata
T5.2: Implement PubMed search (NCBI E-utilities, 3 req/s rate limit)
T5.3: Entity extraction from abstracts (Pydantic AI structured output)
T5.4: Test — contradiction detector flags KG conflicts
T5.5: KG update proposals + contradiction detection (human review gate)
T5.6: make test-staging — zero failures

GUARDRAILS:
- PubMed rate limit: 3 req/s (NCBI policy) — enforce with sleep
- Entity extraction uses Pydantic AI structured output, NOT regex (Rule #16)
- KG updates NEVER auto-applied — proposals saved for human review
- Follow the create-literature-report skill pattern for agent architecture

Use /tdd-iterate for TDD execution.
Branch: feat/kg-enrichment-agent
```

---

## PR-6: Research Agents (Acquisition + Annotation + Self-Evolving)

**Issues**: #851, #853, #854
**Branch**: `feat/research-agents`
**Effort**: XL (7-10 days)
**Depends on**: PR-1, PR-2, PR-4, PR-5
**Timing**: Can run while GCP factorial executes

### Cold-Start Prompt

```
Execute PR-6 from the pre-GCP housekeeping plan.

PLAN: Read docs/planning/pre-full-gcp-housekeeping-and-qa.xml — PR id="6"
RESEARCH: Read docs/planning/biomedical-agentic-ai-research-report.md — TissueLab, conformal bandit sections

CONTEXT TO READ FIRST:
1. src/minivess/agents/ — all existing agent modules
2. src/minivess/ensemble/ — conformal prediction modules (7+ files)
3. src/minivess/orchestration/flows/ — all flow files (understand flow patterns)
4. src/minivess/orchestration/constants.py — flow name constants
5. knowledge-graph/decisions/L2-architecture/agent_architecture.yaml — Pydantic AI resolved
6. docs/planning/conformal-uq-segmentation-report.md — conformal prediction context
7. docs/planning/interactive-segmentation-report.md — annotation context

WHAT TO BUILD (5 tasks):
T6.1: Conformal bandit acquisition agent (Flow 0b) — Thompson sampling + PCCP budget
T6.2: Active learning annotation agent — volume ranking by model disagreement
T6.3: Self-evolving segmentation agent (TissueLab pattern) — drift → retrain trigger
T6.4: PCCP constraint enforcement layer — all agents must pass confidence gates
T6.5: make test-staging — zero failures

GUARDRAILS:
- All agents use Pydantic AI (NOT LangGraph)
- PCCP enforcement mandatory for ALL agent actions
- Agents NEVER modify data/models without human approval gate
- Self-evolving agent uses Prefect deployments (Rule #17) — no standalone scripts
- Retraining triggers via run_deployment(), not direct training calls
- This is the most ambitious PR — consider splitting into 3 sub-PRs if context gets large

Use /tdd-iterate for TDD execution.
Branch: feat/research-agents
```

---

## How to Launch

**Do NOT use wrapper scripts around `claude -p` (BANNED — see metalearning).**

Copy-paste the cold-start prompt block for each PR directly into a fresh Claude Code
session. Each prompt is self-contained with all context references.

### Group A (Day 1 — 3 parallel Claude Code sessions):

1. Open terminal 1 → `claude` → paste PR-1 cold-start prompt
2. Open terminal 2 → `claude` → paste PR-2 cold-start prompt
3. Open terminal 3 → `claude` → paste PR-3 cold-start prompt

### Group B (after Group A PRs merged to main):

1. `git checkout main && git pull` in each terminal
2. Open terminal 1 → `claude` → paste PR-4 cold-start prompt
3. Open terminal 2 → `claude` → paste PR-5 cold-start prompt

### Sequential (after Group B PRs merged — single session, 1 agent):

Use the cold-start prompt below. Copy-paste into a fresh `claude` session.

```
I have this planning done for pre-GCP housekeeping PRs. Groups A and B are COMPLETE:
- Group A: PRs #865, #866, #867 (merged to main)
- Group B: PRs #868, #869 (review/merge pending)

Now execute PR-6 (Research Agents) — the final PR, sequential.

PLAN: Read docs/planning/pre-full-gcp-housekeeping-and-qa.xml — PR id="6"
PRIOR RESULTS: Read the <execution-history> section in the XML for Groups A+B actuals.

EXECUTION MODEL:
- Single agent (PR-6 is XL, most ambitious — full context budget)
- Read context → TDD (RED-GREEN-VERIFY-FIX) → make test-staging → commit → push → create PR
- Track wall time
- After complete: update XML plan <execution-audit> block with actual wall time + test counts

PR-6: Research Agents (Acquisition + Annotation + Self-Evolving)
  Branch: feat/research-agents
  Issues: #851, #853, #854
  Depends on: PR-1 (MERGED), PR-2 (MERGED), PR-4 (MERGED), PR-5 (MERGED)
  Cold-start context from XML plan PR id="6"
  Research: docs/planning/biomedical-agentic-ai-research-report.md — TissueLab, conformal bandit
  Research: docs/planning/conformal-uq-segmentation-report.md — conformal prediction
  Research: docs/planning/interactive-segmentation-report.md — annotation context

GUARDRAILS:
- from __future__ import annotations at top of every Python file
- Use pathlib.Path, encoding='utf-8', datetime.now(timezone.utc)
- No import re for structured data (Rule #16)
- All agents use Pydantic AI (NOT LangGraph)
- PCCP enforcement mandatory for ALL agent actions
- Agents NEVER modify data/models without human approval gate
- Self-evolving agent uses Prefect deployments (Rule #17) — no standalone scripts
- Retraining triggers via run_deployment(), not direct training calls
- uv run pytest for tests, make test-staging for final verification
- Do NOT enable GitHub Actions CI triggers
- Consider splitting into 3 sub-PRs if context gets large

After PR completes and results logged: stop and report.
I will review, merge to main, then promote main → prod.
```

### After all PRs merged — promote to prod:

```bash
# Create promotion PR: main → prod
gh pr create --base prod --title "Promote: pre-GCP housekeeping complete" --body "All 6 PRs merged. Run make test-prod."
make test-prod  # Full suite verification before merge
```
