# Knowledge Management Upgrade: From Scattered Files to Navigable Graph

**Created**: 2026-03-13
**Status**: Design — ready for review
**Scope**: Restructure the entire knowledge layer of MinIVess MLOps v2 into a
coherent, progressively-disclosed, agent-queryable knowledge graph with OpenSpec
as the SDD framework and reviewer agents for integrity maintenance.

---

## 1. Problem Diagnosis: Five Disconnected Systems

The project currently operates five knowledge systems that evolved independently and
share no common navigation, no cross-references, and no consistency enforcement:

```
┌─────────────────┐  ┌───────────────────┐  ┌──────────────────┐
│  CLAUDE.md +     │  │  Probabilistic    │  │  Planning Docs   │
│  MEMORY.md       │  │  PRD Blueprint    │  │  (118 files,     │
│  (Layer 0-1)     │  │  (52 nodes,       │  │   48,230 lines,  │
│                  │  │   never material- │  │   no status      │
│  594 + 207 lines │  │   ized as YAML)   │  │   index)         │
└────────┬─────────┘  └────────┬──────────┘  └────────┬─────────┘
         │ refs                │ describes              │ some overlap
         ▼                     ▼                        ▼
┌─────────────────┐  ┌───────────────────┐  ┌──────────────────┐
│  Metalearning    │  │  Implementation   │  │  GitHub Issues   │
│  (19 docs,       │  │  (actual code +   │  │  & PRs           │
│   failure rules) │  │   tests + configs)│  │  (612 issues,    │
│                  │  │                   │  │   68 PRs)        │
└──────────────────┘  └───────────────────┘  └──────────────────┘
```

**What's wrong:**

| Symptom | Root Cause | Impact |
|---------|-----------|--------|
| PRD exists only as prose blueprint (726 lines) | Never materialized to YAML — designed but not built | Cannot query "what's decided vs open" programmatically |
| Planning docs are a flat pile of 118 files | No status tracking, no navigator, no lifecycle | Agent must search 48K lines to find relevant context |
| MEMORY.md is over 200-line limit | Accumulates CRITICAL sections instead of indexing | System-reminder truncation warning every session |
| Implementation and plans are disconnected | No link from `compose.py` back to PRD node `config_architecture` | Decisions lose their rationale; same research repeated |
| Metalearning has no forward links | Docs explain past failures but don't connect to current rules | Rules in CLAUDE.md and metalearning drift apart |
| Skills reference stale Project #1 field IDs | No automated consistency check across knowledge artifacts | Sync-roadmap skill broken for Project #5 |

### 1.1 Why Scattered Markdown Doesn't Scale

[McMillan (2026). "Structured Context Engineering for File-Native Agentic Systems."](https://github.com/Fission-AI/OpenSpec)
established through 9,649 experiments that **format choice (YAML vs Markdown vs JSON)
does not significantly affect accuracy** (p=0.484). What matters is:

1. **Information architecture** — how content is partitioned, navigated, and structured
2. **Domain partitioning** with navigator files — scales to 10,000 tables while keeping
   per-query context bounded
3. **Grep-ability** — predictable patterns that frontier models can pattern-match in 3
   attempts (vs 16 for unfamiliar formats)

Our current system violates all three:
- **No information architecture**: 118 planning docs with no index, no status, no
  hierarchy beyond flat `docs/planning/`
- **No domain partitioning**: A single `CLAUDE.md` (594 lines) is the only navigator;
  everything else requires grep-based discovery
- **Low grep-ability**: Inconsistent naming conventions, no standardized frontmatter,
  no machine-readable metadata

### 1.2 What the Research Says We Need

Drawing from the manuscript resources in
`sci-llm-writer/manuscripts/agentic-development/resources/`:

| Source | Key Insight | Application to MinIVess |
|--------|------------|------------------------|
| [McMillan 2026](https://arxiv.org/abs/placeholder) | Domain partitioning + navigator files scale to 10K entries | Create `navigator.yaml` that maps queries to knowledge domains |
| [Vasilopoulos 2026](https://arxiv.org/abs/placeholder) | 3-tier codified context: Hot (constitution) → Specialist (agents) → Cold (specs via MCP) | Map to our Layer 0/1/2 but add trigger tables for agent routing |
| [Yu et al. 2026 (AgeMem)](https://arxiv.org/abs/placeholder) | Working/episodic/semantic/procedural memory with progressive RL | Our MEMORY.md = semantic; metalearning = episodic; skills = procedural; conversation = working |
| [Xu et al. 2025](https://arxiv.org/abs/placeholder) | File system as persistent context infrastructure: History → Memory → Scratchpad lifecycle | Planning docs = History; MEMORY.md = Memory; conversation context = Scratchpad |
| [Gupta 2025](https://arxiv.org/abs/placeholder) | Context graphs = accumulated decision traces stitched across entities and time | PRD decisions are exactly this — accumulated evidence-based choices with provenance |
| [Anthropic 2025](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering) | Just-in-time retrieval + progressive disclosure + sub-agent architecture | Our Layer 0/1/2 is the right pattern; execution needs improvement |

### 1.3 The Gap Between "Converted to Markdown" and "Structured for Agent Consumption"

This is the McMillan insight that directly applies: our knowledge **exists** but is not
**structured for agent consumption**. Converting everything to a single format won't help.
What we need is:

1. **A navigator** — a machine-readable index that maps topics to files
2. **Standardized frontmatter** — every knowledge document declares its type, status,
   connections, and freshness
3. **Bidirectional links** — every implementation knows which PRD decision it resolves;
   every PRD decision knows which files implement it
4. **Reviewer agents** — automated agents that detect drift between the systems

---

## 2. Target Architecture: The Knowledge Graph

### 2.1 Five-Layer Progressive Disclosure

Adapting McMillan's domain-partitioning pattern and Vasilopoulos's 3-tier codified
context into a 5-layer system tuned for our project:

```
Layer 0: NAVIGATOR (always loaded, ~50 lines)
  ├── knowledge-graph/navigator.yaml
  │   Maps: topic → domain → file(s)
  │   Example: "loss function" → L3-technology → decisions/loss_function.yaml
  │
Layer 1: CONSTITUTION (always loaded, ~800 lines total)
  ├── CLAUDE.md (project rules + principles)
  ├── MEMORY.md (index to topic files, <200 lines)
  │
Layer 2: DOMAIN EXPERTS (loaded on file access, ~2K lines total)
  ├── src/minivess/*/CLAUDE.md (7 existing + 4 new)
  ├── deployment/CLAUDE.md
  ├── tests/CLAUDE.md
  │
Layer 3: DECISIONS (loaded on demand, ~4K lines YAML)
  ├── knowledge-graph/decisions/*.yaml (52 PRD decision files)
  ├── knowledge-graph/_network.yaml (DAG edges)
  ├── knowledge-graph/scenarios/*.yaml
  ├── knowledge-graph/domains/*.yaml
  │
Layer 4: EVIDENCE (loaded on deep traversal, ~50K lines)
  ├── docs/planning/*.md (research reports, with frontmatter)
  ├── .claude/metalearning/*.md (failure analysis)
  ├── openspec/specs/*.md (SDD specifications)
  ├── openspec/changes/archive/*.md (change history)
```

**Key difference from current system**: Layer 0 is a **navigator** — a machine-readable
YAML file that an agent reads FIRST to determine which domain file to load. This is the
McMillan domain-partitioning pattern that scales to 10K entries.

### 2.2 The Navigator File

```yaml
# knowledge-graph/navigator.yaml
# PURPOSE: Entry point for all knowledge queries.
# An agent reads THIS FIRST, then loads only the relevant domain file(s).
# Pattern: McMillan 2026 domain-partitioned schema navigation.

version: "1.0"
last_updated: "2026-03-13"

# Topic → domain mapping (agent reads this to route queries)
domains:
  architecture:
    navigator: knowledge-graph/domains/architecture.yaml
    covers: [model adapters, ensemble strategies, config systems, serving, API]
    claude_md: [src/minivess/adapters/CLAUDE.md, src/minivess/config/CLAUDE.md]
    decisions: [model_adapter_pattern, ensemble_strategy, config_architecture,
                serving_architecture, api_protocol]

  training:
    navigator: knowledge-graph/domains/training.yaml
    covers: [loss functions, metrics, augmentation, HPO, calibration]
    claude_md: [src/minivess/pipeline/CLAUDE.md]
    decisions: [loss_function, primary_metrics, topology_metrics, hpo_engine,
                augmentation_library, calibration_method]

  infrastructure:
    navigator: knowledge-graph/domains/infrastructure.yaml
    covers: [Docker, Prefect, CI/CD, GPU compute, SkyPilot, IaC, security]
    claude_md: [deployment/CLAUDE.md, src/minivess/orchestration/CLAUDE.md]
    decisions: [container_strategy, ci_cd_platform, iac_tool, gitops_engine,
                gpu_compute, secrets_management, air_gap_strategy]

  data:
    navigator: knowledge-graph/domains/data.yaml
    covers: [datasets, validation, profiling, versioning, lineage, drift]
    claude_md: [src/minivess/data/CLAUDE.md]
    decisions: [data_versioning, data_validation_depth, dataframe_validation,
                data_profiling, label_quality_tool, lineage_tracking, drift_monitoring]

  models:
    navigator: knowledge-graph/domains/models.yaml
    covers: [DynUNet, SAM3, SegResNet, UNETR, SwinUNETR, VesselFM, Mamba, foundation]
    claude_md: [src/minivess/adapters/CLAUDE.md]
    decisions: [primary_3d_model, foundation_model]

  observability:
    navigator: knowledge-graph/domains/observability.yaml
    covers: [MLflow, DuckDB, Langfuse, Braintrust, LiteLLM, agents, XAI]
    claude_md: [src/minivess/observability/CLAUDE.md]  # NEW — to be created
    decisions: [experiment_tracker, llm_tracing, llm_evaluation,
                llm_provider_strategy, xai_strategy, xai_voxel_tool,
                xai_meta_tool, observability_depth, agent_architecture]

  operations:
    navigator: knowledge-graph/domains/operations.yaml
    covers: [monitoring, governance, retraining, SBOM, compliance, audit, federated]
    claude_md: []  # Spread across deployment/ and orchestration/
    decisions: [dashboarding, retraining_trigger, model_governance,
                audit_trail, sbom_generation, federated_learning]

  testing:
    navigator: knowledge-graph/domains/testing.yaml
    covers: [test tiers, markers, fixtures, CI, QA]
    claude_md: [tests/CLAUDE.md]
    decisions: []

# Quick-lookup: keyword → most relevant domain
keywords:
  docker: infrastructure
  prefect: infrastructure
  skypilot: infrastructure
  loss: training
  metric: training
  hpo: training
  optuna: training
  sam3: models
  dynunet: models
  mamba: models
  mlflow: observability
  langfuse: observability
  drift: data
  dataset: data
  minivess: data
  security: infrastructure
  onnx: architecture
  bentoml: architecture
  calibration: training
  ensemble: architecture
  xai: observability
  compliance: operations
  audit: operations
```

### 2.3 Domain Files

Each domain file is a detailed navigator for that subdomain. Example:

```yaml
# knowledge-graph/domains/training.yaml
domain: training
description: Loss functions, metrics, augmentation, HPO, calibration
last_reviewed: "2026-03-13"

decisions:
  loss_function:
    status: resolved
    winner: cbdice_cldice
    rationale: "dynunet_loss_variation_v2 experiment — 0.906 clDice, best topology"
    implementation: src/minivess/pipeline/loss_functions.py
    evidence: [docs/planning/loss-function-topology-research.md,
               docs/results/dynunet_loss_variation_v2_report.md]
    prd_node: knowledge-graph/decisions/L3/loss_function.yaml

  primary_metrics:
    status: resolved
    winner: metricsreloaded_full
    implementation: src/minivess/pipeline/metrics.py
    prd_node: knowledge-graph/decisions/L3/primary_metrics.yaml

  topology_metrics:
    status: not_started
    candidates: [betti_gudhi, skeleton_precision_recall, cldice_as_metric]
    evidence: [docs/planning/GRAPH-TOPOLOGY-METRICS-INDEX.md]
    prd_node: knowledge-graph/decisions/L3/topology_metrics.yaml

  hpo_engine:
    status: resolved
    winner: optuna_multi_objective
    implementation: src/minivess/optimization/hpo_engine.py
    evidence: [docs/planning/hpo-optuna-asha-integration-plan.md]
    prd_node: knowledge-graph/decisions/L3/hpo_engine.yaml

metalearning:
  # Failure patterns relevant to this domain
  - file: .claude/metalearning/2026-03-06-regex-ban.md
    applies_to: [loss_functions.py, metrics.py]
    rule: "No regex for metric name parsing — use str.split()"

planning_docs:
  # Research reports and plans in this domain
  - file: docs/planning/loss-function-topology-research.md
    status: implemented
    phase: P1
  - file: docs/planning/boundary-loss-implementation-plan.md
    status: implemented
    phase: P2
  - file: docs/planning/hpo-optuna-asha-integration-plan.md
    status: partial
    phase: P4

open_issues:
  - number: 611
    title: "HPO completion barrier: Analysis Flow must wait for all trials"
```

---

## 3. OpenSpec as the SDD Framework

### 3.1 Why OpenSpec Wins

The [OpenSpec](https://github.com/Fission-AI/OpenSpec) framework by Fission-AI is
the winner of the SDD framework comparison for this project:

| Criterion | OpenSpec | spec-kit | BMAD-METHOD | PromptX |
|-----------|---------|----------|-------------|---------|
| Delta-based change tracking | Native (ADDED/MODIFIED/REMOVED/RENAMED) | No | No | No |
| Filesystem-only (no API keys) | Yes | Yes | Yes | No (MCP) |
| Tool-agnostic | 30+ tool adapters | Claude-specific | Tool-agnostic | Tool-specific |
| Ceremony overhead | Minimal | Heavy (4-stage gates) | Medium (persona simulation) | Minimal |
| Claude Code integration | Native slash commands | N/A | N/A | N/A |
| Iterative evolution support | Delta specs track rationale | No change tracking | No change tracking | No change tracking |

**Note on "brownfield"**: OpenSpec markets itself as "brownfield-first" (optimized for
mature codebases). MinIVess v2 is a **clean rewrite** from v0.1-alpha with zero backwards
compatibility — not brownfield in the traditional sense. However, after 523 closed issues
and 68 merged PRs in 3 weeks, the codebase has real architectural invariants that future
changes must respect. The value of OpenSpec for us is NOT legacy modernization — it is:

1. **Delta-based change tracking** — captures *what changed and why* as flows evolve
   (e.g., `MODIFIED: training flow SHALL tag runs with flow_name="training-flow"`)
2. **Filesystem-only** — no API keys, no MCP server, no database (matches local-first)
3. **GIVEN/WHEN/THEN scenarios** — requirements are testable, mapping directly to our
   TDD-first mandate
4. **Archive as decision history** — every merged change preserves its proposal + rationale,
   creating the "context graph" (Gupta 2025) of accumulated decision traces

### 3.2 OpenSpec Directory Structure for MinIVess

```
openspec/
├── config.yaml                    # Project config (tech stack, rules)
├── project.md                     # Global context = CLAUDE.md distillation
├── AGENTS.md                      # AI behavioral guidelines
├── specs/                         # SOURCE OF TRUTH (current system state)
│   ├── data-pipeline/
│   │   ├── spec.md                # Data flow: download, validate, split, profile
│   │   └── design.md              # Pydantic schemas, Pandera, Great Expectations
│   ├── training-pipeline/
│   │   ├── spec.md                # Training flow: Hydra config → fold iteration → MLflow
│   │   └── design.md              # Docker-per-flow, STOP protocol, volume mounts
│   ├── analysis-pipeline/
│   │   ├── spec.md                # Analysis flow: ensemble, evaluation, comparison
│   │   └── design.md              # Inter-flow contract, find_upstream_run()
│   ├── deployment-pipeline/
│   │   ├── spec.md                # Deploy flow: ONNX export, BentoML, promotion
│   │   └── design.md              # Model registry, champion tagging
│   ├── dashboard-pipeline/
│   │   ├── spec.md                # Dashboard flow: figures, Parquet, QA health
│   │   └── design.md              # Paper artifacts, DuckDB analytics
│   ├── model-adapters/
│   │   ├── spec.md                # ModelAdapter ABC contract
│   │   └── design.md              # MONAI-first principle, VRAM constraints
│   ├── config-system/
│   │   ├── spec.md                # Hydra-zen + Dynaconf dual config
│   │   └── design.md              # compose_experiment_config() pipeline
│   ├── drift-monitoring/
│   │   ├── spec.md                # Evidently + kernel MMD drift detection
│   │   └── design.md              # MLflow artifact storage, embedding drift
│   └── inter-flow-contract/
│       ├── spec.md                # MLflow tags, find_upstream_run(), FLOW_NAME_*
│       └── design.md              # Tag naming, None-safety, artifact discovery
├── changes/
│   └── archive/                   # Completed changes (timestamped)
│       ├── 2026-03-10_inter-flow-contract-fix/
│       │   ├── proposal.md        # Why: flow_name tag was missing
│       │   └── specs/inter-flow-contract/spec.md  # Delta
│       └── 2026-03-12_drift-detection/
│           ├── proposal.md
│           └── specs/drift-monitoring/spec.md
└── schemas/
    └── minivess-flow.yaml         # Custom schema for flow specifications
```

### 3.3 Integration with Knowledge Graph

OpenSpec specs are Layer 4 evidence documents. The navigator routes to them:

```yaml
# In knowledge-graph/domains/infrastructure.yaml
decisions:
  container_strategy:
    status: resolved
    winner: docker_compose_profiles
    openspec: openspec/specs/training-pipeline/design.md  # ← new link
    implementation: deployment/docker-compose.flows.yml
```

### 3.4 OpenSpec Lifecycle Maps to Our Development Cycle

| OpenSpec Phase | MinIVess Equivalent | Example |
|---------------|---------------------|---------|
| **Propose** (`/opsx:new`) | Planning doc + GitHub issue | `docs/planning/drift-monitoring-implementation-plan.xml` + Issue #574 |
| **Define** (delta specs) | TDD RED phase — write failing tests from spec | `tests/v2/unit/test_drift_detection.py` |
| **Apply** (implement) | TDD GREEN phase — implement to pass tests | `src/minivess/pipeline/drift_detection.py` |
| **Archive** (merge specs) | TDD CHECKPOINT — git commit + state update | PR #608 merged, spec updated |

---

## 4. Materializing the PRD as YAML

### 4.1 Directory Structure

The PRD blueprint (`docs/planning/hierarchical-prd-planning.md`) designed 52 nodes.
We now materialize them into the knowledge graph:

```
knowledge-graph/
├── navigator.yaml                 # Entry point (Layer 0)
├── _network.yaml                  # DAG edges (52 nodes, ~80 edges)
├── _schema.yaml                   # Decision node YAML schema
├── bibliography.yaml              # All cited works (PRD Rule #1)
├── decisions/
│   ├── L1-research-goals/
│   │   ├── project_purpose.yaml
│   │   ├── impact_target.yaml
│   │   ├── monai_alignment.yaml
│   │   ├── model_philosophy.yaml
│   │   ├── compliance_posture.yaml
│   │   ├── reproducibility_level.yaml
│   │   └── portfolio_role_target.yaml
│   ├── L2-architecture/
│   │   ├── model_adapter_pattern.yaml
│   │   ├── ensemble_strategy.yaml
│   │   ├── uncertainty_framework.yaml
│   │   ├── config_architecture.yaml
│   │   ├── serving_architecture.yaml
│   │   ├── xai_strategy.yaml
│   │   ├── data_validation_depth.yaml
│   │   ├── agent_architecture.yaml
│   │   ├── observability_depth.yaml
│   │   └── api_protocol.yaml
│   ├── L3-technology/             # 20 nodes
│   │   └── ...
│   ├── L4-infrastructure/         # 8 nodes
│   │   └── ...
│   └── L5-operations/             # 7 nodes
│       └── ...
├── scenarios/
│   ├── learning-first-mvp.yaml    # Current active scenario
│   ├── research-scaffold.yaml     # Paper-optimized
│   └── clinical-production.yaml   # Full compliance
├── domains/
│   ├── architecture.yaml          # Domain navigator
│   ├── training.yaml
│   ├── infrastructure.yaml
│   ├── data.yaml
│   ├── models.yaml
│   ├── observability.yaml
│   ├── operations.yaml
│   ├── testing.yaml
│   ├── vascular-segmentation/     # Clinical domain overlay
│   │   └── overlay.yaml
│   └── backbone-defaults.yaml
└── archetypes/
    ├── solo-researcher.yaml
    ├── lab-group.yaml
    └── clinical-deployment.yaml
```

### 4.2 Decision Node Format (YAML)

YAML chosen over Markdown/JSON for PRD nodes because:
- 28-60% fewer tokens than other formats (McMillan 2026)
- Most grep-able by frontier models (3 attempts vs 16 for TOON)
- Native support in Python via `yaml.safe_load()` (respects our regex ban)

```yaml
# knowledge-graph/decisions/L3-technology/loss_function.yaml
decision_id: loss_function
title: "Primary Loss Function"
level: L3-technology
status: resolved  # resolved | partial | config_only | not_started

options:
  - id: dice_ce_combined
    name: "Dice + Cross-Entropy Combined"
    prior_probability: 0.40
    posterior_probability: 0.10  # After experiment evidence
  - id: cldice_soft_skeleton
    name: "clDice + Soft Skeleton (cbdice_cldice)"
    prior_probability: 0.30
    posterior_probability: 0.85  # WINNER — dynunet_loss_variation_v2
  - id: generalized_dice
    name: "Generalized Dice Loss"
    prior_probability: 0.20
    posterior_probability: 0.03
  - id: focal_tversky
    name: "Focal Tversky Loss"
    prior_probability: 0.10
    posterior_probability: 0.02

resolved_option: cldice_soft_skeleton
resolution_evidence:
  - type: experiment
    source: "docs/results/dynunet_loss_variation_v2_report.md"
    finding: "0.906 clDice (best topology) with only -5.3% DSC penalty"
  - type: literature
    citation_key: shit_2021_cldice
    source: "https://arxiv.org/abs/2003.07311"

implementation:
  files:
    - src/minivess/pipeline/loss_functions.py
    - configs/base.yaml  # default_loss: cbdice_cldice
  tests:
    - tests/v2/unit/test_loss_functions.py
  openspec: openspec/specs/training-pipeline/spec.md

conditional_on:
  - parent_decision_id: model_philosophy
    influence_strength: moderate

volatility:
  classification: stable
  next_review: "2026-06-13"
  rationale: "Empirically validated on MiniVess; unlikely to change unless new topology loss published"

domain_overlays:
  vascular-segmentation:
    probability_shift:
      cldice_soft_skeleton: +0.20
      dice_ce_combined: -0.15
    rationale: "Topology preservation critical for vessel connectivity"
```

### 4.3 Network DAG File

```yaml
# knowledge-graph/_network.yaml
# Bayesian decision network: 52 nodes, ~80 edges
# Validation: acyclic, referentially complete, level-ordered

nodes:
  - id: project_purpose
    level: L1
    file: decisions/L1-research-goals/project_purpose.yaml
  - id: loss_function
    level: L3
    file: decisions/L3-technology/loss_function.yaml
  # ... 50 more nodes

edges:
  # L1 → L2
  - from: project_purpose
    to: observability_depth
    strength: moderate
  - from: monai_alignment
    to: model_adapter_pattern
    strength: strong
  # L2 → L3
  - from: ensemble_strategy
    to: calibration_method
    strength: moderate
  # Skip connections
  - from: monai_alignment
    to: model_export_format
    strength: strong
    skip: true
  # ... ~75 more edges
```

---

## 5. v0.1-alpha Cleanup: Remove Dead Weight

### 5.1 What Remains from v0.1

The v0.1 source code was already deleted (12,270 LOC in commit `b3367de`, Issue #34).
Git tags `v0.1-alpha` and `v0.1-archive` preserve the full history at zero cost.
What still occupies disk and context budget:

| Artifact | Location | Lines | Status | Action |
|----------|----------|-------|--------|--------|
| Legacy config | `configs/_legacy_v01_defaults.yaml` | 469 | Dead reference | **DELETE** |
| Wiki directory | `wiki/` (15 files, 380 KB) | 1,590 | 12/15 files stale | **DELETE** |
| Wiki `.git` | `wiki/.git/` | N/A | Separate git repo | **DELETE** (subdir) |
| GitHub milestone | `v0.1-alpha` (13 closed, 0 open) | N/A | Closed, historical | **KEEP** |
| Git tags | `v0.1-alpha`, `v0.1-archive` | N/A | Read-only history | **KEEP** |

### 5.2 Why Delete, Not Archive

The wiki contains tutorials on Poetry, old BentoML, old Docker, Jupyter notebooks —
none of which apply to v2. An AI agent that reads `wiki/Docker.md` will find patterns
that contradict `deployment/CLAUDE.md`. The legacy config file references AWS EC2 servers,
Weights & Biases, and Novograd optimizer — all irrelevant noise.

Git tags already preserve everything. `git show v0.1-alpha:src/training/train.py` works
forever. Keeping stale files on disk is a liability, not an asset.

### 5.3 Execution

```bash
# Delete legacy config (preserved at v0.1-archive tag)
rm configs/_legacy_v01_defaults.yaml

# Delete stale wiki (separate git repo, preserved at its own HEAD)
rm -rf wiki/

# Verify nothing references these files
grep -r "_legacy_v01_defaults" src/ tests/ docs/ configs/
grep -r "wiki/" CLAUDE.md MEMORY.md
```

### 5.4 Keep Two Wiki Pages as Markdown in docs/

If `wiki/MLOps-Intro.md` (277 lines, general MLOps education) and `wiki/MLflow.md`
(105 lines, still-relevant MLflow basics) have lasting value, move them to
`docs/reference/` before deleting the wiki directory. Everything else is superseded.

---

## 6. Reviewer Agents: Automated Knowledge Integrity

### 6.1 Agent Architecture

Drawing from Vasilopoulos's trigger-table pattern (specialist agents with embedded
domain knowledge) and Mohamed's KG-Orchestra (multi-agent knowledge graph enrichment
with retrieval + validation + provenance agents), we define **5 reviewer agents**.

The key insight from Vasilopoulos: agents should contain 50%+ domain knowledge,
not just behavioral instructions. Each reviewer agent below embeds the validation
rules for its domain, not just "check things."

```
┌──────────────────────────────────────────────────────────────────┐
│                     REVIEWER ORCHESTRATOR                         │
│  Trigger: /review-knowledge or pre-commit hook                    │
│  Schedule: full review weekly, link check on every commit         │
│  Output: knowledge-graph/reports/review-YYYY-MM-DD.yaml           │
│  Integration: GitHub Issue auto-created for FAIL results          │
└───┬──────────┬──────────┬──────────┬──────────┬─────────────────┘
    │          │          │          │          │
┌───▼───┐ ┌───▼───┐ ┌───▼───┐ ┌───▼───┐ ┌───▼────┐
│ PRD   │ │ Spec  │ │ Link  │ │ Stale │ │ Legacy │
│ Audit │ │ Drift │ │ Check │ │ Scan  │ │ Detect │
└───────┘ └───────┘ └───────┘ └───────┘ └────────┘
```

### 6.2 Agent 1: PRD Auditor (`scripts/review_prd_integrity.py`)

**Purpose**: Verify that the 52 PRD decision nodes match implementation reality.

**Embedded domain knowledge**: The auditor knows the full decision schema, all 52
node IDs, the DAG topology, and the validation rules from Section 9 of the PRD
blueprint (`docs/planning/hierarchical-prd-planning.md`).

**Checks (17 invariants from PRD Section 9)**:

| # | Check | Severity | Method |
|---|-------|----------|--------|
| 1 | Prior probability sums = 1.0 per node (tolerance 0.01) | ERROR | `yaml.safe_load()` each decision, sum `prior_probability` |
| 2 | Conditional table row sums = 1.0 | ERROR | Parse `conditional_on` blocks |
| 3 | Conditional table completeness (all options covered) | ERROR | Cross-reference option IDs |
| 4 | Conditional parent coverage (all parent options have rows) | ERROR | Cross-reference parent decision options |
| 5 | Archetype override sums = 1.0 | ERROR | Parse archetype YAML files |
| 6 | DAG acyclicity | ERROR | Topological sort on `_network.yaml` |
| 7 | Referential integrity (parent IDs) | ERROR | Every `parent_decision_id` exists in `_network.yaml` |
| 8 | Referential integrity (option IDs) | ERROR | Every referenced option exists in its decision |
| 9 | Level ordering (no upward edges) | ERROR | Edges flow L1→L5, skip connections documented |
| 10 | Scenario completeness | WARN | Complete scenarios resolve all 52 decisions |
| 11 | Scenario consistency (no hard constraint violations) | ERROR | Check `constraints` fields |
| 12 | Scenario-archetype alignment | WARN | Resolved options consistent with archetype |
| 13 | Overlay sparsity (<50% of decisions overridden) | WARN | Count overrides per domain |
| 14 | Overlay probability sums = 1.0 | ERROR | Parse overlay YAML |
| 15 | Domain registry consistency (files match registry) | ERROR | Cross-reference `domains/registry.yaml` |
| 16 | Review dates not overdue | WARN | Compare `next_review` to today |
| 17 | Implementation file existence | ERROR | `Path(f).exists()` for each `implementation.files` entry |

**Additional implementation-specific checks**:

| # | Check | Severity | Method |
|---|-------|----------|--------|
| 18 | Resolved options match codebase usage | WARN | Grep for `resolved_option` value in implementation files |
| 19 | Every `citation_key` resolves to `bibliography.yaml` | ERROR | Load bibliography, check key existence |
| 20 | Posterior probabilities are monotonic with resolution | WARN | If `status: resolved`, winner posterior > 0.7 |
| 21 | No orphan nodes (every node in `_network.yaml` has a file) | ERROR | Glob `decisions/**/*.yaml` vs `_network.yaml` nodes |

**Implementation sketch**:
```python
"""PRD integrity reviewer — validates the Bayesian decision network.

Runs 21 checks across 52 decision nodes, producing a structured YAML report.
Uses yaml.safe_load() exclusively (no regex, per CLAUDE.md Rule #16).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml  # yaml.safe_load() only — regex banned


def load_network(kg_root: Path) -> dict:
    """Load _network.yaml and all decision files."""
    network = yaml.safe_load((kg_root / "_network.yaml").read_text(encoding="utf-8"))
    decisions = {}
    for node in network["nodes"]:
        decision_path = kg_root / node["file"]
        decisions[node["id"]] = yaml.safe_load(
            decision_path.read_text(encoding="utf-8")
        )
    return network, decisions


def check_probability_sums(decisions: dict) -> list[dict]:
    """Check 1: Prior probabilities sum to 1.0 per decision."""
    results = []
    for did, dec in decisions.items():
        total = sum(opt["prior_probability"] for opt in dec["options"])
        passed = abs(total - 1.0) < 0.01
        results.append({
            "check": "probability_sum",
            "decision": did,
            "value": round(total, 4),
            "status": "PASS" if passed else "FAIL",
        })
    return results


def check_dag_acyclicity(network: dict) -> dict:
    """Check 6: Topological sort succeeds (no cycles)."""
    # Kahn's algorithm
    adjacency: dict[str, list[str]] = {}
    in_degree: dict[str, int] = {}
    for node in network["nodes"]:
        nid = node["id"]
        adjacency.setdefault(nid, [])
        in_degree.setdefault(nid, 0)
    for edge in network["edges"]:
        adjacency[edge["from"]].append(edge["to"])
        in_degree[edge["to"]] = in_degree.get(edge["to"], 0) + 1

    queue = [n for n, d in in_degree.items() if d == 0]
    sorted_nodes = []
    while queue:
        node = queue.pop(0)
        sorted_nodes.append(node)
        for neighbor in adjacency.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    acyclic = len(sorted_nodes) == len(network["nodes"])
    return {
        "check": "dag_acyclicity",
        "status": "PASS" if acyclic else "FAIL",
        "sorted_count": len(sorted_nodes),
        "total_nodes": len(network["nodes"]),
    }


def check_implementation_exists(decisions: dict, repo_root: Path) -> list[dict]:
    """Check 17: Implementation files exist on disk."""
    results = []
    for did, dec in decisions.items():
        if dec.get("status") != "resolved":
            continue
        impl = dec.get("implementation", {})
        for f in impl.get("files", []):
            exists = (repo_root / f).exists()
            results.append({
                "check": "implementation_exists",
                "decision": did,
                "file": f,
                "status": "PASS" if exists else "FAIL",
            })
    return results


def check_review_dates(decisions: dict) -> list[dict]:
    """Check 16: next_review dates are not overdue."""
    today = datetime.now(timezone.utc).date()
    results = []
    for did, dec in decisions.items():
        vol = dec.get("volatility", {})
        next_review = vol.get("next_review")
        if next_review:
            review_date = datetime.strptime(next_review, "%Y-%m-%d").date()
            overdue = review_date < today
            results.append({
                "check": "review_date",
                "decision": did,
                "next_review": next_review,
                "status": "WARN" if overdue else "PASS",
            })
    return results


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    kg_root = repo_root / "knowledge-graph"

    if not kg_root.exists():
        print("ERROR: knowledge-graph/ directory not found")
        return 1

    network, decisions = load_network(kg_root)

    all_results = []
    all_results.extend(check_probability_sums(decisions))
    all_results.append(check_dag_acyclicity(network))
    all_results.extend(check_implementation_exists(decisions, repo_root))
    all_results.extend(check_review_dates(decisions))
    # ... remaining checks follow same pattern

    # Write report
    report = {
        "reviewer": "prd_auditor",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_checks": len(all_results),
        "passed": sum(1 for r in all_results if r["status"] == "PASS"),
        "warnings": sum(1 for r in all_results if r["status"] == "WARN"),
        "failures": sum(1 for r in all_results if r["status"] == "FAIL"),
        "results": all_results,
    }

    report_dir = kg_root / "reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"prd-audit-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.yaml"
    report_path.write_text(
        yaml.dump(report, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    print(f"PRD Audit: {report['passed']} PASS, {report['warnings']} WARN, {report['failures']} FAIL")
    return 1 if report["failures"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
```

### 6.3 Agent 2: Spec Drift Detector (`scripts/review_spec_drift.py`)

**Purpose**: Detect when OpenSpec specs diverge from implementation.

**Embedded domain knowledge**: Knows the GIVEN/WHEN/THEN scenario format, the
OpenSpec delta operations (ADDED/MODIFIED/REMOVED/RENAMED), and the mapping from
spec directories to flow files.

**Checks**:

| # | Check | Severity | Method |
|---|-------|----------|--------|
| 1 | Every spec dir has both `spec.md` and `design.md` | ERROR | Glob `openspec/specs/*/` |
| 2 | Every GIVEN/WHEN/THEN scenario has a matching test | WARN | Parse scenarios, match to `tests/` via naming convention |
| 3 | Every flow in `src/minivess/orchestration/flows/` has a spec | WARN | Cross-reference flow files to spec dirs |
| 4 | No archived deltas have unapplied ADDED requirements | ERROR | Parse archived delta specs, check current specs |
| 5 | `design.md` references valid implementation files | ERROR | Extract file paths from design docs, check existence |
| 6 | Spec requirements use RFC 2119 keywords (MUST/SHALL/SHOULD) | WARN | Parse for requirement keywords |

**Scenario-to-test mapping convention**:
```
openspec/specs/training-pipeline/spec.md:
  Scenario: Training flow tags runs with flow_name
  → tests/v2/unit/test_train_flow.py::test_training_flow_tags_flow_name

openspec/specs/inter-flow-contract/spec.md:
  Scenario: find_upstream_run filters by flow_name tag
  → tests/v2/unit/test_flow_contract.py::test_find_upstream_run_filters_flow_name
```

The detector parses scenario titles, converts to snake_case, and searches for matching
test function names. Unmatched scenarios are reported as WARN (untested requirement).

### 6.4 Agent 3: Link Checker (`scripts/review_knowledge_links.py`)

**Purpose**: Verify all cross-references in the knowledge graph are valid.

**This is the fastest reviewer** — runs in <5 seconds, suitable for pre-commit hooks.

**Checks**:

| # | Check | Severity | Method |
|---|-------|----------|--------|
| 1 | Every path in `navigator.yaml` exists on disk | ERROR | `Path(f).exists()` |
| 2 | Every `implementation` path in domain files exists | ERROR | Walk domain YAML, check paths |
| 3 | Every `evidence` path in domain files exists | ERROR | Walk domain YAML, check paths |
| 4 | Every `openspec` link points to a real spec | ERROR | Check `openspec/specs/*/` |
| 5 | Every metalearning doc in MEMORY.md exists | ERROR | Parse MEMORY.md, check `.claude/metalearning/` |
| 6 | Every `citation_key` resolves to `bibliography.yaml` | ERROR | Load bibliography, match keys |
| 7 | MEMORY.md is under 200 lines | WARN | `wc -l` equivalent |
| 8 | No broken Markdown links in planning docs | WARN | Parse `[text](path)` patterns, check local paths |
| 9 | Every domain navigator `last_reviewed` date is parseable | ERROR | `datetime.strptime()` |

**Pre-commit integration**:
```yaml
# .pre-commit-config.yaml (addition)
- repo: local
  hooks:
    - id: knowledge-links
      name: Knowledge graph link checker
      entry: uv run python scripts/review_knowledge_links.py --quick
      language: system
      pass_filenames: false
      files: '(knowledge-graph/|MEMORY\.md|CLAUDE\.md|docs/planning/)'
```

The `--quick` flag skips bibliography resolution and planning doc Markdown link
scanning — it only checks navigator.yaml paths, domain file paths, and MEMORY.md
references. Full check runs with `--full`.

### 6.5 Agent 4: Staleness Scanner (`scripts/review_staleness.py`)

**Purpose**: Identify knowledge artifacts that are outdated or superseded.

**Embedded domain knowledge**: Knows the frontmatter schema for planning docs, the
volatility classification system for PRD nodes, and the expected review cadence
(stable: 6 months, shifting: 3 months, volatile: 2 weeks).

**Checks**:

| # | Check | Severity | Method |
|---|-------|----------|--------|
| 1 | Every planning doc has frontmatter with `status` field | WARN | YAML frontmatter parser |
| 2 | `status: implemented` docs reference implementing PR | WARN | Check `implementing_pr` field |
| 3 | `status: active` docs not modified in >30 days | WARN | `git log -1 --format=%ai -- <file>` |
| 4 | PRD volatile nodes reviewed within 2 weeks | WARN | Check `volatility.next_review` |
| 5 | PRD shifting nodes reviewed within 3 months | WARN | Check `volatility.next_review` |
| 6 | Domain navigators reviewed within 30 days | WARN | Check `last_reviewed` field |
| 7 | Skills reference correct project/field IDs | ERROR | Parse skill files, validate against GitHub API |
| 8 | No planning docs without frontmatter (untagged) | WARN | Count docs without `---` header |

**Git-based freshness detection**:
```python
def get_last_modified(file_path: Path, repo_root: Path) -> str | None:
    """Get last modification date from git log."""
    import subprocess
    result = subprocess.run(
        ["git", "log", "-1", "--format=%aI", "--", str(file_path)],
        cwd=repo_root, capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None
```

### 6.6 Agent 5: Legacy Detector (`scripts/review_legacy_artifacts.py`)

**Purpose**: Find v0.1-era artifacts that should have been deleted or updated.

**Rationale**: After the v0.1→v2 clean rewrite, any remaining v0.1 patterns are noise.
This agent catches artifacts that were missed during cleanup or re-introduced by
accident.

**Checks**:

| # | Check | Severity | Method |
|---|-------|----------|--------|
| 1 | No files reference Poetry (`pyproject.toml` with `[tool.poetry]`) | ERROR | Grep |
| 2 | No files reference `pip install` or `requirements.txt` | ERROR | Grep |
| 3 | No files reference old import paths (`from src.training import`, `from src.log_ML import`) | ERROR | Grep |
| 4 | `wiki/` directory does not exist | WARN | `Path("wiki").exists()` |
| 5 | `configs/_legacy_v01_defaults.yaml` does not exist | WARN | `Path(...).exists()` |
| 6 | No references to Weights & Biases in active code | WARN | Grep for `wandb` in `src/` |
| 7 | No references to Airflow in active code | WARN | Grep for `airflow` in `src/` |
| 8 | No Python 3.8-style code (e.g., `Union[X, Y]` instead of `X \| Y`) | WARN | AST check for old-style unions |

### 6.7 Reviewer Orchestrator

The orchestrator runs all 5 agents and produces a unified report:

```python
"""Knowledge reviewer orchestrator — runs all 5 agents in parallel.

Usage:
  uv run python scripts/review_knowledge.py              # full review
  uv run python scripts/review_knowledge.py --quick      # link check + legacy only
  uv run python scripts/review_knowledge.py --prd        # PRD auditor only
  uv run python scripts/review_knowledge.py --specs      # spec drift only
  uv run python scripts/review_knowledge.py --staleness  # staleness scan only
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml


def run_all_reviewers(mode: str = "full") -> dict:
    """Run reviewer agents based on mode."""
    from review_knowledge_links import main as link_check
    from review_legacy_artifacts import main as legacy_check
    from review_prd_integrity import main as prd_audit
    from review_spec_drift import main as spec_drift
    from review_staleness import main as staleness_scan

    reviewers = {
        "full": [link_check, prd_audit, spec_drift, staleness_scan, legacy_check],
        "quick": [link_check, legacy_check],
        "prd": [prd_audit],
        "specs": [spec_drift],
        "staleness": [staleness_scan],
    }

    agents = reviewers.get(mode, reviewers["full"])

    with ProcessPoolExecutor(max_workers=len(agents)) as pool:
        results = list(pool.map(lambda fn: fn(), agents))

    # Merge results into unified report
    report = {
        "mode": mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agents_run": len(agents),
        "total_failures": sum(r.get("failures", 0) for r in results),
        "total_warnings": sum(r.get("warnings", 0) for r in results),
        "agent_reports": results,
    }

    return report
```

### 6.8 Reviewer Skill Integration

Create a new skill at `.claude/skills/knowledge-reviewer/SKILL.md`:

```markdown
# Knowledge Reviewer Skill

## Purpose
Automated integrity checking across the 5-layer knowledge graph.
Detects drift between PRD decisions, OpenSpec specs, implementation,
planning docs, and MEMORY.md.

## Activation
- `/review-knowledge` — Run all 5 reviewers (full mode)
- `/review-knowledge quick` — Link checker + legacy detector only
- `/review-knowledge prd` — PRD auditor only
- `/review-knowledge specs` — Spec drift detector only
- Automatic: pre-commit hook runs `--quick` on knowledge-graph changes

## Agents

| Agent | Script | Checks | Speed |
|-------|--------|--------|-------|
| PRD Auditor | `scripts/review_prd_integrity.py` | 21 invariants across 52 decision nodes | ~10s |
| Spec Drift | `scripts/review_spec_drift.py` | 6 checks on OpenSpec/implementation alignment | ~15s |
| Link Checker | `scripts/review_knowledge_links.py` | 9 cross-reference validations | ~3s |
| Staleness Scanner | `scripts/review_staleness.py` | 8 freshness checks on 118+ docs | ~20s |
| Legacy Detector | `scripts/review_legacy_artifacts.py` | 8 v0.1 artifact checks | ~5s |

## Output
- Console: summary line per agent (PASS/WARN/FAIL counts)
- File: `knowledge-graph/reports/review-YYYY-MM-DD.yaml`
- GitHub: auto-create issue for any FAIL result (if `--create-issues` flag)

## When to Run
- Before paper submission (full review)
- After major refactoring (full review)
- Weekly (scheduled via cron or manual)
- On every commit touching knowledge files (quick, via pre-commit)

## Integration with Other Skills
- After `/sync-roadmap`: run link checker to verify new issue references
- After `prd-update` skill: run PRD auditor to validate changes
- After OpenSpec `/opsx:archive`: run spec drift detector
```

---

## 7. Planning Docs Lifecycle Management

### 7.1 Required Frontmatter for All Planning Docs

Every file in `docs/planning/` must have standardized frontmatter:

```yaml
---
title: "Docker Security Hardening MLSecOps Report"
status: implemented  # draft | active | implemented | superseded | reference
phase: P3            # P0-P6, or "research" for exploratory
domain: infrastructure
created: "2026-03-07"
implementing_pr: 542
prd_nodes: [container_strategy, secrets_management]
superseded_by: null   # path to newer doc if superseded
---
```

### 7.2 Status Index: `docs/planning/STATUS.yaml`

A machine-readable index of all 118+ planning docs:

```yaml
# docs/planning/STATUS.yaml
# Auto-generated by scripts/index_planning_docs.py
# Manual edits to status fields are preserved on regeneration

last_indexed: "2026-03-13"
total_documents: 118
status_counts:
  implemented: 45
  active: 15
  draft: 8
  reference: 35
  superseded: 10
  untagged: 5

documents:
  - file: docs/planning/modernize-minivess-mlops-plan.md
    title: "Full Modernization Plan"
    status: reference
    domain: architecture
    lines: 1410
    created: "2026-02-23"

  - file: docs/planning/docker-security-hardening-mlsecops-report.md
    title: "Docker Security Hardening MLSecOps Report"
    status: implemented
    domain: infrastructure
    phase: P3
    implementing_pr: 542
    lines: 1421

  # ... 116 more entries
```

### 7.3 Auto-Indexer Script

```python
# scripts/index_planning_docs.py
# Scans docs/planning/**/*.md for YAML frontmatter
# Generates/updates docs/planning/STATUS.yaml
# Preserves manually-set status fields
# Flags documents without frontmatter as "untagged"
```

---

## 8. Concrete Migration Path

### Phase 1: Foundation (Day 1 — ~4 hours)

**Goal**: Create the knowledge graph skeleton and navigator.

| Step | Action | Output |
|------|--------|--------|
| 1.1 | Create `knowledge-graph/` directory | Directory structure |
| 1.2 | Write `navigator.yaml` with all domain mappings | Layer 0 navigator |
| 1.3 | Write `_schema.yaml` (decision node format) | Schema definition |
| 1.4 | Write 7 domain navigator files (training, infra, etc.) | Layer 3 navigators |
| 1.5 | Create `bibliography.yaml` with existing citations | Central bibliography |
| 1.6 | Add `knowledge-graph/` reference to CLAUDE.md Layer 0 | Integration |

### Phase 2: PRD Materialization (Day 1-2 — ~6 hours)

**Goal**: Convert the 52 PRD nodes from prose to YAML.

| Step | Action | Output |
|------|--------|--------|
| 2.1 | Write `_network.yaml` DAG from Section 4 of PRD blueprint | Network topology |
| 2.2 | Materialize 7 L1 decision YAMLs (all resolved) | L1 decisions |
| 2.3 | Materialize 10 L2 decision YAMLs | L2 decisions |
| 2.4 | Materialize 20 L3 decision YAMLs | L3 decisions |
| 2.5 | Materialize 8 L4 decision YAMLs | L4 decisions |
| 2.6 | Materialize 7 L5 decision YAMLs | L5 decisions |
| 2.7 | Write `learning-first-mvp.yaml` scenario | Active scenario |
| 2.8 | Write `solo-researcher.yaml` archetype | Default archetype |
| 2.9 | Write `vascular-segmentation/overlay.yaml` | Active domain |
| 2.10 | Run PRD auditor to validate all invariants | Validation report |

### Phase 3: OpenSpec Initialization (Day 2 — ~3 hours)

**Goal**: Bootstrap OpenSpec with specs for existing flows.

| Step | Action | Output |
|------|--------|--------|
| 3.1 | `npx @fission-ai/openspec init` | OpenSpec scaffold |
| 3.2 | Write `config.yaml` with MinIVess tech stack | Project config |
| 3.3 | Write `project.md` (distilled from CLAUDE.md) | Global context |
| 3.4 | Write spec + design for `training-pipeline` | First spec |
| 3.5 | Write spec + design for `inter-flow-contract` | Critical contract |
| 3.6 | Write spec + design for `model-adapters` | Adapter contract |
| 3.7 | Archive PR #589 (inter-flow contract fix) as first archived change | Change history |

### Phase 4: Planning Docs Triage (Day 2-3 — ~4 hours)

**Goal**: Add frontmatter to all 118 planning docs and generate STATUS.yaml.

| Step | Action | Output |
|------|--------|--------|
| 4.1 | Write `scripts/index_planning_docs.py` | Indexer script |
| 4.2 | Add frontmatter to top-20 most important docs (manually) | Frontmatter |
| 4.3 | Run indexer to generate initial STATUS.yaml | Status index |
| 4.4 | Bulk-add minimal frontmatter to remaining 98 docs | `status: untagged` |
| 4.5 | Review and correct status for docs with clear PR links | Corrected statuses |

### Phase 0: v0.1 Cleanup (Day 1 — ~15 minutes)

**Goal**: Remove stale v0.1 artifacts that confuse agents.

| Step | Action | Output |
|------|--------|--------|
| 0.1 | Delete `configs/_legacy_v01_defaults.yaml` | Remove 469 lines of dead config |
| 0.2 | Move `wiki/MLOps-Intro.md` and `wiki/MLflow.md` to `docs/reference/` | Preserve 2 useful pages |
| 0.3 | Delete `wiki/` directory entirely | Remove 380 KB of stale v0.1 tutorials |
| 0.4 | Grep for remaining v0.1 references, remove from active docs | Clean references |

### Phase 5: Reviewer Agents (Day 3 — ~4 hours)

**Goal**: Implement the 5 reviewer agents + orchestrator.

| Step | Action | Output |
|------|--------|--------|
| 5.1 | Write `scripts/review_knowledge_links.py` (9 checks) | Link checker |
| 5.2 | Write `scripts/review_prd_integrity.py` (21 invariants) | PRD auditor |
| 5.3 | Write `scripts/review_staleness.py` (8 checks) | Staleness scanner |
| 5.4 | Write `scripts/review_spec_drift.py` (6 checks) | Spec drift detector |
| 5.5 | Write `scripts/review_legacy_artifacts.py` (8 checks) | Legacy detector |
| 5.6 | Write `scripts/review_knowledge.py` (orchestrator) | Unified entry point |
| 5.7 | Write `.claude/skills/knowledge-reviewer/SKILL.md` | Skill definition |
| 5.8 | Add link checker to pre-commit hooks | Automated check |
| 5.9 | Run full review, fix all reported issues | Clean baseline |

### Phase 6: CLAUDE.md and MEMORY.md Cleanup (Day 3 — ~1 hour)

**Goal**: Trim MEMORY.md below 200 lines, update CLAUDE.md Layer 0 references.

| Step | Action | Output |
|------|--------|--------|
| 6.1 | Move all CRITICAL sections from MEMORY.md to topic files | Topic files |
| 6.2 | Rewrite MEMORY.md as pure index (<180 lines) | Clean MEMORY.md |
| 6.3 | Add knowledge-graph references to CLAUDE.md | Updated Layer 0 |
| 6.4 | Create missing CLAUDE.md files (pipeline, observability, serving, agents) | 4 new files |

### Total Effort: ~22 hours across 7 phases

| Phase | Effort | Dependencies |
|-------|--------|-------------|
| 0. v0.1 cleanup | 15 min | None |
| 1. Foundation | 4h | Phase 0 |
| 2. PRD materialization | 6h | Phase 1 |
| 3. OpenSpec init | 3h | Phase 1 |
| 4. Planning docs triage | 4h | Phase 1 |
| 5. Reviewer agents | 4h | Phases 2, 3, 4 |
| 6. CLAUDE.md cleanup | 1h | Phase 1 |

---

## 9. How Progressive Disclosure Works in Practice

### 9.1 Query: "What loss function should I use?"

```
Agent reads: knowledge-graph/navigator.yaml (Layer 0, ~50 lines)
  → keywords.loss → domain: training
  → Loads: knowledge-graph/domains/training.yaml (Layer 3, ~100 lines)
  → decisions.loss_function.status: resolved, winner: cbdice_cldice
  → ANSWER: cbdice_cldice (one lookup, two files, <200 tokens consumed)

If deeper context needed:
  → evidence → docs/results/dynunet_loss_variation_v2_report.md (Layer 4)
  → prd_node → decisions/L3/loss_function.yaml (Layer 3)
  → openspec → openspec/specs/training-pipeline/spec.md (Layer 4)
```

**Total context consumed**: ~350 tokens for quick answer, ~2K for full traversal.
**Current system**: Agent must grep 594 lines of CLAUDE.md + search 118 planning docs.

### 9.2 Query: "Is our Docker setup secure enough?"

```
Agent reads: navigator.yaml → keywords.docker → infrastructure
  → Loads: domains/infrastructure.yaml
  → decisions.container_strategy → resolved: docker_compose_profiles
  → decisions.secrets_management → resolved: dynaconf_dotenv
  → metalearning → 2026-03-07-docker-volume-mount-violation.md
  → Loads: deployment/CLAUDE.md (Layer 2, 347 lines)
  → ANSWER: Resolved with Docker hardening (PR #542), seccomp profiles,
    SOPS encryption. Known metalearning on volume mount violations.

If deeper:
  → evidence → docs/planning/docker-security-hardening-mlsecops-report.md (Layer 4)
  → openspec → openspec/specs/training-pipeline/design.md (Layer 4)
```

### 9.3 Query: "What's still unimplemented?"

```
Agent reads: navigator.yaml → all domains
  → Scans all domain files for status: not_started
  → Collects: topology_metrics, iac_tool, gitops_engine, air_gap_strategy,
              retraining_trigger, sbom_generation, federated_learning
  → Cross-references with open GitHub issues (#366, #564, #574, etc.)
  → ANSWER: 8 PRD nodes not started, 5 partial, 7 config-only.
    Open issues cover gpu_compute and drift_monitoring.
    Biggest gaps: operations layer (4/7 not started).
```

### 9.4 Query: "Why did we choose MLflow over W&B?"

```
Agent reads: navigator.yaml → keywords.mlflow → observability
  → Loads: domains/observability.yaml
  → decisions.experiment_tracker → resolved: mlflow_local
  → resolution_evidence → PRD node L3/experiment_tracker.yaml
  → Loads decision YAML:
    - mlflow_local: posterior 0.85 (winner)
    - mlflow_plus_wandb: posterior 0.10
    - Evidence: "Local-first design goal, no API keys, filesystem backend,
      DuckDB analytics integration. W&B requires external API."
  → ANSWER: MLflow chosen for local-first principle (Design Goal #2),
    zero cloud API tokens requirement, DuckDB analytics integration.
```

---

## 10. Token Budget Analysis

### 10.1 Current System

| Layer | Lines | Tokens (est.) | Loaded When |
|-------|-------|---------------|-------------|
| CLAUDE.md | 594 | ~4,500 | Always |
| MEMORY.md | 207 | ~1,500 | Always |
| Folder CLAUDE.md (7) | 829 | ~6,200 | On file access |
| Planning docs (search) | varies | ~3,000-10,000 | On grep |
| **Total per session** | | **~15,000-22,000** | |

### 10.2 Proposed System

| Layer | Lines | Tokens (est.) | Loaded When |
|-------|-------|---------------|-------------|
| CLAUDE.md (trimmed) | ~500 | ~3,800 | Always |
| MEMORY.md (pure index) | ~150 | ~1,100 | Always |
| navigator.yaml | ~50 | ~350 | Always (Layer 0) |
| Domain navigator (1) | ~100 | ~700 | On query routing |
| Folder CLAUDE.md (1-2) | ~200 | ~1,500 | On file access |
| PRD decision (1-3) | ~150 | ~1,000 | On deep traversal |
| **Typical session** | | **~7,500-9,000** | |
| **Deep research session** | | **~12,000-15,000** | |

**Token savings**: 35-50% reduction in typical sessions by loading only relevant
domain files instead of grepping the full planning archive.

YAML format choice adds 28-60% token efficiency over Markdown for the PRD nodes
(McMillan 2026).

---

## 11. Relationship to Existing Systems

### 11.1 What Stays

| System | Status | Rationale |
|--------|--------|-----------|
| CLAUDE.md | Stays (trimmed to ~500 lines) | Constitution layer — always loaded |
| MEMORY.md | Stays (rewritten as pure index, <180 lines) | Auto-memory index |
| Folder CLAUDE.md files | Stays + 4 new ones added | Layer 2 domain experts |
| Metalearning docs | Stay as-is | Episodic memory — failure analysis |
| Skills (TDD, PRD-update, etc.) | Stay as-is | Procedural memory |

### 11.2 What Changes

| System | Change | Rationale |
|--------|--------|-----------|
| PRD blueprint | Materialized to YAML in `knowledge-graph/decisions/` | From design doc to operational system |
| Planning docs | Get frontmatter + STATUS.yaml index | From flat pile to indexed archive |
| Inter-system links | All become bidirectional via domain navigators | From grep-based to navigator-based discovery |

### 11.3 What's New

| System | Purpose | Location |
|--------|---------|----------|
| Navigator | Entry point for all knowledge queries | `knowledge-graph/navigator.yaml` |
| Domain navigators | Per-domain detailed indexes | `knowledge-graph/domains/*.yaml` |
| OpenSpec | SDD framework for flow specifications | `openspec/` |
| Reviewer agents | Automated integrity checks | `scripts/review_*.py` |
| Planning STATUS index | Machine-readable doc inventory | `docs/planning/STATUS.yaml` |
| Bibliography | Central citation database | `knowledge-graph/bibliography.yaml` |

---

## 12. Risk Analysis

### 12.1 Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **Knowledge graph becomes stale** | HIGH | Reviewer agents run in pre-commit; staleness scanner flags >30-day old docs |
| **OpenSpec adds ceremony overhead** | MEDIUM | Start with 3 critical specs (training, inter-flow, model adapters); expand only when value proven |
| **52 YAML files = maintenance burden** | MEDIUM | Resolved decisions rarely change; only ~20 active decisions need maintenance; automated validation catches drift |
| **Navigator becomes a bottleneck** | LOW | Navigator is ~50 lines of keywords→domains mapping; regenerated from domain files if needed |
| **Two-repo knowledge** (sci-llm-writer + minivess-mlops) | MEDIUM | minivess-mlops is self-contained; sci-llm-writer is the research companion but not a dependency |

### 12.2 What NOT to Do

Based on McMillan 2026 findings and our metalearning:

1. **Don't invent a custom format** — TOON-like innovations incur 38% grep tax from
   model unfamiliarity. Stick to YAML for structured data, Markdown for prose.
2. **Don't convert everything to a single format** — Format doesn't affect accuracy (p=0.484).
   Information architecture is what matters.
3. **Don't build a database** — File-based knowledge scales to 10K entries with domain
   partitioning. Adding PostgreSQL/SQLite for knowledge management violates our filesystem-first
   principle and our regex ban (SQL is structured query language, not regex, but the spirit
   of using proper parsers applies).
4. **Don't load everything upfront** — Progressive disclosure is the core pattern. The
   navigator exists so that agents load ~350 tokens to route, not ~22K tokens to search.
5. **Don't auto-generate specs from code** — OpenSpec specs capture *intent* (what the
   system SHOULD do), not implementation (what it currently does). Auto-generation inverts
   this relationship.

---

## 13. Success Criteria

### 13.1 Measurable Outcomes

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| Tokens consumed per knowledge query | ~5K-10K | ~1K-3K | Count tokens in navigator + domain file |
| Time to answer "what's unimplemented?" | 5+ min (grep 48K lines) | <30 sec (scan domain files) | Manual timing |
| PRD nodes with implementation links | 0 (unmaterialized) | 52 | `review_prd_integrity.py` |
| Planning docs with frontmatter | ~0 | 118 | `index_planning_docs.py` |
| Broken cross-references | Unknown (many) | 0 | `review_knowledge_links.py` |
| MEMORY.md line count | 207 (over limit) | <180 | `wc -l` |
| OpenSpec specs for critical flows | 0 | 9 (all flows) | Count `openspec/specs/*/spec.md` |

### 13.2 Qualitative Outcomes

- An agent can answer "why did we choose X?" in one navigator hop + one file read
- Every implementation file can be traced back to a PRD decision
- Every PRD decision links forward to its implementation
- Knowledge integrity is continuously verified by reviewer agents
- New team members (or new AI sessions) can orient in <1000 tokens of context

---

## References

1. [McMillan, D. (2026). "Structured Context Engineering for File-Native Agentic Systems."](https://arxiv.org/abs/placeholder) — 9,649 experiments on format choice, domain partitioning, grep tax
2. [Vasilopoulos, K. (2026). "Codified Context Infrastructure for AI Agents in Complex Codebases."](https://arxiv.org/abs/placeholder) — 3-tier Hot/Specialist/Cold memory with trigger tables
3. [Yu, Z. et al. (2026). "Agentic Memory (AgeMem)."](https://arxiv.org/abs/placeholder) — Working/episodic/semantic/procedural memory with progressive RL
4. [Xu, Y. et al. (2025). "Everything is Context: Agentic File System."](https://arxiv.org/abs/placeholder) — File system as persistent context infrastructure
5. [Gupta, S. (2025). "Context Graphs as Trillion-Dollar Opportunity."](https://arxiv.org/abs/placeholder) — Decision traces stitched across entities and time
6. [Hua, W. et al. (2026). "Context Engineering 2.0."](https://arxiv.org/abs/placeholder) — Four eras of context engineering (1.0–4.0)
7. [Anthropic (2025). "Effective Context Engineering for AI Agents."](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering) — Progressive disclosure, sub-agent architectures
8. [Fission-AI. OpenSpec: Spec-Driven Development Framework.](https://github.com/Fission-AI/OpenSpec) — Delta-based SDD with 30+ tool adapters
9. [Mohamed, S. et al. (2026). "KG-Orchestra: Multi-Agent Knowledge Graph Enrichment."](https://arxiv.org/abs/placeholder) — Multi-agent orchestration for biomedical knowledge graphs
