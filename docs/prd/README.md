# MinIVess MLOps v2 — Probabilistic PRD

## Overview

This directory contains a **hierarchical probabilistic Product Requirements Document**
for MinIVess MLOps v2. Rather than prescribing a single implementation path, the PRD
encodes technology decisions as a **Bayesian decision network** with prior probabilities,
conditional dependencies, and team-specific modulations.

## Why Probabilistic?

Traditional PRDs assume you know what to build. MinIVess v2 is a **learning-first**
project where:
- The primary goal is maximizing tool exposure and skill building
- Multiple valid paths exist for every decision
- New research papers and tools constantly shift the landscape
- Different team profiles (solo researcher vs. clinical deployment) lead to different optimal paths

The Bayesian approach encodes this uncertainty explicitly, making it possible to:
1. Reason about which decisions are resolved vs. open
2. Propagate evidence (new papers, benchmarks) through the network
3. Compose specific implementation paths (scenarios) from the network
4. Adapt recommendations to different team profiles (archetypes)

## Structure

```
prd/
├── README.md                    ← You are here
├── schema.yaml                  # Frontmatter schema for narrative PRDs
├── llm-context.md               # System prompt for AI assistants
│
├── decisions/                   # Bayesian decision network
│   ├── _schema.yaml             # JSON Schema for decision nodes
│   ├── _network.yaml            # DAG topology (52 nodes, ~80 edges)
│   ├── L1-research-goals/       # 7 strategic decisions (root nodes)
│   ├── L2-architecture/         # 10 architectural decisions
│   ├── L3-technology/           # 20 technology stack decisions
│   ├── L4-infrastructure/       # 8 infrastructure decisions
│   └── L5-operations/           # 7 operational decisions
│
├── archetypes/                  # Team profiles that modulate probabilities
│   ├── solo-researcher.archetype.yaml
│   ├── lab-group.archetype.yaml
│   └── clinical-deployment.archetype.yaml
│
├── scenarios/                   # Composed decision paths
│   ├── learning-first-mvp.scenario.yaml    ← ACTIVE
│   ├── research-scaffold.scenario.yaml
│   └── clinical-production.scenario.yaml
│
└── domains/                     # Domain-specific overlays
    ├── registry.yaml
    ├── backbone-defaults.yaml
    ├── vascular-segmentation/overlay.yaml  ← ACTIVE
    ├── cardiac-imaging/overlay.yaml
    ├── neuroimaging/overlay.yaml
    └── general-medical/overlay.yaml
```

## Quick Navigation

### By Role

**If you're exploring what to build next:**
→ Start with `scenarios/learning-first-mvp.scenario.yaml` (active path)
→ Check `decisions/L1-research-goals/` for strategic context

**If you're adding a new tool or paper:**
→ Use the PRD-update skill: `.claude/skills/prd-update/`
→ Run `ingest-paper` protocol to extract relevant decisions

**If you're a researcher joining the project:**
→ Read `llm-context.md` for full context
→ Check `archetypes/` to find your team profile
→ Review `domains/vascular-segmentation/overlay.yaml` for domain specifics

**If you're an AI assistant:**
→ Load `llm-context.md` as system context
→ The `_network.yaml` gives you the full decision DAG

### By Decision Level

| Level | Scope | Nodes | Description |
|-------|-------|-------|-------------|
| **L1** | Research Goals | 7 | Project purpose, impact targets, MONAI alignment |
| **L2** | Architecture | 10 | Model strategy, UQ, serving, config, XAI |
| **L3** | Technology | 20 | Specific tools: models, losses, metrics, tracking |
| **L4** | Infrastructure | 8 | Compute, containers, CI/CD, IaC, export |
| **L5** | Operations | 7 | Monitoring, drift, governance, retraining |

### Implementation Status

| Status | Count | Meaning |
|--------|-------|---------|
| **resolved** | ~20 | Already implemented in codebase |
| **config_only** | ~8 | Configuration exists, not integrated |
| **partial** | ~5 | Some options implemented |
| **not_started** | ~19 | Open for future implementation |

## Concepts

### Decision Node
A single technology or architecture choice with 2-5 options, each carrying
a prior probability. See `decisions/_schema.yaml` for the full specification.

### Conditional Probability Table (CPT)
When a parent decision is made, it shifts the probabilities of child decisions.
For example, choosing `monai_native` alignment strongly boosts `monai_bundle`
export format and `monai_fl` federated learning.

### Archetype
A team profile (solo researcher, lab group, clinical deployment) that provides
alternative probability distributions. The same decision network serves all
archetypes — only the weights change.

### Scenario
A fully-resolved path through the decision network. The "Learning-First MVP"
scenario represents the current active implementation. Scenarios have a
`joint_probability` indicating how likely that specific combination is.

### Domain Overlay
Domain-specific adjustments to prior probabilities, metrics, and losses.
The vascular segmentation overlay boosts topology-aware metrics (clDice)
and vessel-specific augmentations.

## Maintenance

The PRD is maintained using the **prd-update** Claude Code skill at
`.claude/skills/prd-update/`. Available operations:

- `add-decision` — Add a new decision node to the network
- `update-priors` — Update probabilities based on new evidence
- `add-option` — Add a new option to an existing decision
- `create-scenario` — Create a new composed scenario
- `ingest-paper` — Read a paper and extract relevant decisions
- `validate` — Check DAG integrity, probability sums, cross-references

## References

- [Modernization Plan](../../modernize-minivess-mlops-plan.md) — Original 60KB implementation plan
- [Architecture Decision Records](../../adr/) — 5 ADRs from Phase 0-6
- [Claude Code Patterns](../../claude-code-patterns.md) — Development patterns documentation
- [Phase Tracker](../../../.claude/phase-tracker.md) — Implementation status (P0-P6 complete)
