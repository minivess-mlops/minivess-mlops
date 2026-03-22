---
name: plan-context-load
version: 1.0.0
description: >
  Mandatory pre-planning context load. Executes the 7-step SOP before ANY plan creation:
  navigator routing, domain loading, metalearning search, decision registry check,
  memory scan, source-of-truth reading, plan archive check. Prevents context amnesia
  and re-asking decided questions. Use BEFORE any planning task, factorial design, or
  architectural decision. Do NOT use for: code implementation (use self-learning-iterative-coder),
  literature research (use create-literature-report).
last_updated: 2026-03-22
activation: proactive
invocation: /plan-context-load
metadata:
  category: planning
  tags: [context-management, planning-sop, decision-registry, metalearning]
  relations:
    compose_with: [search-metalearning, knowledge-reviewer, prd-update]
    depend_on: []
    similar_to: [read-memories]
    belong_to: [context-management-upgrade]
---

# /plan-context-load — Pre-Planning Context Load SOP

## Purpose

Prevent context amnesia by loading ALL relevant context BEFORE creating any plan.
This skill is MANDATORY before ANY planning task (Issue #906).

## When to Use

- Before creating any plan document
- Before factorial design decisions
- Before architectural decisions
- Before writing metalearning docs
- When starting a new session that continues prior planning work

## Execution Steps (ALL MANDATORY)

### Step 1: Navigator Routing

Read `knowledge-graph/navigator.yaml` and identify which domain(s) are relevant
to the current task. Use the `keywords` section for routing.

```
Read knowledge-graph/navigator.yaml
→ Identify relevant domains (e.g., "training", "cloud", "models")
```

### Step 2: Domain Loading

Read the relevant domain YAML(s) from `knowledge-graph/domains/`.
Focus on resolved decisions (posterior >= 0.80).

```
Read knowledge-graph/domains/{relevant_domain}.yaml
→ Note resolved decisions and their evidence
```

### Step 3: Metalearning Search

Search metalearning docs for keywords related to the current task.
Load top-5 results and read the most relevant ones.

```bash
uv run python scripts/build_metalearning_index.py --query "<task keywords>" --top 5
```

Read each relevant doc. Note prevention rules.

### Step 4: Decision Registry Check

Read the decision registry. Check if any question you're about to ask
has already been decided.

```
Read knowledge-graph/decisions/registry.yaml
→ For each DO_NOT_RE_ASK: true entry, NEVER ask this question
```

### Step 5: Memory Scan

Read MEMORY.md and relevant topic files.

```
Read MEMORY.md
→ Check for prior session decisions on this topic
```

### Step 6: Source of Truth

Read the authoritative document for the specific task:

| Task Type | Source of Truth |
|-----------|----------------|
| Factorial design | `docs/planning/pre-gcp-master-plan.xml` line 16 |
| Flow merger | `docs/planning/training-and-post-training-into-two-subflows-under-one-flow.md` |
| Debug scope | `CLAUDE.md` Rule 27 |
| Cloud architecture | `CLAUDE.md` + `.claude/rules/` |
| Model details | `knowledge-graph/domains/models.yaml` |

### Step 7: Plan Archive Check

Search the plan archive for existing plans related to the current task.
Avoid duplicating work that has already been planned.

```bash
# Check navigator for theme health
cat docs/planning/v0-2_archive/navigator.yaml

# Search for existing plans on the topic
uv run python scripts/build_plan_archive.py --search "<task keywords>"
```

If existing plans are found with `status: implemented`, read them before
creating new plans. If found with `status: planned`, consider updating
rather than creating new plans.

## Output Format

After completing all 7 steps, present a summary:

```
=== Pre-Planning Context Load Complete ===

Domains loaded: training, cloud
Metalearning hits: 3 relevant docs
  - 2026-03-20-full-factorial-is-not-24-cells.md (CRITICAL)
  - 2026-03-22-debug-equals-production-8th-violation.md
  - 2026-03-22-wrong-metalearning-doc-failure-mode.md

Decided questions (DO NOT RE-ASK):
  - post_training_methods: only "none" and "swag"
  - debug_scope: 1 fold, 2 epochs, half data ONLY
  - post_training_execution_model: same SkyPilot job, iterate internally

Source of truth read: pre-gcp-master-plan.xml

Ready to proceed with planning.
```

## Absolute Rules

1. **NEVER skip any step** — each step prevents a different failure mode
2. **NEVER ask a DO_NOT_RE_ASK question** — the answer is in the registry
3. **NEVER write a plan without citing the source of truth**
4. **NEVER write metalearning in panic** — wait for user confirmation
5. **Present what you THINK first** — let user correct before proceeding

## Cross-References

- Issue #906 (context compounding failure)
- `.claude/context-management-upgrade-plan.md`
- `.claude/rules/planning-sop.md`
- `knowledge-graph/decisions/registry.yaml`
- `scripts/build_metalearning_index.py`
