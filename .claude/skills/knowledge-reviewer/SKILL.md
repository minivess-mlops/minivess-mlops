---
name: knowledge-reviewer
version: 1.1.0
description: >
  Validate knowledge-graph integrity, PRD consistency, legacy cleanup, and staleness
  after any knowledge artifact change. Use when modifying knowledge-graph/, docs/planning/,
  CLAUDE.md, or MEMORY.md. Runs automatically via knowledge-links pre-commit hook (quick mode).
  Do NOT use for: code implementation (use self-learning-iterative-coder) or
  literature research (use create-literature-report).
last_updated: 2026-03-19
activation: manual
invocation: /knowledge-reviewer
metadata:
  category: knowledge
  tags: [knowledge-graph, validation, prd, pre-commit, staleness]
  relations:
    compose_with:
      - kg-sync
      - prd-update
    depend_on: []
    similar_to: []
    belong_to: []
---

# Knowledge Reviewer Skill

> Validate knowledge-graph integrity, PRD consistency, legacy cleanup, and staleness
> after any knowledge artifact change.

## When to Use This Skill

Run the knowledge reviewer after modifying any of these:
- `knowledge-graph/` -- domain files, navigator, decisions, bibliography, scenarios
- `docs/planning/` -- planning docs, PRD files, frontmatter
- `CLAUDE.md` or `MEMORY.md` -- top-level project instructions

The `knowledge-links` pre-commit hook runs automatically on commits touching these
paths (quick mode only). Use the full orchestrator for deeper validation.

---

## Reviewer Agents

Four independent agents, each with a single responsibility. All return a structured
report dict with `agent_name`, `failures`, `warnings`, `total_checks`, and `checks`.

### 1. Link Checker (`scripts/review_knowledge_links.py`)

Validates all cross-references resolve to real files.

| Check | Severity | What it verifies |
|-------|----------|------------------|
| Navigator paths | ERROR | Every `navigator` and `claude_md` path in `navigator.yaml` exists on disk |
| Domain implementation paths | WARN | `implementation` file paths in domain YAML files exist |
| Domain evidence paths | WARN | `evidence` file paths in domain YAML files exist |
| PRD node paths | ERROR | `prd_node` references in domain decisions point to real files |
| Network node files | ERROR | Every node in `_network.yaml` has a corresponding decision file |
| Bibliography keys | ERROR | Every `citation_key` in decision files resolves in `bibliography.yaml` |
| Domain dates | ERROR | Every domain `last_reviewed` date is parseable (YYYY-MM-DD) |
| MEMORY.md length | WARN | `MEMORY.md` stays under 200 lines |

Supports `--quick` mode which skips bibliography key resolution (faster, used by pre-commit).

```bash
uv run python scripts/review_knowledge_links.py          # full check
uv run python scripts/review_knowledge_links.py --quick   # paths only, skip bibliography
```

### 2. PRD Auditor (`scripts/review_prd_integrity.py`)

Validates the 52-node Bayesian decision network structure.

| Check | Severity | What it verifies |
|-------|----------|------------------|
| Node count | ERROR | Network has exactly 52 nodes |
| Node-file consistency | ERROR | Every network node has a matching `.yaml` decision file |
| ID matching | ERROR/WARN | `decision_id` in files matches node `id` in network |
| DAG acyclicity | ERROR | No cycles in the decision graph (Kahn's algorithm) |
| Edge references | ERROR | Every edge's `from` and `to` reference existing nodes |
| Level ordering | WARN | Edges flow from lower to higher levels (L1 -> L5) |
| Probability sums | ERROR | Option `prior_probability` values sum to ~1.0 per decision |
| Resolved winners | ERROR | Every resolved decision has a `resolved_option` |
| Status distribution | INFO | Reports how many decisions are open/resolved/deferred |

```bash
uv run python scripts/review_prd_integrity.py
```

### 3. Legacy Detector (`scripts/review_legacy_artifacts.py`)

Finds v0.1-era patterns that should have been removed in the v2 rewrite.

| Check | Severity | What it verifies |
|-------|----------|------------------|
| No Poetry | ERROR | `[tool.poetry]` not in `pyproject.toml` |
| No pip install | ERROR | No `pip install` or `requirements.txt` references in `src/` |
| No old imports | ERROR | No `from src.training import` / `from src.log_ML import` / `from src.datasets import` |
| wiki/ deleted | WARN | `wiki/` directory removed |
| Legacy config deleted | WARN | `configs/_legacy_v01_defaults.yaml` removed |
| No wandb | WARN | No `import wandb` in `src/` |
| No airflow | WARN | No `import airflow` in `src/` |

Note: Adapter error messages that document external install instructions (e.g., SAM3
from-source) are excluded from the pip-install check.

```bash
uv run python scripts/review_legacy_artifacts.py
```

### 4. Staleness Scanner (`scripts/review_staleness.py`)

Identifies knowledge artifacts that are overdue for review.

| Check | Severity | What it verifies |
|-------|----------|------------------|
| Volatile node reviews | WARN | Volatile/evolving PRD decisions reviewed before `next_review` date |
| Domain navigator freshness | WARN | Domain navigators `last_reviewed` within 30 days |
| Planning frontmatter | WARN | All `docs/planning/*.md` files have YAML frontmatter |
| Scenario freshness | WARN | Active scenario file modified within 30 days (git log) |

```bash
uv run python scripts/review_staleness.py
```

---

## Orchestrator

`scripts/review_knowledge.py` runs agents in combination based on mode.

| Mode | Flag | Agents Run | Use Case |
|------|------|------------|----------|
| Full | `--full` (default) | All 4 agents | After major knowledge-graph changes |
| Quick | `--quick` | Link checker (quick) + Legacy detector | Pre-commit, fast feedback |
| PRD | `--prd` | PRD auditor only | After editing decision files |
| Staleness | `--staleness` | Staleness scanner only | Periodic freshness audit |

```bash
uv run python scripts/review_knowledge.py              # full review (all 4 agents)
uv run python scripts/review_knowledge.py --quick       # link check + legacy only
uv run python scripts/review_knowledge.py --prd         # PRD auditor only
uv run python scripts/review_knowledge.py --staleness   # staleness scan only
```

Exit codes:
- `0` -- all checks passed (warnings are non-blocking)
- `1` -- one or more ERROR-severity checks failed

---

## Quick Command Reference

```bash
# Full validation (recommended after any knowledge-graph/ edit)
uv run python scripts/review_knowledge.py

# Fast check (pre-commit equivalent, ~2s)
uv run python scripts/review_knowledge.py --quick

# After editing PRD decision files
uv run python scripts/review_knowledge.py --prd

# Monthly staleness audit
uv run python scripts/review_knowledge.py --staleness

# Individual agents (for debugging a specific failure)
uv run python scripts/review_knowledge_links.py
uv run python scripts/review_prd_integrity.py
uv run python scripts/review_legacy_artifacts.py
uv run python scripts/review_staleness.py
```

---

## When Checks Fail

Every failure must be resolved before merging. Two acceptable outcomes:

1. **Fix immediately** -- if the root cause is clear and the fix is under 5 minutes
   (e.g., broken file path, missing bibliography key, probability sum off by 0.1).

2. **Create a GitHub issue** -- if the fix requires design work or touches multiple
   files. The issue must include:
   - Which agent and check failed
   - The exact error message
   - Affected file paths
   - Priority label (`P0-critical` for ERROR severity, `P2-medium` for WARN)

"Pre-existing" is not a valid excuse to skip a failure. See CLAUDE.md Rule #20.

---

## Pre-Commit Integration

The `knowledge-links` hook is registered in `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: knowledge-links
      name: Knowledge graph link checker
      entry: uv run python scripts/review_knowledge_links.py --quick
      language: system
      pass_filenames: false
      files: '(knowledge-graph/|MEMORY\.md|CLAUDE\.md|docs/planning/)'
```

This hook:
- Triggers only when files in `knowledge-graph/`, `MEMORY.md`, `CLAUDE.md`, or
  `docs/planning/` are staged for commit
- Runs in quick mode (skips bibliography resolution for speed)
- Blocks the commit on any ERROR-severity failure
- Does NOT run the PRD auditor or staleness scanner (use `--full` manually for those)

---

## Related

- `.claude/skills/prd-update/SKILL.md` -- PRD maintenance operations (add decisions, update priors)
- `knowledge-graph/navigator.yaml` -- Top-level navigation index for the knowledge graph
- `knowledge-graph/_network.yaml` -- DAG topology (52 nodes, source of truth for PRD auditor)
- `knowledge-graph/bibliography.yaml` -- Central bibliography for citation key resolution
- `docs/planning/knowledge-management-upgrade.md` -- Design doc that introduced this system
