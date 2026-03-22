---
name: search-metalearning
version: 1.0.0
description: >
  Search past metalearning documents for relevant failure patterns, lessons learned,
  and decided constraints. Uses DuckDB full-text search over 90+ metalearning docs.
  Use PROACTIVELY before planning, before writing metalearning, and when encountering
  a problem that might have been solved before.
  Do NOT use for: code-level searches (use Grep), config lookups (use navigator.yaml).
last_updated: 2026-03-22
activation: proactive
invocation: /search-metalearning
metadata:
  category: knowledge
  tags: [metalearning, search, context-management, failure-patterns]
  relations:
    compose_with: [plan-context-load, knowledge-reviewer]
    depend_on: []
    similar_to: [read-memories]
    belong_to: [context-management-upgrade]
---

# /search-metalearning — Search Failure Pattern Database

## When to Use

- **Before planning**: Search for prior decisions and mistakes on the topic
- **Before writing metalearning**: Check if a similar doc already exists
- **When encountering errors**: Search for known failure patterns
- **During AskUserQuestion**: Verify the question hasn't been answered before

## How to Execute

### Step 1: Run the search script

```bash
uv run python scripts/build_metalearning_index.py --query "<keywords>" --top 5
```

### Step 2: Read relevant docs

For each high-scoring result, read the full metalearning doc:
```
Read .claude/metalearning/<filename>
```

### Step 3: Apply findings

- If a doc addresses the current problem → follow its prevention rules
- If a doc is stale or wrong → update or delete it
- If no doc exists for this pattern → create one after resolving

## Rebuild Index

The index auto-rebuilds when no query is given:
```bash
uv run python scripts/build_metalearning_index.py
```

## Examples

```bash
# Search for factorial design issues
uv run python scripts/build_metalearning_index.py -q "factorial design"

# Search for Docker-related failures
uv run python scripts/build_metalearning_index.py -q "docker resistance"

# Search for debug vs production confusion
uv run python scripts/build_metalearning_index.py -q "debug production scope"
```

## Cross-References

- Issue #906 (context compounding failure)
- `.claude/context-management-upgrade-plan.md` (Phase 2)
- `scripts/build_metalearning_index.py` (index builder)
- `knowledge-graph/decisions/registry.yaml` (decided questions)
