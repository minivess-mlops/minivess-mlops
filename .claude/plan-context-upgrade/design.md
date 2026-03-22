# Plan Context Upgrade: Architecture

## Problem Statement

347 planning documents in `docs/planning/` represent significant institutional knowledge
but are currently unindexed, unclassified, and inaccessible to Claude Code without
reading them all (100k+ tokens).

## Architecture

### Components

```
docs/planning/v0-2_archive/
  navigator.yaml          -- L1: Theme index with health scores
  plan_archive.duckdb     -- L2: Full-text searchable DuckDB index
  original_docs/          -- L3: Actual planning documents (347 files)
  themes/                 -- L4: Symlink dirs by theme (optional)

scripts/
  classify_plan_docs.py   -- Classification engine (13 themes, keyword + manual)
  build_plan_archive.py   -- DuckDB builder (full rebuild)
  update_plan_archive.py  -- Incremental updater (upsert + stale detection)
  generate_audit_report.py -- Health dashboard + phantom detection
```

### Data Flow

```
original_docs/ --[classify]--> theme assignments
                  --[build]---> plan_archive.duckdb
                  --[audit]---> health scores --> navigator.yaml
```

### Classification Strategy

1. **Keyword matching**: Each of 13 themes has a keyword list. First match wins.
2. **Manual overrides**: Ambiguous docs get explicit theme assignment.
3. **Unclassified bucket**: Docs matching no theme go to "unclassified" for review.

### 13 Themes

training, cloud, infrastructure, models, observability, testing, architecture,
operations, manuscript, evaluation, deployment, harness, data

### Health Scores

Code-verified (not guessed). For each theme:
- Count of docs with `status: implemented` in KG domain files
- Count of docs that exist on disk and are readable
- Count of docs referenced but missing (phantoms)
- Score = (implemented + existing) / total_referenced * 100

### Integration Points

1. **plan-context-load skill**: Step 7 checks plan archive before planning
2. **.claude/settings.json**: PostToolUse hook updates archive on file writes
3. **KG navigator.yaml**: plan_archive section routes to archive navigator
4. **Audit report**: Periodic health check via generate_audit_report.py

## Design Decisions

1. **DuckDB over SQLite**: Consistent with project's DuckDB-for-analytics pattern
2. **Keyword classification over LLM**: Deterministic, fast, no API cost
3. **Manual overrides dict**: Transparent, version-controlled, easy to audit
4. **Incremental updates**: Don't rebuild 347-doc index for one new file
5. **Health scores in YAML**: Human-readable, diffable, part of KG layer
