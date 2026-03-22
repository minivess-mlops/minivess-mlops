---
name: plan-index
version: 1.0.0
description: >
  Search and navigate the plan archive (347 planning documents, 13 themes).
  Use when you need to find existing plans before creating new ones, check
  what has been planned/implemented, or understand theme health scores.
last_updated: 2026-03-22
activation: reactive
invocation: /plan-index
metadata:
  category: context-management
  tags: [planning, archive, search, duckdb]
  relations:
    compose_with: [plan-context-load]
    depend_on: []
    similar_to: [search-metalearning]
    belong_to: [context-management-upgrade]
---

# /plan-index -- Plan Archive Search

## Purpose

Navigate and search 347 archived planning documents without reading them all.
Three-layer access: navigator YAML -> DuckDB full-text search -> original files.

## Quick Commands

```bash
# Theme overview with health scores
cat docs/planning/v0-2_archive/navigator.yaml

# Full-text search
uv run python scripts/build_plan_archive.py --search "loss function"

# Classification summary
uv run python scripts/classify_plan_docs.py --summary

# Health audit
uv run python scripts/generate_audit_report.py

# Stale index check
uv run python scripts/update_plan_archive.py --stale
```

## When to Use

- Before creating a new plan (check if one already exists)
- When you need to understand what has been planned in a domain
- When checking implementation status of past plans
- During context load (Step 7 of plan-context-load)

## Workflow

1. **Read navigator**: `docs/planning/v0-2_archive/navigator.yaml`
2. **Check health score**: Is the relevant theme well-covered or sparse?
3. **Search if needed**: `--search "topic"` for specific content
4. **Read originals**: Only when you need the full content of a specific plan

## 13 Themes

training (92), cloud (80), infrastructure (78), models (75), observability (72),
testing (70), architecture (68), operations (65), manuscript (58), evaluation (55),
deployment (55), harness (50), data (45)

## Files

| File | Purpose |
|------|---------|
| `docs/planning/v0-2_archive/navigator.yaml` | Theme index + health scores |
| `docs/planning/v0-2_archive/plan_archive.duckdb` | Full-text DuckDB index |
| `scripts/classify_plan_docs.py` | Classification engine |
| `scripts/build_plan_archive.py` | Index builder |
| `scripts/update_plan_archive.py` | Incremental updater |
| `scripts/generate_audit_report.py` | Audit report generator |
