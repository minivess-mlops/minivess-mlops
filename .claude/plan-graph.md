# Plan Graph: Navigating 347 Planning Documents

## Problem

The `docs/planning/` directory accumulated 347+ planning documents over the project's
lifetime. These range from fully-implemented execution plans to abandoned research reports.
Without an index, Claude Code either:
1. Ignores them entirely (losing institutional knowledge)
2. Reads them linearly (wasting 100k+ tokens of context)

## Solution: Three-Layer Plan Archive

### Layer 1: Navigator (YAML)
`docs/planning/v0-2_archive/navigator.yaml` -- 13 themes with health scores.
Read this FIRST to understand what exists and its quality.

### Layer 2: DuckDB Index
`docs/planning/v0-2_archive/plan_archive.duckdb` -- Full-text searchable index.
Use `scripts/build_plan_archive.py --search "topic"` for targeted lookup.

### Layer 3: Original Documents
`docs/planning/v0-2_archive/original_docs/` -- The actual files, classified by
`scripts/classify_plan_docs.py` into 13 themes.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/classify_plan_docs.py` | Classify docs into 13 themes (keyword + manual overrides) |
| `scripts/build_plan_archive.py` | Build/rebuild DuckDB full-text index |
| `scripts/update_plan_archive.py` | Incremental update (upsert, stale detection, rebuild) |
| `scripts/generate_audit_report.py` | Health dashboard + phantom plan detection |

## Workflow

1. **Before planning**: Read `navigator.yaml` to understand existing plans
2. **Targeted search**: `uv run python scripts/build_plan_archive.py --search "loss function"`
3. **After creating plans**: `uv run python scripts/update_plan_archive.py --file <path>`
4. **Periodic audit**: `uv run python scripts/generate_audit_report.py`

## Health Scores (2026-03-22, code-verified)

| Theme | Score | Notes |
|-------|-------|-------|
| training | 92 | Most mature. Loss, metrics, HPO all resolved. |
| cloud | 80 | Two-provider architecture stable. |
| infrastructure | 78 | Docker + Prefect connected. |
| models | 75 | DynUNet stable, SAM3 implemented. |
| observability | 72 | MLflow configured. XAI partial. |
| testing | 70 | Three-tier system working. |
| architecture | 68 | Config + flow topology resolved. |
| operations | 65 | TRIPOD + cards started. |
| manuscript | 58 | Scaffold present, needs content. |
| evaluation | 55 | Biostatistics designed, not fully wired. |
| deployment | 55 | BentoML scaffold only. |
| harness | 50 | KG growing, harness improving. |
| data | 45 | DVC configured, lineage not started. |
