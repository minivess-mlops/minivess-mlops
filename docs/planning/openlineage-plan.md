# OpenLineage/Marquez Lineage Tracking — Implementation Plan (Issue #44)

## Current State
- Marquez service in docker-compose (full profile) but no PostgreSQL backend
- `openlineage-python` not in pyproject.toml
- DVC pipeline has download → preprocess → validate_data stages
- AuditTrail + ExperimentTracker exist but emit no lineage events
- No OpenLineage client or event builders

## Architecture

### New Module: `src/minivess/observability/lineage.py`

1. **LineageEmitter** — Wrapper around OpenLineage client with pipeline-aware events
2. **emit_dataset_event()** — Emits input/output dataset facets
3. **emit_job_event()** — Emits job start/complete/fail events
4. **pipeline_run()** — Context manager for a full pipeline lineage trace

### Integration Points
- Pipeline stage boundaries: discover → preprocess → train → export
- Each stage emits START → COMPLETE/FAIL job events
- Dataset facets include file paths, schemas, row counts
- Falls back to no-op when Marquez is not available (local dev)

### OpenLineage Event Model
- **Job**: Named pipeline stage (e.g., "minivess.preprocess")
- **Run**: Single execution with UUID
- **Dataset**: Input/output data (namespace + name)
- **Facets**: Schema, stats, source location metadata

## New Files
- `src/minivess/observability/lineage.py`

## Modified Files
- `src/minivess/observability/__init__.py` — add lineage exports

## Dependencies
- `openlineage-python>=1.26` (add to pyproject.toml)

## Test Plan
- `tests/v2/unit/test_lineage.py` (~15 tests)
