# Model Registry with Promotion Stages — Implementation Plan (Issue #50)

## Current State
- ExperimentTracker wraps MLflow with basic register_model()
- AuditTrail logs model training/deployment events
- ModelCard generates markdown documentation
- IEC 62304 lifecycle stages defined
- No structured promotion workflow (dev → staging → production → archived)

## Architecture

### New Module: `src/minivess/observability/model_registry.py`
- **ModelStage** — StrEnum: DEVELOPMENT, STAGING, PRODUCTION, ARCHIVED
- **ModelVersion** — Dataclass: name, version, stage, metrics, semantic version parts
- **PromotionCriteria** — Dataclass: metric thresholds that must be met for promotion
- **PromotionResult** — Dataclass: approved/rejected with reason and metrics comparison
- **ModelRegistry** — Orchestrator:
  - register_version() — register a new model version in DEVELOPMENT
  - evaluate_promotion() — compare metrics against criteria
  - promote() — transition stage with audit logging
  - get_production_model() — retrieve current production version
  - list_versions() — list all versions of a model
  - to_markdown() — generate registry report

## Test Plan
- `tests/v2/unit/test_model_registry.py` (~12 tests)
  - TestModelStage: enum values
  - TestModelVersion: construction, semantic versioning
  - TestPromotionCriteria: threshold checking
  - TestModelRegistry: register, promote, reject, production lookup, markdown
