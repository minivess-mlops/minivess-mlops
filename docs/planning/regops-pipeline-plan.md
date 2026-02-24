# RegOps CI/CD Pipeline Extension — Implementation Plan (Issue #21)

## Current State
- RegulatoryDocGenerator exists (IEC 62304 DHF, risk analysis, SRS, validation)
- AuditTrail with JSON persistence
- GitHub Actions CI (ci-v2.yml) has lint/type/test only
- No compliance gates or regulatory artifact generation in CI

## Architecture

### New Module: `src/minivess/compliance/regops.py`
- **CIContext** — Dataclass capturing CI/CD environment (commit SHA, actor, ref, run ID)
- **RegOpsPipeline** — Orchestrator that:
  - Builds AuditTrail from CI context
  - Generates all regulatory documents via RegulatoryDocGenerator
  - Generates EU AI Act compliance report
  - Writes artifacts to output directory
  - Returns manifest of generated files
- **generate_ci_audit_entry()** — Create AuditEntry from CI environment vars
- **generate_regulatory_artifacts()** — Convenience function for CI scripts

### New Workflow: `.github/workflows/regulatory-artifacts.yml`
- Triggered on push to main + manual dispatch
- Calls RegOpsPipeline to generate docs
- Uploads artifacts with retention

## Test Plan
- `tests/v2/unit/test_regops.py` (~12 tests)
  - TestCIContext: construction, from_env, defaults
  - TestRegOpsPipeline: artifact generation, manifest, output files
  - TestCIAuditEntry: entry creation, metadata fields
