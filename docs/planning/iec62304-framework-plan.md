# Full IEC 62304 Compliance Framework — Implementation Plan (Issue #46)

## Current State
- SaMDRiskClass enum exists (Class I, IIa, IIb, III)
- RegulatoryDocGenerator produces DHF, risk analysis, SRS, validation summary
- AuditTrail with JSON persistence
- No IEC 62304 lifecycle stage tracking
- No software classification (Class A/B/C)
- No traceability matrix
- No PCCP template

## Architecture

### New Module: `src/minivess/compliance/iec62304.py`
- **SoftwareSafetyClass** — StrEnum: CLASS_A, CLASS_B, CLASS_C
- **LifecycleStage** — StrEnum: DEVELOPMENT, VERIFICATION, VALIDATION, RELEASE, MAINTENANCE
- **TraceabilityEntry** — Dataclass: requirement_id → implementation_ref → test_ref
- **TraceabilityMatrix** — Collection of TraceabilityEntry with to_markdown()
- **PCCPTemplate** — Predetermined Change Control Plan template with to_markdown()
- **IEC62304Framework** — Orchestrator combining lifecycle, traceability, PCCP

## Test Plan
- `tests/v2/unit/test_iec62304.py` (~12 tests)
  - TestSoftwareSafetyClass: enum values, classification
  - TestLifecycleStage: enum values, ordering
  - TestTraceabilityMatrix: construction, add entries, markdown, coverage
  - TestPCCPTemplate: construction, markdown, change types
  - TestIEC62304Framework: orchestration, lifecycle tracking
