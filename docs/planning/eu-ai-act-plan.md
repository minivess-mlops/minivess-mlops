# EU AI Act Compliance Checklist — Implementation Plan (Issue #20)

## Current State
- SaMDRiskClass enum exists (Class I, IIa, IIb, III)
- IEC 62304 regulatory doc generator exists
- No EU AI Act specific compliance

## Architecture

### New Module: `src/minivess/compliance/eu_ai_act.py`
- **EUAIActRiskLevel** — StrEnum: UNACCEPTABLE, HIGH, LIMITED, MINIMAL
- **EUAIActChecklist** — Dataclass mapping MinIVess to EU AI Act requirements
  - risk_classification, data_governance, transparency, human_oversight, etc.
- **classify_risk_level()** — Determine EU AI Act risk level for medical AI
- **generate_compliance_report()** — Produce gap analysis markdown

## Test Plan
- `tests/v2/unit/test_eu_ai_act.py` (~10 tests)
  - TestRiskLevel: enum values, medical AI classification
  - TestChecklist: construction, markdown, required fields
  - TestComplianceReport: generation, sections
