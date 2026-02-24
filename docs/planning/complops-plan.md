# ComplOps Regulatory Automation — Implementation Plan (Issue #49)

## Current State
- RegOps CI/CD pipeline (#21) generates IEC 62304 docs
- RegulatoryDocGenerator produces DHF, risk analysis, SRS, validation
- EU AI Act compliance checklist (#20) implemented
- No 510(k) or EU MDR technical file templates
- No LLM-assisted compliance validation

## Architecture

### New Module: `src/minivess/compliance/complops.py`
- **RegulatoryTemplate** — StrEnum: FDA_510K, EU_MDR_TECH_FILE, IEC_62304_FULL
- **ComplianceCheckResult** — Dataclass: template, gaps, score, recommendations
- **generate_510k_summary()** — FDA 510(k) predicate comparison template
- **generate_eu_mdr_technical_file()** — EU MDR Annex II/III technical doc template
- **assess_compliance_gaps()** — Automated gap analysis across all frameworks

## Test Plan
- `tests/v2/unit/test_complops.py` (~10 tests)
  - TestRegulatoryTemplate: enum values
  - TestComplianceCheckResult: construction, score, gaps
  - Test510kSummary: generation, sections
  - TestEUMDRTechFile: generation, sections
  - TestComplianceGapAssessment: gap detection, scoring
