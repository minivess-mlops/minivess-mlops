# Automated Regulatory Documentation Generation — Implementation Plan (Issue #8)

## Current State
- `AuditTrail` tracks IEC 62304 lifecycle events (data access, training, deployment, test)
- `ModelCard` generates lite markdown documentation
- No IEC 62304 document templates
- No design history file generation
- No risk analysis document generation

## Architecture

### New Module: `src/minivess/compliance/regulatory_docs.py`

1. **RegulatoryDocGenerator** — Generates IEC 62304 document templates from AuditTrail
2. **generate_design_history()** — Design History File from audit trail events
3. **generate_risk_analysis()** — SaMD risk analysis template with classification
4. **generate_srs()** — Software Requirements Specification template
5. **SaMDRiskClass** — EU MDR / FDA risk classification enum

### Document Templates Generated
- **Design History File (DHF)**: Chronological development record from audit trail
- **Risk Analysis**: SaMD risk classification + mitigation strategies
- **Software Requirements Specification (SRS)**: IEC 62304 §5.2 template
- **Validation Summary**: Test metrics + coverage from audit trail

### Integration Points
- Reads from AuditTrail entries (already collecting lifecycle events)
- Outputs markdown documents (consistent with ModelCard)
- Designed for CI/CD integration (generate docs on each release)

## New Files
- `src/minivess/compliance/regulatory_docs.py`

## Modified Files
- `src/minivess/compliance/__init__.py` — add regulatory doc exports

## Test Plan
- `tests/v2/unit/test_regulatory_docs.py` (~15 tests)
