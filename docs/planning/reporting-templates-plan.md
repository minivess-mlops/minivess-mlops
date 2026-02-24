# Reporting Standard Templates — Implementation Plan (Issue #14)

## Current State
- ModelCard exists with basic to_markdown()
- RegulatoryDocGenerator covers IEC 62304 artifacts
- No CONSORT-AI or MI-CLEAR-LLM templates

## Architecture

### New Module: `src/minivess/compliance/reporting_templates.py`
- **ConsortAIChecklist** — Dataclass for CONSORT-AI minimum reporting items
  - Model description, training data, performance metrics, CI, clinical context
  - `to_markdown()` for document generation
- **MiClearLLMChecklist** — Dataclass for MI-CLEAR-LLM items
  - LLM model details, prompt strategy, hallucination handling, output validation
  - `to_markdown()` for document generation
- **generate_consort_ai_report()** — Generate from experiment data
- **generate_miclear_llm_report()** — Generate from LLM agent config

## Test Plan
- `tests/v2/unit/test_reporting_templates.py` (~12 tests)
  - TestConsortAI: construction, markdown, required fields
  - TestMiClearLLM: construction, markdown, required fields
  - TestIntegration: generate from data, sections present
