# MedSAM3 Interactive Annotation Adapter — Implementation Plan (Issue #22)

## Current State
- Sam3Adapter stub in `adapters/sam3.py` (requires segment-anything package)
- Label Studio integration referenced in PRD for annotation platform
- No interactive annotation workflow infrastructure
- No medical concept-aware prompting

## Architecture

### New Module: `src/minivess/adapters/medsam3.py`
- **PromptType** — StrEnum: POINT, BOX, MASK, CONCEPT
- **MedicalConcept** — StrEnum: VESSEL, TUMOR, ORGAN, TISSUE, LESION
- **AnnotationPrompt** — Dataclass: prompt type, coordinates, concept label
- **MedSAM3Config** — Dataclass: model settings, concept vocabulary
- **MedSAM3Predictor** — Interactive annotation predictor:
  - add_prompt() — register point/box/concept prompts
  - predict() — run inference with accumulated prompts
  - reset() — clear prompts for new annotation
  - to_annotation_record() — export annotation metadata

### New Module: `src/minivess/data/annotation_session.py`
- **AnnotationSession** — Manages interactive annotation workflow:
  - start() — begin session for a volume
  - add_interaction() — record prompt-result pair
  - compute_agreement() — compare with reference annotation
  - to_markdown() — generate session report

## Test Plan
- `tests/v2/unit/test_medsam3.py` (~12 tests)
  - TestPromptType: enum values
  - TestMedicalConcept: enum values, vessel membership
  - TestAnnotationPrompt: point, box, concept prompts
  - TestMedSAM3Config: construction, defaults
  - TestMedSAM3Predictor: add prompt, reset, annotation record
  - TestAnnotationSession: interactions, agreement, markdown
