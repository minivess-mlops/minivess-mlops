# AtlasSegFM One-Shot Foundation Model Customization — Implementation Plan (Issue #15)

## Current State
- ModelAdapter ABC in `adapters/base.py` with 7 concrete implementations
- LoRA wrapper in `adapters/lora.py` for parameter-efficient fine-tuning
- VesselFM and VISTA-3D foundation model adapters exist
- PRD decision `foundation_model_integration` covers LoRA strategy
- No atlas-guided registration/adaptation infrastructure

## Architecture

### New Module: `src/minivess/adapters/atlas.py`
- **AtlasRegistrationMethod** — StrEnum: AFFINE, DEFORMABLE, LANDMARK
- **AtlasConfig** — Dataclass: atlas volume path, registration method, one-shot settings
- **AtlasRegistrationResult** — Dataclass: warped atlas, deformation field, similarity score
- **register_atlas()** — Compute registration between atlas and target volume
- **AtlasSegFMAdapter** — ModelAdapter subclass:
  - Wraps a base ModelAdapter (like LoRA wraps any adapter)
  - Injects atlas-derived spatial priors as auxiliary input channel
  - Supports one-shot customization via atlas registration
  - get_config() returns atlas-specific metadata

### New Module: `src/minivess/adapters/adaptation_comparison.py`
- **AdaptationMethod** — StrEnum: FULL_FINETUNE, LORA, ATLAS_ONESHOT, ZERO_SHOT
- **AdaptationResult** — Dataclass: method, metrics, parameter count, training time
- **compare_adaptation_methods()** — Tabulate results across methods
- **FeasibilityReport** — generates markdown feasibility analysis

## Test Plan
- `tests/v2/unit/test_atlas.py` (~12 tests)
  - TestAtlasRegistrationMethod: enum values
  - TestAtlasConfig: construction, defaults
  - TestAtlasRegistrationResult: construction, similarity
  - TestRegisterAtlas: affine registration, deformable, identity
  - TestAtlasSegFMAdapter: construction, config, trainable params
  - TestAdaptationComparison: method enum, results, comparison table, feasibility
