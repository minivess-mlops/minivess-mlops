# SynthICL Domain Randomization Augmentation — Implementation Plan (Issue #17)

## Current State
- MONAI spatial transforms in `data/transforms.py`
- TorchIO intensity augmentation in `data/augmentation.py`
- Synthetic drift generation in `data/drift_synthetic.py` (for monitoring)
- PRD `augmentation_stack` resolved with MONAI+TorchIO
- No domain randomization for synthetic training data

## Architecture

### New Module: `src/minivess/data/domain_randomization.py`
- **RandomizationParam** — StrEnum: INTENSITY, CONTRAST, NOISE, BLUR, SPACING
- **DomainRandomizationConfig** — Dataclass: parameter ranges, seed, num_augmented
- **SyntheticVesselGenerator** — Generate synthetic vessel-like structures:
  - random_tubular_mask() — synthetic tubular binary masks
  - randomize_intensity() — random intensity profiles
  - randomize_contrast() — random contrast transformations
- **DomainRandomizationPipeline** — Orchestrates multi-parameter randomization:
  - apply() — apply full randomization to a volume+mask pair
  - generate_batch() — produce N randomized synthetic samples
  - to_markdown() — report randomization settings

## Test Plan
- `tests/v2/unit/test_domain_randomization.py` (~12 tests)
  - TestRandomizationParam: enum values
  - TestDomainRandomizationConfig: construction, defaults
  - TestSyntheticVesselGenerator: tubular mask, intensity, contrast
  - TestDomainRandomizationPipeline: apply, batch, markdown report
