# Generative UQ Methods — Implementation Plan (Issue #51)

## Current State
- MC Dropout in `ensemble/mc_dropout.py`
- Deep Ensembles in `ensemble/deep_ensembles.py`
- Conformal prediction in `ensemble/conformal.py`
- No multi-rater data support, no generative UQ

## Architecture

### New Module: `src/minivess/ensemble/generative_uq.py`
- **GenerativeUQMethod** — StrEnum: PROB_UNET, PHISEG, SSN
- **MultiRaterData** — Dataclass: volume_id, masks (list of rater annotations)
- **GenerativeUQConfig** — Dataclass: method, latent_dim, num_samples
- **generalized_energy_distance()** — GED metric for multi-rater evaluation
- **q_dice()** — Quantized Dice from QUBIQ (staged thresholds)
- **GenerativeUQEvaluator** — Evaluate generative UQ methods:
  - add_prediction_samples() — register model samples
  - add_rater_annotations() — register ground-truth raters
  - compute_ged() — GED between samples and raters
  - compute_q_dice() — Q-Dice metric
  - to_markdown() — evaluation report

## Test Plan
- `tests/v2/unit/test_generative_uq.py` (~12 tests)
  - TestGenerativeUQMethod: enum values
  - TestMultiRaterData: construction, rater count
  - TestGenerativeUQConfig: construction, defaults
  - TestGED: identical, different, symmetric
  - TestQDice: perfect, imperfect, empty
  - TestGenerativeUQEvaluator: add data, compute metrics, markdown
