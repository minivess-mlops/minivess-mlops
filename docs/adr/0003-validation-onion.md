# ADR-0003: Multi-Layer Validation ("Validation Onion")

## Status

Accepted

## Date

2026-02-23

## Context

Biomedical segmentation pipelines handle data at many levels of abstraction: raw NIfTI files on disk, metadata DataFrames extracted from headers, tensor batches in GPU memory, training metric logs, model weight diagnostics, and prediction drift statistics. No single validation tool covers all of these levels.

The v0.1-alpha codebase had no systematic validation. Misconfigured voxel spacings, corrupted NIfTI headers, and silently degraded training metrics went undetected until manual inspection. For SaMD-principled development (IEC 62304), every data transformation boundary must be validated, and the validation evidence must be auditable.

## Decision

We implement a multi-layer validation architecture where each layer uses the tool best suited to its data type and validation semantics:

| Layer | Tool | Scope | Location |
|-------|------|-------|----------|
| 1 | Pydantic v2 | Config schemas, API request/response | `config/models.py` |
| 2 | Pandera | DataFrame schemas (NIfTI metadata, metrics) | `validation/schemas.py` |
| 3 | Great Expectations | Batch quality expectations (row counts, distributions) | `validation/expectations.py` |
| 4 | Hypothesis | Property-based tests for config edge cases | `tests/v2/unit/test_config_models.py` |
| 5 | MONAI transforms | Spatial/intensity preprocessing invariants | `data/transforms.py` |
| 6 | TorchIO augmentation | Augmentation parameter bounds validation | `data/augmentation.py` |
| 7 | Deepchecks Vision | Image data integrity, train/test distribution | `validation/deepchecks_vision.py` |
| 8 | WeightWatcher | Model weight spectral diagnostics (alpha metric) | `ensemble/weightwatcher.py` |
| 9 | Evidently AI | Prediction drift detection (KS, PSI) | `validation/drift.py` |
| 10 | TorchMetrics | GPU-accelerated metric computation bounds | `pipeline/metrics.py` |
| 11 | Calibration checks | ECE/MCE within bounds, temperature scaling | `ensemble/calibration.py` |
| 12 | SaMD audit trail | SHA-256 data hashing, lifecycle event logging | `compliance/audit.py` |

Each layer validates at its natural boundary:

- **Layers 1-3**: Configuration and tabular metadata (before any compute).
- **Layer 4**: Generative testing of config invariants (at test time).
- **Layers 5-6**: Data preprocessing pipeline (before model ingestion).
- **Layers 7-9**: Data and model quality (after training, before deployment).
- **Layers 10-11**: Metric and calibration integrity (during/after evaluation).
- **Layer 12**: Compliance evidence (throughout the lifecycle).

## Consequences

**Positive:**

- Each validation tool operates at its optimal abstraction level rather than forcing one tool to cover everything.
- Validation failures are caught at the earliest possible pipeline stage, reducing wasted compute.
- SaMD audit trail captures validation evidence from all layers, supporting IEC 62304 traceability.
- Pandera and Great Expectations cover complementary concerns: schema conformance vs. batch distribution quality.

**Negative:**

- Twelve validation layers require twelve sets of test fixtures and configuration.
- Contributors must understand which layer is appropriate for a new validation rule.
- Some layers have overlapping concerns (e.g., Pandera range checks vs. Great Expectations value bounds), requiring discipline to avoid redundant checks.

**Neutral:**

- Deepchecks Vision, WeightWatcher, and Evidently are optional dependencies. Tests for these layers are skipped when the packages are not installed.
