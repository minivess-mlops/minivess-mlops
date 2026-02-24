# Dataset Use Plan: MiniVess NIfTI Integration

**Issue**: #35 — Wire real MiniVess NIfTI data into v2 training pipeline
**Date**: 2026-02-24
**Status**: Draft → Review → Approved → Implementation

---

## 1. Problem Statement

The v2 pipeline trains on synthetic tensors (`torch.randn`) only. The real MiniVess
dataset (70 two-photon fluorescence microscopy volumes of rodent cerebrovasculature,
Poon et al. 2023) exists as a 939 MB ZIP but is not wired into the training pipeline.
No automated download, no data reorganisation, no DVC versioning of real data, and no
synthetic drift generation for monitoring validation.

---

## 2. Dataset Access (Verified)

**The MiniVess dataset is publicly downloadable without registration.**

| Property | Value |
|----------|-------|
| Source | EBRAINS Knowledge Graph |
| Dataset ID | `bf268b89-1420-476b-b428-b85a913eb523` |
| Paper DOI | `10.1038/s41597-023-02048-8` |
| Licence | CC BY-NC-SA |
| Total size | 939 MB (211 files) |
| Format | NIfTI compressed (`.nii.gz`) + JSON metadata |
| Volumes | 70 (uint16 raw images + uint8 binary segmentation masks) |
| Voxel spacing | ~1.0 µm × 1.0 µm × 1.0 µm (per-volume JSON) |
| Typical shape | 512 × 512 × 22-to-100+ slices |

### Directory Structure in ZIP

```
d-bf268b89-1420-476b-b428-b85a913eb523.zip
├── raw/mv01.nii.gz … mv70.nii.gz       # 70 fluorescence image volumes
├── seg/mv01_y.nii.gz … mv70_y.nii.gz   # 70 binary vessel segmentation masks
├── json/mv01.json … mv70.json           # 70 per-volume metadata files
└── Licence (CCBY-NC-SA).pdf
```

### Programmatic Download

```
EBRAINS Data-Proxy API (no auth required):
  List:     GET https://data-proxy.ebrains.eu/api/v1/datasets/{ID}?limit=300
  Download: GET https://data-proxy.ebrains.eu/api/v1/datasets/{ID}/{path}
  Stats:    GET https://data-proxy.ebrains.eu/api/v1/datasets/{ID}/stat

Direct CSCS storage:
  https://rgw.cscs.ch/ebrains:d-{ID}/{path}
```

---

## 3. Current State Analysis

### What works
- `discover_nifti_pairs(data_dir)` — expects `imagesTr/` + `labelsTr/` subdirectories
- MONAI `CacheDataset` with `LoadImaged` → `Spacingd` → `NormalizeIntensityd` → augmentation
- `DataConfig.data_dir` defaults to `Path("data/raw")`
- DVC configured with local MinIO remote + AWS S3 remote
- `data/minivess.dvc` tracks 211 files (984 MB) — but this references old v0.1 layout

### Gaps
| Gap | Description |
|-----|-------------|
| **Structure mismatch** | EBRAINS uses `raw/`, `seg/`; loader expects `imagesTr/`, `labelsTr/` |
| **No download script** | No way to fetch data programmatically |
| **No extraction** | ZIP must be extracted and reorganised |
| **Label naming** | EBRAINS: `mv01_y.nii.gz`; loader expects matching names: `mv01.nii.gz` |
| **No preprocess.py** | Referenced in `dvc.yaml` but file does not exist |
| **No synthetic NIfTI** | Tests use tensor fixtures, never exercise `discover_nifti_pairs()` |
| **No drift simulation** | No synthetic corrupted data for Evidently/monitoring validation |
| **DVC manifest stale** | `minivess.dvc` references old layout hash |

---

## 4. Implementation Plan

### Task 1: Download Script (`scripts/download_minivess.py`)

**Purpose**: Fetch the MiniVess dataset from EBRAINS and extract into the expected
directory structure.

**Behaviour**:
1. Check if `data/raw/minivess/imagesTr/` already has 70 `.nii.gz` files → skip
2. Check if `dataset_local/d-bf268b89-...zip` exists → extract from local ZIP
3. Otherwise → download individual files via EBRAINS data-proxy API
4. Reorganise into loader-expected structure:
   ```
   data/raw/minivess/
   ├── imagesTr/
   │   ├── mv01.nii.gz → mv70.nii.gz    (from raw/)
   ├── labelsTr/
   │   ├── mv01.nii.gz → mv70.nii.gz    (from seg/, renamed: mv01_y → mv01)
   └── metadata/
       ├── mv01.json → mv70.json         (from json/)
       └── licence.pdf
   ```
5. Verify integrity: 70 image files, 70 label files, matching names
6. Print summary statistics (total size, voxel spacing range, shape range)

**Design decisions**:
- Use `httpx` (already in project deps via LangGraph) for async download with progress
- Checksum verification via EBRAINS API `stat` endpoint
- Idempotent: re-running skips completed downloads
- No `requests` dependency — keep it lean

**Tests** (RED phase):
- `test_download_reorganise_structure`: Given extracted raw/seg/json dirs, verify
  reorganisation creates correct imagesTr/labelsTr structure with renamed labels
- `test_download_idempotent`: Second call is a no-op when data exists
- `test_download_integrity_check`: Verify file count and name matching

### Task 2: Data Preprocessing (`src/minivess/data/preprocess.py`)

**Purpose**: Implement the preprocessing stage referenced in `dvc.yaml`.

**Behaviour**:
1. Read all NIfTI pairs from `data/raw/minivess/`
2. Validate each volume against `NiftiMetadataSchema`
3. Resample to uniform voxel spacing (1.0 µm isotropic) if needed
4. Write processed volumes to `data/processed/minivess/`
5. Generate `data/validation_report.json` with per-volume statistics

**Tests**:
- `test_preprocess_creates_output_dir`: Verify output structure
- `test_preprocess_validates_schema`: Bad volume → raises ValidationError
- `test_preprocess_report_generated`: JSON report has expected fields

### Task 3: Synthetic NIfTI Fixtures (`tests/v2/fixtures/synthetic_nifti.py`)

**Purpose**: Generate small synthetic NIfTI files on disk for testing
`discover_nifti_pairs()` and the full MONAI loader pipeline without requiring
real 939 MB dataset.

**Behaviour**:
- `create_synthetic_nifti_dataset(tmp_path, n_volumes=3, spatial_size=(32,32,8))`
- Creates `imagesTr/` and `labelsTr/` with matching `.nii.gz` files
- Labels have ~10% foreground voxels (simulating sparse vessel annotations)
- Returns the data directory path

**Tests**:
- `test_discover_nifti_pairs_synthetic`: Verify pairs discovered from synthetic files
- `test_create_train_loader_synthetic`: Full MONAI loader on synthetic NIfTI
- `test_create_val_loader_synthetic`: Validation loader produces expected shapes

### Task 4: Synthetic Drift Generation (`src/minivess/data/drift_synthetic.py`)

**Purpose**: Generate controlled distribution shifts for validating drift detection
(Evidently/Alibi-Detect). This is a key novel contribution — topology-aware drift
signals for vascular segmentation.

**Drift types**:

| Drift Type | Implementation | Detection Signal |
|------------|----------------|-----------------|
| **Intensity shift** | Global brightness ±30% | Evidently feature drift (KS test) |
| **Noise injection** | Gaussian noise σ=0.05-0.20 | Evidently data quality |
| **Resolution degradation** | Downsample + upsample (blur) | Embedding drift (Alibi-Detect MMD) |
| **Topology corruption** | Erode/dilate vessel masks → broken connectivity | clDice drift, Betti number change |
| **Covariate shift** | Shift voxel spacing metadata | Schema validation failure |
| **Gradual drift** | Linear interpolation of above over N batches | Time-series drift detection |

**API**:
```python
def apply_drift(
    volume: torch.Tensor,
    drift_type: DriftType,
    severity: float = 0.5,   # 0.0 = no drift, 1.0 = extreme
    seed: int | None = None,
) -> torch.Tensor: ...

def generate_drifted_dataset(
    source_dir: Path,
    output_dir: Path,
    drift_config: DriftConfig,
) -> DriftManifest: ...
```

**Tests**:
- `test_intensity_drift_changes_distribution`: KS test p-value < 0.05 for severity > 0.3
- `test_topology_drift_changes_betti`: Betti-0 changes for erosion drift
- `test_gradual_drift_monotonic`: Increasing severity over batches
- `test_no_drift_preserves_distribution`: severity=0 → KS p-value > 0.05

### Task 5: DVC Pipeline Update

**Purpose**: Update DVC to version the real dataset and preprocessing outputs.

**Changes**:
1. Update `data/minivess.dvc` to track new directory structure
2. Update `dvc.yaml` preprocess stage to use actual `preprocess.py`
3. Add `download` stage to `dvc.yaml`:
   ```yaml
   stages:
     download:
       cmd: uv run python scripts/download_minivess.py
       outs:
         - data/raw/minivess/
       frozen: true  # Don't re-download on every dvc repro
   ```

**Tests**:
- `test_dvc_yaml_valid`: Parse dvc.yaml, verify stage dependencies are consistent
- `test_dvc_config_minio`: Verify MinIO remote configuration

### Task 6: Loader Enhancement

**Purpose**: Make `discover_nifti_pairs()` handle the MiniVess naming convention
natively while remaining backward-compatible.

**Changes**:
- Add `raw/` + `seg/` as additional discovery patterns (alongside `imagesTr/` + `labelsTr/`)
- Handle label suffix stripping (`mv01_y.nii.gz` → match `mv01.nii.gz`)
- Add metadata loading from `json/` directory (optional)

**Tests**:
- `test_discover_ebrains_layout`: raw/ + seg/ directory structure
- `test_discover_suffix_stripping`: `_y` suffix labels matched to images
- `test_discover_backward_compatible`: imagesTr/ + labelsTr/ still works

---

## 5. Directory Structure After Implementation

```
minivess-mlops/
├── data/
│   ├── raw/
│   │   └── minivess/
│   │       ├── imagesTr/          # 70 NIfTI image volumes (symlinks or copies)
│   │       ├── labelsTr/          # 70 NIfTI label volumes (renamed)
│   │       └── metadata/          # 70 JSON metadata + licence
│   ├── processed/
│   │   └── minivess/              # Resampled + validated volumes
│   ├── synthetic_drift/           # Generated drift datasets
│   ├── minivess.dvc               # DVC tracking manifest
│   └── .gitignore                 # Excludes raw/ and processed/
├── dataset_local/                 # Downloaded ZIP (gitignored, not DVC-tracked)
│   └── d-bf268b89-...zip
├── scripts/
│   └── download_minivess.py       # Download + extract + reorganise
├── src/minivess/data/
│   ├── loader.py                  # Enhanced discover_nifti_pairs()
│   ├── preprocess.py              # NEW: preprocessing pipeline
│   ├── drift_synthetic.py         # NEW: drift generation
│   ├── transforms.py              # Existing MONAI transforms
│   └── augmentation.py            # Existing TorchIO augmentation
└── tests/v2/
    ├── fixtures/
    │   └── synthetic_nifti.py     # NIfTI file generation helpers
    ├── unit/
    │   ├── test_download.py       # Download script tests
    │   ├── test_preprocess.py     # Preprocessing tests
    │   ├── test_drift.py          # Drift generation tests
    │   └── test_loader_nifti.py   # NIfTI loader tests
    └── integration/
        └── test_data_pipeline.py  # End-to-end: synthetic NIfTI → loader → batch
```

---

## 6. Task Execution Order

```
Task 3 (synthetic NIfTI fixtures)    ← Foundation — enables all other tests
    ↓
Task 6 (loader enhancement)          ← Wire discover_nifti_pairs for real data
    ↓
Task 1 (download script)             ← Fetch real data
    ↓
Task 2 (preprocessing)               ← Process real data
    ↓
Task 5 (DVC pipeline)                ← Version everything
    ↓
Task 4 (drift generation)            ← Synthetic drift for monitoring
```

---

## 7. Acceptance Criteria (Issue #35)

- [ ] `uv run python scripts/download_minivess.py` downloads and organises data
- [ ] `discover_nifti_pairs()` works with real MiniVess directory structure
- [ ] Full MONAI training pipeline runs on real NIfTI data (at least 1 epoch)
- [ ] Synthetic NIfTI fixtures exercise the file-based loader in CI
- [ ] DVC tracks the raw and processed data directories
- [ ] Drift generation produces detectable distribution shifts
- [ ] All new tests pass: `uv run pytest tests/v2/ -x -q`
- [ ] No regressions: existing 103 tests still pass

---

## 8. Out of Scope (Deferred)

- Evidently/Alibi-Detect integration (issue #38 — drift detection)
- Multi-site data harmonisation (future work)
- FHIR/EHR integration (future work)
- DataLad migration (experimental, issue not yet created)
- Label quality assessment with Cleanlab (separate issue)

---

## Review

### Reviewer 1: Data Engineering Perspective

**Concern**: The download script downloads 939 MB. CI/CD should NOT download
real data on every run. How is this handled?

**Response**: The download stage in `dvc.yaml` is `frozen: true`, meaning
`dvc repro` skips it. CI uses synthetic NIfTI fixtures (Task 3). Real data
download is manual (`uv run python scripts/download_minivess.py`) or
triggered explicitly. The `.gitignore` already excludes `data/raw/` and
`data/processed/`. The `dataset_local/` ZIP is also gitignored.

### Reviewer 2: Reproducibility Perspective

**Concern**: How do we ensure the exact same dataset version is used across
researchers? The EBRAINS API could change.

**Response**: Three layers of reproducibility:
1. DVC manifest (`minivess.dvc`) locks the exact file hashes
2. Download script verifies checksums from the EBRAINS stat endpoint
3. The `dataset_local/` ZIP serves as an offline backup source
4. `frozen: true` prevents accidental re-download

### Reviewer 3: Medical Imaging Perspective

**Concern**: The MiniVess data is two-photon fluorescence microscopy, not MRI.
Voxel spacing is in µm not mm. Does the validation schema handle this?

**Response**: Good catch. The current `NiftiMetadataSchema` allows spacing up
to 10.0 mm for x/y, which would reject µm-scale data if stored in mm units.
The MiniVess JSON metadata gives spacing as ~1.0 µm. The NIfTI header affine
typically stores this in mm (0.001 mm). We need to:
1. Read the JSON metadata to get true physical spacing
2. Either store spacing in the NIfTI header correctly (µm → mm conversion)
   OR relax the schema validation range for microscopy data
3. Add a `microscopy` domain flag to `DataConfig` that adjusts validation bounds

**Action**: Task 2 (preprocessing) will handle spacing unit normalisation.
Schema validation bounds will be parameterised by imaging modality.

### Reviewer 4: Testing Perspective

**Concern**: The synthetic NIfTI fixtures create files on disk. How do we avoid
test pollution and ensure cleanup?

**Response**: All synthetic NIfTI generation uses `tmp_path` (pytest fixture
providing a unique temporary directory per test, automatically cleaned up).
No test writes to the real `data/` directory. The `create_synthetic_nifti_dataset`
function takes `tmp_path` as its first argument, making cleanup automatic.

### Reviewer 5: MLOps Pipeline Perspective

**Concern**: The drift generation (Task 4) creates synthetic corrupted data.
How does this connect to the monitoring stack without implementing Evidently
(which is issue #38)?

**Response**: Task 4 creates the drift data and provides ground-truth drift
labels (which type and severity). It does NOT implement detection — that's
issue #38. The connection point is the `DriftManifest` output which records
what drift was applied to which volumes, so issue #38 can use this as a
test oracle for detection accuracy. Task 4 is self-contained and testable:
we verify drift was actually applied using statistical tests (KS test for
intensity, Betti numbers for topology) without needing Evidently.

---

## Appendix: TDD Task Specs (for self-learning-iterative-coder)

### Task Spec Format

Each task below is formatted for the TDD skill's outer loop.

```yaml
tasks:
  - id: T1_synthetic_nifti_fixtures
    description: "Create pytest fixtures that generate small NIfTI files on disk"
    red_spec:
      test_file: tests/v2/unit/test_loader_nifti.py
      tests:
        - test_create_synthetic_nifti_creates_files
        - test_discover_nifti_pairs_synthetic
        - test_create_train_loader_from_nifti_files
    green_spec:
      files_to_create:
        - tests/v2/fixtures/__init__.py
        - tests/v2/fixtures/synthetic_nifti.py
      files_to_modify: []

  - id: T2_loader_enhancement
    description: "Enhance discover_nifti_pairs() for EBRAINS raw/seg layout"
    red_spec:
      test_file: tests/v2/unit/test_loader_nifti.py
      tests:
        - test_discover_ebrains_layout
        - test_discover_suffix_stripping
        - test_discover_backward_compatible
    green_spec:
      files_to_modify:
        - src/minivess/data/loader.py

  - id: T3_download_script
    description: "Download MiniVess from EBRAINS with auto-extraction"
    red_spec:
      test_file: tests/v2/unit/test_download.py
      tests:
        - test_reorganise_ebrains_to_loader_structure
        - test_download_idempotent
        - test_download_verify_file_counts
    green_spec:
      files_to_create:
        - scripts/download_minivess.py

  - id: T4_preprocess
    description: "Preprocessing pipeline for DVC stage"
    red_spec:
      test_file: tests/v2/unit/test_preprocess.py
      tests:
        - test_preprocess_creates_output
        - test_preprocess_validation_report
        - test_preprocess_rejects_invalid_volume
    green_spec:
      files_to_create:
        - src/minivess/data/preprocess.py

  - id: T5_dvc_update
    description: "Update DVC pipeline and manifest"
    red_spec:
      test_file: tests/v2/unit/test_dvc_config.py
      tests:
        - test_dvc_yaml_stages_consistent
        - test_dvc_config_has_minio_remote
    green_spec:
      files_to_modify:
        - dvc.yaml
        - data/minivess.dvc

  - id: T6_drift_synthetic
    description: "Generate controlled distribution shifts"
    red_spec:
      test_file: tests/v2/unit/test_drift.py
      tests:
        - test_intensity_drift_detectable
        - test_noise_drift_detectable
        - test_topology_drift_changes_connectivity
        - test_no_drift_preserves_distribution
        - test_gradual_drift_monotonic
    green_spec:
      files_to_create:
        - src/minivess/data/drift_synthetic.py

  - id: T7_integration_test
    description: "End-to-end: synthetic NIfTI → loader → training step"
    red_spec:
      test_file: tests/v2/integration/test_data_pipeline.py
      tests:
        - test_nifti_to_training_step
        - test_nifti_to_validation_metrics
    green_spec:
      files_to_modify: []
```

---

*Plan version: 1.0 | Author: Claude Opus 4.6 | Reviewed: 5 synthetic reviewers*
