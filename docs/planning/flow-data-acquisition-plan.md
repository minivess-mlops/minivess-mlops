# Flow 0: Data Acquisition — Implementation Plan

**Issue:** #328 | **Branch:** `feat/flow-data-acquisition` | **Date:** 2026-03-04

## Context

The pipeline currently starts at Flow 1 (Data Engineering) which assumes NIfTI files
already exist on disk. Flow 0 (Data Acquisition) sits before Flow 1 and handles:

1. **Downloading datasets** — Automated where possible, manual instructions where not
2. **Format conversion** — OME-TIFF → NIfTI (the microscopy-native format path)
3. **Provenance logging** — Record acquisition metadata for reproducibility

This is a **placeholder flow** — the real-world acquisition path (adaptive microscope
control, real-time edge inference) is a future L3 PRD decision. For now, we solve the
practical problem: "how does a new PhD student get the data onto their machine?"

### Three Datasets

| Dataset | Volumes | Download | Format | Programmatic? |
|---------|---------|----------|--------|---------------|
| **MiniVess** | 70 | EBRAINS (requires login + approval) | NIfTI | No — manual download, login required |
| **DeepVess** | 1 | Cornell eCommons | TIFF | Partial — page requires click-through |
| **TubeNet-2PM** | 1 | UCL Research Data Repository | TIFF | Partial — Figshare-like API possible |
| **VesselNN** | 12 | GitHub (petteriTeikari/vesselNN) | NIfTI | Yes — `git clone` or `gh` |

### Flow Position in Pipeline

```
Flow 0 (NEW): Acquisition → Flow 1: Data → Flow 2: Train → ...
```

- **Core flow:** Yes (no data = no pipeline)
- **Work pool:** `cpu-pool`
- **Docker image:** `minivess-acquisition:latest`

## Phases

### Phase 1: Configuration & Result Types (2 tasks)

**Task 1.1: AcquisitionConfig dataclass**
- File: `src/minivess/config/acquisition_config.py`
- Fields:
  - `datasets: list[str]` — which datasets to acquire (default: all 4)
  - `output_dir: Path` — where to write NIfTI files (default: `data/raw/`)
  - `skip_existing: bool` — skip if output already exists (default: True)
  - `convert_formats: bool` — run OME-TIFF→NIfTI conversion (default: True)
  - `verify_checksums: bool` — verify downloaded files (default: True)
- Tests: valid config, invalid dataset name, default values

**Task 1.2: AcquisitionResult dataclass**
- File: `src/minivess/orchestration/flows/acquisition_flow.py` (top of file)
- Fields:
  - `datasets_acquired: dict[str, DatasetAcquisitionStatus]`
  - `total_volumes: int`
  - `conversion_log: list[str]`
  - `provenance: dict[str, Any]`
- `DatasetAcquisitionStatus`: enum-like with `READY`, `DOWNLOADED`, `MANUAL_REQUIRED`, `FAILED`

### Phase 2: Dataset Acquisition Registry (3 tasks)

**Task 2.1: DatasetAcquisitionEntry dataclass + registry**
- File: `src/minivess/data/acquisition_registry.py`
- Extends `ExternalDatasetConfig` concept with acquisition-specific fields:
  - `download_method: str` — `"manual"`, `"git_clone"`, `"http_download"`, `"api"`
  - `requires_auth: bool`
  - `source_format: str` — `"nifti"`, `"tiff"`, `"ome_tiff"`
  - `manual_instructions: str` — human-readable download steps
  - `expected_checksums: dict[str, str] | None` — filename → SHA256
- Registry covers all 4 datasets with accurate download methods
- Tests: registry completeness, all datasets have instructions

**Task 2.2: Download check task**
- File: `src/minivess/data/acquisition_registry.py`
- `check_dataset_availability(dataset_name: str, output_dir: Path) -> DatasetAcquisitionStatus`
- Checks if expected files already exist in output_dir
- Returns READY if all volumes present, MANUAL_REQUIRED if missing
- Tests: existing dir → READY, missing dir → MANUAL_REQUIRED

**Task 2.3: VesselNN git clone downloader**
- File: `src/minivess/data/downloaders.py`
- `download_vesselnn(target_dir: Path) -> Path`
- Uses `subprocess.run(["git", "clone", "--depth", "1", url, str(target_dir)])`
- Only dataset with fully automated download
- Tests: mock subprocess, verify directory creation

### Phase 3: Format Conversion (2 tasks)

**Task 3.1: TIFF → NIfTI converter**
- File: `src/minivess/data/format_conversion.py`
- `convert_tiff_to_nifti(input_path: Path, output_path: Path, voxel_spacing: tuple[float, float, float]) -> Path`
- Uses `tifffile` (already in deps) to read + `nibabel` to write NIfTI
- Preserves voxel spacing from dataset registry (NOT from TIFF metadata, which is unreliable)
- Handles 3D stacks and multi-page TIFFs
- Idempotent: skip if output exists and skip_existing=True
- Tests: synthetic 3D array round-trip, spacing preservation, skip-existing

**Task 3.2: Batch conversion task**
- File: `src/minivess/data/format_conversion.py`
- `convert_dataset_formats(dataset_name: str, input_dir: Path, output_dir: Path, registry: dict) -> list[str]`
- Walks input_dir, converts all TIFF/OME-TIFF to NIfTI
- Returns log of conversions performed
- Tests: mock filesystem with mixed formats

### Phase 4: Prefect Flow (3 tasks)

**Task 4.1: @task functions**
- File: `src/minivess/orchestration/flows/acquisition_flow.py`
- Tasks:
  - `check_dataset_status_task(dataset_name, output_dir) -> DatasetAcquisitionStatus`
  - `download_dataset_task(dataset_name, output_dir) -> DatasetAcquisitionStatus`
  - `convert_formats_task(dataset_name, input_dir, output_dir) -> list[str]`
  - `log_acquisition_provenance_task(results) -> dict[str, Any]`
  - `print_manual_instructions_task(datasets: list[str]) -> None` — logs instructions for manual datasets

**Task 4.2: @flow orchestrator**
- `run_acquisition_flow(config: AcquisitionConfig) -> AcquisitionResult`
- Steps:
  1. For each dataset: check status → download if possible → convert if needed
  2. Collect manual-required datasets and log instructions
  3. Log provenance to MLflow
  4. Return AcquisitionResult
- Quality gate: at least MiniVess must be READY (others are external test sets)
- Tests: mock all tasks, verify flow orchestration logic

**Task 4.3: Register in trigger chain + deployments**
- Update `trigger.py`: add `"acquisition"` as first flow in `_DEFAULT_FLOWS`
- Update `deployments.py`: add `"acquisition"` to `FLOW_WORK_POOL_MAP` (cpu-pool) and `FLOW_IMAGE_MAP`
- Tests: verify flow ordering, verify deployment config

### Phase 5: Integration & Dockerfile (2 tasks)

**Task 5.1: Integration test**
- File: `tests/integration/test_acquisition_flow_integration.py`
- Test the full flow with mocked downloads
- Verify: flow produces AcquisitionResult, provenance dict has expected keys
- Verify: manual instructions printed for non-automatable datasets

**Task 5.2: Dockerfile.acquisition**
- File: `deployment/docker/Dockerfile.acquisition`
- Follows pattern of existing Dockerfiles (FROM minivess-base)
- CPU-only (no CUDA needed for format conversion)

## File Inventory

### New Files
| File | Purpose |
|------|---------|
| `src/minivess/config/acquisition_config.py` | AcquisitionConfig dataclass |
| `src/minivess/data/acquisition_registry.py` | Dataset registry with download methods |
| `src/minivess/data/downloaders.py` | Automated download functions (VesselNN) |
| `src/minivess/data/format_conversion.py` | TIFF → NIfTI conversion |
| `src/minivess/orchestration/flows/acquisition_flow.py` | Prefect Flow 0 |
| `deployment/docker/Dockerfile.acquisition` | Docker image |
| `tests/unit/test_acquisition_config.py` | Config tests |
| `tests/unit/test_acquisition_registry.py` | Registry tests |
| `tests/unit/test_downloaders.py` | Downloader tests |
| `tests/unit/test_format_conversion.py` | Conversion tests |
| `tests/unit/test_acquisition_flow.py` | Flow unit tests |
| `tests/integration/test_acquisition_flow_integration.py` | Integration test |

### Modified Files
| File | Change |
|------|--------|
| `src/minivess/orchestration/trigger.py` | Add "acquisition" to _DEFAULT_FLOWS |
| `src/minivess/orchestration/deployments.py` | Add acquisition to work pool + image maps |

## Non-Goals (Explicitly Deferred)

- **Adaptive microscope control** — L3 PRD decision (conformal bandit), not Flow 0
- **OME-TIFF metadata extraction** — Future enhancement when we have real OME-TIFF data
- **DVC integration** — Already handled in Flow 1, not duplicated here
- **Label Studio upload** — Separate annotation flow, not acquisition
- **Programmatic EBRAINS download** — Requires OAuth2 + data use agreement, manual only
- **HTTP download for DeepVess/TubeNet** — Repositories require click-through; provide URLs + instructions

## TDD Order

Each task follows RED → GREEN → VERIFY → CHECKPOINT:

1. Task 1.1 → Task 1.2 (config + result types, no deps)
2. Task 2.1 → Task 2.2 → Task 2.3 (registry, then checks, then downloader)
3. Task 3.1 → Task 3.2 (converter, then batch)
4. Task 4.1 → Task 4.2 → Task 4.3 (tasks, flow, integration into chain)
5. Task 5.1 → Task 5.2 (integration test, Dockerfile)

## Estimated Scope

- ~12 tasks across 5 phases
- ~12 new files (6 source + 6 test)
- ~2 modified files
- Predominantly placeholder/scaffold code — the hard acquisition logic (microscope
  control, real-time inference) lives in future PRD decisions
