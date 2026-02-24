# MONAI Deploy Clinical Deployment Pathway — Implementation Plan (Issue #47)

## Current State
- BentoML REST service in `serving/bento_service.py`
- ONNX Runtime inference in `serving/onnx_inference.py`
- Gradio demo in `serving/gradio_demo.py`
- ServingConfig in `config/models.py` (host, port, batch, timeout)
- No DICOM/FHIR support, no clinical deployment packaging

## Architecture

### New Module: `src/minivess/serving/clinical_deploy.py`
- **DeploymentTarget** — StrEnum: RESEARCH, STAGING, CLINICAL, PACS
- **DICOMConfig** — Dataclass: AE title, host, port, DICOM SCP settings
- **ClinicalDeployConfig** — Dataclass: model path, deployment target, DICOM settings
- **DICOMHandler** — DICOM I/O abstraction:
  - validate_dicom_metadata() — validate required DICOM tags
  - create_dicom_sr() — generate DICOM Structured Report from predictions
- **MonaiDeployManifest** — MAP (MONAI Application Package) manifest:
  - to_dict() — export as manifest dictionary
  - to_markdown() — human-readable deployment report
- **ClinicalDeploymentPipeline** — Orchestrates clinical deployment:
  - validate() — pre-deployment validation checks
  - generate_manifest() — create MAP manifest
  - to_markdown() — deployment summary report

## Test Plan
- `tests/v2/unit/test_clinical_deploy.py` (~12 tests)
  - TestDeploymentTarget: enum values
  - TestDICOMConfig: construction, defaults
  - TestDICOMHandler: validate metadata, create SR
  - TestMonaiDeployManifest: construction, to_dict, markdown
  - TestClinicalDeploymentPipeline: validate, manifest, markdown
