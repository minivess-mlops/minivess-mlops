# BentoML + ONNX + Gradio Serving Plan

**Issue**: #36 — Make BentoML/ONNX/Gradio serving production-ready
**Date**: 2026-02-24
**Status**: Draft → Implementation

---

## 1. Problem Statement

The serving stack (BentoML, ONNX export, Gradio demo) was scaffolded in the
legacy phase but never tested end-to-end. All 8 serving tests are skipped due
to missing optional dependencies. We cannot currently:

- Export a trained SegResNet to ONNX
- Serve predictions via BentoML REST API
- Demonstrate segmentation via Gradio UI

---

## 2. Current State

| Component | File | Status |
|-----------|------|--------|
| BentoML service | `serving/bento_service.py` | Scaffolded, uses `output.prediction` (wrong — should use raw model output) |
| ONNX inference | `serving/onnx_inference.py` | Scaffolded, untested |
| Gradio demo | `serving/gradio_demo.py` | 2D-slice only, untested |
| ONNX export | `adapters/segresnet.py:export_onnx()` | Exists, untested |
| Tests | `test_serving.py` | 8 tests, all skipped (importorskip) |

**Key bugs in current code**:
- `bento_service.py:60` calls `output.prediction` — but the BentoML service
  calls `self.model(tensor)` which would return a `SegmentationOutput` dataclass
  only via the adapter. The service loads a raw PyTorch model, so output is a
  plain tensor.
- Gradio demo only handles 2D slices, not NIfTI volumes.

---

## 3. Implementation Plan

### Task T1: Install serving deps + fix test skips

Install `onnxruntime` and `gradio` into the dev environment so tests actually
run. Keep them as optional in pyproject.toml (academic users may not need serving).

```bash
uv sync --extra serving --extra dev
```

**Tests**: Existing 8 tests in `test_serving.py` should no longer skip.

### Task T2: ONNX export + round-trip test

Add tests that:
1. Create a SegResNetAdapter with known config
2. Export to ONNX via `adapter.export_onnx()`
3. Load with OnnxSegmentationInference
4. Run inference on a random volume
5. Verify output shape and value range

**New tests** (`test_serving.py`):
- `test_onnx_export_creates_file`
- `test_onnx_roundtrip_inference`
- `test_onnx_output_shape_matches_pytorch`

### Task T3: Fix BentoML service

The BentoML service needs to work with ONNX models (not PyTorch) for
production serving. Fix the predict method to:
1. Accept numpy volume
2. Run through OnnxSegmentationInference
3. Return segmentation dict

Also fix the service to load from an ONNX file path rather than BentoML model
store (simpler for academic use — no `bentoml models` registry needed).

**New tests**:
- `test_bento_predict_with_onnx_model`
- `test_bento_health_endpoint`

### Task T4: Gradio demo — NIfTI upload + slice viewer

Enhance the Gradio demo to:
1. Accept NIfTI file upload (`.nii.gz`)
2. Load volume with nibabel
3. Show slice slider for axial/sagittal/coronal views
4. Run segmentation (ONNX or dummy) on selected slice or full volume
5. Overlay segmentation mask on the image

**New tests**:
- `test_gradio_demo_creates_blocks`
- `test_gradio_predict_slice_dummy_mode`
- `test_gradio_nifti_loading` (synthetic NIfTI → slice extraction)

### Task T5: Docker compose service definition

Update `deployment/docker-compose.yml` to include a BentoML serving container.
Basic smoke test that the compose file parses correctly.

**Tests**:
- `test_docker_compose_valid_yaml`
- `test_docker_compose_has_bentoml_service`

### Task T6: Integration test — export → serve → predict

End-to-end: create model → export ONNX → load in inference engine → predict.

**Tests**:
- `test_export_serve_predict_roundtrip`

---

## 4. Execution Order

```
T1 (install deps) → T2 (ONNX export) → T3 (BentoML fix) → T4 (Gradio) → T5 (Docker) → T6 (integration)
```

---

## 5. Decision: Gradio for Demo UI

**Choice**: Gradio (already in project, HF ecosystem alignment)

**Rationale**:
- Already scaffolded in `serving/gradio_demo.py`
- Good enough for academic lab intranet (Docker deployment, no cloud)
- NIfTI upload + slice viewer covers the core use case
- Lab members can extend with standard Python — no frontend knowledge needed
- Integrates well with BentoML backend via API calls

**Alternatives considered** (from knowledge base):
- NiceGUI: More flexible for full web apps, but overkill for a demo/inference tool
- Streamlit: Global re-run model problematic for stateful 3D slice navigation
- Panel/HoloViz: Better for dashboards, worse for ML inference demos

---

## 6. Out of Scope

- GPU-optimised batch inference (future issue)
- Model registry integration with BentoML model store
- Public deployment / HF Spaces
- Real-time monitoring dashboard (separate from Evidently issue #38)
