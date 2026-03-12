# Serving — BentoML + ONNX Runtime

## Architecture

BentoML service definitions for ONNX Runtime inference serving.
Models exported via Deploy Flow as ONNX, served via BentoML.

## Key Files

| File | Purpose |
|------|---------|
| `bento_service.py` | BentoML service definition |
| `onnx_runner.py` | ONNX Runtime inference runner |

## Deployment

- BentoML builds bentos from service definitions
- ONNX models loaded from MLflow model registry
- Gradio demo UI for interactive testing

## Key Rules

- ONNX is the export format (resolved PRD decision: model_export_format)
- Models come from MLflow artifact store, NOT from local filesystem
- No direct PyTorch inference in serving — ONNX only
