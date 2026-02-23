# ADR-0001: Model Adapter Abstract Base Class

## Status

Accepted

## Date

2026-02-23

## Context

MinIVess MLOps v2 is designed as a model-agnostic biomedical segmentation platform. The v0.1-alpha codebase was tightly coupled to a single MONAI model family (U-Net, SegResNet), making it impractical to swap in newer architectures such as VISTA-3D, SAMv3, or SwinUNETR without rewriting training, evaluation, and serving code.

The v2 roadmap targets support for heterogeneous model families including MONAI-native architectures, foundation models with LoRA adapters (SAMv3), and custom research models. All must share a common pipeline for training, evaluation, ensembling, ONNX export, and serving.

## Decision

We define a `ModelAdapter` abstract base class (`src/minivess/adapters/base.py`) that extends both `abc.ABC` and `torch.nn.Module`. Every segmentation model in the system must subclass `ModelAdapter` and implement the following abstract methods:

- `forward(images: Tensor, **kwargs) -> SegmentationOutput` -- inference producing standardized output
- `get_config() -> dict[str, Any]` -- serializable configuration for reproducibility
- `load_checkpoint(path: Path) -> None` -- weight loading
- `save_checkpoint(path: Path) -> None` -- weight persistence
- `trainable_parameters() -> int` -- parameter count for diagnostics
- `export_onnx(path: Path, example_input: Tensor) -> None` -- ONNX export for serving

`SegmentationOutput` is a standardized dataclass with `prediction` (softmax probabilities), `logits` (raw outputs), and `metadata` fields.

The design uses inheritance (`ABC + nn.Module`) rather than a Python Protocol because:

1. Adapters must carry trainable state (weights), making `nn.Module` inheritance unavoidable.
2. ABC enforcement catches missing method implementations at class definition time, not at runtime.
3. The `forward()` contract enables standard PyTorch training loops, `torch.compile`, and ONNX tracing without adapter-specific branching.

## Consequences

**Positive:**

- Any new model family (MONAI, SAM, custom) is a single file implementing six methods.
- The training engine, ensemble module, evaluation pipeline, BentoML service, and ONNX export all program against `ModelAdapter`, eliminating model-specific conditional logic.
- `SegmentationOutput` standardizes the interface between inference and downstream consumers (metrics, calibration, conformal prediction).
- WeightWatcher spectral diagnostics and Deepchecks Vision operate on any adapter via `nn.Module` inspection.

**Negative:**

- Models that do not naturally fit the `(B, C, D, H, W)` tensor contract (e.g., 2D slice-based models, point-cloud models) require wrapper logic inside their adapter.
- The `**kwargs` escape hatch in `forward()` (needed for SAMv3 prompts) reduces type safety compared to a fully typed signature.

**Neutral:**

- Existing MONAI adapters (SegResNet, SwinUNETR) are thin wrappers of approximately 80 lines each, confirming low overhead.
