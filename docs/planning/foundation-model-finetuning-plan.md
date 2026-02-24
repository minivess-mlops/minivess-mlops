# Foundation Model Fine-Tuning — Implementation Plan (Issue #42)

## Current State
- `ModelAdapter` ABC defined with 2 implementations (SegResNet, SwinUNETR)
- `ModelFamily` enum includes MONAI_VISTA3D and SAM3_LORA stubs
- `ModelConfig` has lora_rank, lora_alpha, lora_dropout fields
- `peft>=0.14` in pyproject.toml (not yet used)
- MONAI >=1.4 (supports VISTA-3D natively)
- No segment-anything dependency (SAM3 is exploratory)

## Architecture

### 1. Vista3dAdapter (`adapters/vista3d.py`)
- Uses MONAI's SegResNetDS2 backbone (VISTA-3D's auto-seg backbone)
- Standard forward: images → logits → softmax predictions
- No interactive prompting in v1 (future work)
- Supports checkpoint loading for pretrained VISTA-3D weights

### 2. LoraModelAdapter (`adapters/lora.py`)
- Generic PEFT LoRA wrapper that wraps any ModelAdapter
- Applies LoRA to targeted linear/conv layers via PEFT config
- Exposes trainable_parameters() showing only LoRA params
- save/load LoRA adapter weights separately
- ONNX export by merging LoRA weights first

### 3. Sam3Adapter (`adapters/sam3.py`) — Exploratory
- Skeleton adapter with conditional import guard
- Config-aware (uses ModelConfig LoRA fields)
- Raises clear error if segment-anything not installed

## New Files
1. `src/minivess/adapters/vista3d.py`
2. `src/minivess/adapters/lora.py`
3. `src/minivess/adapters/sam3.py`
4. `tests/v2/unit/test_foundation_models.py`

## Modified Files
- `src/minivess/adapters/__init__.py` — add new exports
