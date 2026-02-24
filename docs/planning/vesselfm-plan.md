# vesselFM Foundation Model Adapter — Implementation Plan (Issue #3)

## Current State
- DynUNetAdapter exists with configurable filters
- No vesselFM integration
- huggingface_hub already in project deps

## Architecture

### New Module: `src/minivess/adapters/vesselfm.py`
- **VesselFMAdapter** — ModelAdapter wrapping vesselFM pre-trained DynUNet
  - Downloads checkpoint from HuggingFace hub (bwittmann/vesselFM)
  - Uses DynUNet with [32, 64, 128, 256, 320, 320] filters
  - Binary segmentation (1 output channel, sigmoid post-processing)
  - Converts to 2-class SegmentationOutput for API compatibility
- Config: `ModelFamily.VESSEL_FM` enum value

### Key Details
- vesselFM = DynUNet with 6 levels, [32,64,128,256,320,320] filters
- Input: (B, 1, D, H, W), Output: (B, 1, D, H, W) logits (sigmoid)
- Pre-trained on 17 vascular datasets (includes MiniVess)
- HuggingFace: bwittmann/vesselFM, filename: vesselFM_base.pt

## Test Plan
- `tests/v2/unit/test_vesselfm.py` (~10 tests)
  - TestVesselFMAdapter: isinstance, config, enum
  - TestVesselFMConfig: filters, channels
  - Mock-based tests (no actual HF download in CI)
