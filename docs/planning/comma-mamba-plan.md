# COMMA/Mamba Architecture Adapter — Implementation Plan (Issue #9)

## Current State
- `ModelAdapter` ABC in `adapters/base.py` with 6 abstract methods
- 4 working adapters: SegResNet, SwinUNETR, Vista3D, LoRA wrapper
- 1 stub: Sam3 (exploratory)
- `ModelFamily` enum in `config/models.py` has 5 options
- No SSM/Mamba architecture in the project

## Architecture

### New Module: `src/minivess/adapters/comma.py`

1. **MambaBlock** — Pure-PyTorch selective state-space block (no CUDA dependency)
   - Linear expansion → 1D depthwise conv → SiLU gate → output projection
   - Processes flattened spatial sequences (D*H*W) for long-range dependencies

2. **CoordinateEmbedding** — 3D coordinate-aware positional embedding
   - Adds learnable coordinate maps (the "Coordinate" in COMMA)

3. **CommaBlock** — COMMA encoder/decoder block combining Conv3d + MambaBlock

4. **CommaAdapter** — Full ModelAdapter implementing COMMA UNet-like architecture
   - Encoder: Conv3d downsampling + MambaBlock at each resolution
   - Decoder: TransposeConv3d upsampling + skip connections
   - Configurable init_filters, n_levels, d_state

### Modified Files
- `src/minivess/config/models.py` — Add `COMMA_MAMBA = "comma_mamba"` to ModelFamily
- `src/minivess/adapters/__init__.py` — Export CommaAdapter

### Design Decisions
- Pure PyTorch implementation (no `mamba-ssm` CUDA dependency) for CPU/CI compatibility
- SSM block approximates Mamba's selective scan using 1D conv + gating
- Small default architecture (init_filters=32, d_state=16) for tractable unit tests
- Follows existing adapter patterns exactly (SegResNet as reference)

## Test Plan
- `tests/v2/unit/test_comma_adapter.py` (~15 tests)
  - TestCommaAdapter: isinstance, forward shape, probability output, batch size
  - TestMambaBlock: standalone forward, sequence processing
  - TestCoordinateEmbedding: shape preservation, learnable params
  - TestCheckpoint: save/load round-trip, config serialization
  - TestONNX: export (if supported)
