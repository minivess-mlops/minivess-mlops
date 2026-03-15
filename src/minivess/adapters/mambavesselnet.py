"""MambaVesselNet++ adapter — ModelAdapter ABC implementation.

Architecture: Chen et al. 2024 (ACM MM) / Xu & Chen et al. 2025 (ACM TOMM)
Reference:    https://github.com/CC0117/MambaVesselNet  (MIT license)
DOI MM2024:   https://doi.org/10.1145/3696409.3700231
DOI TOMM2025: https://doi.org/10.1145/3757324

Design decisions:
- D01: Inline backbone (MIT-licensed code adapted into this module)
- D02: BidirMamba = two standard Mamba modules (fwd + bwd.flip), no vendored fork
- D03: No permutation — native (B,C,H,W,D) MONAI convention throughout
- D04: AMP conservative — train ON, val OFF (mirrors SAM3, MONAI #4243)
- D05: eval_roi_size = (64,64,64) — matches training patch
- D06: All MONAI blocks require norm_name="instance"
- D07: LayerNorm INSIDE BidirectionalMambaLayer only (no double-LN)
- D08: _mamba_available() lives in model_builder, re-exported here for tests
- D09: GPU tests in tests/gpu_instance/ (not tests/v2/unit/)
- D10: ONNX export raises NotImplementedError (mamba selective scan not traceable)

PLATFORM PAPER — NOT SOTA RACE.
MambaVesselNet demonstrates the SSM family integrates via ModelAdapter ABC.
BANNED: "outperforms / surpasses / beats DynUNet"
"""

from __future__ import annotations

# Re-export availability check for test imports (D08)
from minivess.adapters.model_builder import _mamba_available as _mamba_available

__all__ = ["_mamba_available"]
