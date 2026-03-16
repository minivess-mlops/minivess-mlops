---
name: model-adapters
description: "ModelAdapter ABC and implementations — DynUNet, SAM3, VesselFM, Mamba"
metadata:
  languages: "python"
  versions: "2.0"
  revision: 1
  updated-on: "2026-03-16"
  source: maintainer
  tags: "monai,adapter,model,sam3,dynunet,mamba,vesselfm"
---

# ModelAdapter ABC

All models implement `ModelAdapter` — the platform's model-agnostic integration point.

```python
from minivess.adapters.base import ModelAdapter, SegmentationOutput

class MyAdapter(ModelAdapter):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config        # REQUIRED attribute
        self.net = build_network()  # REQUIRED attribute

    def forward(self, x: Tensor) -> SegmentationOutput:
        # x: (B, C, H, W, D) — 5D input
        return SegmentationOutput(logits=self.net(x))
```

## Implemented Models

| Model | Adapter | VRAM (train) | Notes |
|-------|---------|-------------|-------|
| DynUNet | `DynUNetAdapter` | ~3.5 GB | MONAI native, default |
| SegResNet | `SegResNetAdapter` | ~2 GB | MONAI native |
| SwinUNETR | `SwinUNETRAdapter` | ~4 GB | MONAI native |
| UNETR | `UNETRAdapter` | ~3 GB | MONAI native |
| AttentionUNet | `AttentionUNetAdapter` | ~2.5 GB | MONAI native |
| SAM3 Vanilla | `Sam3VanillaAdapter` | ~3.5 GB | Requires pretrained weights |
| SAM3 TopoLoRA | `Sam3TopoLoraAdapter` | >=16 GB | Cloud GPU required |
| SAM3 Hybrid | `Sam3HybridAdapter` | ~7.2 GB | Cloud GPU required |
| VesselFM | `VesselFMAdapter` | ~10 GB | HuggingFace weights |
| MambaVesselNet | `MambaVesselNetAdapter` | ~5-8 GB | Requires mamba-ssm CUDA |

## Key Constraints

- `self.config` + `self.net` REQUIRED on every adapter
- SAM3 ALWAYS requires real pretrained weights (ViT-32L, 648M params)
- SAM3 MUST use BF16 on Ampere+ (NEVER FP16 — overflow → NaN)
- Mamba requires CUDA-compiled mamba-ssm (INSTALL_MAMBA=1 Docker build arg)
- `_mamba_available()` and `_sam3_package_available()` in `model_builder.py`
- Config: `configs/method_capabilities.yaml` for model-loss compatibility

## Factory Function

```python
from minivess.adapters.model_builder import build_adapter

# Build any model by name — dispatches to correct adapter
model = build_adapter("dynunet", experiment_config)
model = build_adapter("sam3_vanilla", experiment_config)
```

See: `src/minivess/adapters/CLAUDE.md` for VRAM tables and SAM3 details.
