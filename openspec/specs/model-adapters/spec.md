# Model Adapters Specification

## Overview

All models implement the ModelAdapter ABC. The platform is model-agnostic —
any model that implements train/predict/export integrates automatically.

## Scenarios

### Scenario: ModelAdapter ABC enforces contract
- GIVEN a new model class
- WHEN it subclasses ModelAdapter
- THEN it MUST implement forward(B,C,H,W,D) → SegmentationOutput
- AND it MUST implement get_eval_roi_size() for sliding_window_inference

### Scenario: MONAI-native models use MONAI directly
- GIVEN a MONAI model (DynUNet, SegResNet, SwinUNETR)
- WHEN it is wrapped in a ModelAdapter
- THEN the adapter uses MONAI classes directly with zero custom code
- AND dimension order is (B,C,H,W,D) — depth LAST

### Scenario: External models adapt to MONAI conventions
- GIVEN an external model (SAM3)
- WHEN it is wrapped in a ModelAdapter
- THEN the adapter translates external conventions to MONAI
- AND the adapter NEVER asks MONAI to adapt to the external model

### Scenario: SAM3 requires SDPA attention
- GIVEN the SAM3 adapter is instantiated
- WHEN attention implementation is configured
- THEN attn_implementation='sdpa' is mandatory
- AND eager attention causes OOM on 8 GB GPUs

### Scenario: Models report VRAM requirements
- GIVEN a ModelAdapter subclass
- WHEN the model is built
- THEN sam3_vram_check.py verifies sufficient GPU VRAM (≥16 GB for SAM3)
- AND the check runs at build time, not at training time

## Requirements

- ModelAdapter ABC is the ONLY integration pattern
- MONAI dimension order: (B,C,H,W,D) — depth LAST, always
- If MONAI has it, use it directly — zero custom code
- SAM3 ≠ SAM2 (different architecture, different weights)
- SAM3 ALWAYS requires real pretrained weights (ViT-32L, 648M params)
