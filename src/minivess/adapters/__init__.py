"""Model adapters -- ModelAdapter protocol and concrete implementations."""

from __future__ import annotations

from minivess.adapters.base import ModelAdapter, SegmentationOutput
from minivess.adapters.lora import LoraModelAdapter
from minivess.adapters.segresnet import SegResNetAdapter
from minivess.adapters.swinunetr import SwinUNETRAdapter
from minivess.adapters.vista3d import Vista3dAdapter

__all__ = [
    "LoraModelAdapter",
    "ModelAdapter",
    "SegmentationOutput",
    "SegResNetAdapter",
    "SwinUNETRAdapter",
    "Vista3dAdapter",
]
