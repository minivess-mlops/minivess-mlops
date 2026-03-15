"""Tests for LoRALinear layer in sam3_topolora adapter.

T0.1: Verify LoRA only supports nn.Linear, rejects Conv2d,
and handles FP16→FP32 dtype casting correctly.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from minivess.adapters.sam3_topolora import LoRALinear


class TestLoRALinearTypeValidation:
    """LoRALinear should only support nn.Linear layers."""

    def test_accepts_linear(self) -> None:
        """LoRALinear wraps nn.Linear without error."""
        linear = nn.Linear(64, 32)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        assert lora.rank == 4

    def test_rejects_conv2d(self) -> None:
        """LoRALinear raises TypeError for Conv2d input."""
        conv = nn.Conv2d(3, 16, kernel_size=3)
        with pytest.raises(TypeError, match="only supports nn.Linear"):
            LoRALinear(conv, rank=4)

    def test_rejects_other_modules(self) -> None:
        """LoRALinear raises TypeError for non-Linear modules."""
        bn = nn.BatchNorm1d(64)
        with pytest.raises(TypeError, match="only supports nn.Linear"):
            LoRALinear(bn, rank=4)


class TestLoRALinearForward:
    """LoRALinear forward pass correctness."""

    def test_forward_shape_preserved(self) -> None:
        """Output shape matches original layer output."""
        linear = nn.Linear(64, 32)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        x = torch.randn(2, 64)
        out = lora(x)
        assert out.shape == (2, 32)

    def test_forward_fp16_input(self) -> None:
        """LoRA handles FP16 input with FP32 LoRA params."""
        linear = nn.Linear(64, 32)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        # Simulate SAM3 FP16 encoder output
        x = torch.randn(2, 64, dtype=torch.float16)
        linear.weight.data = linear.weight.data.half()
        if linear.bias is not None:
            linear.bias.data = linear.bias.data.half()
        out = lora(x)
        # Output should match input dtype
        assert out.dtype == torch.float16

    def test_original_frozen(self) -> None:
        """Original layer parameters are frozen after wrapping."""
        linear = nn.Linear(64, 32)
        lora = LoRALinear(linear, rank=4)
        for param in lora.original.parameters():
            assert not param.requires_grad

    def test_lora_params_trainable(self) -> None:
        """LoRA A and B matrices are trainable."""
        linear = nn.Linear(64, 32)
        lora = LoRALinear(linear, rank=4)
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_weight_property_delegates(self) -> None:
        """weight property returns original layer's weight."""
        linear = nn.Linear(64, 32)
        lora = LoRALinear(linear, rank=4)
        assert lora.weight is linear.weight
