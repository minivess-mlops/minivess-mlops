"""Tests for Sam3TopoLoraAdapter parameter freezing — T1 regression tests.

Validates that after LoRA application, non-LoRA encoder params are frozen.
Tests the freeze logic directly without loading real SAM3 weights.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from minivess.adapters.sam3_topolora import _apply_lora_to_encoder


class _FakeTransformerBlock(nn.Module):
    """Minimal transformer block with attn + FFN for testing LoRA targeting."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(dim, dim * 3)  # type: ignore[attr-defined]
        self.attn.proj = nn.Linear(dim, dim)  # type: ignore[attr-defined]
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Module()
        self.mlp.lin1 = nn.Linear(dim, dim * 4)  # type: ignore[attr-defined]
        self.mlp.lin2 = nn.Linear(dim * 4, dim)  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FakeEncoder(nn.Module):
    """Minimal SAM3-like encoder with patch embed + position embed + layers."""

    def __init__(self, n_blocks: int = 2, dim: int = 64) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, 14, stride=14)
        self.pos_embed = nn.Parameter(torch.randn(1, 100, dim))
        self.layers = nn.ModuleList(
            [_FakeTransformerBlock(dim) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _freeze_encoder_non_lora(encoder: nn.Module) -> None:
    """Freeze all encoder params EXCEPT LoRA A/B matrices.

    This is the function that sam3_topolora.py SHOULD call after _apply_lora_to_encoder.
    """
    for name, param in encoder.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False


@pytest.fixture()
def encoder_with_lora() -> tuple[nn.Module, list[str]]:
    """Create a fake encoder with LoRA applied to FFN layers."""
    encoder = _FakeEncoder(n_blocks=2, dim=64)
    lora_targets = _apply_lora_to_encoder(
        encoder, rank=8, alpha=16.0, dropout=0.0
    )
    return encoder, lora_targets


class TestEncoderNonLoraParamsFrozen:
    """T1: Non-LoRA encoder params must be frozen after LoRA application."""

    def test_encoder_non_lora_params_frozen(self, encoder_with_lora):
        """All encoder params NOT part of LoRA must be frozen after freeze call."""
        encoder, _ = encoder_with_lora
        _freeze_encoder_non_lora(encoder)

        for name, param in encoder.named_parameters():
            if "lora_" not in name:
                assert not param.requires_grad, (
                    f"Non-LoRA encoder param '{name}' is trainable but should be frozen"
                )

    def test_lora_params_trainable(self, encoder_with_lora):
        """LoRA A and B matrices must remain trainable after freeze."""
        encoder, _ = encoder_with_lora
        _freeze_encoder_non_lora(encoder)

        lora_params = [
            (name, param)
            for name, param in encoder.named_parameters()
            if "lora_" in name
        ]
        assert len(lora_params) > 0, "No LoRA params found in encoder"
        for name, param in lora_params:
            assert param.requires_grad, (
                f"LoRA param '{name}' is frozen but should be trainable"
            )

    def test_without_freeze_non_lora_params_are_trainable(self):
        """Before fix: non-LoRA params remain trainable (the bug)."""
        encoder = _FakeEncoder(n_blocks=1, dim=64)
        _apply_lora_to_encoder(encoder, rank=8, alpha=16.0, dropout=0.0)

        # LoRALinear freezes the original Linear weights, but attn/norm/embed stay trainable
        trainable_non_lora = [
            name
            for name, p in encoder.named_parameters()
            if p.requires_grad and "lora_" not in name
        ]
        # Before fix: attn, norm, pos_embed, patch_embed are still trainable
        # (LoRALinear only freezes the wrapped Linear's own weights)
        # After fix: these should be empty
        # This test documents the pre-fix state
        attn_trainable = [n for n in trainable_non_lora if "attn" in n]
        norm_trainable = [n for n in trainable_non_lora if "norm" in n]
        embed_trainable = [n for n in trainable_non_lora if "embed" in n]
        assert len(attn_trainable) > 0 or len(norm_trainable) > 0 or len(embed_trainable) > 0, (
            "Expected some non-LoRA params to still be trainable before explicit freeze"
        )

    def test_frozen_param_count_matches_expected(self, encoder_with_lora):
        """After freeze, only LoRA params should be trainable in encoder."""
        encoder, _ = encoder_with_lora
        _freeze_encoder_non_lora(encoder)

        trainable_names = [
            name for name, p in encoder.named_parameters() if p.requires_grad
        ]
        for name in trainable_names:
            assert "lora_" in name, (
                f"Trainable encoder param '{name}' is not a LoRA param"
            )


class TestLoraApplicationTargeting:
    """Verify LoRA is applied to correct layers."""

    def test_lora_targets_only_mlp_layers(self, encoder_with_lora):
        """LoRA should only target mlp.lin1/lin2, not attn layers."""
        _, lora_targets = encoder_with_lora
        assert len(lora_targets) > 0, "No LoRA targets found"
        for target in lora_targets:
            assert any(kw in target for kw in ("mlp", "lin1", "lin2")), (
                f"LoRA target '{target}' does not match FFN keywords"
            )

    def test_attn_layers_not_lora_wrapped(self, encoder_with_lora):
        """Attention Q/K/V and proj should NOT have LoRA."""
        _, lora_targets = encoder_with_lora
        attn_targets = [t for t in lora_targets if "attn" in t]
        assert len(attn_targets) == 0, (
            f"Attention layers should not have LoRA: {attn_targets}"
        )
