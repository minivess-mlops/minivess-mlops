"""Tests for sam3_topolora adapter construction — LoRA application logic.

Catches Glitch #9: _apply_lora_to_encoder must only wrap nn.Linear, NOT nn.Conv2d.
Also tests LoRA zero-init equivalence (TorchTune best practice).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from minivess.adapters.sam3_topolora import LoRALinear, _apply_lora_to_encoder


@pytest.mark.model_construction
class TestApplyLoraToEncoder:
    """_apply_lora_to_encoder must only target nn.Linear layers."""

    def test_lora_only_targets_ffn_linear(self) -> None:
        """LoRA targets FFN Linear layers only, not Conv2d or attention (paper fidelity)."""

        class MockEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mlp = nn.Module()
                self.mlp.lin1 = nn.Linear(64, 256)  # FFN — should get LoRA
                self.mlp.lin2 = nn.Linear(256, 64)  # FFN — should get LoRA
                self.attn_qkv = nn.Linear(64, 192)  # attention — should NOT
                self.conv = nn.Conv2d(64, 64, 3)  # Conv2d — should NOT

        encoder = MockEncoder()
        targets = _apply_lora_to_encoder(encoder, rank=8, alpha=16.0, dropout=0.0)
        assert "mlp.lin1" in targets, f"Expected 'mlp.lin1' in targets, got {targets}"
        assert "mlp.lin2" in targets, f"Expected 'mlp.lin2' in targets, got {targets}"
        assert "attn_qkv" not in targets, (
            f"Attention should NOT be targeted, got {targets}"
        )
        assert "conv" not in targets, f"Conv2d should NOT be targeted, got {targets}"

    def test_lora_skips_small_linear_layers(self) -> None:
        """LoRA skips Linear layers with in_features < rank."""

        class TinyEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mlp = nn.Module()
                self.mlp.lin1 = nn.Linear(4, 4)  # too small (4 < rank=8)
                self.mlp.lin2 = nn.Linear(64, 64)  # big enough

        encoder = TinyEncoder()
        targets = _apply_lora_to_encoder(encoder, rank=8, alpha=16.0, dropout=0.0)
        assert "mlp.lin1" not in targets, "Tiny layer (4 < rank=8) should be skipped"
        assert "mlp.lin2" in targets, "Big enough layer should be targeted"

    def test_lora_replaces_with_lora_linear(self) -> None:
        """Targeted modules are replaced with LoRALinear instances."""

        class SimpleEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mlp = nn.Module()
                self.mlp.fc1 = nn.Linear(64, 32)

        encoder = SimpleEncoder()
        _apply_lora_to_encoder(encoder, rank=8, alpha=16.0, dropout=0.0)
        assert isinstance(encoder.mlp.fc1, LoRALinear), (
            f"Expected LoRALinear, got {type(encoder.mlp.fc1).__name__}"
        )

    def test_nested_modules_targeted(self) -> None:
        """LoRA finds FFN Linear layers inside nested modules."""

        class NestedEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                )
                self.attn = nn.Linear(64, 64)  # attention — should NOT
                self.conv = nn.Conv2d(64, 64, 3)

        encoder = NestedEncoder()
        targets = _apply_lora_to_encoder(encoder, rank=8, alpha=16.0, dropout=0.0)
        # Should find the two Linear layers inside mlp (FFN keyword match)
        assert len(targets) == 2, f"Expected 2 targets, got {len(targets)}: {targets}"
        assert all("mlp" in t for t in targets), f"Non-mlp targeted: {targets}"


@pytest.mark.model_construction
class TestLoRAZeroInitEquivalence:
    """At initialization, LoRA output MUST equal base model output.

    This is because lora_B is initialized to zeros, so the LoRA contribution
    is zero at init. Violation indicates a bug in the initialization.
    Source: TorchTune LoRA test patterns.
    """

    def test_zero_init_output_matches_base(self) -> None:
        """LoRA(x) == base(x) at initialization."""
        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=8, alpha=16.0, dropout=0.0)
        x = torch.randn(2, 64)
        with torch.no_grad():
            base_out = original(x)
            lora_out = lora(x)
        torch.testing.assert_close(base_out, lora_out, rtol=1e-5, atol=1e-5)

    def test_zero_init_lora_b_is_zero(self) -> None:
        """lora_B must be initialized to zeros."""
        linear = nn.Linear(64, 32)
        lora = LoRALinear(linear, rank=8, alpha=16.0)
        assert torch.all(lora.lora_B == 0), "lora_B must be zero-initialized"
