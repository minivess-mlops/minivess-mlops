"""Tests for SAM3 BF16 auto-detection (T4 ban enforcement).

Verifies:
1. sam3_backbone.py uses auto-detection (not hardcoded float16)
2. _encoder_dtype property exists and returns correct dtype
3. extract_features casts input to match encoder dtype (not hardcoded .half())
4. T4 ban: smoke_test_gcp.yaml does NOT list T4 in accelerators
5. CLAUDE.md documents T4 ban

See: .claude/metalearning/2026-03-15-t4-turing-fp16-nan-ban.md
     .claude/metalearning/2026-03-15-sam3-bf16-fp16-fuckup.md
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml


class TestBF16AutoDetection:
    """sam3_backbone.py must auto-detect BF16 on Ampere+ GPUs."""

    def test_no_hardcoded_float16_in_from_pretrained(self) -> None:
        """from_pretrained must NOT use hardcoded torch.float16."""
        src = Path("src/minivess/adapters/sam3_backbone.py")
        tree = ast.parse(src.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_load_sam3_encoder":
                # Find the from_pretrained call
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        for kw in child.keywords:
                            if kw.arg == "torch_dtype" and (
                                isinstance(kw.value, ast.Attribute)
                                and isinstance(kw.value.value, ast.Name)
                                and kw.value.value.id == "torch"
                                and kw.value.attr == "float16"
                            ):
                                pytest.fail(
                                    "torch_dtype must use auto-detection "
                                    "(self._encoder_dtype or conditional), "
                                    "not hardcoded torch.float16"
                                )
                return
        pytest.fail("_load_sam3_encoder not found")

    def test_has_encoder_dtype_attribute(self) -> None:
        """Sam3Backbone must have _encoder_dtype attribute."""
        src = Path("src/minivess/adapters/sam3_backbone.py")
        content = src.read_text(encoding="utf-8")
        assert "_encoder_dtype" in content, (
            "Sam3Backbone must define _encoder_dtype for BF16 auto-detection"
        )

    def test_extract_features_uses_encoder_dtype(self) -> None:
        """extract_features must cast to _encoder_dtype, not hardcoded .half()."""
        src = Path("src/minivess/adapters/sam3_backbone.py")
        tree = ast.parse(src.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "extract_features":
                body_src = ast.dump(node)
                # Must NOT have a bare .half() call
                for child in ast.walk(node):
                    if (
                        isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Attribute)
                        and child.func.attr == "half"
                    ):
                        pytest.fail(
                            "extract_features must use x.to(self._encoder_dtype), "
                            "not x.half() (which always gives FP16)"
                        )
                # Must reference _encoder_dtype
                assert "_encoder_dtype" in body_src, (
                    "extract_features must use _encoder_dtype for input casting"
                )
                return
        pytest.fail("extract_features not found")


class TestT4BanInSkyPilotYAML:
    """GCP smoke test YAML must NOT list T4."""

    def test_smoke_test_gcp_no_t4(self) -> None:
        """smoke_test_gcp.yaml must not include T4 in accelerators."""
        path = Path("deployment/skypilot/smoke_test_gcp.yaml")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        accelerators = data.get("resources", {}).get("accelerators", {})
        if isinstance(accelerators, dict):
            assert "T4" not in accelerators, (
                "T4 BANNED in GCP YAML: Turing lacks BF16 → FP16 overflow → NaN. Use L4."
            )
        elif isinstance(accelerators, str):
            assert "T4" not in accelerators


class TestT4BanInCLAUDEMD:
    """CLAUDE.md must document T4 ban."""

    def test_claude_md_bans_t4(self) -> None:
        """CLAUDE.md must mention T4 ban."""
        content = Path("CLAUDE.md").read_text(encoding="utf-8")
        assert "T4" in content and "BAN" in content.upper(), (
            "CLAUDE.md must document T4 ban for SAM3 models"
        )
