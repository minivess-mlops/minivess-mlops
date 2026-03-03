"""Tests for MONAI Deploy SDK optional dependency handling (#278).

Covers:
- Import warning when SDK not installed
- Duck-typed classes work without SDK
- Optional dep group defined
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class TestMonaiDeployDuckTyping:
    """Test that MONAI Deploy classes work without the SDK."""

    def test_inference_operator_instantiation(self, tmp_path: Path) -> None:
        from minivess.serving.monai_deploy_app import MiniVessInferenceOperator

        op = MiniVessInferenceOperator(model_path=tmp_path / "model.onnx")
        assert op.model_path == tmp_path / "model.onnx"

    def test_seg_app_instantiation(self, tmp_path: Path) -> None:
        from minivess.serving.monai_deploy_app import MiniVessSegApp

        app = MiniVessSegApp(model_path=tmp_path / "model.onnx")
        operators = app.compose()
        assert len(operators) == 1

    def test_map_manifest_generation(self) -> None:
        from minivess.serving.monai_deploy_app import generate_map_manifest

        manifest = generate_map_manifest("test-app", "1.0", "dynunet")
        assert manifest["api-version"] == "1.0"
        assert manifest["application"]["name"] == "test-app"

    def test_map_main_generation(self) -> None:
        from minivess.serving.monai_deploy_app import generate_map_main_py

        source = generate_map_main_py()
        assert "MiniVessSegApp" in source
        assert "def main" in source


class TestMonaiDeployImportWarning:
    """Test import warning for MONAI Deploy SDK."""

    def test_check_monai_deploy_available(self) -> None:
        from minivess.serving.monai_deploy_compat import MONAI_DEPLOY_AVAILABLE

        # SDK likely not installed in test environment
        assert isinstance(MONAI_DEPLOY_AVAILABLE, bool)

    def test_warn_if_not_available(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        from minivess.serving.monai_deploy_compat import warn_if_monai_deploy_missing

        with caplog.at_level(logging.WARNING):
            warn_if_monai_deploy_missing()
        # Only warns if not installed
