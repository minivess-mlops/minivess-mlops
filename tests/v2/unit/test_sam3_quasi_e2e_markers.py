"""Tests that SAM3 quasi-e2e combos are marked slow + model_loading (#596).

SAM3 model loading (ViT-32L, 848M params, HuggingFace download) takes >60 min
on dev machines. These combos must be marked so they're excluded from staging/prod
tiers and only run on GPU instances or when explicitly selected.
"""

from __future__ import annotations

import ast
from pathlib import Path

from minivess.testing.quasi_e2e_runner import _SAM3_MODELS

CONFTEST_PATH = Path("tests/v2/quasi_e2e/conftest.py")


class TestSam3QuasiE2eMarkers:
    """Verify SAM3 quasi-e2e tests get appropriate markers."""

    def test_conftest_adds_slow_marker_to_sam3(self) -> None:
        """pytest_collection_modifyitems must add 'slow' marker to SAM3 combos."""
        source = CONFTEST_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Find the pytest_collection_modifyitems function
        func_found = False
        marks_slow = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "pytest_collection_modifyitems"
            ):
                func_found = True
                # Walk the function body for pytest.mark.slow references
                func_source = ast.dump(node)
                if "slow" in func_source:
                    marks_slow = True
                break

        assert func_found, "pytest_collection_modifyitems not found in conftest"
        assert marks_slow, (
            "pytest_collection_modifyitems does not add 'slow' marker to SAM3 combos"
        )

    def test_conftest_adds_model_loading_marker_to_sam3(self) -> None:
        """pytest_collection_modifyitems must add 'model_loading' marker to SAM3 combos."""
        source = CONFTEST_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)

        func_found = False
        marks_model_loading = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "pytest_collection_modifyitems"
            ):
                func_found = True
                func_source = ast.dump(node)
                if "model_loading" in func_source:
                    marks_model_loading = True
                break

        assert func_found, "pytest_collection_modifyitems not found in conftest"
        assert marks_model_loading, (
            "pytest_collection_modifyitems does not add 'model_loading' marker to SAM3 combos"
        )

    def test_conftest_checks_all_sam3_models(self) -> None:
        """The conftest must reference _SAM3_MODELS or check all three SAM3 model names."""
        source = CONFTEST_PATH.read_text(encoding="utf-8")

        # Either imports _SAM3_MODELS or checks all three model names
        uses_frozenset = "_SAM3_MODELS" in source
        checks_all_names = all(name in source for name in _SAM3_MODELS)

        assert uses_frozenset or checks_all_names, (
            f"conftest must check all SAM3 models: {sorted(_SAM3_MODELS)}. "
            "Import _SAM3_MODELS from quasi_e2e_runner or check each name."
        )

    def test_sam3_models_frozenset_has_expected_members(self) -> None:
        """_SAM3_MODELS must contain exactly the known SAM3 variants."""
        expected = {"sam3_vanilla", "sam3_topolora", "sam3_hybrid"}
        assert expected == _SAM3_MODELS, (
            f"_SAM3_MODELS mismatch: got {_SAM3_MODELS}, expected {expected}"
        )
