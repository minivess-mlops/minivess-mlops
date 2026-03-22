"""Tests for DeepVess external test data pipeline.

P0 PUBLICATION BLOCKER: DeepVess is the only external test dataset for the
Nature Protocols platform paper. These tests verify the full pipeline from
registry config through analysis flow wiring to biostatistics split.

RED PHASE: All tests should FAIL until implementation is complete.
"""

from __future__ import annotations

import pytest

from minivess.data.external_datasets import (
    DVC_CONFIGS,
    EXTERNAL_DATASETS,
    discover_external_test_pairs,
    validate_external_config,
)


class TestDeepVessRegistryConfig:
    """Verify DeepVess registry entry is accurate."""

    def test_deepvess_exists_in_registry(self) -> None:
        assert "deepvess" in EXTERNAL_DATASETS

    def test_deepvess_n_volumes_is_correct(self) -> None:
        """n_volumes should reflect labeled volumes available for evaluation.

        DeepVess eCommons README (verified 2026-03-22):
        - 24 total TIFF volumes (6 per group × 4 groups)
        - 1 ground truth labeled pair (image + segmentation)
        - For our external test evaluation, we use the 1 labeled pair.
        """
        config = EXTERNAL_DATASETS["deepvess"]
        # The registry should have the VERIFIED count, not a confabulated one
        assert config.n_volumes == 1, (
            f"n_volumes should be 1 (1 labeled GT pair from eCommons). "
            f"Got {config.n_volumes}. See README: '6 samples for each of four groups' "
            f"but only 1 ground truth image+label pair for training/evaluation."
        )

    def test_deepvess_license_is_cc_by_4(self) -> None:
        """License should be CC-BY 4.0, not 'eCommons-educational'."""
        config = EXTERNAL_DATASETS["deepvess"]
        assert config.license == "CC-BY-4.0", (
            f"DeepVess license is CC-BY 4.0 (verified from eCommons README). "
            f"Got '{config.license}'"
        )

    def test_deepvess_license_verified_true(self) -> None:
        """License has been verified from eCommons README."""
        config = EXTERNAL_DATASETS["deepvess"]
        assert config.license_verified is True

    def test_deepvess_role_is_external_test(self) -> None:
        config = EXTERNAL_DATASETS["deepvess"]
        assert config.role == "external_test"

    def test_deepvess_modality_valid(self) -> None:
        config = EXTERNAL_DATASETS["deepvess"]
        errors = validate_external_config(config)
        assert not errors, f"Validation errors: {errors}"

    def test_deepvess_dvc_config_exists(self) -> None:
        assert "deepvess" in DVC_CONFIGS

    def test_deepvess_source_url_is_ecommons(self) -> None:
        config = EXTERNAL_DATASETS["deepvess"]
        assert "ecommons.cornell.edu" in config.source_url


class TestAnalysisFlowExternalDataLoaders:
    """Test that _build_dataloaders_from_config returns actual DataLoaders, not raw pairs."""

    def test_build_dataloaders_returns_dataloader_not_list(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The return value must be DataLoader objects, not list[dict]."""

        # Create synthetic test data
        data_dir = tmp_path / "external"  # type: ignore[operator]
        deepvess_dir = data_dir / "deepvess" / "images"
        labels_dir = data_dir / "deepvess" / "labels"
        deepvess_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        (deepvess_dir / "vol_001.tif").write_bytes(b"fake")
        (labels_dir / "vol_001.tif").write_bytes(b"fake")

        monkeypatch.setenv("EXTERNAL_DATA_DIR", str(data_dir))
        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")

        from minivess.orchestration.flows.analysis_flow import (
            _build_dataloaders_from_config,
        )

        result = _build_dataloaders_from_config({})

        # The result must have DataLoader objects, not raw pair dicts
        assert "deepvess" in result
        deepvess_data = result["deepvess"]
        assert "all" in deepvess_data

        # This is the critical check: "all" must be a DataLoader, not a list
        all_value = deepvess_data["all"]
        assert not isinstance(all_value, list), (
            f"_build_dataloaders_from_config returns raw pairs (list), not DataLoader. "
            f"Got type: {type(all_value)}. Fix analysis_flow.py line 2122."
        )


class TestBiostatisticsTestSplit:
    """Test that biostatistics flow processes test split for DeepVess."""

    def test_build_per_volume_data_supports_test_split(self) -> None:
        """The function must accept split='test' and process test/ prefix metrics."""
        import inspect

        from minivess.orchestration.flows.biostatistics_flow import (
            _build_per_volume_data,
        )

        sig = inspect.signature(_build_per_volume_data)
        assert "split" in sig.parameters, (
            "_build_per_volume_data must have a 'split' parameter"
        )
        # Default should be "trainval" for backward compat
        default = sig.parameters["split"].default
        assert default == "trainval", (
            f"Default split should be 'trainval', got {default}"
        )

    def test_flow_body_calls_test_split(self) -> None:
        """The flow body must call _build_per_volume_data with split='test'.

        This is the critical wiring gap: the function supports split='test'
        but the flow body never calls it that way.
        """
        import ast
        from pathlib import Path

        flow_file = Path("src/minivess/orchestration/flows/biostatistics_flow.py")
        tree = ast.parse(flow_file.read_text(encoding="utf-8"))

        # Find all calls to _build_per_volume_data
        calls_with_test_split: list[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                func_name = None
                if isinstance(func, ast.Name):
                    func_name = func.id
                elif isinstance(func, ast.Attribute):
                    func_name = func.attr
                if func_name == "_build_per_volume_data":
                    for kw in node.keywords:
                        if (
                            kw.arg == "split"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value == "test"
                        ):
                            calls_with_test_split.append(node.lineno)

        assert calls_with_test_split, (
            "biostatistics_flow.py never calls _build_per_volume_data(split='test'). "
            "DeepVess test metrics are excluded from all statistical analysis. "
            "Add a second call with split='test' after line 436."
        )


class TestDeepVessDiscovery:
    """Test that discover_external_test_pairs works for DeepVess layout."""

    def test_discover_raises_on_missing_directory(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Must raise FileNotFoundError, not return empty (Rule 25: loud failures)."""
        nonexistent = tmp_path / "nonexistent"  # type: ignore[operator]
        with pytest.raises(FileNotFoundError):
            discover_external_test_pairs(nonexistent, "deepvess")

    def test_discover_finds_tiff_pairs(self, tmp_path: pytest.TempPathFactory) -> None:
        """DeepVess data is TIFF format. Discovery must handle .tif/.tiff."""
        data_dir = tmp_path / "deepvess"  # type: ignore[operator]
        images_dir = data_dir / "images"
        labels_dir = data_dir / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        # Create synthetic TIFF pairs
        (images_dir / "vol_001.tif").write_bytes(b"fake tiff")
        (labels_dir / "vol_001.tif").write_bytes(b"fake tiff label")

        pairs = discover_external_test_pairs(data_dir, "deepvess")
        assert len(pairs) == 1
        assert "image" in pairs[0]
        assert "label" in pairs[0]
        assert pairs[0]["image"].endswith("vol_001.tif")

    def test_discover_finds_nifti_pairs(self, tmp_path: pytest.TempPathFactory) -> None:
        """Also support NIfTI in case data is converted."""
        data_dir = tmp_path / "deepvess_nifti"  # type: ignore[operator]
        images_dir = data_dir / "images"
        labels_dir = data_dir / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        (images_dir / "vol_001.nii.gz").write_bytes(b"fake nifti")
        (labels_dir / "vol_001.nii.gz").write_bytes(b"fake nifti label")

        pairs = discover_external_test_pairs(data_dir, "deepvess")
        assert len(pairs) == 1

    def test_discover_returns_empty_on_no_labels_dir(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """If images/ exists but labels/ doesn't, return empty with warning."""
        data_dir = tmp_path / "deepvess_no_labels"  # type: ignore[operator]
        images_dir = data_dir / "images"
        images_dir.mkdir(parents=True)

        pairs = discover_external_test_pairs(data_dir, "deepvess")
        assert pairs == []
