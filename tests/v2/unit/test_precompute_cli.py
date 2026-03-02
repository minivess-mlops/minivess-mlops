"""Tests for precompute_targets.py CLI script (T6 — topology real-data plan)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path


class TestDiscoverVolumes:
    """Test volume discovery from data directory."""

    def test_discovers_labels_from_minivess_dir(self, tmp_path: Path) -> None:
        """Finds all .nii.gz labels in labelsTr/ subdirectory."""
        import sys

        sys.path.insert(0, "scripts")
        from precompute_targets import discover_volumes

        labels_dir = tmp_path / "labelsTr"
        labels_dir.mkdir()
        for i in range(3):
            (labels_dir / f"mv{i + 1:02d}.nii.gz").touch()

        volumes = discover_volumes(tmp_path)
        assert len(volumes) == 3
        assert all("label" in v and "volume_id" in v for v in volumes)
        volume_ids = [v["volume_id"] for v in volumes]
        assert "mv01" in volume_ids
        assert "mv02" in volume_ids
        assert "mv03" in volume_ids

    def test_returns_empty_when_no_labels(self, tmp_path: Path) -> None:
        """Returns empty list when labelsTr/ is missing."""
        import sys

        sys.path.insert(0, "scripts")
        from precompute_targets import discover_volumes

        volumes = discover_volumes(tmp_path)
        assert volumes == []


class TestResolveTargetConfigs:
    """Test mapping from YAML config to AuxTargetConfig objects."""

    def test_resolves_sdf_config(self) -> None:
        """Resolves 'compute_sdf_from_mask' to the actual function."""
        import sys

        sys.path.insert(0, "scripts")
        from precompute_targets import resolve_target_configs

        yaml_configs: list[dict[str, Any]] = [
            {"name": "sdf", "suffix": "sdf", "compute_fn": "compute_sdf_from_mask"},
        ]
        configs = resolve_target_configs(yaml_configs)
        assert len(configs) == 1
        assert configs[0].name == "sdf"
        assert callable(configs[0].compute_fn)

    def test_resolves_centreline_config(self) -> None:
        """Resolves 'compute_centreline_distance_map' to the actual function."""
        import sys

        sys.path.insert(0, "scripts")
        from precompute_targets import resolve_target_configs

        yaml_configs: list[dict[str, Any]] = [
            {
                "name": "centerline_dist",
                "suffix": "centerline_dist",
                "compute_fn": "compute_centreline_distance_map",
            },
        ]
        configs = resolve_target_configs(yaml_configs)
        assert len(configs) == 1
        assert configs[0].name == "centerline_dist"


class TestMainEntrypoint:
    """Test that the CLI main() function drives precomputation."""

    def test_main_calls_precompute(self, tmp_path: Path) -> None:
        """main() loads config, discovers volumes, calls precompute."""
        import sys

        sys.path.insert(0, "scripts")
        # Create a minimal config pointing to tmp_path
        import yaml
        from precompute_targets import main

        config = {
            "experiment_name": "test",
            "conditions": [],
            "data_dir": str(tmp_path),
            "precompute_targets": [
                {"name": "sdf", "suffix": "sdf", "compute_fn": "compute_sdf_from_mask"},
            ],
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config), encoding="utf-8")

        # No labelsTr/ → no volumes → precompute should be called with empty list
        with patch(
            "precompute_targets.precompute_auxiliary_targets"
        ) as mock_precompute:
            mock_precompute.return_value = {"computed": 0, "skipped": 0}
            main(["--config", str(config_path)])
            mock_precompute.assert_called_once()
