"""Cross-file GCS bucket name consistency tests.

Validates that GCS bucket names in preflight_gcp.py match .dvc/config,
SkyPilot YAMLs, and cloud config files. Drift causes DVC pull failures
and checkpoint sync errors.

Source of truth: scripts/preflight_gcp.py::GCS_BUCKET, CHECKPOINT_BUCKET
"""

from __future__ import annotations

import ast
import configparser
from pathlib import Path

import pytest
import yaml

PREFLIGHT_PATH = Path("scripts/preflight_gcp.py")
DVC_CONFIG = Path(".dvc/config")
SKYPILOT_DIR = Path("deployment/skypilot")


def _extract_constant(name: str) -> str:
    """Extract a string constant from preflight_gcp.py via AST."""
    source = PREFLIGHT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    if isinstance(node.value, ast.Constant):
                        return str(node.value.value)
    msg = f"{name} constant not found in preflight_gcp.py"
    raise ValueError(msg)


def _strip_gs_prefix(bucket_url: str) -> str:
    """Strip gs:// prefix for comparison."""
    return bucket_url.removeprefix("gs://")


class TestGCSBucketConsistency:
    """GCS bucket names must be consistent across all config files."""

    def test_dvc_bucket_source_constant(self) -> None:
        """preflight_gcp.py must define GCS_BUCKET with expected name."""
        gcs_bucket = _extract_constant("GCS_BUCKET")
        assert "minivess-mlops-dvc-data" in gcs_bucket

    def test_checkpoint_bucket_source_constant(self) -> None:
        """preflight_gcp.py must define CHECKPOINT_BUCKET."""
        ckpt_bucket = _extract_constant("CHECKPOINT_BUCKET")
        assert "minivess-mlops-checkpoints" in ckpt_bucket

    def test_dvc_config_matches_preflight(self) -> None:
        """DVC config remote URL must match GCS_BUCKET."""
        gcs_bucket = _extract_constant("GCS_BUCKET")
        gcs_bucket_name = _strip_gs_prefix(gcs_bucket)

        # Parse .dvc/config (INI-like format)
        config = configparser.ConfigParser()
        config.read(str(DVC_CONFIG), encoding="utf-8")

        # Find the GCS remote
        found_match = False
        for section in config.sections():
            if "remote" in section:
                url = config.get(section, "url", fallback="")
                if "gs://" in url:
                    url_name = _strip_gs_prefix(url)
                    assert url_name == gcs_bucket_name, (
                        f".dvc/config [{section}] url '{url}' does not match "
                        f"GCS_BUCKET '{gcs_bucket}'"
                    )
                    found_match = True

        assert found_match, "No GCS remote found in .dvc/config"

    @pytest.mark.parametrize(
        "yaml_path",
        [
            SKYPILOT_DIR / "train_factorial.yaml",
            SKYPILOT_DIR / "train_production.yaml",
            SKYPILOT_DIR / "train_hpo.yaml",
        ],
        ids=lambda p: p.name,
    )
    def test_skypilot_checkpoint_mount_matches(self, yaml_path: Path) -> None:
        """SkyPilot file_mounts for checkpoints must use CHECKPOINT_BUCKET."""
        ckpt_bucket = _extract_constant("CHECKPOINT_BUCKET")
        ckpt_bucket_name = _strip_gs_prefix(ckpt_bucket)

        with yaml_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)

        file_mounts = config.get("file_mounts", {})
        if not file_mounts:
            pytest.skip(f"{yaml_path.name} has no file_mounts")

        # Check any mount that references the checkpoint bucket
        for mount_point, mount_config in file_mounts.items():
            if isinstance(mount_config, dict):
                source = mount_config.get("source", "")
            elif isinstance(mount_config, str):
                source = mount_config
            else:
                continue

            if "checkpoint" in source.lower() or "checkpoint" in mount_point.lower():
                source_name = _strip_gs_prefix(source)
                assert ckpt_bucket_name in source_name, (
                    f"{yaml_path.name}: checkpoint mount source '{source}' "
                    f"does not match CHECKPOINT_BUCKET '{ckpt_bucket}'"
                )
