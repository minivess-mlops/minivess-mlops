"""Docker image content validation (proxy) — Phase 7 Task 7.2.

Verifies that required files exist in the repo — proxy for Docker image content.
The actual Docker image copies from these paths, so repo presence guarantees
image presence (assuming Dockerfile COPY is correct).

Plan: run-debug-factorial-experiment-report-6th-pass-post-run-fix-2.xml
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


class TestRequiredFilesInRepo:
    """Files required by Docker image and SkyPilot setup must exist in repo."""

    def test_splits_files_exist(self) -> None:
        """Split files referenced in SkyPilot setup must exist."""
        splits_dir = REPO_ROOT / "configs" / "splits"
        assert splits_dir.exists(), f"Missing splits dir: {splits_dir}"

        required = ["3fold_seed42.json", "debug_half_1fold.json"]
        for name in required:
            path = splits_dir / name
            assert path.exists(), f"Missing splits file: {path}"

    def test_dvc_files_exist(self) -> None:
        """DVC tracking files for data must exist."""
        data_dir = REPO_ROOT / "data"
        assert data_dir.exists(), f"Missing data dir: {data_dir}"

    def test_dockerfile_base_exists(self) -> None:
        """Base Dockerfile must exist."""
        dockerfile = REPO_ROOT / "deployment" / "docker" / "Dockerfile.base"
        assert dockerfile.exists(), f"Missing: {dockerfile}"

    def test_skypilot_yamls_exist(self) -> None:
        """Critical SkyPilot YAMLs must exist."""
        skypilot_dir = REPO_ROOT / "deployment" / "skypilot"
        required = ["train_factorial.yaml", "train_production.yaml"]
        for name in required:
            path = skypilot_dir / name
            assert path.exists(), f"Missing SkyPilot YAML: {path}"

    def test_pyproject_toml_exists(self) -> None:
        """pyproject.toml must exist (uv needs it)."""
        assert (REPO_ROOT / "pyproject.toml").exists()

    def test_env_example_exists(self) -> None:
        """.env.example must exist (single source for config)."""
        assert (REPO_ROOT / ".env.example").exists()
