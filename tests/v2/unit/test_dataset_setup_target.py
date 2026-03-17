"""Tests for dataset-setup Makefile target (#753)."""

from __future__ import annotations

from pathlib import Path


class TestDatasetSetupTarget:
    """Verify dataset-setup Makefile target exists and is correct."""

    def test_makefile_has_dataset_setup_target(self) -> None:
        """Makefile should have a dataset-setup target."""
        content = Path("Makefile").read_text(encoding="utf-8")
        assert "dataset-setup:" in content

    def test_dataset_setup_has_three_steps(self) -> None:
        """Target should have 3 steps: verify, DVC, upload."""
        content = Path("Makefile").read_text(encoding="utf-8")
        assert "Step 1/3" in content
        assert "Step 2/3" in content
        assert "Step 3/3" in content

    def test_dataset_setup_uses_dvc(self) -> None:
        """Target must use DVC for versioning."""
        content = Path("Makefile").read_text(encoding="utf-8")
        assert "dvc add" in content

    def test_dataset_setup_uses_sky_rsync(self) -> None:
        """Target must use sky rsync for RunPod upload."""
        content = Path("Makefile").read_text(encoding="utf-8")
        # Find the dataset-setup section
        idx = content.find("dataset-setup:")
        section = content[idx : idx + 1200]
        assert "sky rsync up" in section

    def test_dataset_setup_in_help(self) -> None:
        """Target should appear in help output."""
        content = Path("Makefile").read_text(encoding="utf-8")
        assert (
            "dataset-setup" in content.split("help:")[0] or "dataset-setup" in content
        )
