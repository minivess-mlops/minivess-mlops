"""Tests for combined topology experiment config (T14 — #241) and discussion notes (T17 — #244)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _load_config() -> dict[str, Any]:
    from minivess.config.compose import compose_experiment_config

    return compose_experiment_config(experiment_name="dynunet_topology_all_approaches")


class TestCombinedTopologyConfig:
    """Tests for combined all-approaches experiment config (T14)."""

    def test_combined_config_loads(self) -> None:
        """YAML config loads without error."""
        config = _load_config()
        assert config["experiment_name"] == "dynunet_topology_all_approaches_v1"

    def test_combined_config_six_conditions(self) -> None:
        """All 6 conditions present."""
        config = _load_config()
        conditions = config["conditions"]
        assert len(conditions) == 6
        names = [c["name"] for c in conditions]
        assert "baseline" in names
        assert "d2c_only" in names
        assert "multitask" in names
        assert "multitask_d2c" in names
        assert "tffm" in names
        assert "full_pipeline" in names

    def test_combined_config_valid_losses(self) -> None:
        """All loss names recognized."""
        config = _load_config()
        assert config["loss"] == "cbdice_cldice"

    def test_combined_config_3_folds(self) -> None:
        """Each condition has 3 folds."""
        config = _load_config()
        assert config["folds"] == 3

    def test_combined_config_no_sam_references(self) -> None:
        """Zero SAM model families."""
        config = _load_config()
        config_str = yaml.dump(config)
        assert "sam3" not in config_str.lower()
        assert "sam_lora" not in config_str.lower()


class TestDiscussionNotes:
    """Tests for paper discussion section notes (T17 — #244)."""

    def test_discussion_notes_exists(self) -> None:
        """File exists and has content."""
        path = Path(
            "docs/planning/v0-2_archive/original_docs/topology-approaches-discussion-notes.md"
        )
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert len(content) > 100

    def test_discussion_notes_has_sections(self) -> None:
        """Implemented, deferred, and 3D rationale sections present."""
        path = Path(
            "docs/planning/v0-2_archive/original_docs/topology-approaches-discussion-notes.md"
        )
        content = path.read_text(encoding="utf-8")
        assert "implemented" in content.lower() or "approach" in content.lower()
        assert "deferred" in content.lower() or "future" in content.lower()
        assert "3d" in content.lower() or "3D" in content

    def test_discussion_notes_no_sam_implementation(self) -> None:
        """No SAM implementation details."""
        path = Path(
            "docs/planning/v0-2_archive/original_docs/topology-approaches-discussion-notes.md"
        )
        content = path.read_text(encoding="utf-8")
        # Should mention SAM3 as parallel/future, not as implemented
        assert (
            "sam3" not in content.lower()
            or "parallel" in content.lower()
            or "future" in content.lower()
        )
