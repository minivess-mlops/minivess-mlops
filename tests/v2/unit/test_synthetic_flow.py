"""Tests for synthetic generation Prefect flow (T-D6)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest  # noqa: TC002 — used at runtime as fixture decorator

if TYPE_CHECKING:
    from pathlib import Path


class TestSyntheticGenerationFlow:
    """Test synthetic volume generation flow tasks."""

    def test_generate_volumes_task(self, tmp_path: Path) -> None:
        """Task should generate (image, mask) pairs for a method."""
        from minivess.orchestration.flows.synthetic_generation_flow import (
            generate_volumes_task,
        )

        result = generate_volumes_task.fn(
            method="debug",
            n_volumes=2,
            output_dir=str(tmp_path / "synthetic"),
        )
        assert result["n_volumes"] == 2
        assert result["method"] == "debug"

    def test_generate_volumes_saves_files(self, tmp_path: Path) -> None:
        """Generated volumes should be saved as .npy files."""
        from minivess.orchestration.flows.synthetic_generation_flow import (
            generate_volumes_task,
        )

        output_dir = tmp_path / "synthetic"
        generate_volumes_task.fn(
            method="debug",
            n_volumes=3,
            output_dir=str(output_dir),
        )
        assert output_dir.exists()
        npy_files = list(output_dir.glob("*.npy"))
        assert len(npy_files) >= 3  # at least image files

    def test_generate_volumes_all_methods(self, tmp_path: Path) -> None:
        """Should work for all registered methods."""
        from minivess.data.synthetic import list_generators
        from minivess.orchestration.flows.synthetic_generation_flow import (
            generate_volumes_task,
        )

        for method in list_generators():
            result = generate_volumes_task.fn(
                method=method,
                n_volumes=1,
                output_dir=str(tmp_path / method),
                config={"seed": 42},
            )
            assert result["n_volumes"] == 1

    def test_flow_definition_exists(self) -> None:
        """The flow function should be importable."""
        from minivess.orchestration.flows.synthetic_generation_flow import (
            synthetic_generation_flow,
        )

        assert callable(synthetic_generation_flow)

    def test_flow_runs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Flow should complete without error."""
        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        monkeypatch.setenv("PREFECT_DISABLED", "1")

        from minivess.orchestration.flows.synthetic_generation_flow import (
            synthetic_generation_flow,
        )

        result = synthetic_generation_flow.fn(
            method="debug",
            n_volumes=2,
            output_dir=str(tmp_path / "output"),
        )
        assert result["status"] == "completed"
