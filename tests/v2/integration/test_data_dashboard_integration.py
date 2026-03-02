"""Integration tests for data flow → dashboard pipeline.

Covers Tasks 6.1 and 6.2 of data-engineering-improvement-plan.xml.
Closes #181.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Task 6.1: Wire simulator into data flow
# ---------------------------------------------------------------------------


class TestDataFlowWithSyntheticDrift:
    """Data flow processes synthetic drift data correctly."""

    def test_simulator_output_passes_data_flow(self, tmp_path: Path) -> None:
        """Synthetic timepoints can be saved and discovered by data flow."""
        import torch

        from minivess.data.acquisition_simulator import (
            AcquisitionSimulatorConfig,
            DriftSchedule,
            SyntheticAcquisitionSimulator,
        )
        from minivess.data.drift_synthetic import DriftType
        from minivess.orchestration.flows.data_flow import (
            discover_data_task,
        )

        # Generate synthetic data
        config = AcquisitionSimulatorConfig(
            n_timepoints=3,
            base_volume_shape=(1, 8, 8, 8),
            schedules=[
                DriftSchedule(
                    drift_type=DriftType.NOISE_INJECTION,
                    start_timepoint=0,
                    end_timepoint=3,
                    severity_curve="linear",
                ),
            ],
            seed=42,
        )
        sim = SyntheticAcquisitionSimulator(config)
        batch = sim.generate_batch()

        # Save to NIfTI-like files (fake for discovery)
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        for i, vol in enumerate(batch["volumes"]):
            # Write fake .nii.gz files (discovery only checks existence)
            (images_dir / f"synth_{i:03d}.nii.gz").write_bytes(vol.numpy().tobytes())
            (labels_dir / f"synth_{i:03d}.nii.gz").write_bytes(
                torch.zeros_like(vol).numpy().tobytes()
            )

        # Discover
        pairs = discover_data_task(data_dir=tmp_path)
        assert len(pairs) == 3

    def test_simulator_metadata_available_for_dashboard(self) -> None:
        """Simulator metadata can populate DataDashboardSection."""
        from minivess.data.acquisition_simulator import (
            AcquisitionSimulatorConfig,
            DriftSchedule,
            SyntheticAcquisitionSimulator,
        )
        from minivess.data.drift_synthetic import DriftType
        from minivess.orchestration.flows.dashboard_sections import (
            DataDashboardSection,
        )

        config = AcquisitionSimulatorConfig(
            n_timepoints=5,
            base_volume_shape=(1, 4, 4, 4),
            schedules=[
                DriftSchedule(
                    drift_type=DriftType.NOISE_INJECTION,
                    start_timepoint=0,
                    end_timepoint=5,
                    severity_curve="linear",
                ),
            ],
            seed=42,
        )
        sim = SyntheticAcquisitionSimulator(config)
        batch = sim.generate_batch()

        # Extract drift summary from metadata
        drift_summary: dict[str, float] = {}
        for meta in batch["metadata"]:
            for drift in meta["active_drifts"]:
                drift_type = drift["drift_type"]
                drift_summary[drift_type] = max(
                    drift_summary.get(drift_type, 0), drift["severity"]
                )

        section = DataDashboardSection(
            n_volumes=config.n_timepoints,
            n_external_datasets=0,
            quality_gate_passed=True,
            drift_summary=drift_summary,
            external_datasets={},
        )
        assert section.n_volumes == 5
        assert len(section.drift_summary) > 0


# ---------------------------------------------------------------------------
# Task 6.2: Full integration test — data flow → dashboard
# ---------------------------------------------------------------------------


class TestDataToDashboardIntegration:
    """End-to-end: data flow → everything dashboard."""

    def test_data_flow_feeds_dashboard(self, tmp_path: Path) -> None:
        """DataFlowResult feeds into run_everything_dashboard_flow."""
        from minivess.orchestration.flows.dashboard_flow import (
            run_everything_dashboard_flow,
        )
        from minivess.orchestration.flows.data_flow import run_data_flow

        # Create minimal dataset
        data_dir = tmp_path / "data"
        (data_dir / "images").mkdir(parents=True)
        (data_dir / "labels").mkdir(parents=True)
        for i in range(4):
            (data_dir / "images" / f"vol_{i:03d}.nii.gz").write_bytes(b"fake")
            (data_dir / "labels" / f"vol_{i:03d}.nii.gz").write_bytes(b"fake")

        # Run data flow
        data_result = run_data_flow(data_dir=data_dir, n_folds=2, seed=42)

        # Feed into everything dashboard
        output_dir = tmp_path / "dashboard"
        result = run_everything_dashboard_flow(
            output_dir=output_dir,
            n_volumes=len(data_result.pairs),
            quality_gate_passed=data_result.quality_passed,
            external_datasets=data_result.external_datasets,
            drift_summary={},
            environment="test",
            experiment_config="test.yaml",
            model_profile_name="dynunet",
            architecture_name="DynUNet",
            param_count=0,
            onnx_exported=False,
            champion_category="",
            loss_name="cbdice_cldice",
            flow_results={"data": "success"},
            last_data_version="0.0.0",
            last_training_run_id="",
            trigger_source="integration_test",
        )

        assert result["report_path"].exists()
        assert result["metadata_path"].exists()

    def test_dashboard_report_has_all_sections(self, tmp_path: Path) -> None:
        """Generated report markdown has all 4 H2 sections."""
        from minivess.orchestration.flows.dashboard_flow import (
            run_everything_dashboard_flow,
        )

        result = run_everything_dashboard_flow(
            output_dir=tmp_path,
            n_volumes=10,
            quality_gate_passed=True,
            external_datasets={"deepvess": 1},
            drift_summary={"noise": 0.5},
            environment="integration",
            experiment_config="test.yaml",
            model_profile_name="dynunet",
            architecture_name="DynUNet",
            param_count=1_000_000,
            onnx_exported=True,
            champion_category="balanced",
            loss_name="cbdice_cldice",
            flow_results={
                "data": "success",
                "train": "success",
                "analyze": "success",
                "deploy": "success",
                "dashboard": "success",
            },
            last_data_version="0.1.0",
            last_training_run_id="run_abc123",
            trigger_source="integration_test",
        )

        content = result["report_path"].read_text(encoding="utf-8")
        assert "## Data" in content
        assert "## Configuration" in content
        assert "## Model" in content
        assert "## Pipeline" in content
        assert "cbdice_cldice" in content

    def test_dashboard_metadata_schema(self, tmp_path: Path) -> None:
        """Generated JSON metadata has all expected keys."""
        from minivess.orchestration.flows.dashboard_flow import (
            run_everything_dashboard_flow,
        )

        result = run_everything_dashboard_flow(
            output_dir=tmp_path,
            n_volumes=10,
            quality_gate_passed=True,
            environment="test",
            flow_results={"data": "success"},
            trigger_source="test",
        )

        data = json.loads(result["metadata_path"].read_text(encoding="utf-8"))
        assert "generated_at" in data
        assert "data" in data
        assert "config" in data
        assert "model" in data
        assert "pipeline" in data
        assert data["data"]["n_volumes"] == 10
        assert data["pipeline"]["trigger_source"] == "test"

    def test_trigger_chain_data_to_dashboard(self, tmp_path: Path) -> None:
        """Trigger chain runs data→dashboard without error."""
        from minivess.orchestration.trigger import (
            FlowTriggerConfig,
            PipelineTriggerChain,
        )

        executed: list[str] = []

        def mock_data_flow(**kwargs: object) -> None:
            executed.append("data")

        def mock_dashboard_flow(**kwargs: object) -> None:
            executed.append("dashboard")

        chain = PipelineTriggerChain()
        chain.register_flow("data", mock_data_flow, is_core=True)
        chain.register_flow("dashboard", mock_dashboard_flow, is_core=False)

        config = FlowTriggerConfig(skip_flows=["train", "analyze", "deploy"])
        results = chain.run_chain(trigger_source="integration_test", config=config)

        assert "data" in executed
        assert "dashboard" in executed
        # All executed flows should succeed
        for r in results:
            assert r.status == "success", f"{r.flow_name} failed: {r.error}"

    def test_dvc_change_triggers_chain(self, tmp_path: Path) -> None:
        """DVC version change detection triggers the pipeline."""
        from minivess.data.versioning import (
            create_data_version_tag,
            detect_dvc_change,
        )
        from minivess.orchestration.trigger import (
            FlowTriggerConfig,
            PipelineTriggerChain,
        )

        # Simulate DVC change
        dvc_file = tmp_path / "data.dvc"
        dvc_file.write_text("md5: new_hash_abc\n", encoding="utf-8")
        assert detect_dvc_change(data_dir=tmp_path) is True

        # Create version tag
        tag = create_data_version_tag("minivess", "0.2.0")
        assert tag == "data/minivess/v0.2.0"

        # Run trigger chain in dry-run mode
        chain = PipelineTriggerChain()
        config = FlowTriggerConfig(dry_run=True)
        results = chain.run_chain(trigger_source="dvc_version_change", config=config)
        assert len(results) == 5
        assert all(r.status == "skipped" for r in results)
