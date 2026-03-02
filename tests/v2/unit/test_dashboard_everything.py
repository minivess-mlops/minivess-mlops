"""Tests for Everything Dashboard tasks and flow.

Covers Tasks 4.2 and 4.3 of data-engineering-improvement-plan.xml.
Closes #179.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Task 4.2: Section collection tasks
# ---------------------------------------------------------------------------


class TestCollectDataSectionTask:
    """collect_data_section_task returns DataDashboardSection."""

    def test_collect_data_section_returns_section(self) -> None:
        from minivess.orchestration.flows.dashboard_flow import (
            collect_data_section_task,
        )
        from minivess.orchestration.flows.dashboard_sections import (
            DataDashboardSection,
        )

        section = collect_data_section_task(
            n_volumes=10,
            quality_gate_passed=True,
            external_datasets={"deepvess": 1},
            drift_summary={"noise": 0.3},
        )
        assert isinstance(section, DataDashboardSection)
        assert section.n_volumes == 10


class TestCollectConfigSectionTask:
    """collect_config_section_task returns ConfigDashboardSection."""

    def test_collect_config_section_returns_section(self) -> None:
        from minivess.orchestration.flows.dashboard_flow import (
            collect_config_section_task,
        )
        from minivess.orchestration.flows.dashboard_sections import (
            ConfigDashboardSection,
        )

        section = collect_config_section_task(
            environment="development",
            experiment_config="dynunet_losses.yaml",
            model_profile_name="dynunet",
        )
        assert isinstance(section, ConfigDashboardSection)
        assert section.environment == "development"


class TestCollectModelSectionTask:
    """collect_model_section_task returns ModelDashboardSection."""

    def test_collect_model_section_returns_section(self) -> None:
        from minivess.orchestration.flows.dashboard_flow import (
            collect_model_section_task,
        )
        from minivess.orchestration.flows.dashboard_sections import (
            ModelDashboardSection,
        )

        section = collect_model_section_task(
            architecture_name="DynUNet",
            param_count=1_500_000,
            onnx_exported=True,
            champion_category="balanced",
            loss_name="cbdice_cldice",
        )
        assert isinstance(section, ModelDashboardSection)
        assert section.architecture_name == "DynUNet"


class TestCollectPipelineSectionTask:
    """collect_pipeline_section_task returns PipelineDashboardSection."""

    def test_collect_pipeline_section_returns_section(self) -> None:
        from minivess.orchestration.flows.dashboard_flow import (
            collect_pipeline_section_task,
        )
        from minivess.orchestration.flows.dashboard_sections import (
            PipelineDashboardSection,
        )

        section = collect_pipeline_section_task(
            flow_results={"data": "success", "train": "pending"},
            last_data_version="0.1.0",
            last_training_run_id="run123",
            trigger_source="manual",
        )
        assert isinstance(section, PipelineDashboardSection)
        assert section.flow_results["data"] == "success"


# ---------------------------------------------------------------------------
# Task 4.2: Report/metadata generation with everything dashboard
# ---------------------------------------------------------------------------


class TestGenerateEverythingReport:
    """generate_everything_report includes all sections in markdown."""

    def test_report_has_data_section(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.dashboard_flow import (
            generate_everything_report,
        )
        from minivess.orchestration.flows.dashboard_sections import (
            ConfigDashboardSection,
            DataDashboardSection,
            EverythingDashboard,
            ModelDashboardSection,
            PipelineDashboardSection,
        )

        dashboard = EverythingDashboard(
            data=DataDashboardSection(
                n_volumes=10,
                n_external_datasets=2,
                quality_gate_passed=True,
                drift_summary={},
                external_datasets={"deepvess": 1},
            ),
            config=ConfigDashboardSection(
                environment="dev",
                dynaconf_params={"debug": True},
                experiment_config="test.yaml",
                model_profile_name="dynunet",
            ),
            model=ModelDashboardSection(
                architecture_name="DynUNet",
                param_count=1_000_000,
                onnx_exported=False,
                champion_category="balanced",
                loss_name="cbdice_cldice",
            ),
            pipeline=PipelineDashboardSection(
                flow_results={"data": "success"},
                last_data_version="0.1.0",
                last_training_run_id="run123",
                trigger_source="manual",
            ),
            generated_at=datetime.now(UTC).isoformat(),
        )

        report_path = generate_everything_report(
            dashboard=dashboard, output_dir=tmp_path
        )
        content = report_path.read_text(encoding="utf-8")
        assert "## Data" in content
        assert "## Configuration" in content
        assert "## Model" in content
        assert "## Pipeline" in content


class TestExportEverythingMetadata:
    """export_everything_metadata includes all section keys in JSON."""

    def test_metadata_has_all_keys(self, tmp_path: Path) -> None:
        import json

        from minivess.orchestration.flows.dashboard_flow import (
            export_everything_metadata,
        )
        from minivess.orchestration.flows.dashboard_sections import (
            ConfigDashboardSection,
            DataDashboardSection,
            EverythingDashboard,
            ModelDashboardSection,
            PipelineDashboardSection,
        )

        dashboard = EverythingDashboard(
            data=DataDashboardSection(
                n_volumes=10,
                n_external_datasets=2,
                quality_gate_passed=True,
                drift_summary={},
                external_datasets={},
            ),
            config=ConfigDashboardSection(
                environment="dev",
                dynaconf_params={},
                experiment_config="",
                model_profile_name="",
            ),
            model=ModelDashboardSection(
                architecture_name="DynUNet",
                param_count=0,
                onnx_exported=False,
                champion_category="",
                loss_name="",
            ),
            pipeline=PipelineDashboardSection(
                flow_results={},
                last_data_version="",
                last_training_run_id="",
                trigger_source="manual",
            ),
            generated_at=datetime.now(UTC).isoformat(),
        )

        json_path = export_everything_metadata(dashboard=dashboard, output_dir=tmp_path)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "data" in data
        assert "config" in data
        assert "model" in data
        assert "pipeline" in data
        assert "generated_at" in data


# ---------------------------------------------------------------------------
# Task 4.3: Updated run_everything_dashboard_flow
# ---------------------------------------------------------------------------


class TestRunEverythingDashboardFlow:
    """run_everything_dashboard_flow with all sections."""

    def test_everything_flow_returns_dict(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.dashboard_flow import (
            run_everything_dashboard_flow,
        )

        result = run_everything_dashboard_flow(
            output_dir=tmp_path,
            n_volumes=10,
            quality_gate_passed=True,
            external_datasets={},
            drift_summary={},
            environment="dev",
            experiment_config="test.yaml",
            model_profile_name="dynunet",
            architecture_name="DynUNet",
            param_count=0,
            onnx_exported=False,
            champion_category="",
            loss_name="",
            flow_results={},
            last_data_version="",
            last_training_run_id="",
            trigger_source="manual",
        )
        assert isinstance(result, dict)
        assert "report_path" in result
        assert "metadata_path" in result

    def test_everything_flow_creates_report(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.dashboard_flow import (
            run_everything_dashboard_flow,
        )

        result = run_everything_dashboard_flow(
            output_dir=tmp_path,
            n_volumes=10,
            quality_gate_passed=True,
            external_datasets={},
            drift_summary={},
            environment="dev",
            experiment_config="test.yaml",
            model_profile_name="dynunet",
            architecture_name="DynUNet",
            param_count=0,
            onnx_exported=False,
            champion_category="",
            loss_name="",
            flow_results={"data": "success"},
            last_data_version="0.1.0",
            last_training_run_id="",
            trigger_source="manual",
        )
        report_path = result["report_path"]
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "## Data" in content

    def test_backwards_compat_original_flow_still_works(self, tmp_path: Path) -> None:
        """Original run_dashboard_flow still works (imports don't break)."""
        from minivess.orchestration.flows.dashboard_flow import (
            run_dashboard_flow,
        )

        # Just verify it's importable — it requires comparison_table
        # which we don't mock here, so we just check it exists
        assert callable(run_dashboard_flow)
