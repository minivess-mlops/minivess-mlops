"""Tests for dashboard section dataclasses.

Covers Task 4.1 of data-engineering-improvement-plan.xml.
Closes #179 (partial).
"""

from __future__ import annotations

from datetime import UTC, datetime


class TestDataDashboardSection:
    """DataDashboardSection fields."""

    def test_data_section_has_profile(self) -> None:
        from minivess.orchestration.flows.dashboard_sections import (
            DataDashboardSection,
        )

        section = DataDashboardSection(
            n_volumes=10,
            n_external_datasets=2,
            quality_gate_passed=True,
            drift_summary={},
            external_datasets={},
        )
        assert section.n_volumes == 10

    def test_data_section_has_drift_summary(self) -> None:
        from minivess.orchestration.flows.dashboard_sections import (
            DataDashboardSection,
        )

        section = DataDashboardSection(
            n_volumes=5,
            n_external_datasets=0,
            quality_gate_passed=True,
            drift_summary={"noise_injection": 0.5},
            external_datasets={},
        )
        assert isinstance(section.drift_summary, dict)
        assert "noise_injection" in section.drift_summary

    def test_data_section_has_external_datasets(self) -> None:
        from minivess.orchestration.flows.dashboard_sections import (
            DataDashboardSection,
        )

        section = DataDashboardSection(
            n_volumes=5,
            n_external_datasets=2,
            quality_gate_passed=True,
            drift_summary={},
            external_datasets={"deepvess": 1, "vesselnn": 1},
        )
        assert section.n_external_datasets == 2
        assert "deepvess" in section.external_datasets


class TestConfigDashboardSection:
    """ConfigDashboardSection fields."""

    def test_config_section_has_environment(self) -> None:
        from minivess.orchestration.flows.dashboard_sections import (
            ConfigDashboardSection,
        )

        section = ConfigDashboardSection(
            environment="development",
            dynaconf_params={"debug": True},
            experiment_config="dynunet_losses.yaml",
            model_profile_name="dynunet",
        )
        assert section.environment == "development"

    def test_config_section_has_dynaconf_params(self) -> None:
        from minivess.orchestration.flows.dashboard_sections import (
            ConfigDashboardSection,
        )

        section = ConfigDashboardSection(
            environment="staging",
            dynaconf_params={"debug": False, "test_markers": "unit"},
            experiment_config="",
            model_profile_name="",
        )
        assert isinstance(section.dynaconf_params, dict)
        assert "debug" in section.dynaconf_params


class TestModelDashboardSection:
    """ModelDashboardSection fields."""

    def test_model_section_has_architecture(self) -> None:
        from minivess.orchestration.flows.dashboard_sections import (
            ModelDashboardSection,
        )

        section = ModelDashboardSection(
            architecture_name="DynUNet",
            param_count=1_500_000,
            onnx_exported=True,
            champion_category="balanced",
            loss_name="cbdice_cldice",
        )
        assert section.architecture_name == "DynUNet"

    def test_model_section_tracks_onnx(self) -> None:
        from minivess.orchestration.flows.dashboard_sections import (
            ModelDashboardSection,
        )

        section = ModelDashboardSection(
            architecture_name="DynUNet",
            param_count=0,
            onnx_exported=False,
            champion_category="",
            loss_name="",
        )
        assert section.onnx_exported is False


class TestPipelineDashboardSection:
    """PipelineDashboardSection fields."""

    def test_pipeline_section_has_flow_status(self) -> None:
        from minivess.orchestration.flows.dashboard_sections import (
            PipelineDashboardSection,
        )

        section = PipelineDashboardSection(
            flow_results={"data": "success", "train": "pending"},
            last_data_version="0.1.0",
            last_training_run_id="abc123",
            trigger_source="manual",
        )
        assert isinstance(section.flow_results, dict)
        assert section.flow_results["data"] == "success"

    def test_pipeline_section_has_trigger_source(self) -> None:
        from minivess.orchestration.flows.dashboard_sections import (
            PipelineDashboardSection,
        )

        section = PipelineDashboardSection(
            flow_results={},
            last_data_version="",
            last_training_run_id="",
            trigger_source="dvc_version_change",
        )
        assert section.trigger_source == "dvc_version_change"


class TestEverythingDashboard:
    """EverythingDashboard combines all sections."""

    def test_everything_dashboard_has_all_sections(self) -> None:
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
        assert dashboard.data.n_volumes == 10
        assert dashboard.config.environment == "dev"
        assert dashboard.model.architecture_name == "DynUNet"
        assert isinstance(dashboard.pipeline.flow_results, dict)
        assert dashboard.generated_at is not None
