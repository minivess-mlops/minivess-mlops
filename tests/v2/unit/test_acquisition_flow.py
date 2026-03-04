"""Tests for Prefect Data Acquisition Flow (Flow 0).

Phase 4, Tasks 4.1–4.3 of flow-data-acquisition-plan.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Task 4.1: @task function tests
# ---------------------------------------------------------------------------


class TestCheckDatasetStatusTask:
    """check_dataset_status_task wraps check_dataset_availability."""

    def test_returns_status(self, tmp_path: Path) -> None:
        from minivess.config.acquisition_config import DatasetAcquisitionStatus
        from minivess.orchestration.flows.acquisition_flow import (
            check_dataset_status_task,
        )

        status = check_dataset_status_task(
            dataset_name="minivess", output_dir=tmp_path / "nonexistent"
        )
        assert isinstance(status, DatasetAcquisitionStatus)

    def test_ready_when_files_exist(self, tmp_path: Path) -> None:
        from minivess.config.acquisition_config import DatasetAcquisitionStatus
        from minivess.orchestration.flows.acquisition_flow import (
            check_dataset_status_task,
        )

        d = tmp_path / "vesselnn"
        (d / "images").mkdir(parents=True)
        (d / "labels").mkdir(parents=True)
        (d / "images" / "vol_001.nii.gz").write_bytes(b"fake")
        (d / "labels" / "vol_001.nii.gz").write_bytes(b"fake")

        status = check_dataset_status_task(dataset_name="vesselnn", output_dir=d)
        assert status == DatasetAcquisitionStatus.READY


class TestDownloadDatasetTask:
    """download_dataset_task tries automated download if available."""

    def test_manual_dataset_returns_manual_required(self, tmp_path: Path) -> None:
        from minivess.config.acquisition_config import DatasetAcquisitionStatus
        from minivess.orchestration.flows.acquisition_flow import (
            download_dataset_task,
        )

        status = download_dataset_task(
            dataset_name="minivess", output_dir=tmp_path / "minivess"
        )
        assert status == DatasetAcquisitionStatus.MANUAL_REQUIRED

    def test_vesselnn_calls_downloader(self, tmp_path: Path) -> None:
        from minivess.config.acquisition_config import DatasetAcquisitionStatus
        from minivess.orchestration.flows.acquisition_flow import (
            download_dataset_task,
        )

        target = tmp_path / "vesselnn"
        with patch(
            "minivess.orchestration.flows.acquisition_flow.get_downloader"
        ) as mock_get:
            mock_get.return_value = lambda target_dir, **kw: target_dir
            status = download_dataset_task(dataset_name="vesselnn", output_dir=target)
        assert status == DatasetAcquisitionStatus.DOWNLOADED


class TestConvertFormatsTask:
    """convert_formats_task wraps convert_dataset_formats."""

    def test_returns_log_list(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.acquisition_flow import (
            convert_formats_task,
        )

        log = convert_formats_task(
            dataset_name="minivess",
            input_dir=tmp_path / "nonexistent",
            output_dir=tmp_path / "output",
        )
        assert isinstance(log, list)


class TestLogAcquisitionProvenanceTask:
    """log_acquisition_provenance_task produces MLflow-compatible dict."""

    def test_returns_dict_with_keys(self) -> None:
        from minivess.config.acquisition_config import DatasetAcquisitionStatus
        from minivess.orchestration.flows.acquisition_flow import (
            log_acquisition_provenance_task,
        )

        result = log_acquisition_provenance_task(
            datasets_acquired={"vesselnn": DatasetAcquisitionStatus.READY},
            total_volumes=12,
        )
        assert isinstance(result, dict)
        assert "acq_n_datasets" in result
        assert "acq_total_volumes" in result


class TestPrintManualInstructionsTask:
    """print_manual_instructions_task logs instructions for manual datasets."""

    def test_returns_instructions_dict(self) -> None:
        from minivess.orchestration.flows.acquisition_flow import (
            print_manual_instructions_task,
        )

        result = print_manual_instructions_task(
            manual_datasets=["minivess", "deepvess"]
        )
        assert isinstance(result, dict)
        assert "minivess" in result
        assert "deepvess" in result


# ---------------------------------------------------------------------------
# Task 4.2: @flow orchestrator tests
# ---------------------------------------------------------------------------


class TestAcquisitionFlowOrchestrator:
    """run_acquisition_flow orchestrates the full flow."""

    def test_returns_acquisition_result(self, tmp_path: Path) -> None:
        from minivess.config.acquisition_config import (
            AcquisitionConfig,
            AcquisitionResult,
        )
        from minivess.orchestration.flows.acquisition_flow import (
            run_acquisition_flow,
        )

        config = AcquisitionConfig(
            datasets=["minivess"],
            output_dir=tmp_path,
            convert_formats=False,
        )
        result = run_acquisition_flow(config=config)
        assert isinstance(result, AcquisitionResult)

    def test_ready_dataset_skips_download(self, tmp_path: Path) -> None:
        from minivess.config.acquisition_config import (
            AcquisitionConfig,
            AcquisitionResult,
            DatasetAcquisitionStatus,
        )
        from minivess.orchestration.flows.acquisition_flow import (
            run_acquisition_flow,
        )

        # Create pre-existing dataset
        d = tmp_path / "vesselnn"
        (d / "images").mkdir(parents=True)
        (d / "labels").mkdir(parents=True)
        (d / "images" / "vol_001.nii.gz").write_bytes(b"fake")
        (d / "labels" / "vol_001.nii.gz").write_bytes(b"fake")

        config = AcquisitionConfig(
            datasets=["vesselnn"],
            output_dir=tmp_path,
            convert_formats=False,
        )
        result = run_acquisition_flow(config=config)
        assert isinstance(result, AcquisitionResult)
        assert result.datasets_acquired["vesselnn"] == DatasetAcquisitionStatus.READY

    def test_flow_accepts_trigger_source(self, tmp_path: Path) -> None:
        from minivess.config.acquisition_config import AcquisitionConfig
        from minivess.orchestration.flows.acquisition_flow import (
            run_acquisition_flow,
        )

        config = AcquisitionConfig(
            datasets=["minivess"],
            output_dir=tmp_path,
            convert_formats=False,
        )
        # Should not raise — trigger_source is accepted via **kwargs
        result = run_acquisition_flow(config=config, trigger_source="manual")
        assert result is not None


# ---------------------------------------------------------------------------
# Task 4.3: Trigger chain + deployments integration
# ---------------------------------------------------------------------------


class TestTriggerChainRegistration:
    """Acquisition flow is registered in the trigger chain."""

    def test_acquisition_in_default_flows(self) -> None:
        from minivess.orchestration.trigger import PipelineTriggerChain

        assert "acquisition" in PipelineTriggerChain._DEFAULT_FLOWS

    def test_acquisition_is_first_flow(self) -> None:
        from minivess.orchestration.trigger import PipelineTriggerChain

        assert PipelineTriggerChain._DEFAULT_FLOWS[0] == "acquisition"

    def test_acquisition_is_core(self) -> None:
        from minivess.orchestration.trigger import PipelineTriggerChain

        chain = PipelineTriggerChain()
        entry = chain._flows["acquisition"]
        assert entry.is_core is True


class TestDeploymentsRegistration:
    """Acquisition flow has deployment configuration."""

    def test_acquisition_in_work_pool_map(self) -> None:
        from minivess.orchestration.deployments import FLOW_WORK_POOL_MAP

        assert "acquisition" in FLOW_WORK_POOL_MAP
        assert FLOW_WORK_POOL_MAP["acquisition"] == "cpu-pool"

    def test_acquisition_in_image_map(self) -> None:
        from minivess.orchestration.deployments import FLOW_IMAGE_MAP

        assert "acquisition" in FLOW_IMAGE_MAP
        assert FLOW_IMAGE_MAP["acquisition"] == "minivess-acquisition:latest"

    def test_deployment_config(self) -> None:
        from minivess.orchestration.deployments import get_flow_deployment_config

        config = get_flow_deployment_config("acquisition")
        assert config["work_pool"] == "cpu-pool"
        assert config["image"] == "minivess-acquisition:latest"
