"""Integration tests for the Data Acquisition Flow (Flow 0).

Tests the full flow with mocked downloads, verifying end-to-end
orchestration produces correct AcquisitionResult and provenance.

Phase 5, Task 5.1 of flow-data-acquisition-plan.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


class TestAcquisitionFlowIntegration:
    """Full flow integration with mocked external downloads."""

    def test_full_flow_with_existing_minivess(self, tmp_path: Path) -> None:
        """Flow handles pre-existing MiniVess data (most common case)."""
        from minivess.config.acquisition_config import (
            AcquisitionConfig,
            AcquisitionResult,
            DatasetAcquisitionStatus,
        )
        from minivess.orchestration.flows.acquisition_flow import (
            run_acquisition_flow,
        )

        # Simulate MiniVess already downloaded
        mv = tmp_path / "minivess"
        (mv / "images").mkdir(parents=True)
        (mv / "labels").mkdir(parents=True)
        for i in range(3):
            (mv / "images" / f"vol_{i:03d}.nii.gz").write_bytes(b"fake")
            (mv / "labels" / f"vol_{i:03d}.nii.gz").write_bytes(b"fake")

        config = AcquisitionConfig(
            datasets=["minivess"],
            output_dir=tmp_path,
            convert_formats=False,
        )
        result = run_acquisition_flow(config=config)

        assert isinstance(result, AcquisitionResult)
        assert result.datasets_acquired["minivess"] == DatasetAcquisitionStatus.READY
        assert result.provenance["acq_n_datasets"] == 1

    def test_flow_with_mocked_vesselnn_download(self, tmp_path: Path) -> None:
        """Flow uses git clone for VesselNN."""
        from minivess.config.acquisition_config import (
            AcquisitionConfig,
            AcquisitionResult,
            DatasetAcquisitionStatus,
        )
        from minivess.orchestration.flows.acquisition_flow import (
            run_acquisition_flow,
        )

        def fake_clone(target_dir: Path, **_kwargs: object) -> Path:
            target_dir.mkdir(parents=True, exist_ok=True)
            return target_dir

        config = AcquisitionConfig(
            datasets=["vesselnn"],
            output_dir=tmp_path,
            convert_formats=False,
        )

        with patch(
            "minivess.orchestration.flows.acquisition_flow.get_downloader"
        ) as mock:
            mock.return_value = fake_clone
            result = run_acquisition_flow(config=config)

        assert isinstance(result, AcquisitionResult)
        assert (
            result.datasets_acquired["vesselnn"] == DatasetAcquisitionStatus.DOWNLOADED
        )

    def test_flow_collects_manual_instructions(self, tmp_path: Path) -> None:
        """Flow prints instructions for manual-download datasets."""
        from minivess.config.acquisition_config import (
            AcquisitionConfig,
            AcquisitionResult,
            DatasetAcquisitionStatus,
        )
        from minivess.orchestration.flows.acquisition_flow import (
            run_acquisition_flow,
        )

        # tubenet_2pm excluded: olfactory bulb, different organ, only 1 2PM volume
        config = AcquisitionConfig(
            datasets=["deepvess"],
            output_dir=tmp_path,
            convert_formats=False,
        )
        result = run_acquisition_flow(config=config)

        assert isinstance(result, AcquisitionResult)
        assert (
            result.datasets_acquired["deepvess"]
            == DatasetAcquisitionStatus.MANUAL_REQUIRED
        )

    def test_flow_provenance_has_timestamp(self, tmp_path: Path) -> None:
        """Provenance dict includes an ISO timestamp."""
        from minivess.config.acquisition_config import AcquisitionConfig
        from minivess.orchestration.flows.acquisition_flow import (
            run_acquisition_flow,
        )

        config = AcquisitionConfig(
            datasets=["minivess"],
            output_dir=tmp_path,
            convert_formats=False,
        )
        result = run_acquisition_flow(config=config)
        assert "acq_timestamp" in result.provenance

    def test_flow_with_tiff_conversion(self, tmp_path: Path) -> None:
        """Flow converts TIFF → NIfTI for DeepVess."""
        import tifffile

        from minivess.config.acquisition_config import (
            AcquisitionConfig,
            AcquisitionResult,
        )
        from minivess.orchestration.flows.acquisition_flow import (
            run_acquisition_flow,
        )

        # Simulate DeepVess TIFF data already downloaded
        dv = tmp_path / "deepvess"
        (dv / "images").mkdir(parents=True)
        (dv / "labels").mkdir(parents=True)

        data = np.zeros((5, 8, 8), dtype=np.uint8)
        tifffile.imwrite(str(dv / "images" / "vol_001.tif"), data)
        tifffile.imwrite(str(dv / "labels" / "vol_001.tif"), data)

        # Pre-populate so status is READY (has image+label files)
        config = AcquisitionConfig(
            datasets=["deepvess"],
            output_dir=tmp_path,
            convert_formats=True,
        )
        result = run_acquisition_flow(config=config)

        assert isinstance(result, AcquisitionResult)
        assert len(result.conversion_log) == 2  # images + labels
