"""Tests for MONAI Deploy clinical deployment pathway (Issue #47)."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# T1: DeploymentTarget enum
# ---------------------------------------------------------------------------


class TestDeploymentTarget:
    """Test deployment target enum."""

    def test_enum_values(self) -> None:
        """DeploymentTarget should have four targets."""
        from minivess.serving.clinical_deploy import DeploymentTarget

        assert DeploymentTarget.RESEARCH == "research"
        assert DeploymentTarget.STAGING == "staging"
        assert DeploymentTarget.CLINICAL == "clinical"
        assert DeploymentTarget.PACS == "pacs"


# ---------------------------------------------------------------------------
# T2: DICOMConfig
# ---------------------------------------------------------------------------


class TestDICOMConfig:
    """Test DICOM configuration."""

    def test_construction(self) -> None:
        """DICOMConfig should capture DICOM settings."""
        from minivess.serving.clinical_deploy import DICOMConfig

        config = DICOMConfig(
            ae_title="MINIVESS",
            host="localhost",
            port=11112,
        )
        assert config.ae_title == "MINIVESS"
        assert config.port == 11112

    def test_defaults(self) -> None:
        """DICOMConfig should have sensible defaults."""
        from minivess.serving.clinical_deploy import DICOMConfig

        config = DICOMConfig()
        assert config.ae_title == "MINIVESS_SEG"
        assert config.port == 11112
        assert config.host == "0.0.0.0"


# ---------------------------------------------------------------------------
# T3: DICOMHandler
# ---------------------------------------------------------------------------


class TestDICOMHandler:
    """Test DICOM I/O handler."""

    def test_validate_metadata_complete(self) -> None:
        """Complete DICOM metadata should pass validation."""
        from minivess.serving.clinical_deploy import DICOMHandler

        handler = DICOMHandler()
        metadata = {
            "PatientID": "P001",
            "StudyInstanceUID": "1.2.3.4",
            "SeriesInstanceUID": "1.2.3.4.5",
            "Modality": "MR",
        }
        result = handler.validate_dicom_metadata(metadata)
        assert result.is_valid is True
        assert len(result.missing_tags) == 0

    def test_validate_metadata_missing(self) -> None:
        """Missing required tags should fail validation."""
        from minivess.serving.clinical_deploy import DICOMHandler

        handler = DICOMHandler()
        metadata = {"PatientID": "P001"}
        result = handler.validate_dicom_metadata(metadata)
        assert result.is_valid is False
        assert len(result.missing_tags) > 0

    def test_create_dicom_sr(self) -> None:
        """create_dicom_sr should produce a structured report."""
        from minivess.serving.clinical_deploy import DICOMHandler

        handler = DICOMHandler()
        sr = handler.create_dicom_sr(
            study_uid="1.2.3.4",
            series_uid="1.2.3.4.5",
            findings={"vessel_volume_mm3": 1234.5, "num_segments": 3},
        )
        assert sr["study_uid"] == "1.2.3.4"
        assert "findings" in sr
        assert sr["sr_type"] == "COMPREHENSIVE"


# ---------------------------------------------------------------------------
# T4: MonaiDeployManifest
# ---------------------------------------------------------------------------


class TestMonaiDeployManifest:
    """Test MONAI Application Package manifest."""

    def test_construction(self) -> None:
        """MonaiDeployManifest should capture MAP metadata."""
        from minivess.serving.clinical_deploy import MonaiDeployManifest

        manifest = MonaiDeployManifest(
            app_name="minivess-segmentation",
            version="1.0.0",
            model_name="segresnet_v1",
        )
        assert manifest.app_name == "minivess-segmentation"
        assert manifest.version == "1.0.0"

    def test_to_dict(self) -> None:
        """to_dict should produce a serializable manifest."""
        from minivess.serving.clinical_deploy import MonaiDeployManifest

        manifest = MonaiDeployManifest(
            app_name="minivess-segmentation",
            version="1.0.0",
            model_name="segresnet_v1",
        )
        d = manifest.to_dict()
        assert d["app_name"] == "minivess-segmentation"
        assert "api_version" in d

    def test_to_markdown(self) -> None:
        """to_markdown should produce a readable manifest report."""
        from minivess.serving.clinical_deploy import MonaiDeployManifest

        manifest = MonaiDeployManifest(
            app_name="minivess-segmentation",
            version="1.0.0",
            model_name="segresnet_v1",
        )
        md = manifest.to_markdown()
        assert "MONAI" in md
        assert "minivess-segmentation" in md


# ---------------------------------------------------------------------------
# T5: ClinicalDeploymentPipeline
# ---------------------------------------------------------------------------


class TestClinicalDeploymentPipeline:
    """Test clinical deployment pipeline."""

    def test_validate_passes(self) -> None:
        """validate should pass with complete config."""
        from minivess.serving.clinical_deploy import (
            ClinicalDeployConfig,
            ClinicalDeploymentPipeline,
        )

        config = ClinicalDeployConfig(
            model_path="/models/segresnet.onnx",
            model_name="segresnet_v1",
            version="1.0.0",
            deployment_target="clinical",
        )
        pipeline = ClinicalDeploymentPipeline(config)
        result = pipeline.validate()
        assert result.is_valid is True

    def test_validate_fails_missing_model(self) -> None:
        """validate should fail without model name."""
        from minivess.serving.clinical_deploy import (
            ClinicalDeployConfig,
            ClinicalDeploymentPipeline,
        )

        config = ClinicalDeployConfig(
            model_path="/models/segresnet.onnx",
            model_name="",
            version="1.0.0",
        )
        pipeline = ClinicalDeploymentPipeline(config)
        result = pipeline.validate()
        assert result.is_valid is False

    def test_to_markdown(self) -> None:
        """to_markdown should produce a deployment report."""
        from minivess.serving.clinical_deploy import (
            ClinicalDeployConfig,
            ClinicalDeploymentPipeline,
        )

        config = ClinicalDeployConfig(
            model_path="/models/segresnet.onnx",
            model_name="segresnet_v1",
            version="1.0.0",
            deployment_target="clinical",
        )
        pipeline = ClinicalDeploymentPipeline(config)
        md = pipeline.to_markdown()
        assert "Clinical" in md
        assert "segresnet_v1" in md
