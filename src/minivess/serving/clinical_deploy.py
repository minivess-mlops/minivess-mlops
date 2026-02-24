"""MONAI Deploy clinical deployment pathway.

Provides infrastructure for clinical-grade deployment of segmentation
models via MONAI Application Package (MAP) format, DICOM I/O handling,
and deployment validation for regulated environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class DeploymentTarget(StrEnum):
    """Clinical deployment target environments."""

    RESEARCH = "research"
    STAGING = "staging"
    CLINICAL = "clinical"
    PACS = "pacs"


@dataclass
class DICOMConfig:
    """DICOM network configuration.

    Parameters
    ----------
    ae_title:
        Application Entity title for DICOM SCP.
    host:
        Network host to bind.
    port:
        DICOM port (default: 11112).
    """

    ae_title: str = "MINIVESS_SEG"
    host: str = "0.0.0.0"
    port: int = 11112


@dataclass
class ValidationResult:
    """Result of a deployment validation check.

    Parameters
    ----------
    is_valid:
        Whether validation passed.
    missing_tags:
        List of missing required items.
    warnings:
        Non-blocking warnings.
    """

    is_valid: bool
    missing_tags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_REQUIRED_DICOM_TAGS = frozenset({
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Modality",
})


class DICOMHandler:
    """DICOM I/O abstraction for clinical deployment.

    Validates DICOM metadata and generates Structured Reports (SR)
    from segmentation predictions.
    """

    def validate_dicom_metadata(
        self, metadata: dict[str, Any],
    ) -> ValidationResult:
        """Validate that required DICOM tags are present.

        Parameters
        ----------
        metadata:
            Dictionary of DICOM tag names to values.
        """
        missing = [
            tag for tag in sorted(_REQUIRED_DICOM_TAGS)
            if tag not in metadata
        ]
        return ValidationResult(
            is_valid=len(missing) == 0,
            missing_tags=missing,
        )

    def create_dicom_sr(
        self,
        study_uid: str,
        series_uid: str,
        findings: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a DICOM Structured Report from segmentation findings.

        Parameters
        ----------
        study_uid:
            Study Instance UID.
        series_uid:
            Series Instance UID.
        findings:
            Dictionary of segmentation findings (metrics, counts).
        """
        return {
            "study_uid": study_uid,
            "series_uid": series_uid,
            "sr_type": "COMPREHENSIVE",
            "findings": findings,
            "created": datetime.now(UTC).isoformat(),
            "software": "minivess-segmentation",
        }


@dataclass
class ClinicalDeployConfig:
    """Configuration for clinical deployment.

    Parameters
    ----------
    model_path:
        Path to the exported model (ONNX or TorchScript).
    model_name:
        Model identifier.
    version:
        Semantic version string.
    deployment_target:
        Target environment.
    dicom:
        DICOM configuration.
    """

    model_path: str = ""
    model_name: str = ""
    version: str = "0.0.0"
    deployment_target: str = "research"
    dicom: DICOMConfig = field(default_factory=DICOMConfig)


@dataclass
class MonaiDeployManifest:
    """MONAI Application Package (MAP) manifest.

    Parameters
    ----------
    app_name:
        Application name.
    version:
        Application version.
    model_name:
        Underlying model identifier.
    api_version:
        MAP API version.
    """

    app_name: str
    version: str
    model_name: str
    api_version: str = "0.6.0"
    sdk_version: str = "3.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Export as a serializable manifest dictionary."""
        return {
            "api_version": self.api_version,
            "sdk_version": self.sdk_version,
            "app_name": self.app_name,
            "version": self.version,
            "model_name": self.model_name,
            "command": f"python -m monai.deploy.runner {self.app_name}",
            "input": {"formats": ["dicom"]},
            "output": {"formats": ["dicom-sr", "nifti"]},
            "resources": {
                "cpu": 4,
                "memory": "8Gi",
                "gpu": 1,
            },
        }

    def to_markdown(self) -> str:
        """Generate a human-readable manifest report."""
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        sections = [
            "# MONAI Application Package Manifest",
            "",
            f"**Generated:** {now}",
            f"**Application:** {self.app_name}",
            f"**Version:** {self.version}",
            f"**Model:** {self.model_name}",
            f"**MAP API:** {self.api_version}",
            f"**SDK:** {self.sdk_version}",
            "",
            "## I/O Specification",
            "",
            "- **Input:** DICOM series",
            "- **Output:** DICOM SR + NIfTI segmentation",
            "",
            "## Resource Requirements",
            "",
            "| Resource | Requirement |",
            "|----------|-------------|",
            "| CPU | 4 cores |",
            "| Memory | 8 GiB |",
            "| GPU | 1 NVIDIA dGPU |",
            "",
        ]
        return "\n".join(sections)


class ClinicalDeploymentPipeline:
    """Orchestrates clinical deployment validation and packaging.

    Parameters
    ----------
    config:
        Clinical deployment configuration.
    """

    def __init__(self, config: ClinicalDeployConfig) -> None:
        self.config = config

    def validate(self) -> ValidationResult:
        """Run pre-deployment validation checks."""
        missing: list[str] = []
        warnings: list[str] = []

        if not self.config.model_name:
            missing.append("model_name")
        if not self.config.model_path:
            missing.append("model_path")
        if self.config.version == "0.0.0":
            warnings.append("Version is default (0.0.0)")

        if (
            self.config.deployment_target == "clinical"
            and self.config.dicom.ae_title == "MINIVESS_SEG"
        ):
            warnings.append(
                "Using default AE title for clinical deployment"
            )

        return ValidationResult(
            is_valid=len(missing) == 0,
            missing_tags=missing,
            warnings=warnings,
        )

    def generate_manifest(self) -> MonaiDeployManifest:
        """Generate a MAP manifest from configuration."""
        return MonaiDeployManifest(
            app_name=f"minivess-{self.config.model_name}",
            version=self.config.version,
            model_name=self.config.model_name,
        )

    def to_markdown(self) -> str:
        """Generate a deployment summary report."""
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        validation = self.validate()

        sections = [
            "# Clinical Deployment Summary",
            "",
            f"**Generated:** {now}",
            f"**Model:** {self.config.model_name}",
            f"**Version:** {self.config.version}",
            f"**Target:** {self.config.deployment_target}",
            f"**Model Path:** {self.config.model_path}",
            "",
            "## DICOM Configuration",
            "",
            f"- **AE Title:** {self.config.dicom.ae_title}",
            f"- **Host:** {self.config.dicom.host}",
            f"- **Port:** {self.config.dicom.port}",
            "",
            "## Validation",
            "",
            f"- **Status:** {'PASS' if validation.is_valid else 'FAIL'}",
        ]

        if validation.missing_tags:
            sections.append(
                f"- **Missing:** {', '.join(validation.missing_tags)}"
            )
        if validation.warnings:
            for w in validation.warnings:
                sections.append(f"- **Warning:** {w}")

        sections.append("")
        return "\n".join(sections)
