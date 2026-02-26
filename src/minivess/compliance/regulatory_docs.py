"""Automated regulatory documentation generation for IEC 62304 compliance.

Generates document templates from AuditTrail data:
- Design History File (DHF)
- Risk Analysis (ISO 14971)
- Software Requirements Specification (SRS)
- Validation Summary
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minivess.compliance.audit import AuditTrail


class SaMDRiskClass(StrEnum):
    """EU MDR / FDA SaMD risk classification."""

    CLASS_I = "class_i"
    CLASS_IIA = "class_iia"
    CLASS_IIB = "class_iib"
    CLASS_III = "class_iii"


_RISK_CLASS_LABELS = {
    SaMDRiskClass.CLASS_I: "Class I",
    SaMDRiskClass.CLASS_IIA: "Class IIa",
    SaMDRiskClass.CLASS_IIB: "Class IIb",
    SaMDRiskClass.CLASS_III: "Class III",
}


class RegulatoryDocGenerator:
    """Generates IEC 62304 regulatory document templates from AuditTrail.

    Parameters
    ----------
    audit_trail:
        AuditTrail containing lifecycle events.
    product_name:
        Name of the SaMD product.
    product_version:
        Version string.
    risk_class:
        EU MDR risk classification (default: Class I).
    """

    def __init__(
        self,
        audit_trail: AuditTrail,
        product_name: str,
        product_version: str,
        risk_class: SaMDRiskClass = SaMDRiskClass.CLASS_I,
    ) -> None:
        self.audit_trail = audit_trail
        self.product_name = product_name
        self.product_version = product_version
        self.risk_class = risk_class

    def _header(self, title: str) -> str:
        """Generate standard document header."""
        now = datetime.now(UTC).strftime("%Y-%m-%d")
        return (
            f"# {title}\n\n"
            f"**Product:** {self.product_name}\n"
            f"**Version:** {self.product_version}\n"
            f"**Date:** {now}\n"
            f"**Classification:** {_RISK_CLASS_LABELS[self.risk_class]}\n"
        )

    def generate_design_history(self) -> str:
        """Generate a Design History File from audit trail events.

        Returns
        -------
        Markdown string documenting the development chronology.
        """
        lines = [self._header("Design History File")]
        lines.append("\n## Development Chronology\n")

        if not self.audit_trail.entries:
            lines.append("No events recorded.\n")
            return "\n".join(lines)

        lines.append("| Timestamp | Event Type | Actor | Description |")
        lines.append("|-----------|-----------|-------|-------------|")

        for entry in self.audit_trail.entries:
            lines.append(
                f"| {entry.timestamp} | {entry.event_type} "
                f"| {entry.actor} | {entry.description} |"
            )

        lines.append("\n## Data Integrity\n")
        hash_entries = [e for e in self.audit_trail.entries if e.data_hash]
        if hash_entries:
            for entry in hash_entries:
                lines.append(f"- **{entry.event_type}**: SHA-256 `{entry.data_hash}`")
        else:
            lines.append("No data integrity hashes recorded.")

        return "\n".join(lines)

    def generate_risk_analysis(self) -> str:
        """Generate a SaMD risk analysis document.

        Returns
        -------
        Markdown string with risk classification and mitigation strategies.
        """
        risk_label = _RISK_CLASS_LABELS[self.risk_class]
        lines = [self._header("Risk Analysis")]

        lines.append("\n## Risk Classification\n")
        lines.append(
            f"This software is classified as **{risk_label}** "
            f"under EU MDR 2017/745 Annex VIII Rule 11.\n"
        )

        lines.append("## Identified Risks\n")
        lines.append("| Risk ID | Description | Severity | Probability | Mitigation |")
        lines.append("|---------|-------------|----------|-------------|------------|")
        lines.append(
            "| R-001 | Incorrect segmentation leading to misdiagnosis "
            "| High | Medium | Human-in-the-loop review, uncertainty quantification |"
        )
        lines.append(
            "| R-002 | Out-of-distribution input data "
            "| Medium | Medium | Data validation gates, drift detection |"
        )
        lines.append(
            "| R-003 | Model degradation over time "
            "| Medium | Low | Continuous monitoring, retraining pipeline |"
        )

        lines.append("\n## Mitigation Strategies\n")
        lines.append(
            "1. **Uncertainty Quantification**: MC Dropout + conformal prediction sets"
        )
        lines.append("2. **Data Validation**: Great Expectations batch quality gates")
        lines.append(
            "3. **Drift Detection**: Statistical drift monitoring (KS test, PSI)"
        )
        lines.append(
            "4. **Audit Trail**: Full lifecycle traceability (IEC 62304 Clause 8)"
        )

        return "\n".join(lines)

    def generate_srs(self) -> str:
        """Generate a Software Requirements Specification template.

        Returns
        -------
        Markdown string following IEC 62304 Clause 5.2.
        """
        lines = [self._header("Software Requirements Specification")]

        lines.append("\n## 1. Scope\n")
        lines.append(
            f"{self.product_name} v{self.product_version} provides automated "
            f"3D biomedical vessel segmentation using deep learning.\n"
        )

        lines.append("## 2. Functional Requirements\n")
        lines.append("| Req ID | Description | Priority | Verification |")
        lines.append("|--------|-------------|----------|--------------|")
        lines.append("| FR-001 | Accept NIfTI input volumes | Must | Unit test |")
        lines.append(
            "| FR-002 | Produce segmentation masks with class probabilities "
            "| Must | Integration test |"
        )
        lines.append(
            "| FR-003 | Report uncertainty estimates per voxel | Should | Unit test |"
        )
        lines.append(
            "| FR-004 | Log all processing to audit trail | Must | Unit test |"
        )

        lines.append("\n## 3. Non-Functional Requirements\n")
        lines.append("| Req ID | Description | Metric |")
        lines.append("|--------|-------------|--------|")
        lines.append("| NFR-001 | Inference latency | < 30s per volume |")
        lines.append("| NFR-002 | Dice score on validation set | >= 0.80 |")
        lines.append("| NFR-003 | Data integrity | SHA-256 hash verification |")

        lines.append("\n## 4. Traceability\n")
        lines.append(
            "Requirements are traced to test cases via pytest markers "
            "and to audit trail events."
        )

        return "\n".join(lines)

    def generate_validation_summary(self) -> str:
        """Generate a validation summary from test evaluation events.

        Returns
        -------
        Markdown string summarizing test results.
        """
        lines = [self._header("Validation Summary")]

        test_events = [
            e for e in self.audit_trail.entries if e.event_type == "TEST_EVALUATION"
        ]

        if not test_events:
            lines.append("\nNo test evaluation events recorded in audit trail.\n")
            return "\n".join(lines)

        lines.append("\n## Test Results\n")
        for event in test_events:
            lines.append(f"### {event.description}\n")
            lines.append(f"- **Timestamp:** {event.timestamp}")
            lines.append(f"- **Actor:** {event.actor}")
            if event.metadata:
                for key, value in event.metadata.items():
                    if isinstance(value, float):
                        lines.append(f"- **{key}:** {value:.4f}")
                    else:
                        lines.append(f"- **{key}:** {value}")
            lines.append("")

        return "\n".join(lines)
