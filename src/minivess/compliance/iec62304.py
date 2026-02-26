"""IEC 62304 compliance framework for medical device software lifecycle.

Implements software safety classification (Class A/B/C), lifecycle stage
tracking, requirements traceability matrix, and predetermined change
control plan (PCCP) per FDA December 2024 guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class SoftwareSafetyClass(StrEnum):
    """IEC 62304 software safety classification.

    - Class A: No injury or damage to health possible
    - Class B: Non-serious injury possible
    - Class C: Death or serious injury possible
    """

    CLASS_A = "class_a"
    CLASS_B = "class_b"
    CLASS_C = "class_c"


class LifecycleStage(StrEnum):
    """IEC 62304 software lifecycle stages (Clause 5–9)."""

    DEVELOPMENT = "development"
    VERIFICATION = "verification"
    VALIDATION = "validation"
    RELEASE = "release"
    MAINTENANCE = "maintenance"


@dataclass
class TraceabilityEntry:
    """Single entry in the requirements traceability matrix.

    Parameters
    ----------
    requirement_id:
        Requirement identifier (e.g., FR-001).
    description:
        Brief description of the requirement.
    implementation_ref:
        Reference to the implementation (file, function, class).
    test_ref:
        Reference to the verification test. Empty string if untested.
    """

    requirement_id: str
    description: str
    implementation_ref: str
    test_ref: str = ""


class TraceabilityMatrix:
    """Requirements traceability matrix (IEC 62304 Clause 5.7).

    Maps requirements → implementation → tests for bidirectional
    traceability as required by IEC 62304 and ISO 13485.
    """

    def __init__(self) -> None:
        self.entries: list[TraceabilityEntry] = []

    def add_entry(
        self,
        *,
        requirement_id: str,
        description: str,
        implementation_ref: str,
        test_ref: str = "",
    ) -> TraceabilityEntry:
        """Add a traceability record."""
        entry = TraceabilityEntry(
            requirement_id=requirement_id,
            description=description,
            implementation_ref=implementation_ref,
            test_ref=test_ref,
        )
        self.entries.append(entry)
        return entry

    def coverage_report(self) -> dict[str, Any]:
        """Compute traceability coverage statistics.

        Returns
        -------
        Dictionary with total, tested, untested counts and gap list.
        """
        tested = [e for e in self.entries if e.test_ref]
        untested = [e for e in self.entries if not e.test_ref]
        return {
            "total": len(self.entries),
            "tested": len(tested),
            "untested": len(untested),
            "coverage": len(tested) / len(self.entries) if self.entries else 0.0,
            "gaps": [e.requirement_id for e in untested],
        }

    def to_markdown(self) -> str:
        """Generate traceability matrix as markdown."""
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        sections = [
            "# Requirements Traceability Matrix",
            "",
            f"**Generated:** {now}",
            "",
        ]

        if not self.entries:
            sections.append("No requirements tracked.")
            sections.append("")
            return "\n".join(sections)

        sections.extend(
            [
                "| Req ID | Description | Implementation | Test |",
                "|--------|-------------|----------------|------|",
            ]
        )

        for entry in self.entries:
            test_cell = entry.test_ref if entry.test_ref else "*GAP*"
            sections.append(
                f"| {entry.requirement_id} | {entry.description} "
                f"| {entry.implementation_ref} | {test_cell} |"
            )

        # Coverage summary
        report = self.coverage_report()
        sections.extend(
            [
                "",
                "## Coverage Summary",
                "",
                f"- **Total Requirements:** {report['total']}",
                f"- **Tested:** {report['tested']}",
                f"- **Untested:** {report['untested']}",
                f"- **Coverage:** {report['coverage']:.0%}",
            ]
        )

        if report["gaps"]:
            sections.extend(
                [
                    "",
                    "## Gaps",
                    "",
                ]
            )
            for gap_id in report["gaps"]:
                sections.append(f"- {gap_id}")

        sections.append("")
        return "\n".join(sections)


@dataclass
class PCCPTemplate:
    """Predetermined Change Control Plan (FDA December 2024 guidance).

    Documents permitted changes that do not require a new regulatory
    submission, along with verification methods for each change type.

    Parameters
    ----------
    product_name:
        Name of the SaMD product.
    product_version:
        Current version string.
    permitted_changes:
        List of pre-approved change descriptions.
    """

    product_name: str
    product_version: str
    permitted_changes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate PCCP document as markdown."""
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        sections = [
            "# Predetermined Change Control Plan (PCCP)",
            "",
            f"**Product:** {self.product_name}",
            f"**Version:** {self.product_version}",
            f"**Generated:** {now}",
            "",
            "## Scope",
            "",
            "This PCCP documents modifications to the AI/ML-enabled SaMD that are "
            "pre-specified and do not require a new regulatory submission, per "
            "FDA guidance (December 2024).",
            "",
            "## Permitted Changes",
            "",
        ]

        if self.permitted_changes:
            for idx, change in enumerate(self.permitted_changes, 1):
                sections.append(f"{idx}. {change}")
        else:
            sections.append("No permitted changes defined.")

        sections.extend(
            [
                "",
                "## Verification Protocol",
                "",
                "All permitted changes must satisfy:",
                "",
                "1. Automated regression test suite passes (>= 95% coverage)",
                "2. Performance metrics remain within predefined acceptance criteria",
                "3. Audit trail documents the change with full traceability",
                "4. Risk analysis updated if change affects safety classification",
                "",
            ]
        )

        return "\n".join(sections)
