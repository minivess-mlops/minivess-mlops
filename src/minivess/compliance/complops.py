"""ComplOps regulatory automation tooling (Batista et al., 2025).

Provides automated regulatory artifact generation for FDA 510(k),
EU MDR technical files, and cross-framework compliance gap analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class RegulatoryTemplate(StrEnum):
    """Supported regulatory submission templates."""

    FDA_510K = "fda_510k"
    EU_MDR_TECH_FILE = "eu_mdr_tech_file"
    IEC_62304_FULL = "iec_62304_full"


@dataclass
class ComplianceCheckResult:
    """Result of an automated compliance gap assessment.

    Parameters
    ----------
    template:
        Regulatory template checked against.
    total_requirements:
        Total number of requirements in the template.
    satisfied:
        Number of requirements currently satisfied.
    gaps:
        List of unsatisfied requirement descriptions.
    recommendations:
        Suggested actions to close gaps.
    """

    template: str
    total_requirements: int
    satisfied: int
    gaps: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Compliance score as a fraction (0.0–1.0)."""
        if self.total_requirements == 0:
            return 0.0
        return self.satisfied / self.total_requirements


def generate_510k_summary(
    *,
    device_name: str,
    predicate_device: str,
    intended_use: str,
    technological_characteristics: str,
    performance_data: str = "",
    biocompatibility: str = "",
    software_description: str = "",
) -> str:
    """Generate an FDA 510(k) predicate comparison summary.

    Parameters
    ----------
    device_name:
        Name of the subject device.
    predicate_device:
        Name or 510(k) number of the predicate device.
    intended_use:
        Intended use statement.
    technological_characteristics:
        Description of the technology.
    """
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    sections = [
        "# FDA 510(k) Summary",
        "",
        f"**Device:** {device_name}",
        f"**Generated:** {now}",
        "",
        "## 1. Intended Use",
        "",
        intended_use,
        "",
        "## 2. Predicate Device Comparison",
        "",
        f"**Predicate:** {predicate_device}",
        "",
        "## 3. Technological Characteristics",
        "",
        technological_characteristics,
    ]

    if performance_data:
        sections.extend(["", "## 4. Performance Data", "", performance_data])

    if biocompatibility:
        sections.extend(["", "## 5. Biocompatibility", "", biocompatibility])

    if software_description:
        sections.extend(
            [
                "",
                "## 6. Software Description (IEC 62304)",
                "",
                software_description,
            ]
        )

    sections.append("")
    return "\n".join(sections)


def generate_eu_mdr_technical_file(
    *,
    device_name: str,
    manufacturer: str,
    device_class: str,
    intended_purpose: str,
    clinical_evaluation: str = "",
    risk_management: str = "",
    design_verification: str = "",
) -> str:
    """Generate an EU MDR Annex II/III technical documentation template.

    Parameters
    ----------
    device_name:
        Name of the medical device.
    manufacturer:
        Legal manufacturer name.
    device_class:
        EU MDR classification (I, IIa, IIb, III).
    intended_purpose:
        Intended purpose of the device.
    """
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    sections = [
        "# EU MDR Technical Documentation",
        "",
        f"**Device:** {device_name}",
        f"**Manufacturer:** {manufacturer}",
        f"**Classification:** Class {device_class}",
        f"**Generated:** {now}",
        "",
        "## Annex II — Technical Documentation",
        "",
        "### 1. Device Description and Specification",
        "",
        f"**Intended Purpose:** {intended_purpose}",
        "",
        "### 2. Information Supplied by the Manufacturer",
        "",
        "Label and instructions for use per Annex I Chapter III.",
    ]

    if clinical_evaluation:
        sections.extend(
            [
                "",
                "### 3. Clinical Evaluation (Annex XIV)",
                "",
                clinical_evaluation,
            ]
        )

    if risk_management:
        sections.extend(
            [
                "",
                "### 4. Risk Management (ISO 14971)",
                "",
                risk_management,
            ]
        )

    if design_verification:
        sections.extend(
            [
                "",
                "### 5. Design Verification and Validation",
                "",
                design_verification,
            ]
        )

    sections.extend(
        [
            "",
            "## Annex III — Post-Market Surveillance",
            "",
            "Post-market surveillance plan per Article 83–86.",
            "",
        ]
    )

    return "\n".join(sections)


def assess_compliance_gaps(
    *,
    template: str,
    provided_items: dict[str, bool],
) -> ComplianceCheckResult:
    """Assess compliance gaps against a regulatory template.

    Parameters
    ----------
    template:
        Regulatory template identifier.
    provided_items:
        Item name → whether it has been provided (True/False).
    """
    total = len(provided_items)
    satisfied = sum(1 for v in provided_items.values() if v)
    gaps = [name for name, present in provided_items.items() if not present]
    recommendations = [f"Provide documentation for: {gap}" for gap in gaps]

    return ComplianceCheckResult(
        template=template,
        total_requirements=total,
        satisfied=satisfied,
        gaps=gaps,
        recommendations=recommendations,
    )
