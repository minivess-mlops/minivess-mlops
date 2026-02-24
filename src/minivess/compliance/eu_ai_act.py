"""EU AI Act compliance checklist and risk classification.

Implements the four-tier risk classification from the EU AI Act (Regulation
2024/1689) and provides a gap-analysis checklist for high-risk medical AI
systems such as SaMD (Software as a Medical Device).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum


class EUAIActRiskLevel(StrEnum):
    """EU AI Act four-tier risk classification."""

    UNACCEPTABLE = "unacceptable"
    HIGH = "high"
    LIMITED = "limited"
    MINIMAL = "minimal"


def classify_risk_level(
    *,
    is_medical_device: bool = False,
) -> EUAIActRiskLevel:
    """Determine EU AI Act risk level.

    Medical devices (SaMD) are classified as HIGH risk under Annex I,
    Section A of the EU AI Act.

    Parameters
    ----------
    is_medical_device:
        Whether the AI system qualifies as a medical device.
    """
    if is_medical_device:
        return EUAIActRiskLevel.HIGH
    return EUAIActRiskLevel.LIMITED


@dataclass
class EUAIActChecklist:
    """EU AI Act compliance checklist for an AI system.

    Parameters
    ----------
    system_name:
        Name of the AI system.
    risk_level:
        EU AI Act risk tier (unacceptable, high, limited, minimal).
    intended_purpose:
        Description of the system's intended purpose.
    data_governance:
        Data governance measures (Article 10).
    transparency:
        Transparency provisions (Article 13).
    human_oversight:
        Human oversight mechanisms (Article 14).
    robustness:
        Accuracy, robustness, and cybersecurity measures (Article 15).
    risk_management:
        Risk management system description (Article 9).
    technical_documentation:
        Technical documentation (Article 11).
    record_keeping:
        Automatic logging / record-keeping (Article 12).
    conformity_assessment:
        Conformity assessment procedure (Article 43).
    """

    system_name: str
    risk_level: str
    intended_purpose: str
    data_governance: str = ""
    transparency: str = ""
    human_oversight: str = ""
    robustness: str = ""
    risk_management: str = ""
    technical_documentation: str = ""
    record_keeping: str = ""
    conformity_assessment: str = ""

    def to_markdown(self) -> str:
        """Generate EU AI Act compliance checklist as markdown."""
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        sections = [
            "# EU AI Act Compliance Checklist",
            "",
            f"**System:** {self.system_name}",
            f"**Risk Level:** {self.risk_level.upper()}",
            f"**Generated:** {now}",
            "",
            "## 1. Intended Purpose",
            "",
            self.intended_purpose,
        ]

        high_risk_articles = [
            ("2. Risk Management (Article 9)", self.risk_management),
            ("3. Data Governance (Article 10)", self.data_governance),
            ("4. Technical Documentation (Article 11)", self.technical_documentation),
            ("5. Record-Keeping (Article 12)", self.record_keeping),
            ("6. Transparency (Article 13)", self.transparency),
            ("7. Human Oversight (Article 14)", self.human_oversight),
            ("8. Robustness (Article 15)", self.robustness),
            ("9. Conformity Assessment (Article 43)", self.conformity_assessment),
        ]

        for heading, content in high_risk_articles:
            sections.extend(["", f"## {heading}", ""])
            if content:
                sections.append(content)
            else:
                sections.append("*Not provided — gap identified.*")

        sections.append("")
        return "\n".join(sections)


def generate_compliance_report(
    *,
    system_name: str,
    intended_purpose: str,
    is_medical_device: bool = False,
    data_governance: str = "",
    transparency: str = "",
    human_oversight: str = "",
    robustness: str = "",
    risk_management: str = "",
) -> str:
    """Generate an EU AI Act compliance gap-analysis report.

    Parameters
    ----------
    system_name:
        Name of the AI system.
    intended_purpose:
        System's intended purpose.
    is_medical_device:
        Whether the system qualifies as a medical device.
    """
    risk_level = classify_risk_level(is_medical_device=is_medical_device)
    checklist = EUAIActChecklist(
        system_name=system_name,
        risk_level=risk_level.value,
        intended_purpose=intended_purpose,
        data_governance=data_governance,
        transparency=transparency,
        human_oversight=human_oversight,
        robustness=robustness,
        risk_management=risk_management,
    )
    return checklist.to_markdown()
