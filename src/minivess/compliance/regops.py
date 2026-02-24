"""RegOps CI/CD pipeline extension for regulatory artifact generation.

Orchestrates automated generation of IEC 62304 regulatory documents,
EU AI Act compliance reports, and audit trails from CI/CD context
(Lähteenmäki et al., 2023; Rosmarino et al., 2025).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from minivess.compliance.audit import AuditEntry, AuditTrail

if TYPE_CHECKING:
    from pathlib import Path
from minivess.compliance.eu_ai_act import generate_compliance_report
from minivess.compliance.regulatory_docs import RegulatoryDocGenerator, SaMDRiskClass


@dataclass
class CIContext:
    """CI/CD environment context for regulatory traceability.

    Parameters
    ----------
    commit_sha:
        Git commit SHA triggering the pipeline.
    actor:
        User or bot that triggered the pipeline.
    ref:
        Git ref (branch or tag).
    run_id:
        CI/CD pipeline run identifier.
    repository:
        Repository identifier (owner/name).
    """

    commit_sha: str
    actor: str = "ci"
    ref: str = ""
    run_id: str = ""
    repository: str = ""

    @classmethod
    def from_env(cls) -> CIContext:
        """Create CIContext from GitHub Actions environment variables."""
        return cls(
            commit_sha=os.environ.get("GITHUB_SHA", "unknown"),
            actor=os.environ.get("GITHUB_ACTOR", "ci"),
            ref=os.environ.get("GITHUB_REF", ""),
            run_id=os.environ.get("GITHUB_RUN_ID", ""),
            repository=os.environ.get("GITHUB_REPOSITORY", ""),
        )


def generate_ci_audit_entry(
    trail: AuditTrail,
    ci_context: CIContext,
) -> AuditEntry:
    """Log a CI/CD pipeline event to the audit trail.

    Parameters
    ----------
    trail:
        Audit trail to append to.
    ci_context:
        CI/CD environment context.
    """
    return trail.log_event(
        "CI_PIPELINE",
        f"CI/CD pipeline run for commit {ci_context.commit_sha}",
        actor=ci_context.actor,
        metadata={
            "commit_sha": ci_context.commit_sha,
            "ref": ci_context.ref,
            "run_id": ci_context.run_id,
            "repository": ci_context.repository,
        },
    )


@dataclass
class RegOpsPipeline:
    """Orchestrates regulatory artifact generation from CI/CD context.

    Parameters
    ----------
    ci_context:
        CI/CD environment metadata.
    product_name:
        Name of the SaMD product.
    product_version:
        Version string for the product.
    risk_class:
        EU MDR risk classification.
    is_medical_device:
        Whether the system is a medical device (for EU AI Act).
    """

    ci_context: CIContext
    product_name: str
    product_version: str
    risk_class: SaMDRiskClass = SaMDRiskClass.CLASS_I
    is_medical_device: bool = True
    _trail: AuditTrail = field(default_factory=AuditTrail, init=False, repr=False)

    def generate_artifacts(self, output_dir: Path) -> list[Path]:
        """Generate all regulatory artifacts and write to output directory.

        Parameters
        ----------
        output_dir:
            Directory to write generated documents to.

        Returns
        -------
        List of paths to generated files (the manifest).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest: list[Path] = []

        # 1. Log CI pipeline event
        generate_ci_audit_entry(self._trail, self.ci_context)

        # 2. Generate IEC 62304 regulatory documents
        doc_gen = RegulatoryDocGenerator(
            audit_trail=self._trail,
            product_name=self.product_name,
            product_version=self.product_version,
            risk_class=self.risk_class,
        )

        docs = {
            "design-history.md": doc_gen.generate_design_history(),
            "risk-analysis.md": doc_gen.generate_risk_analysis(),
            "srs.md": doc_gen.generate_srs(),
            "validation-summary.md": doc_gen.generate_validation_summary(),
        }

        for filename, content in docs.items():
            path = output_dir / filename
            path.write_text(content, encoding="utf-8")
            manifest.append(path)

        # 3. Generate EU AI Act compliance report
        eu_report = generate_compliance_report(
            system_name=self.product_name,
            intended_purpose=f"{self.product_name} v{self.product_version} — "
            "3D biomedical vessel segmentation",
            is_medical_device=self.is_medical_device,
        )
        eu_path = output_dir / "eu-ai-act-compliance.md"
        eu_path.write_text(eu_report, encoding="utf-8")
        manifest.append(eu_path)

        # 4. Save audit trail
        trail_path = output_dir / "audit-trail.json"
        self._trail.save(trail_path)
        manifest.append(trail_path)

        return manifest
