"""Compliance — Audit trails, model cards, and SaMD (IEC 62304) lifecycle hooks."""

from __future__ import annotations

from minivess.compliance.audit import AuditEntry, AuditTrail
from minivess.compliance.fairness import (
    FairnessReport,
    SubgroupMetrics,
    compute_disparity,
    evaluate_subgroup_fairness,
    generate_audit_report,
)
from minivess.compliance.model_card import ModelCard
from minivess.compliance.regulatory_docs import (
    RegulatoryDocGenerator,
    SaMDRiskClass,
)

__all__ = [
    "AuditEntry",
    "AuditTrail",
    "FairnessReport",
    "ModelCard",
    "RegulatoryDocGenerator",
    "SaMDRiskClass",
    "SubgroupMetrics",
    "compute_disparity",
    "evaluate_subgroup_fairness",
    "generate_audit_report",
]
