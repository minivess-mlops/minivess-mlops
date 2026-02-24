"""Compliance — Audit trails, model cards, and SaMD (IEC 62304) lifecycle hooks."""

from __future__ import annotations

from minivess.compliance.audit import AuditEntry, AuditTrail
from minivess.compliance.eu_ai_act import (
    EUAIActChecklist,
    EUAIActRiskLevel,
    classify_risk_level,
    generate_compliance_report,
)
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
from minivess.compliance.reporting_templates import (
    ConsortAIChecklist,
    MiClearLLMChecklist,
    generate_consort_ai_report,
    generate_miclear_llm_report,
)

__all__ = [
    "AuditEntry",
    "AuditTrail",
    "ConsortAIChecklist",
    "EUAIActChecklist",
    "EUAIActRiskLevel",
    "FairnessReport",
    "MiClearLLMChecklist",
    "ModelCard",
    "RegulatoryDocGenerator",
    "SaMDRiskClass",
    "SubgroupMetrics",
    "classify_risk_level",
    "compute_disparity",
    "evaluate_subgroup_fairness",
    "generate_audit_report",
    "generate_compliance_report",
    "generate_consort_ai_report",
    "generate_miclear_llm_report",
]
