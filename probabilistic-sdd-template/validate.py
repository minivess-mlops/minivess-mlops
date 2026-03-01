#!/usr/bin/env python3
"""Probabilistic SDD Structural Validator.

Validates the integrity of a probabilistic Software Design Document (SDD)
instance. Designed to run as:
  1. Pre-commit hook (on staged SDD files)
  2. CI/CD check (on all SDD files)
  3. Manual validation (via `python validate.py --sdd-root <path>`)

Checks:
  1. DAG acyclicity: no cycles in the decision network
  2. Probability sums: all priors, conditional tables, archetype weights sum to 1.0
  3. Cross-reference resolution: all edges reference existing nodes
  4. Missing files: every node has a corresponding .decision.yaml file
  5. Schema compliance: required fields present (basic checks)

Exit codes:
  0 = all checks pass
  1 = FAIL-level violations (blocks commit)
  2 = WARN-level issues only (informational)

Usage:
  python validate.py --sdd-root <path-to-sdd>
  python validate.py --help
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TOLERANCE = 0.01  # Probability sum tolerance


@dataclass
class ValidationReport:
    """Structured report of SDD validation results."""

    dag_acyclicity: bool = True
    probability_sums: list[str] = field(default_factory=list)
    cross_references: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    schema_violations: list[str] = field(default_factory=list)
    overall_pass: bool = True

    def add_error(self, category: str, message: str) -> None:
        """Add an error to the appropriate category."""
        getattr(self, category).append(message)
        self.overall_pass = False

    def print_report(self) -> None:
        """Print structured validation report to stdout."""
        print("SDD Validation Report")
        print("=" * 50)

        dag_status = "PASS" if self.dag_acyclicity else "FAIL"
        print(f"DAG Acyclicity:        {dag_status}")

        prob_status = (
            "PASS"
            if not self.probability_sums
            else f"FAIL ({len(self.probability_sums)} violations)"
        )
        print(f"Probability Sums:      {prob_status}")

        ref_status = (
            "PASS"
            if not self.cross_references
            else f"FAIL ({len(self.cross_references)} broken refs)"
        )
        print(f"Cross-References:      {ref_status}")

        file_status = (
            "PASS"
            if not self.missing_files
            else f"FAIL ({len(self.missing_files)} missing)"
        )
        print(f"Missing Files:         {file_status}")

        schema_status = (
            "PASS"
            if not self.schema_violations
            else f"FAIL ({len(self.schema_violations)} violations)"
        )
        print(f"Schema Compliance:     {schema_status}")

        print("-" * 50)

        # Print error details
        all_issues = (
            self.probability_sums
            + self.cross_references
            + self.missing_files
            + self.schema_violations
        )
        for issue in all_issues:
            print(f"  {issue}")
        if not self.dag_acyclicity:
            print("  FAIL: Cycle detected in decision network DAG")

        overall = "PASS" if self.overall_pass else "FAIL"
        print(f"\nOverall:               {overall}")


def load_yaml(path: Path) -> dict[str, Any] | None:
    """Load a YAML file, returning None if it doesn't exist."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict):
        return data
    return None


def find_decision_files(decisions_dir: Path) -> list[Path]:
    """Find all .decision.yaml files in the decisions directory."""
    files = []
    for level_dir in sorted(decisions_dir.iterdir()):
        if level_dir.is_dir() and level_dir.name.startswith("L"):
            files.extend(sorted(level_dir.glob("*.decision.yaml")))
    return files


def check_dag_acyclicity(network: dict[str, Any]) -> bool:
    """Check that the decision network is a valid DAG (no cycles).

    Uses Kahn's algorithm for topological sorting.
    Returns True if acyclic, False if cycle detected.
    """
    nodes = {n["id"] for n in network.get("nodes", [])}
    edges = network.get("edges", [])

    if not edges:
        return True

    # Build adjacency list and in-degree count
    adj: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {n: 0 for n in nodes}

    for edge in edges:
        src = edge["from"]
        dst = edge["to"]
        adj[src].append(dst)
        if dst in in_degree:
            in_degree[dst] += 1

    # Kahn's algorithm
    queue = [n for n in nodes if in_degree.get(n, 0) == 0]
    visited = 0

    while queue:
        node = queue.pop(0)
        visited += 1
        for neighbor in adj.get(node, []):
            if neighbor in in_degree:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

    return visited == len(nodes)


def check_probability_sums(decision: dict[str, Any], filename: str) -> list[str]:
    """Check that all probability distributions sum to 1.0."""
    errors = []

    # Check option priors
    options = decision.get("options", [])
    if options:
        total = sum(opt.get("prior_probability", 0) for opt in options)
        if abs(total - 1.0) > TOLERANCE:
            errors.append(
                f"FAIL: {filename}: option prior_probability sum = {total:.4f} (expected 1.0)"
            )

    # Check conditional tables
    for cond in decision.get("conditional_on", []):
        for row in cond.get("conditional_table", []):
            probs = row.get("then_probabilities", {})
            if probs:
                total = sum(probs.values())
                if abs(total - 1.0) > TOLERANCE:
                    parent = row.get("given_parent_option", "?")
                    errors.append(
                        f"FAIL: {filename}: conditional_table[{parent}] sum = {total:.4f}"
                    )

    # Check archetype weights
    for arch_name, arch_data in decision.get("archetype_weights", {}).items():
        overrides = arch_data.get("probability_overrides", {})
        if overrides:
            total = sum(overrides.values())
            if abs(total - 1.0) > TOLERANCE:
                errors.append(
                    f"FAIL: {filename}: archetype '{arch_name}' overrides sum = {total:.4f}"
                )

    return errors


def check_cross_references(network: dict[str, Any], node_ids: set[str]) -> list[str]:
    """Check that all edge references point to existing nodes."""
    errors = []
    for edge in network.get("edges", []):
        src = edge.get("from", "")
        dst = edge.get("to", "")
        if src not in node_ids:
            errors.append(f"FAIL: Edge from unknown node '{src}'")
        if dst not in node_ids:
            errors.append(f"FAIL: Edge to unknown node '{dst}'")
    return errors


def check_missing_files(network: dict[str, Any], decisions_dir: Path) -> list[str]:
    """Check that every node in the network has a corresponding .decision.yaml file."""
    errors = []
    for node in network.get("nodes", []):
        file_path = decisions_dir / node.get("file", "")
        if not file_path.exists():
            errors.append(
                f"FAIL: Node '{node['id']}' references missing file: {node.get('file', '')}"
            )
    return errors


def check_schema_compliance(decision: dict[str, Any], filename: str) -> list[str]:
    """Basic schema compliance checks (required fields)."""
    errors = []
    required = [
        "decision_id",
        "title",
        "description",
        "decision_level",
        "status",
        "options",
        "volatility",
        "domain_applicability",
    ]
    for field_name in required:
        if field_name not in decision:
            errors.append(f"FAIL: {filename}: missing required field '{field_name}'")

    # Check options have required fields
    for i, opt in enumerate(decision.get("options", [])):
        for opt_field in [
            "option_id",
            "title",
            "description",
            "prior_probability",
            "status",
        ]:
            if opt_field not in opt:
                errors.append(f"FAIL: {filename}: option[{i}] missing '{opt_field}'")

    return errors


def validate_sdd(sdd_root: Path) -> ValidationReport:
    """Run all validation checks on an SDD instance.

    Parameters
    ----------
    sdd_root : Path
        Root directory of the SDD instance. Expected structure:
        sdd_root/
          decisions/
            _network.yaml
            _schema.yaml
            L1-*/  L2-*/  ... L5-*/
              *.decision.yaml
          bibliography.yaml

    Returns
    -------
    ValidationReport
        Structured report with pass/fail status per check category.
    """
    report = ValidationReport()

    decisions_dir = sdd_root / "decisions"
    network_path = decisions_dir / "_network.yaml"

    # Load network
    network_data = load_yaml(network_path)
    if not network_data or "network" not in network_data:
        report.add_error(
            "schema_violations", "FAIL: _network.yaml not found or invalid"
        )
        return report

    network = network_data["network"]
    node_ids = {n["id"] for n in network.get("nodes", [])}

    # Check 1: DAG Acyclicity
    if not check_dag_acyclicity(network):
        report.dag_acyclicity = False
        report.overall_pass = False

    # Check 2: Cross-references
    xref_errors = check_cross_references(network, node_ids)
    for err in xref_errors:
        report.add_error("cross_references", err)

    # Check 3: Missing files
    file_errors = check_missing_files(network, decisions_dir)
    for err in file_errors:
        report.add_error("missing_files", err)

    # Check 4-5: Per-file checks (probability sums + schema)
    decision_files = find_decision_files(decisions_dir)
    for path in decision_files:
        decision = load_yaml(path)
        if not decision:
            report.add_error("schema_violations", f"FAIL: {path.name}: invalid YAML")
            continue

        # Probability sums
        prob_errors = check_probability_sums(decision, path.name)
        for err in prob_errors:
            report.add_error("probability_sums", err)

        # Schema compliance
        schema_errors = check_schema_compliance(decision, path.name)
        for err in schema_errors:
            report.add_error("schema_violations", err)

    return report


def main() -> int:
    """CLI entry point for the SDD validator."""
    parser = argparse.ArgumentParser(
        description="Validate a probabilistic SDD (Software Design Document) instance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate.py --sdd-root ./my-sdd
  python validate.py --sdd-root ./examples/minivess
        """,
    )
    parser.add_argument(
        "--sdd-root",
        type=Path,
        required=True,
        help="Root directory of the SDD instance to validate",
    )
    args = parser.parse_args()

    report = validate_sdd(args.sdd_root)
    report.print_report()

    if not report.overall_pass:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
