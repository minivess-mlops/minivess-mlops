#!/usr/bin/env python3
"""PRD Citation Integrity Validator.

Validates academic citation standards across the MinIVess MLOps v2
hierarchical probabilistic PRD. Designed to run as:
  1. Pre-commit hook (on staged PRD files)
  2. CI/CD check (on all PRD files)
  3. Manual validation (via `uv run python scripts/validate_prd_citations.py`)

Checks:
  1. Bibliography resolution: all citation_keys in decision files exist in bibliography.yaml
  2. No citation loss: detect removed citations vs. git HEAD (pre-commit mode)
  3. In-text citation format: author-year pattern in rationale fields
  4. Completeness: every decision file should have at least one reference
  5. Cross-topic consistency: bibliography topics match actual usage

Exit codes:
  0 = all checks pass
  1 = FAIL-level violations (blocks commit)
  2 = WARN-level issues only (informational)

Usage:
  uv run python scripts/validate_prd_citations.py [--strict] [--check-loss]
    --strict     Treat warnings as failures (exit 1)
    --check-loss Compare against git HEAD to detect citation removals
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# pyyaml is an existing dependency via hydra-zen/omegaconf
import yaml

PRD_ROOT = Path(__file__).resolve().parent.parent / "docs" / "prd"
DECISIONS_DIR = PRD_ROOT / "decisions"
BIBLIOGRAPHY_PATH = PRD_ROOT / "bibliography.yaml"

# Author-year pattern: matches (Surname, 2024), (Surname et al., 2024),
# Surname (2024), Surname et al. (2024), Surname & Other (2024)
AUTHOR_YEAR_PATTERN = re.compile(
    r"(?:"
    r"[A-Z][a-z]+(?:\s+et\s+al\.)?(?:\s*&\s*[A-Z][a-z]+)?\s*\(\d{4}\)"  # Surname (2024)
    r"|"
    r"\([A-Z][a-z]+(?:\s+et\s+al\.)?(?:\s*&\s*[A-Z][a-z]+)?,?\s*\d{4}\)"  # (Surname, 2024)
    r")"
)


def load_yaml(path: Path) -> dict[str, Any] | None:
    """Load a YAML file, returning None if it doesn't exist or isn't a dict."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict):
        return data
    return None


def find_decision_files() -> list[Path]:
    """Find all .decision.yaml files in the PRD."""
    files = []
    for level_dir in sorted(DECISIONS_DIR.iterdir()):
        if level_dir.is_dir() and level_dir.name.startswith("L"):
            files.extend(sorted(level_dir.glob("*.decision.yaml")))
    return files


def load_bibliography() -> dict[str, dict[str, Any]]:
    """Load bibliography.yaml and return a dict keyed by citation_key."""
    data = load_yaml(BIBLIOGRAPHY_PATH)
    if not data or "bibliography" not in data:
        return {}
    return {entry["citation_key"]: entry for entry in data["bibliography"]}


def extract_citation_keys(decision: dict[str, Any]) -> list[str]:
    """Extract all citation_keys from a decision file's references array."""
    refs = decision.get("references", [])
    if not refs or not isinstance(refs, list):
        return []
    keys = []
    for ref in refs:
        if isinstance(ref, dict) and "citation_key" in ref:
            keys.append(ref["citation_key"])
        elif isinstance(ref, str):
            keys.append(ref)
    return keys


def check_bibliography_resolution(
    decision_files: list[Path], bib_keys: set[str]
) -> list[str]:
    """Check that all citation_keys in decision files exist in bibliography."""
    errors = []
    for path in decision_files:
        decision = load_yaml(path)
        if not decision:
            continue
        for key in extract_citation_keys(decision):
            if key not in bib_keys:
                errors.append(
                    f"FAIL: {path.name}: citation_key '{key}' not found in bibliography.yaml"
                )
    return errors


def check_completeness(decision_files: list[Path]) -> list[str]:
    """Check that every decision file has at least one reference."""
    warnings = []
    for path in decision_files:
        decision = load_yaml(path)
        if not decision:
            continue
        keys = extract_citation_keys(decision)
        if not keys:
            warnings.append(
                f"WARN: {path.name}: no references — every decision should cite evidence"
            )
    return warnings


def check_in_text_citations(decision_files: list[Path]) -> list[str]:
    """Check that rationale fields contain author-year citations."""
    warnings = []
    for path in decision_files:
        decision = load_yaml(path)
        if not decision:
            continue
        rationale = decision.get("rationale", "")
        if rationale and not AUTHOR_YEAR_PATTERN.search(rationale):
            warnings.append(
                f"WARN: {path.name}: rationale has no author-year citations"
            )
    return warnings


def check_bibliography_quality(bib_entries: dict[str, dict[str, Any]]) -> list[str]:
    """Check that bibliography entries have required fields."""
    issues = []
    for key, entry in bib_entries.items():
        if not entry.get("authors"):
            issues.append(f"WARN: bibliography '{key}': missing authors")
        if not entry.get("title"):
            issues.append(f"WARN: bibliography '{key}': missing title")
        if not entry.get("doi") and not entry.get("url"):
            issues.append(f"WARN: bibliography '{key}': missing both doi and url")
        if not entry.get("inline_citation"):
            issues.append(f"WARN: bibliography '{key}': missing inline_citation")
    return issues


def check_cross_topic_consistency(
    decision_files: list[Path], bib_entries: dict[str, dict[str, Any]]
) -> list[str]:
    """Check that bibliography topics match actual decision file usage."""
    # Build actual usage map: citation_key -> set of decision_ids that cite it
    actual_usage: dict[str, set[str]] = {}
    for path in decision_files:
        decision = load_yaml(path)
        if not decision:
            continue
        decision_id = decision.get("decision_id", path.stem)
        for key in extract_citation_keys(decision):
            actual_usage.setdefault(key, set()).add(decision_id)

    warnings = []
    for key, entry in bib_entries.items():
        declared_topics = set(entry.get("topics", []))
        actual_topics = actual_usage.get(key, set())
        missing_in_bib = actual_topics - declared_topics
        if missing_in_bib:
            warnings.append(
                f"WARN: bibliography '{key}' topics missing: {sorted(missing_in_bib)}"
            )
    return warnings


def check_citation_loss(decision_files: list[Path]) -> list[str]:
    """Compare current references against git HEAD to detect removals."""
    errors = []
    for path in decision_files:
        # Get current keys
        current = load_yaml(path)
        if not current:
            continue
        current_keys = set(extract_citation_keys(current))

        # Get HEAD keys via git show
        relative = path.relative_to(Path(__file__).resolve().parent.parent)
        try:
            result = subprocess.run(
                ["git", "show", f"HEAD:{relative}"],
                capture_output=True,
                text=True,
                cwd=path.parent,
                check=False,
            )
            if result.returncode != 0:
                continue  # File is new, no previous state
            head_data = yaml.safe_load(result.stdout)
            if not head_data:
                continue
            head_keys = set(extract_citation_keys(head_data))
        except (subprocess.SubprocessError, yaml.YAMLError):
            continue

        removed = head_keys - current_keys
        if removed:
            errors.append(
                f"FAIL: {path.name}: citations REMOVED (requires user approval): {sorted(removed)}"
            )
    return errors


def main() -> int:
    """Run all citation validation checks."""
    strict = "--strict" in sys.argv
    check_loss = "--check-loss" in sys.argv

    print("PRD Citation Validation")
    print("=" * 50)

    # Load bibliography
    bib_entries = load_bibliography()
    if not bib_entries:
        print("FAIL: bibliography.yaml not found or empty")
        return 1
    bib_keys = set(bib_entries.keys())
    print(f"Bibliography: {len(bib_entries)} entries loaded")

    # Find decision files
    decision_files = find_decision_files()
    print(f"Decision files: {len(decision_files)} found")
    print("-" * 50)

    all_errors: list[str] = []
    all_warnings: list[str] = []

    # Check 1: Bibliography resolution
    errors = check_bibliography_resolution(decision_files, bib_keys)
    all_errors.extend(errors)
    status = "FAIL" if errors else "PASS"
    print(f"Bibliography Resolution: {status} ({len(errors)} unresolved)")

    # Check 2: Completeness
    warnings = check_completeness(decision_files)
    all_warnings.extend(warnings)
    status = "WARN" if warnings else "PASS"
    print(f"Citation Completeness:   {status} ({len(warnings)} missing)")

    # Check 3: In-text citations
    warnings = check_in_text_citations(decision_files)
    all_warnings.extend(warnings)
    status = "WARN" if warnings else "PASS"
    print(f"In-Text Format:          {status} ({len(warnings)} issues)")

    # Check 4: Bibliography quality
    warnings = check_bibliography_quality(bib_entries)
    all_warnings.extend(warnings)
    status = "WARN" if warnings else "PASS"
    print(f"Bibliography Quality:    {status} ({len(warnings)} issues)")

    # Check 5: Cross-topic consistency
    warnings = check_cross_topic_consistency(decision_files, bib_entries)
    all_warnings.extend(warnings)
    status = "WARN" if warnings else "PASS"
    print(f"Cross-Topic:             {status} ({len(warnings)} mismatches)")

    # Check 6: Citation loss (optional, git-dependent)
    if check_loss:
        errors = check_citation_loss(decision_files)
        all_errors.extend(errors)
        status = "FAIL" if errors else "PASS"
        print(f"No Citation Loss:        {status} ({len(errors)} removals)")

    print("-" * 50)

    # Print details
    for msg in all_errors:
        print(f"  {msg}")
    for msg in all_warnings:
        print(f"  {msg}")

    # Determine exit code
    if all_errors:
        print(
            f"\nOVERALL: FAIL ({len(all_errors)} errors, {len(all_warnings)} warnings)"
        )
        return 1
    elif all_warnings and strict:
        print(f"\nOVERALL: FAIL (strict mode, {len(all_warnings)} warnings)")
        return 1
    elif all_warnings:
        print(f"\nOVERALL: PASS with warnings ({len(all_warnings)} warnings)")
        return 2
    else:
        print("\nOVERALL: PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
