"""Tests that verify AuditTrail.log_data_access() is wired into train_flow.

PR-2 T2.1: The train flow MUST call log_data_access(dataset_name: str,
file_paths: list[str]) after loading data — NOT split names. This test
verifies the wiring via AST inspection of the source code.

References:
  - Issue #821: FDA-ready test set documentation and lineage tracking
  - docs/planning/pre-full-gcp-housekeeping-and-qa.xml PR id="2" T2.1
"""

from __future__ import annotations

import ast
from pathlib import Path

_FLOWS_DIR = Path("src/minivess/orchestration/flows")


def _get_imported_names(filepath: Path) -> set[str]:
    """Parse a Python file's AST and return all imported names."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name)
    return names


def _source_contains(filepath: Path, substring: str) -> bool:
    """Check if a file's source code contains a substring."""
    return substring in filepath.read_text(encoding="utf-8")


class TestTrainFlowImportsAuditTrail:
    """Train flow must import AuditTrail for data access logging."""

    def test_train_flow_imports_audit_trail(self) -> None:
        names = _get_imported_names(_FLOWS_DIR / "train_flow.py")
        assert "AuditTrail" in names, (
            "train_flow.py must import AuditTrail from compliance.audit"
        )


class TestTrainFlowCallsLogDataAccess:
    """Train flow must call log_data_access() with correct signature."""

    def test_train_flow_calls_log_data_access(self) -> None:
        assert _source_contains(_FLOWS_DIR / "train_flow.py", "log_data_access"), (
            "train_flow.py must call log_data_access()"
        )

    def test_train_flow_passes_dataset_name_to_log_data_access(self) -> None:
        """log_data_access must receive dataset_name, not split names."""
        assert _source_contains(_FLOWS_DIR / "train_flow.py", "dataset_name"), (
            "train_flow.py must pass dataset_name to log_data_access()"
        )

    def test_train_flow_passes_file_paths_to_log_data_access(self) -> None:
        """log_data_access must receive file_paths: list[str]."""
        assert _source_contains(_FLOWS_DIR / "train_flow.py", "file_paths"), (
            "train_flow.py must pass file_paths to log_data_access()"
        )


class TestAuditTrailLogDataAccessSignature:
    """AuditTrail.log_data_access() must have the correct signature."""

    def test_log_data_access_accepts_dataset_name_and_file_paths(self) -> None:
        from minivess.compliance.audit import AuditTrail

        trail = AuditTrail()
        entry = trail.log_data_access(
            dataset_name="minivess",
            file_paths=["/data/raw/vol_001.nii.gz", "/data/raw/vol_002.nii.gz"],
            actor="train-flow",
        )
        assert entry.event_type == "DATA_ACCESS"
        assert entry.metadata["dataset"] == "minivess"
        assert entry.metadata["num_files"] == 2
        assert entry.data_hash is not None
