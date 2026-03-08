"""Regression tests for auto-resume: TrainingFlowResult field name bug.

Closes issue #365: result.fold_count crashed at runtime because the field
is named n_folds. These tests prevent regression.

CLAUDE.md Rule #16: AST scanning for field names uses ast.parse(), NOT regex.
"""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

# ---------------------------------------------------------------------------
# T-04.1a: TrainingFlowResult field name tests
# ---------------------------------------------------------------------------


def test_training_flow_result_has_n_folds() -> None:
    """TrainingFlowResult must have an n_folds attribute."""
    from minivess.orchestration.flows.train_flow import TrainingFlowResult

    field_names = {f.name for f in fields(TrainingFlowResult)}
    assert "n_folds" in field_names, (
        "TrainingFlowResult must have 'n_folds' field. Got fields: " + str(field_names)
    )


def test_training_flow_result_no_fold_count() -> None:
    """TrainingFlowResult must NOT have a fold_count attribute (old name, caused crash)."""
    from minivess.orchestration.flows.train_flow import TrainingFlowResult

    field_names = {f.name for f in fields(TrainingFlowResult)}
    assert "fold_count" not in field_names, (
        "TrainingFlowResult must NOT have 'fold_count' field — "
        "that was the old (broken) name. Use 'n_folds' instead."
    )


def test_training_flow_result_n_folds_default() -> None:
    """TrainingFlowResult.n_folds defaults to 0 and accepts a value."""
    from minivess.orchestration.flows.train_flow import TrainingFlowResult

    r = TrainingFlowResult()
    assert r.n_folds == 0

    r2 = TrainingFlowResult(n_folds=3)
    assert r2.n_folds == 3


def test_run_training_flow_script_references_n_folds() -> None:
    """AST scan: scripts/run_training_flow.py must NOT contain 'fold_count'.

    Uses ast.parse() — regex is banned (CLAUDE.md Rule #16).
    """
    script_path = Path("scripts/run_training_flow.py")
    assert script_path.exists(), f"Script not found: {script_path}"

    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script_path))

    bad_accesses: list[str] = []
    for node in ast.walk(tree):
        # Check attribute access: result.fold_count
        if isinstance(node, ast.Attribute) and node.attr == "fold_count":
            bad_accesses.append(f"line {node.lineno}: {ast.unparse(node)}")
        # Check string literals that spell out the wrong name
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "fold_count" in node.value
        ):
            bad_accesses.append(
                f"line {node.lineno}: string literal contains 'fold_count'"
            )

    assert not bad_accesses, (
        "run_training_flow.py uses the old 'fold_count' name — "
        "fix to 'n_folds':\n" + "\n".join(bad_accesses)
    )


# ---------------------------------------------------------------------------
# T-04.1a: Fingerprint determinism tests
# ---------------------------------------------------------------------------


def test_resume_fingerprint_deterministic() -> None:
    """Same config inputs must always produce the same fingerprint."""
    from minivess.pipeline.resume_discovery import compute_config_fingerprint

    fp1 = compute_config_fingerprint(
        loss_name="dice_ce",
        model_family="dynunet",
        fold_id=0,
        max_epochs=100,
        batch_size=2,
    )
    fp2 = compute_config_fingerprint(
        loss_name="dice_ce",
        model_family="dynunet",
        fold_id=0,
        max_epochs=100,
        batch_size=2,
    )
    assert fp1 == fp2, "Fingerprint is not deterministic for identical config"


def test_resume_fingerprint_changes_with_loss() -> None:
    """Different loss_name must produce a different fingerprint."""
    from minivess.pipeline.resume_discovery import compute_config_fingerprint

    fp_dice = compute_config_fingerprint(
        loss_name="dice_ce",
        model_family="dynunet",
        fold_id=0,
        max_epochs=100,
        batch_size=2,
    )
    fp_cbdice = compute_config_fingerprint(
        loss_name="cbdice_cldice",
        model_family="dynunet",
        fold_id=0,
        max_epochs=100,
        batch_size=2,
    )
    assert fp_dice != fp_cbdice, "Fingerprints for different loss functions must differ"


def test_resume_fingerprint_changes_with_fold() -> None:
    """Different fold_id must produce a different fingerprint."""
    from minivess.pipeline.resume_discovery import compute_config_fingerprint

    fp0 = compute_config_fingerprint(
        loss_name="dice_ce",
        model_family="dynunet",
        fold_id=0,
        max_epochs=100,
        batch_size=2,
    )
    fp1 = compute_config_fingerprint(
        loss_name="dice_ce",
        model_family="dynunet",
        fold_id=1,
        max_epochs=100,
        batch_size=2,
    )
    assert fp0 != fp1, "Fingerprints for different fold_ids must differ"


def test_resume_fingerprint_is_16_chars() -> None:
    """Fingerprint must be exactly 16 hex characters."""
    from minivess.pipeline.resume_discovery import compute_config_fingerprint

    fp = compute_config_fingerprint(
        loss_name="dice_ce",
        model_family="dynunet",
        fold_id=0,
        max_epochs=100,
        batch_size=2,
    )
    assert len(fp) == 16, f"Expected 16-char fingerprint, got {len(fp)}: {fp!r}"
    assert all(c in "0123456789abcdef" for c in fp), f"Fingerprint is not hex: {fp!r}"
