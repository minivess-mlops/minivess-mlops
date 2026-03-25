"""Guard test: torch.load(weights_only=False) audit and pickle.load ban.

Ensures that weights_only=False only appears in explicitly allowed files
(self-produced checkpoints), never in adapters (external weights), and
that each occurrence has a # SECURITY comment explaining why.

Also bans pickle.load() / pickle.loads() in production source.

Uses ast.parse() per Rule #16 (no regex for structured data).
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC_ROOT = Path("src/minivess")

# Files that are ALLOWED to have weights_only=False (self-produced checkpoints).
# Each must have a # SECURITY comment explaining why.
ALLOWED_FILES = {
    "checkpoint_utils.py",
    "checkpoint_averaging.py",
    "model_merging.py",
    "subsampled_ensemble.py",
    "swag.py",  # both pipeline and ensemble versions
}


def _find_weights_only_false_calls(tree: ast.AST) -> list[int]:
    """Find line numbers of torch.load() calls with weights_only=False.

    Walks the AST looking for Call nodes that have a keyword argument
    ``weights_only`` with a constant ``False`` value.
    """
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if (
                kw.arg == "weights_only"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is False
            ):
                lines.append(node.lineno)
    return lines


def _find_pickle_load_calls(tree: ast.AST) -> list[int]:
    """Find line numbers of pickle.load() or pickle.loads() calls.

    Detects both ``pickle.load(...)`` and ``pickle.loads(...)`` patterns.
    """
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # pickle.load(...) or pickle.loads(...)
        if (
            isinstance(func, ast.Attribute)
            and func.attr in ("load", "loads")
            and isinstance(func.value, ast.Name)
            and func.value.id == "pickle"
        ):
            lines.append(node.lineno)
    return lines


class TestTorchLoadWeightsOnlyAudit:
    """AST-based audit of torch.load(weights_only=False) across the codebase."""

    def test_weights_only_false_only_in_allowed_files(self) -> None:
        """AST scan src/minivess/ -- weights_only=False only in ALLOWED_FILES."""
        violations: list[str] = []

        for py_file in sorted(_SRC_ROOT.rglob("*.py")):
            source = py_file.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue

            hit_lines = _find_weights_only_false_calls(tree)
            if hit_lines and py_file.name not in ALLOWED_FILES:
                for lineno in hit_lines:
                    violations.append(
                        f"{py_file.relative_to(_SRC_ROOT)}:{lineno} -- "
                        f"weights_only=False in non-allowed file"
                    )

        assert not violations, (
            f"weights_only=False found in {len(violations)} non-allowed location(s):\n"
            + "\n".join(f"  - {v}" for v in violations)
            + f"\n\nAllowed files: {sorted(ALLOWED_FILES)}"
        )

    def test_no_weights_only_false_in_adapters(self) -> None:
        """Adapters load external weights -- must use weights_only=True."""
        adapters_dir = _SRC_ROOT / "adapters"
        if not adapters_dir.exists():
            return  # No adapters dir yet

        violations: list[str] = []
        for py_file in sorted(adapters_dir.rglob("*.py")):
            source = py_file.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue

            hit_lines = _find_weights_only_false_calls(tree)
            for lineno in hit_lines:
                violations.append(
                    f"{py_file.relative_to(_SRC_ROOT)}:{lineno} -- "
                    f"weights_only=False in adapters (external weights)"
                )

        assert not violations, (
            f"weights_only=False found in adapters ({len(violations)} location(s)):\n"
            + "\n".join(f"  - {v}" for v in violations)
            + "\n\nAdapters load external weights and MUST use weights_only=True."
        )

    def test_no_bare_pickle_load_in_production(self) -> None:
        """pickle.load() and pickle.loads() banned in src/."""
        violations: list[str] = []

        for py_file in sorted(_SRC_ROOT.rglob("*.py")):
            source = py_file.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue

            hit_lines = _find_pickle_load_calls(tree)
            for lineno in hit_lines:
                violations.append(
                    f"{py_file.relative_to(_SRC_ROOT)}:{lineno} -- "
                    f"pickle.load/loads call found"
                )

        assert not violations, (
            f"pickle.load/loads found in {len(violations)} location(s):\n"
            + "\n".join(f"  - {v}" for v in violations)
            + "\n\nUse torch.load(weights_only=True) or json.loads() instead."
        )

    def test_all_weights_only_false_have_security_comment(self) -> None:
        """Each weights_only=False must have # SECURITY on a nearby line."""
        missing_comments: list[str] = []

        for py_file in sorted(_SRC_ROOT.rglob("*.py")):
            if py_file.name not in ALLOWED_FILES:
                continue

            source = py_file.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue

            hit_lines = _find_weights_only_false_calls(tree)
            if not hit_lines:
                continue

            source_lines = source.splitlines()
            for lineno in hit_lines:
                # Check the line itself and the 3 lines above for # SECURITY
                found_comment = False
                for check_line in range(max(0, lineno - 4), lineno + 1):
                    if check_line < len(source_lines) and "# SECURITY" in source_lines[check_line]:
                        found_comment = True
                        break
                if not found_comment:
                    missing_comments.append(
                        f"{py_file.relative_to(_SRC_ROOT)}:{lineno} -- "
                        f"weights_only=False without # SECURITY comment"
                    )

        assert not missing_comments, (
            f"Missing # SECURITY comment for {len(missing_comments)} "
            f"weights_only=False call(s):\n"
            + "\n".join(f"  - {v}" for v in missing_comments)
            + "\n\nAdd '# SECURITY: weights_only=False -- self-produced checkpoint, "
            "contains [types].' on or above each call."
        )


class TestCheckpointRoundTrip:
    """Verify that standard checkpoint format works with weights_only=True."""

    def test_standard_checkpoint_loads_with_weights_only_true(
        self, tmp_path: Path
    ) -> None:
        """state_dict + epoch + loss loads fine with weights_only=True."""
        import torch

        ckpt = {
            "epoch": 5,
            "loss": 0.123,
            "state_dict": {"w": torch.tensor([1.0])},
        }
        path = tmp_path / "test.pth"
        torch.save(ckpt, path)
        loaded = torch.load(path, weights_only=True, map_location="cpu")
        assert loaded["epoch"] == 5
        assert loaded["loss"] == 0.123  # noqa: PLR2004
        assert torch.equal(loaded["state_dict"]["w"], torch.tensor([1.0]))
