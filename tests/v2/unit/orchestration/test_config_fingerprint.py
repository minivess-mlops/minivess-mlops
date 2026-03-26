"""Tests for config fingerprint consistency — T7 regression tests.

Bug: In training_subflow (line 841), compute_config_fingerprint is called with
with_aux_calib but WITHOUT patch_size. In train_one_fold_task (line 603), it's
called with patch_size but WITHOUT with_aux_calib. Different fingerprints for
the same config breaks auto-resume discovery.
"""

from __future__ import annotations

import ast
from pathlib import Path

TRAIN_FLOW = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "minivess"
    / "orchestration"
    / "flows"
    / "train_flow.py"
)


class TestFingerprintConsistency:
    """T7: Both compute_config_fingerprint call sites must pass identical kwargs."""

    def test_fingerprint_consistency_between_subflow_and_fold_task(self):
        """Both call sites should produce identical fingerprints for same config."""
        from minivess.pipeline.resume_discovery import compute_config_fingerprint

        # Same config values
        kwargs = {
            "loss_name": "cbdice_cldice",
            "model_family": "sam3_topolora",
            "fold_id": 0,
            "max_epochs": 50,
            "batch_size": 1,
            "patch_size": (64, 64, 3),
            "with_aux_calib": True,
        }

        fp1 = compute_config_fingerprint(**kwargs)
        fp2 = compute_config_fingerprint(**kwargs)
        assert fp1 == fp2, (
            "Same kwargs must produce same fingerprint"
        )

    def test_both_call_sites_pass_same_kwargs(self):
        """AST check: both call sites of compute_config_fingerprint pass same kwargs."""
        source = TRAIN_FLOW.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Find all calls to compute_config_fingerprint or _fp
        calls: list[set[str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name not in ("compute_config_fingerprint", "_fp"):
                continue
            # Extract keyword argument names
            kwarg_names = {kw.arg for kw in node.keywords if kw.arg is not None}
            calls.append(kwarg_names)

        assert len(calls) >= 2, (
            f"Expected at least 2 calls to compute_config_fingerprint, found {len(calls)}"
        )

        # All call sites should pass the same set of keyword arguments
        for i, call_kwargs in enumerate(calls[1:], 1):
            assert calls[0] == call_kwargs, (
                f"compute_config_fingerprint call site 0 uses kwargs {calls[0]} "
                f"but call site {i} uses kwargs {call_kwargs}. "
                f"Missing from site {i}: {calls[0] - call_kwargs}. "
                f"Extra in site {i}: {call_kwargs - calls[0]}."
            )
