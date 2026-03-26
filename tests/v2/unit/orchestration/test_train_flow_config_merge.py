"""Tests for training_flow config_dict extraction — T2 regression tests.

Bug: When training_flow() receives config_dict, gradient_checkpointing and
other factorial-relevant keys are NOT extracted from config_dict at lines
1363-1379. They remain as False (function default) and overwrite config_dict
values during the merge at lines 1452-1456.
"""

from __future__ import annotations

from pathlib import Path

import pytest

TRAIN_FLOW = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "minivess"
    / "orchestration"
    / "flows"
    / "train_flow.py"
)


class TestHydraPathPreservesFactorialKeys:
    """T2: All factorial-relevant keys must be extracted from config_dict."""

    @pytest.fixture()
    def _source(self) -> str:
        return TRAIN_FLOW.read_text(encoding="utf-8")

    def _find_config_dict_extraction_block(self, source: str) -> str:
        """Extract the 'if config_dict is not None:' block from training_flow."""
        lines = source.split("\n")
        in_block = False
        block_lines: list[str] = []
        block_indent = 0
        for line in lines:
            if "if config_dict is not None:" in line and "training_flow" not in line:
                # Find the first extraction block (inside training_flow)
                in_block = True
                block_indent = len(line) - len(line.lstrip())
                block_lines.append(line)
                continue
            if in_block:
                stripped = line.lstrip()
                current_indent = len(line) - len(line.lstrip())
                if stripped and current_indent <= block_indent:
                    break
                block_lines.append(line)
        return "\n".join(block_lines)

    def test_hydra_path_preserves_gradient_checkpointing(self, _source):
        """gradient_checkpointing must be extracted from config_dict."""
        block = self._find_config_dict_extraction_block(_source)
        assert "gradient_checkpointing" in block, (
            "gradient_checkpointing not extracted from config_dict in training_flow. "
            "When config_dict has gradient_checkpointing=True, the function default "
            "(False) will silently override it during merge."
        )

    def test_hydra_path_preserves_gradient_accumulation_steps(self, _source):
        """gradient_accumulation_steps must be extracted from config_dict."""
        block = self._find_config_dict_extraction_block(_source)
        assert "gradient_accumulation_steps" in block, (
            "gradient_accumulation_steps not extracted from config_dict"
        )

    def test_hydra_path_preserves_with_aux_calib(self, _source):
        """with_aux_calib must be extracted from config_dict."""
        block = self._find_config_dict_extraction_block(_source)
        assert "with_aux_calib" in block, (
            "with_aux_calib not extracted from config_dict"
        )

    def test_hydra_path_preserves_max_train_volumes(self, _source):
        """max_train_volumes must be extracted from config_dict."""
        block = self._find_config_dict_extraction_block(_source)
        assert "max_train_volumes" in block, (
            "max_train_volumes not extracted from config_dict"
        )

    def test_merged_config_individual_params_take_precedence(self, _source):
        """merged.update(config) must still give individual params precedence."""
        assert "merged.update(config)" in _source, (
            "Merge logic (merged.update(config)) not found — "
            "individual flow params must take precedence over config_dict"
        )

    def test_merged_config_unextracted_keys_survive(self, _source):
        """Keys in config_dict that are NOT extracted should still survive in merged."""
        # The merge copies all of config_dict first, then overwrites with flow params.
        # Unextracted keys (e.g., custom user keys) should survive.
        assert "merged = dict(config_dict)" in _source, (
            "Merge should copy config_dict first to preserve unextracted keys"
        )
