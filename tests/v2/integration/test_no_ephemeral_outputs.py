"""Tests for T-28: All flows produce non-ephemeral outputs — no /tmp artifacts.

Verifies that:
- Every flow reads its output path from an environment variable (not hardcoded)
- No flow uses /tmp as a default output directory
- data_flow functional test: no files created under /tmp during execution

Uses ast.parse() for source inspection — NO regex (CLAUDE.md Rule #16).
Uses os.walk() over /tmp for functional verification.
Uses pathlib.Path.is_relative_to() for path comparison — no regex.
"""

from __future__ import annotations

import ast
import os
import tempfile
from pathlib import Path

# Flow source files and their expected env var / output path contracts
_FLOW_CONTRACTS = {
    "train_flow": {
        "path": Path("src/minivess/orchestration/flows/train_flow.py"),
        "env_var": "CHECKPOINT_DIR",
        "safe_default_prefix": "/app",
    },
    "post_training_flow": {
        "path": Path("src/minivess/orchestration/flows/post_training_flow.py"),
        "env_var": "POST_TRAINING_OUTPUT_DIR",
        "safe_default_prefix": "/app",
    },
    "analysis_flow": {
        "path": Path("src/minivess/orchestration/flows/analysis_flow.py"),
        "env_var": "ANALYSIS_OUTPUT",
        "safe_default_prefix": "/app",
    },
    "dashboard_flow": {
        "path": Path("src/minivess/orchestration/flows/dashboard_flow.py"),
        "env_var": "DASHBOARD_OUTPUT",
        "safe_default_prefix": "/app",
    },
}


def _extract_string_constants(source: str) -> list[str]:
    """Extract all string constants from Python source using ast.parse()."""
    tree = ast.parse(source)
    strings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.append(node.value)
    return strings


class TestNoEphemeralOutputsSource:
    def test_train_flow_uses_checkpoint_dir_env(self) -> None:
        """training_flow() must read CHECKPOINT_DIR env var — not hardcode /tmp."""
        source = Path("src/minivess/orchestration/flows/train_flow.py").read_text(
            encoding="utf-8"
        )
        assert "CHECKPOINT_DIR" in source, (
            "train_flow.py must read CHECKPOINT_DIR env var for checkpoint output. "
            "Use: checkpoint_base = Path(os.environ.get('CHECKPOINT_DIR', '/app/checkpoints'))"
        )

    def test_train_flow_default_not_tmp(self) -> None:
        """CHECKPOINT_DIR default must not be /tmp — must be /app/... or similar."""
        source = Path("src/minivess/orchestration/flows/train_flow.py").read_text(
            encoding="utf-8"
        )
        strings = _extract_string_constants(source)
        tmp_defaults = [
            s
            for s in strings
            if s.startswith("/tmp")
            and "CHECKPOINT"
            in source[max(0, source.find(s) - 50) : source.find(s) + 50]
        ]
        assert not tmp_defaults, (
            f"train_flow.py must NOT use /tmp as default for CHECKPOINT_DIR. "
            f"Found: {tmp_defaults}"
        )

    def test_post_training_flow_uses_output_dir_env(self) -> None:
        """post_training_flow() must read POST_TRAINING_OUTPUT_DIR env var."""
        source = Path(
            "src/minivess/orchestration/flows/post_training_flow.py"
        ).read_text(encoding="utf-8")
        assert "POST_TRAINING_OUTPUT_DIR" in source, (
            "post_training_flow.py must read POST_TRAINING_OUTPUT_DIR env var. "
            "Use: output_dir = Path(os.environ.get('POST_TRAINING_OUTPUT_DIR', '/app/outputs/post_training'))"
        )

    def test_analysis_flow_uses_output_env(self) -> None:
        """analysis_flow() must read ANALYSIS_OUTPUT env var."""
        source = Path("src/minivess/orchestration/flows/analysis_flow.py").read_text(
            encoding="utf-8"
        )
        assert "ANALYSIS_OUTPUT" in source, (
            "analysis_flow.py must read ANALYSIS_OUTPUT env var. "
            "Use: output_dir = Path(os.environ.get('ANALYSIS_OUTPUT', '/app/outputs/analysis'))"
        )

    def test_dashboard_flow_uses_output_env(self) -> None:
        """dashboard_flow() must read DASHBOARD_OUTPUT env var."""
        source = Path("src/minivess/orchestration/flows/dashboard_flow.py").read_text(
            encoding="utf-8"
        )
        assert "DASHBOARD_OUTPUT" in source, (
            "dashboard_flow.py must read DASHBOARD_OUTPUT env var. "
            "Use: output_dir = Path(os.environ.get('DASHBOARD_OUTPUT', '/app/outputs/dashboard'))"
        )

    def test_no_flow_has_tmp_as_default_output(self) -> None:
        """No flow must use /tmp as a default output path for artifacts."""
        flow_dir = Path("src/minivess/orchestration/flows")
        flows_with_tmp_defaults: list[str] = []

        for flow_file in sorted(flow_dir.glob("*.py")):
            if flow_file.name.startswith("_"):
                continue
            source = flow_file.read_text(encoding="utf-8")
            strings = _extract_string_constants(source)
            tmp_strings = [s for s in strings if s.startswith("/tmp")]
            if tmp_strings:
                flows_with_tmp_defaults.append(f"{flow_file.name}: {tmp_strings}")

        assert not flows_with_tmp_defaults, (
            "These flow files contain /tmp string constants (which would be ephemeral "
            "in Docker containers). Use /app/... defaults with env var overrides:\n"
            + "\n".join(flows_with_tmp_defaults)
        )

    def test_all_flow_defaults_use_app_prefix(self) -> None:
        """All flow output dir defaults must use /app/... prefix (safe in Docker).

        Checks that os.environ.get(ENV_VAR, DEFAULT) calls use /app/... defaults,
        not /tmp/... or other ephemeral paths.
        """
        for flow_name, contract in _FLOW_CONTRACTS.items():
            source = contract["path"].read_text(encoding="utf-8")
            env_var = contract["env_var"]
            safe_prefix = contract["safe_default_prefix"]

            tree = ast.parse(source)
            # Find calls: os.environ.get("ENV_VAR", "DEFAULT_VALUE")
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                # Match os.environ.get(...)
                func = node.func
                if not (
                    isinstance(func, ast.Attribute)
                    and func.attr == "get"
                    and isinstance(func.value, ast.Attribute)
                    and func.value.attr == "environ"
                ):
                    continue
                args = node.args
                if len(args) < 2:
                    continue
                # First arg must be the env var name
                if not (isinstance(args[0], ast.Constant) and args[0].value == env_var):
                    continue
                # Second arg is the default value
                if not isinstance(args[1], ast.Constant):
                    continue
                default = args[1].value
                if not isinstance(default, str):
                    continue
                if not default.startswith("/"):
                    continue
                assert default.startswith(safe_prefix), (
                    f"{flow_name}: os.environ.get('{env_var}', {default!r}) — "
                    f"default {default!r} does not start with {safe_prefix!r}. "
                    "Outputs must go to /app/... to survive container restarts "
                    "via Docker volume mounts."
                )


class TestDataFlowNoTmpArtifacts:
    """Functional tests verifying data_flow produces no /tmp artifacts."""

    def _snapshot_tmp(self) -> set[Path]:
        """Snapshot all files currently under /tmp."""
        files: set[Path] = set()
        try:
            for dirpath, _dirnames, filenames in os.walk("/tmp"):
                for fname in filenames:
                    files.add(Path(dirpath) / fname)
        except PermissionError:
            pass
        return files

    def test_data_flow_no_tmp_artifacts(self) -> None:
        """run_data_flow() must not create files under /tmp."""
        from minivess.orchestration.flows.data_flow import run_data_flow

        before = self._snapshot_tmp()

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data_dir = base / "data"
            (data_dir / "images").mkdir(parents=True)
            (data_dir / "labels").mkdir(parents=True)
            for i in range(3):
                (data_dir / "images" / f"vol_{i:02d}.nii.gz").write_bytes(b"fake")
                (data_dir / "labels" / f"vol_{i:02d}.nii.gz").write_bytes(b"fake")

            splits_dir = base / "splits"
            os.environ["SPLITS_OUTPUT_DIR"] = str(splits_dir)
            os.environ["MLFLOW_TRACKING_URI"] = f"file://{base}/mlruns"
            try:
                run_data_flow(data_dir=data_dir, n_folds=2, seed=42)
            finally:
                del os.environ["SPLITS_OUTPUT_DIR"]
                del os.environ["MLFLOW_TRACKING_URI"]

        after = self._snapshot_tmp()
        new_tmp_files = after - before
        # Filter out system noise (sockets, lock files from unrelated processes)
        new_artifacts = {
            f
            for f in new_tmp_files
            if f.suffix
            in {".json", ".yaml", ".yml", ".pt", ".pth", ".onnx", ".png", ".csv"}
        }
        assert not new_artifacts, (
            f"run_data_flow() wrote artifacts to /tmp (ephemeral in Docker): "
            f"{sorted(new_artifacts)}"
        )
