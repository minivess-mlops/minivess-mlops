"""Tests for T-23: dvc_pull_task in data_flow.py.

Verifies that:
- dvc_pull_task() returns a git commit hash (hex string)
- dvc_pull_task() passes --rev and --remote args to dvc pull subprocess
- data_flow logs data_dvc_commit param to MLflow
- configs/dvc/remotes.yaml is parseable and has required remote entries

NO subprocess invocation — uses unittest.mock.patch.
NOTE: YAML parsing uses yaml.safe_load() — no regex.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

_DATA_FLOW_SRC = Path("src/minivess/orchestration/flows/data_flow.py")
_DVC_REMOTES_YAML = Path("configs/dvc/remotes.yaml")


# ---------------------------------------------------------------------------
# dvc_pull_task functional tests (mocked subprocess)
# ---------------------------------------------------------------------------


class TestDvcPullTask:
    def test_dvc_pull_task_returns_commit_hash(self) -> None:
        """dvc_pull_task() with mock subprocess returns a hex commit hash string."""
        from minivess.orchestration.flows.data_flow import dvc_pull_task

        fake_commit = "abc123def456" * 3  # 36-char hex string

        mock_dvc_result = MagicMock()
        mock_dvc_result.returncode = 0
        mock_dvc_result.stdout = "DVC pulled successfully"

        mock_git_result = MagicMock()
        mock_git_result.returncode = 0
        mock_git_result.stdout = f"{fake_commit}\n"

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [mock_dvc_result, mock_git_result]
            result = dvc_pull_task(Path("/tmp/data"))

        assert result == fake_commit, (
            f"dvc_pull_task must return the git commit hash, got: {result!r}"
        )

    def test_dvc_pull_task_passes_rev(self) -> None:
        """dvc_pull_task with dvc_rev='abc123' must pass --rev abc123 to subprocess."""
        from minivess.orchestration.flows.data_flow import dvc_pull_task

        mock_dvc = MagicMock(returncode=0, stdout="")
        mock_git = MagicMock(returncode=0, stdout="deadbeef\n")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [mock_dvc, mock_git]
            dvc_pull_task(Path("/tmp/data"), dvc_rev="abc123")

        first_call_args = mock_run.call_args_list[0]
        cmd = first_call_args[0][0]  # positional arg: the command list
        assert "--rev" in cmd, f"--rev not in dvc pull command: {cmd}"
        assert "abc123" in cmd, f"abc123 not in dvc pull command: {cmd}"

    def test_dvc_pull_task_passes_remote(self) -> None:
        """dvc_pull_task with remote='s3' must pass --remote s3 to subprocess."""
        from minivess.orchestration.flows.data_flow import dvc_pull_task

        mock_dvc = MagicMock(returncode=0, stdout="")
        mock_git = MagicMock(returncode=0, stdout="deadbeef\n")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [mock_dvc, mock_git]
            dvc_pull_task(Path("/tmp/data"), remote="s3")

        first_call_args = mock_run.call_args_list[0]
        cmd = first_call_args[0][0]
        assert "--remote" in cmd, f"--remote not in dvc pull command: {cmd}"
        assert "s3" in cmd, f"s3 not in dvc pull command: {cmd}"

    def test_dvc_pull_task_no_rev_no_remote(self) -> None:
        """dvc_pull_task with no rev/remote does not pass --rev or --remote."""
        from minivess.orchestration.flows.data_flow import dvc_pull_task

        mock_dvc = MagicMock(returncode=0, stdout="")
        mock_git = MagicMock(returncode=0, stdout="deadbeef\n")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [mock_dvc, mock_git]
            dvc_pull_task(Path("/tmp/data"))

        first_call_args = mock_run.call_args_list[0]
        cmd = first_call_args[0][0]
        assert "--rev" not in cmd, f"Unexpected --rev in default dvc pull: {cmd}"
        assert "--remote" not in cmd, f"Unexpected --remote in default dvc pull: {cmd}"


# ---------------------------------------------------------------------------
# data_flow MLflow logging test (source-level)
# ---------------------------------------------------------------------------


class TestDataFlowDvcCommitLogging:
    def test_data_flow_references_dvc_commit(self) -> None:
        """data_flow.py must reference data_dvc_commit for MLflow logging."""
        source = _DATA_FLOW_SRC.read_text(encoding="utf-8")
        assert "data_dvc_commit" in source, (
            "data_flow.py must log 'data_dvc_commit' param to MLflow. "
            "Call mlflow.log_param('data_dvc_commit', commit_hash) after dvc_pull_task()."
        )

    def test_data_flow_has_dvc_pull_task_import_or_def(self) -> None:
        """data_flow.py must define or reference dvc_pull_task."""
        source = _DATA_FLOW_SRC.read_text(encoding="utf-8")
        assert "dvc_pull_task" in source, (
            "data_flow.py must call or define dvc_pull_task. "
            "Add @task(name='dvc-pull') def dvc_pull_task(...) in data_flow.py."
        )

    def test_data_flow_accepts_dvc_rev_param(self) -> None:
        """run_data_flow() must accept dvc_rev parameter."""
        source = _DATA_FLOW_SRC.read_text(encoding="utf-8")
        assert "dvc_rev" in source, (
            "run_data_flow() must accept dvc_rev parameter for version-locked pulls. "
            "Add dvc_rev: str | None = None to the function signature."
        )


# ---------------------------------------------------------------------------
# configs/dvc/remotes.yaml
# ---------------------------------------------------------------------------


class TestDvcRemotesYaml:
    def test_dvc_remotes_yaml_exists(self) -> None:
        """configs/dvc/remotes.yaml must exist."""
        assert _DVC_REMOTES_YAML.exists(), (
            "configs/dvc/remotes.yaml does not exist. "
            "Create it with local, s3, gcs, azure remote definitions."
        )

    def test_dvc_remotes_yaml_parseable(self) -> None:
        """configs/dvc/remotes.yaml must be parseable via yaml.safe_load()."""
        if not _DVC_REMOTES_YAML.exists():
            return
        data = yaml.safe_load(_DVC_REMOTES_YAML.read_text(encoding="utf-8"))
        assert isinstance(data, dict), (
            f"configs/dvc/remotes.yaml must parse to a dict, got {type(data).__name__}"
        )

    def test_dvc_remotes_yaml_has_remotes_key(self) -> None:
        """configs/dvc/remotes.yaml must have a 'remotes' top-level key."""
        if not _DVC_REMOTES_YAML.exists():
            return
        data = yaml.safe_load(_DVC_REMOTES_YAML.read_text(encoding="utf-8"))
        assert "remotes" in data, (
            "configs/dvc/remotes.yaml must have a top-level 'remotes' key."
        )

    def test_dvc_remotes_yaml_has_local_remote(self) -> None:
        """configs/dvc/remotes.yaml must define a 'local' remote entry."""
        if not _DVC_REMOTES_YAML.exists():
            return
        data = yaml.safe_load(_DVC_REMOTES_YAML.read_text(encoding="utf-8"))
        remotes = data.get("remotes", {})
        assert "local" in remotes, (
            "configs/dvc/remotes.yaml must define a 'local' remote for dev. "
            f"Found keys: {list(remotes.keys())}"
        )

    def test_dvc_remotes_yaml_has_s3_remote(self) -> None:
        """configs/dvc/remotes.yaml must define an 's3' remote entry."""
        if not _DVC_REMOTES_YAML.exists():
            return
        data = yaml.safe_load(_DVC_REMOTES_YAML.read_text(encoding="utf-8"))
        remotes = data.get("remotes", {})
        assert "s3" in remotes, (
            "configs/dvc/remotes.yaml must define an 's3' remote (MinIO-compatible). "
            f"Found keys: {list(remotes.keys())}"
        )
