"""Docker image freshness gate — verifies GAR image GIT_COMMIT matches HEAD.

Prevents launching cloud jobs with stale Docker images that don't contain
the latest code changes. Uses 'docker buildx imagetools inspect' to read
the org.opencontainers.image.revision label from the registry.

See: .claude/metalearning/2026-03-25-stale-docker-image-launch.md
Plan: docs/planning/sam3-batch-size-1-and-robustification-plan.xml
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PREFLIGHT = Path("scripts/preflight_gcp.py")


def _import_preflight():
    """Import preflight module dynamically."""
    spec = importlib.util.spec_from_file_location("preflight", PREFLIGHT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestDockerFreshnessGateExists:
    """Structural tests — the function must exist and be wired in."""

    def test_check_docker_image_freshness_function_exists(self) -> None:
        source = PREFLIGHT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        func_names = [
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        assert "check_docker_image_freshness" in func_names, (
            "preflight_gcp.py must define check_docker_image_freshness() — "
            "stale Docker image incident 2026-03-25"
        )

    def test_freshness_check_in_main_checks_list(self) -> None:
        source = PREFLIGHT.read_text(encoding="utf-8")
        assert "check_docker_image_freshness" in source
        # Verify it appears after checks = [ and before ]
        lines = source.splitlines()
        in_checks = False
        for line in lines:
            if "checks = [" in line:
                in_checks = True
            if in_checks and "check_docker_image_freshness" in line:
                return
            if in_checks and line.strip() == "]":
                break
        pytest.fail("check_docker_image_freshness not in main checks list")


class TestDockerFreshnessGateBehavior:
    """Behavioral tests with mocked subprocess calls."""

    def _make_mock_run(
        self,
        image_commit: str = "abc123def456",
        head_commit: str = "abc123def456",
        imagetools_ok: bool = True,
        git_ok: bool = True,
    ):
        """Create a mock _run function with configurable responses."""

        def fake_run(cmd, **kwargs):
            mock = MagicMock()
            cmd_str = " ".join(cmd)
            if "imagetools" in cmd_str:
                if imagetools_ok:
                    mock.returncode = 0
                    mock.stdout = json.dumps({
                        "image": {
                            "config": {
                                "Labels": {
                                    "org.opencontainers.image.revision": image_commit,
                                }
                            }
                        }
                    })
                else:
                    mock.returncode = 1
                    mock.stdout = ""
                    mock.stderr = "command not found"
            elif "rev-parse" in cmd_str:
                if git_ok:
                    mock.returncode = 0
                    mock.stdout = head_commit + "\n"
                else:
                    mock.returncode = 1
                    mock.stdout = ""
            else:
                mock.returncode = 1
                mock.stdout = ""
            return mock

        return fake_run

    def test_returns_tuple_bool_str(self) -> None:
        mod = _import_preflight()
        # Just verify the function returns a tuple — actual result depends on env
        result = mod.check_docker_image_freshness()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_passes_when_commits_match(self) -> None:
        mod = _import_preflight()
        commit = "abc123def456789"
        from unittest.mock import patch

        with patch.object(
            mod, "_run", side_effect=self._make_mock_run(commit, commit)
        ):
            ok, msg = mod.check_docker_image_freshness()
        assert ok is True
        assert commit[:12] in msg

    def test_fails_when_commits_differ(self) -> None:
        mod = _import_preflight()
        from unittest.mock import patch

        with patch.object(
            mod,
            "_run",
            side_effect=self._make_mock_run("old_commit_abc", "new_commit_def"),
        ):
            ok, msg = mod.check_docker_image_freshness()
        assert ok is False
        assert "old_commit_a" in msg
        assert "new_commit_d" in msg
        assert "make build-base-gpu" in msg

    def test_fails_when_image_has_unknown_commit(self) -> None:
        mod = _import_preflight()
        from unittest.mock import patch

        with patch.object(
            mod,
            "_run",
            side_effect=self._make_mock_run("unknown", "abc123"),
        ):
            ok, msg = mod.check_docker_image_freshness()
        assert ok is False
        assert "unknown" in msg.lower()

    def test_skips_when_imagetools_unavailable(self) -> None:
        mod = _import_preflight()
        from unittest.mock import patch

        with patch.object(
            mod,
            "_run",
            side_effect=self._make_mock_run(imagetools_ok=False),
        ):
            ok, msg = mod.check_docker_image_freshness()
        # Graceful degradation: pass with warning
        assert ok is True
        assert "skip" in msg.lower() or "cannot verify" in msg.lower()

    def test_skips_when_git_unavailable(self) -> None:
        mod = _import_preflight()
        from unittest.mock import patch

        with patch.object(
            mod,
            "_run",
            side_effect=self._make_mock_run(git_ok=False),
        ):
            ok, msg = mod.check_docker_image_freshness()
        assert ok is True
        assert "skip" in msg.lower() or "cannot" in msg.lower()
