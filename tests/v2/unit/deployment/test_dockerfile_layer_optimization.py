"""Tests for Dockerfile layer optimization (#751).

The chmod anti-pattern: a separate RUN chmod on a 9+ GB venv COPY
creates a duplicate layer. Use --chmod in COPY instead.
"""

from __future__ import annotations

from pathlib import Path


class TestDockerfileLayerOptimization:
    """Verify no duplicate layers from chmod on large COPY."""

    def test_no_separate_chmod_on_venv(self) -> None:
        """Dockerfile.base must NOT have 'RUN chmod' after COPY .venv.

        A separate chmod layer duplicates the entire venv (9+ GB).
        Use COPY --chmod=755 instead.
        """
        content = Path("deployment/docker/Dockerfile.base").read_text(encoding="utf-8")
        runner_start = content.find("AS runner")
        runner_content = content[runner_start:]
        # Check actual RUN instructions, not comments
        for line in runner_content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert not (stripped.startswith("RUN") and "chmod" in stripped), (
                f"Dockerfile.base runner has 'RUN chmod' which creates a duplicate "
                f"layer of the .venv directory (9+ GB wasted). Use COPY --chmod instead. "
                f"Line: {stripped}"
            )

    def test_copy_uses_chmod_flag(self) -> None:
        """COPY --chmod must be used for .venv, src, configs."""
        content = Path("deployment/docker/Dockerfile.base").read_text(encoding="utf-8")
        assert "--chmod=" in content, (
            "Dockerfile.base should use COPY --chmod to set permissions "
            "during copy, avoiding a separate RUN chmod layer."
        )

    def test_no_duplicate_venv_layers(self) -> None:
        """Runner stage should have only ONE layer-creating op touching .venv."""
        content = Path("deployment/docker/Dockerfile.base").read_text(encoding="utf-8")
        runner_start = content.find("AS runner")
        runner_content = content[runner_start:]
        # Count only COPY/RUN lines referencing .venv (not comments or ENV)
        venv_layer_ops = sum(
            1
            for line in runner_content.splitlines()
            if "/app/.venv" in line
            and not line.strip().startswith("#")
            and (line.strip().startswith("COPY") or line.strip().startswith("RUN"))
        )
        assert venv_layer_ops == 1, (
            f"Expected 1 layer-creating operation on /app/.venv, got {venv_layer_ops}. "
            "A separate RUN chmod after COPY creates a 9+ GB duplicate layer."
        )
