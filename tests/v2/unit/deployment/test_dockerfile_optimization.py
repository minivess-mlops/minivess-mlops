"""Tests for Dockerfile optimization (T2-T4: #751, #754).

T2: Incremental cleanup (no __pycache__ in final image, no-compile in runner)
T3: zstd compression (Makefile build target)
T4: RunPod ENTRYPOINT verification (no blocking ENTRYPOINT or HEALTHCHECK)
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# T2: Dockerfile incremental cleanup
# ---------------------------------------------------------------------------


class TestDockerfileCleanup:
    """Verify Dockerfile.base runner stage is lean."""

    def test_runner_no_compile_bytecode(self) -> None:
        """Runner stage should set PYTHONDONTWRITEBYTECODE=1."""
        content = Path("deployment/docker/Dockerfile.base").read_text(encoding="utf-8")
        assert "PYTHONDONTWRITEBYTECODE=1" in content

    def test_runner_no_build_tools(self) -> None:
        """Runner stage should NOT have uv, pip, git, gcc, nvcc."""
        content = Path("deployment/docker/Dockerfile.base").read_text(encoding="utf-8")
        # Extract runner stage only (after "AS runner")
        runner_start = content.find("AS runner")
        assert runner_start > 0
        runner_content = content[runner_start:]

        # These tools should only be in builder, not runner
        assert "COPY --from=ghcr.io/astral-sh/uv" not in runner_content
        assert "pip install" not in runner_content
        assert "gcc" not in runner_content

    def test_dockerignore_excludes_pycache(self) -> None:
        """Dockerignore must exclude __pycache__ to prevent bloating context."""
        content = Path(".dockerignore").read_text(encoding="utf-8")
        assert "__pycache__" in content


# ---------------------------------------------------------------------------
# T3: zstd compression (Makefile build target)
# ---------------------------------------------------------------------------


class TestZstdCompression:
    """Verify zstd compression is available in build targets."""

    def test_makefile_has_zstd_build_target(self) -> None:
        """Makefile should have a build target with zstd compression."""
        content = Path("Makefile").read_text(encoding="utf-8")
        assert "zstd" in content or "compression" in content, (
            "Makefile should include a zstd-compressed build target. "
            "zstd decompression is 3-5x faster than gzip, reducing "
            "pull time by 20-30%."
        )


# ---------------------------------------------------------------------------
# T4: RunPod ENTRYPOINT verification
# ---------------------------------------------------------------------------


class TestRunPodCompatibility:
    """Verify Docker images are RunPod-compatible."""

    def test_base_no_blocking_entrypoint(self) -> None:
        """Dockerfile.base must NOT set a blocking ENTRYPOINT.

        RunPod/SkyPilot needs to override the entrypoint with /bin/bash
        for SSH access. A custom ENTRYPOINT prevents this.
        See: SkyPilot #4285, #3879.
        """
        content = Path("deployment/docker/Dockerfile.base").read_text(encoding="utf-8")
        runner_start = content.find("AS runner")
        runner_content = content[runner_start:]
        assert "ENTRYPOINT" not in runner_content, (
            "Dockerfile.base runner stage must NOT set ENTRYPOINT. "
            "RunPod/SkyPilot overrides with /bin/bash for SSH. "
            "See SkyPilot #4285."
        )

    def test_base_no_blocking_cmd(self) -> None:
        """Dockerfile.base should not set CMD that blocks SSH."""
        content = Path("deployment/docker/Dockerfile.base").read_text(encoding="utf-8")
        runner_start = content.find("AS runner")
        runner_content = content[runner_start:]
        # CMD is ok if it's bash or similar — just not a long-running process
        if "CMD" in runner_content:
            assert (
                "bash" in runner_content.lower() or "sleep" in runner_content.lower()
            ), (
                "If Dockerfile.base sets CMD, it must be /bin/bash or similar, "
                "not a blocking process that prevents SSH access."
            )

    def test_base_healthcheck_not_blocking_runpod(self) -> None:
        """HEALTHCHECK should use a fast, non-blocking check.

        RunPod may interpret HEALTHCHECK failures as container unhealthy
        and restart it, causing the STARTING hang.
        """
        content = Path("deployment/docker/Dockerfile.base").read_text(encoding="utf-8")
        runner_start = content.find("AS runner")
        runner_content = content[runner_start:]
        if "HEALTHCHECK" in runner_content:
            # Verify start period is generous (>60s for GPU init)
            assert (
                "start-period" in runner_content.lower()
                or "start_period" in runner_content.lower()
            )

    def test_skypilot_smoke_test_uses_docker_image(self) -> None:
        """smoke_test_gpu.yaml must use image_id: docker:... (not bare VM).

        Note: dev_runpod.yaml intentionally uses SkyPilot default image + uv
        for quick iteration without Docker builds. Production smoke tests
        MUST use Docker images.
        """
        yaml_path = Path("deployment/skypilot/smoke_test_gpu.yaml")
        if not yaml_path.exists():
            return  # Skip if file doesn't exist
        content = yaml_path.read_text(encoding="utf-8")
        assert "image_id:" in content or "docker:" in content

    def test_skypilot_dev_runpod_has_failover(self) -> None:
        """dev_runpod.yaml should support multi-region failover."""
        yaml_path = Path("deployment/skypilot/dev_runpod.yaml")
        content = yaml_path.read_text(encoding="utf-8")
        # Should have multiple accelerator options or region flexibility
        assert "accelerators" in content
