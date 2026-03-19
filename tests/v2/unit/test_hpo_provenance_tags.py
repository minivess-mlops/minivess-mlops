"""Tests for MLflow provenance tags in HPO grid runs.

T1.5: Validate that grid training runs log provenance tags:
- grid_config_hash: SHA-256 of the factorial YAML config
- git_sha: current git commit (or "unknown" in Docker)
- docker_image_digest: image digest from DOCKER_IMAGE_DIGEST env var
"""

from __future__ import annotations

from pathlib import Path


class TestProvenanceTagFunctions:
    """Test provenance tag computation functions."""

    def test_compute_config_hash_deterministic(self, tmp_path: Path) -> None:
        """Same YAML content produces same hash."""
        from minivess.optimization.grid_partitioning import compute_config_hash

        config_path = tmp_path / "test.yaml"
        config_path.write_text("a: 1\nb: 2\n", encoding="utf-8")

        hash1 = compute_config_hash(config_path)
        hash2 = compute_config_hash(config_path)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest

    def test_compute_config_hash_different_content(self, tmp_path: Path) -> None:
        """Different YAML content produces different hash."""
        from minivess.optimization.grid_partitioning import compute_config_hash

        path1 = tmp_path / "a.yaml"
        path2 = tmp_path / "b.yaml"
        path1.write_text("a: 1\n", encoding="utf-8")
        path2.write_text("a: 2\n", encoding="utf-8")

        assert compute_config_hash(path1) != compute_config_hash(path2)

    def test_get_git_sha_returns_string(self) -> None:
        """get_git_sha() returns a hex string or 'unknown'."""
        from minivess.optimization.grid_partitioning import get_git_sha

        sha = get_git_sha()
        assert isinstance(sha, str)
        assert len(sha) > 0

    def test_get_docker_image_digest_from_env(self, monkeypatch: object) -> None:
        """get_docker_image_digest() reads DOCKER_IMAGE_DIGEST env var."""
        # Monkeypatch is a pytest fixture
        import os

        from minivess.optimization.grid_partitioning import get_docker_image_digest

        monkeypatch.setattr(
            os, "environ", {**os.environ, "DOCKER_IMAGE_DIGEST": "sha256:abc123"}
        )  # type: ignore[attr-defined]
        digest = get_docker_image_digest()
        assert digest == "sha256:abc123"

    def test_get_docker_image_digest_unknown(self, monkeypatch: object) -> None:
        """get_docker_image_digest() returns 'unknown' when env var not set."""
        import os

        from minivess.optimization.grid_partitioning import get_docker_image_digest

        env = dict(os.environ)
        env.pop("DOCKER_IMAGE_DIGEST", None)
        monkeypatch.setattr(os, "environ", env)  # type: ignore[attr-defined]
        digest = get_docker_image_digest()
        assert digest == "unknown"

    def test_build_provenance_tags(self, tmp_path: Path) -> None:
        """build_provenance_tags() returns dict with all three tags."""
        from minivess.optimization.grid_partitioning import build_provenance_tags

        config_path = tmp_path / "config.yaml"
        config_path.write_text("experiment: test\n", encoding="utf-8")

        tags = build_provenance_tags(config_path)
        assert "grid_config_hash" in tags
        assert "git_sha" in tags
        assert "docker_image_digest" in tags
        assert len(tags["grid_config_hash"]) == 64
