"""Tests for RegOps CI/CD pipeline extension (Issue #21)."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# T1: CIContext
# ---------------------------------------------------------------------------


class TestCIContext:
    """Test CI/CD context dataclass."""

    def test_construction(self) -> None:
        """CIContext should be constructible with CI metadata."""
        from minivess.compliance.regops import CIContext

        ctx = CIContext(
            commit_sha="abc123def456",
            actor="github-actions[bot]",
            ref="refs/heads/main",
            run_id="12345",
            repository="minivess-mlops/minivess-mlops",
        )
        assert ctx.commit_sha == "abc123def456"
        assert ctx.actor == "github-actions[bot]"

    def test_defaults(self) -> None:
        """Optional fields should have sensible defaults."""
        from minivess.compliance.regops import CIContext

        ctx = CIContext(commit_sha="abc123")
        assert ctx.actor == "ci"
        assert ctx.ref == ""
        assert ctx.run_id == ""
        assert ctx.repository == ""

    def test_from_env(self, monkeypatch: object) -> None:
        """from_env should read GitHub Actions environment variables."""
        import os

        from minivess.compliance.regops import CIContext

        # Monkeypatch is a pytest fixture
        monkeypatch.setattr(
            os,
            "environ",
            {  # type: ignore[attr-defined]
                "GITHUB_SHA": "deadbeef1234",
                "GITHUB_ACTOR": "petteri",
                "GITHUB_REF": "refs/tags/v1.0.0",
                "GITHUB_RUN_ID": "99999",
                "GITHUB_REPOSITORY": "minivess-mlops/minivess-mlops",
            },
        )
        ctx = CIContext.from_env()
        assert ctx.commit_sha == "deadbeef1234"
        assert ctx.actor == "petteri"
        assert ctx.ref == "refs/tags/v1.0.0"
        assert ctx.run_id == "99999"

    def test_from_env_missing(self, monkeypatch: object) -> None:
        """from_env should handle missing env vars gracefully."""
        import os

        from minivess.compliance.regops import CIContext

        monkeypatch.setattr(os, "environ", {})  # type: ignore[attr-defined]
        ctx = CIContext.from_env()
        assert ctx.commit_sha == "unknown"
        assert ctx.actor == "ci"


# ---------------------------------------------------------------------------
# T2: generate_ci_audit_entry
# ---------------------------------------------------------------------------


class TestCIAuditEntry:
    """Test CI audit entry creation."""

    def test_creates_entry(self) -> None:
        """generate_ci_audit_entry should create a CI_PIPELINE entry."""
        from minivess.compliance.audit import AuditTrail
        from minivess.compliance.regops import CIContext, generate_ci_audit_entry

        trail = AuditTrail()
        ctx = CIContext(commit_sha="abc123", actor="bot", ref="refs/heads/main")
        entry = generate_ci_audit_entry(trail, ctx)
        assert entry.event_type == "CI_PIPELINE"
        assert "abc123" in entry.description
        assert entry.actor == "bot"

    def test_metadata_contains_ci_fields(self) -> None:
        """Audit entry metadata should include CI context fields."""
        from minivess.compliance.audit import AuditTrail
        from minivess.compliance.regops import CIContext, generate_ci_audit_entry

        trail = AuditTrail()
        ctx = CIContext(
            commit_sha="abc123",
            ref="refs/tags/v2.0",
            run_id="42",
            repository="org/repo",
        )
        entry = generate_ci_audit_entry(trail, ctx)
        assert entry.metadata["commit_sha"] == "abc123"
        assert entry.metadata["ref"] == "refs/tags/v2.0"
        assert entry.metadata["run_id"] == "42"
        assert entry.metadata["repository"] == "org/repo"


# ---------------------------------------------------------------------------
# T3: RegOpsPipeline
# ---------------------------------------------------------------------------


class TestRegOpsPipeline:
    """Test RegOps orchestration pipeline."""

    def test_construction(self) -> None:
        """RegOpsPipeline should be constructible."""
        from minivess.compliance.regops import CIContext, RegOpsPipeline

        ctx = CIContext(commit_sha="abc123")
        pipeline = RegOpsPipeline(
            ci_context=ctx,
            product_name="MiniVess",
            product_version="2.0.0",
        )
        assert pipeline.product_name == "MiniVess"

    def test_generate_artifacts(self, tmp_path: object) -> None:
        """generate_artifacts should produce regulatory docs in output dir."""
        from pathlib import Path

        from minivess.compliance.regops import CIContext, RegOpsPipeline

        ctx = CIContext(commit_sha="abc123", actor="ci-bot")
        pipeline = RegOpsPipeline(
            ci_context=ctx,
            product_name="MiniVess Segmentation",
            product_version="2.0.0",
        )
        output_dir = Path(str(tmp_path))
        manifest = pipeline.generate_artifacts(output_dir)

        # Should generate multiple regulatory documents
        assert len(manifest) >= 4
        for filepath in manifest:
            assert Path(filepath).exists()
            assert Path(filepath).stat().st_size > 0

    def test_manifest_contains_expected_docs(self, tmp_path: object) -> None:
        """Manifest should include DHF, risk analysis, SRS, validation, EU AI Act."""
        from pathlib import Path

        from minivess.compliance.regops import CIContext, RegOpsPipeline

        ctx = CIContext(commit_sha="abc123")
        pipeline = RegOpsPipeline(
            ci_context=ctx,
            product_name="Test",
            product_version="1.0",
        )
        output_dir = Path(str(tmp_path))
        manifest = pipeline.generate_artifacts(output_dir)
        filenames = [Path(f).name for f in manifest]

        assert "design-history.md" in filenames
        assert "risk-analysis.md" in filenames
        assert "srs.md" in filenames
        assert "validation-summary.md" in filenames
        assert "eu-ai-act-compliance.md" in filenames

    def test_audit_trail_saved(self, tmp_path: object) -> None:
        """generate_artifacts should save the audit trail JSON."""
        from pathlib import Path

        from minivess.compliance.regops import CIContext, RegOpsPipeline

        ctx = CIContext(commit_sha="abc123")
        pipeline = RegOpsPipeline(
            ci_context=ctx,
            product_name="Test",
            product_version="1.0",
        )
        output_dir = Path(str(tmp_path))
        manifest = pipeline.generate_artifacts(output_dir)
        filenames = [Path(f).name for f in manifest]

        assert "audit-trail.json" in filenames
        audit_path = output_dir / "audit-trail.json"
        assert audit_path.exists()
