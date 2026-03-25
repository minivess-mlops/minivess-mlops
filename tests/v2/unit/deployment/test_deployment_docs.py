"""Tests for deployment/CLAUDE.md production runbook completeness.

T-04.3: All operational workflows must be documented in deployment/CLAUDE.md:
- shm_size requirement (MONAI DataLoader Bus error)
- nvidia-ctk version check (CVE-2025-23266)
- Trivy scan
- minio-init bucket creation
- Multi-stage build (builder/runner)

Rule #16: No regex. Use str.split(), str methods, pathlib.Path.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.parent
CLAUDE_MD = ROOT / "deployment" / "CLAUDE.md"


def _content() -> str:
    return CLAUDE_MD.read_text(encoding="utf-8")


def test_claude_md_documents_shm_size() -> None:
    """deployment/CLAUDE.md must document --shm-size requirement for docker compose run."""
    assert "--shm-size" in _content(), (
        "deployment/CLAUDE.md missing '--shm-size' documentation. "
        "MONAI 3D DataLoader uses /dev/shm for IPC — 'docker compose run' "
        "ignores shm_size from compose file, so --shm-size must be passed explicitly. "
        "Add a note in the 'Running Flows' section."
    )


def test_claude_md_documents_nvidia_ctk() -> None:
    """deployment/CLAUDE.md must document nvidia-ctk version check (CVE-2025-23266)."""
    assert "nvidia-ctk" in _content(), (
        "deployment/CLAUDE.md missing nvidia-ctk version check documentation. "
        "CVE-2025-23266 (CVSS 9.0) — container-to-host escape in NVIDIA CTK < 1.17.8. "
        "Add: 'nvidia-ctk --version' check before running GPU containers."
    )


def test_claude_md_documents_trivy() -> None:
    """deployment/CLAUDE.md must document trivy image scanning."""
    assert "trivy" in _content().lower(), (
        "deployment/CLAUDE.md missing Trivy documentation. "
        "Add 'make scan' or trivy usage instructions to the deployment runbook."
    )


def test_claude_md_documents_minio_init() -> None:
    """deployment/CLAUDE.md must document minio-init bucket creation."""
    content = _content()
    has_minio_init = "minio-init" in content or "make init-buckets" in content
    assert has_minio_init, (
        "deployment/CLAUDE.md missing minio-init documentation. "
        "The minio-init service auto-creates mlflow-artifacts bucket on stack startup. "
        "Add a note in the 'One-Time Stack Setup' section."
    )


def test_claude_md_documents_multi_stage() -> None:
    """deployment/CLAUDE.md must document multi-stage build (builder/runner stages)."""
    content = _content()
    has_multi_stage = "builder" in content and "runner" in content
    assert has_multi_stage, (
        "deployment/CLAUDE.md missing multi-stage build documentation. "
        "Dockerfile.base uses builder (devel) → runner (runtime) stages. "
        "Document in the 'Building' section."
    )


# ── T-01.2: Network isolation docs + rootless Docker note (#552, #549) ────────


def test_claude_md_has_network_isolation_section() -> None:
    """deployment/CLAUDE.md must have a Network Isolation Strategy section."""
    content = _content()
    assert "Network Isolation" in content, (
        "deployment/CLAUDE.md missing 'Network Isolation' section. "
        "Document why icc:false is banned and the correct isolation strategy."
    )


def test_claude_md_documents_icc_false_ban() -> None:
    """deployment/CLAUDE.md must explain why icc:false is BANNED."""
    content = _content()
    assert "icc" in content and "false" in content, (
        "deployment/CLAUDE.md must document that icc:false is BANNED. "
        "icc:false breaks minivess-network inter-container communication."
    )


def test_claude_md_shows_network_topology() -> None:
    """deployment/CLAUDE.md must include service names in network topology."""
    content = _content()
    assert "minivess-mlflow" in content, (
        "deployment/CLAUDE.md missing network topology showing minivess-mlflow. "
        "Add ASCII diagram of service-to-service connections."
    )
    assert "minivess-prefect" in content or "prefect" in content.lower(), (
        "deployment/CLAUDE.md missing Prefect in network topology."
    )


def test_claude_md_documents_rootless_docker_blocker() -> None:
    """deployment/CLAUDE.md must document rootless Docker is blocked (#549)."""
    content = _content()
    assert "rootless" in content.lower(), (
        "deployment/CLAUDE.md missing rootless Docker blocker note. "
        "NVIDIA CTK Ubuntu 24.04 bug blocks rootless Docker — document it (#549)."
    )


# ── T-3.7: Spot/on-demand fallback design document (#964) ────────────────────


SPOT_FALLBACK_DOC = ROOT / "docs" / "planning" / "spot-ondemand-fallback-design.md"


class TestSpotOnDemandFallbackDesign:
    """Verify the spot/on-demand fallback design document exists and has required sections."""

    def test_design_doc_exists(self) -> None:
        assert SPOT_FALLBACK_DOC.exists(), (
            "docs/planning/spot-ondemand-fallback-design.md does not exist. "
            "Create the design document for spot-preferred/on-demand fallback strategy."
        )

    def test_design_doc_has_decision_section(self) -> None:
        content = SPOT_FALLBACK_DOC.read_text(encoding="utf-8")
        assert "## Decision" in content or "## Recommendation" in content, (
            "Design doc missing '## Decision' or '## Recommendation' section. "
            "The document must contain a clear recommendation for the user."
        )

    def test_design_doc_has_cost_comparison(self) -> None:
        content = SPOT_FALLBACK_DOC.read_text(encoding="utf-8")
        assert "$" in content, (
            "Design doc missing cost comparison (no '$' found). "
            "Must include spot vs on-demand cost estimates for debug and production runs."
        )
