"""Evals for the kg-sync Skill.

Tests that the knowledge graph structure supports the kg-sync workflow:
code-structure YAML files exist, experiment YAML files are valid, and
the projection system (KG → .tex) has correct cross-references.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
KG_DIR = REPO_ROOT / "knowledge-graph"


class TestCodeStructureFilesEval:
    """Eval: code-structure YAML files must exist and parse."""

    @pytest.fixture(
        params=[
            "flows.yaml",
            "adapters.yaml",
            "config-schema.yaml",
            "test-coverage.yaml",
        ]
    )
    def code_structure_file(self, request: pytest.FixtureRequest) -> Path:
        return KG_DIR / "code-structure" / str(request.param)

    def test_file_exists(self, code_structure_file: Path) -> None:
        assert code_structure_file.exists(), f"Missing: {code_structure_file.name}"

    def test_file_is_valid_yaml(self, code_structure_file: Path) -> None:
        data = yaml.safe_load(code_structure_file.read_text(encoding="utf-8"))
        assert isinstance(data, dict | list), (
            f"{code_structure_file.name} must be YAML dict or list"
        )


class TestDomainFilesEval:
    """Eval: all domain YAML files must be valid and have required fields."""

    def _domain_files(self) -> list[Path]:
        return sorted((KG_DIR / "domains").glob("*.yaml"))

    def test_at_least_8_domains(self) -> None:
        domains = self._domain_files()
        assert len(domains) >= 8, f"Expected >=8 domain files, got {len(domains)}"

    def test_all_domains_have_description(self) -> None:
        for domain_file in self._domain_files():
            data = yaml.safe_load(domain_file.read_text(encoding="utf-8"))
            assert "description" in data, f"{domain_file.name} missing 'description'"
            assert "domain" in data, f"{domain_file.name} missing 'domain'"

    def test_cloud_domain_has_providers(self) -> None:
        """Cloud domain must list providers (prevents AWS S3 repeat)."""
        cloud = yaml.safe_load(
            (KG_DIR / "domains" / "cloud.yaml").read_text(encoding="utf-8")
        )
        assert "providers" in cloud, "cloud.yaml must have 'providers' section"
        assert "gcp" in cloud["providers"], "cloud.yaml must list GCP provider"
        assert "runpod" in cloud["providers"], "cloud.yaml must list RunPod provider"

    def test_cloud_domain_has_gcs_buckets(self) -> None:
        """Cloud domain must document GCS bucket names."""
        cloud = yaml.safe_load(
            (KG_DIR / "domains" / "cloud.yaml").read_text(encoding="utf-8")
        )
        gcp = cloud.get("providers", {}).get("gcp", {})
        buckets = gcp.get("gcs_buckets", {})
        assert "dvc_data" in buckets, "cloud.yaml GCP must have dvc_data bucket"
        assert "minivess-mlops-dvc-data" in str(buckets.get("dvc_data", "")), (
            "DVC bucket must be minivess-mlops-dvc-data (GCS), not AWS S3"
        )


class TestNavigatorKeywordsEval:
    """Eval: navigator keywords must route to correct domains."""

    def test_cloud_keywords_route_correctly(self) -> None:
        """Cloud-related keywords must route to 'cloud' domain, not 'infrastructure'."""
        nav = yaml.safe_load((KG_DIR / "navigator.yaml").read_text(encoding="utf-8"))
        keywords = nav.get("keywords", {})
        cloud_keywords = ["gcp", "gcs", "pulumi", "runpod", "skypilot"]
        for kw in cloud_keywords:
            assert keywords.get(kw) == "cloud", (
                f"Keyword '{kw}' must route to 'cloud' domain, got '{keywords.get(kw)}'"
            )

    def test_openspec_keywords_exist(self) -> None:
        """OpenSpec keywords must exist in navigator."""
        nav = yaml.safe_load((KG_DIR / "navigator.yaml").read_text(encoding="utf-8"))
        keywords = nav.get("keywords", {})
        assert "openspec" in keywords, "Navigator must have 'openspec' keyword"
        assert "sdd" in keywords, "Navigator must have 'sdd' keyword"
