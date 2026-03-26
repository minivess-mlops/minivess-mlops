"""Cross-file HuggingFace repo name consistency tests.

Validates that HF repo names in SkyPilot YAML setup blocks match
the source-of-truth constants in adapter source code.

Root cause: PR #940 fixed 'facebook/sam3-hiera-large' (404) →
'facebook/sam3', but a merge conflict resolution in commit 993768f1
silently reverted the fix. ALL SAM3 jobs in the 10th debug pass
failed at setup with 404. This test prevents that class of regression.

See: docs/planning/v0-2_archive/original_docs/run-debug-factorial-experiment-report-10th-pass-report.md
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Source-of-truth constants from adapter source code
# ---------------------------------------------------------------------------

ADAPTERS_DIR = Path("src/minivess/adapters")
SKYPILOT_DIR = Path("deployment/skypilot")

# SkyPilot YAMLs that pre-cache model weights in their setup blocks
SKYPILOT_YAMLS_WITH_HF_DOWNLOAD = [
    SKYPILOT_DIR / "train_factorial.yaml",
    SKYPILOT_DIR / "train_production.yaml",
    SKYPILOT_DIR / "train_hpo.yaml",
    SKYPILOT_DIR / "dev_runpod.yaml",
    SKYPILOT_DIR / "smoke_test_gpu.yaml",
    SKYPILOT_DIR / "smoke_test_gcp.yaml",
]


def _extract_hf_constant(filepath: Path, constant_name: str) -> str | None:
    """Extract a string constant from a Python source file using AST."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
        ):
            if node.target.id == constant_name and isinstance(
                node.value, ast.Constant
            ):
                return node.value.value
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == constant_name:
                    if isinstance(node.value, ast.Constant):
                        return node.value.value
    return None


def _extract_hf_repos_from_yaml_setup(yaml_path: Path) -> list[str]:
    """Extract HF repo names from hf_hub_download calls in YAML setup block."""
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    setup = cfg.get("setup", "")
    repos = []
    for line in setup.splitlines():
        if "hf_hub_download(" in line:
            # Extract repo name from hf_hub_download('repo', 'file')
            # Use string splitting (Rule 16: no regex for structured data)
            after_call = line.split("hf_hub_download(")[1]
            repo_str = after_call.split(",")[0].strip().strip("'\"")
            repos.append(repo_str)
        if "from_pretrained(" in line and "Sam3" in line:
            after_call = line.split("from_pretrained(")[1]
            repo_str = after_call.split(")")[0].strip().strip("'\"")
            repos.append(repo_str)
    return repos


# ---------------------------------------------------------------------------
# SAM3 HF repo consistency
# ---------------------------------------------------------------------------


class TestSam3HfRepoConsistency:
    """SAM3 HF repo name must be identical across source and all SkyPilot YAMLs."""

    def test_sam3_source_constant_exists(self) -> None:
        """SAM3_HF_MODEL_ID must be defined in sam3_backbone.py."""
        val = _extract_hf_constant(
            ADAPTERS_DIR / "sam3_backbone.py", "SAM3_HF_MODEL_ID"
        )
        assert val is not None, "SAM3_HF_MODEL_ID not found in sam3_backbone.py"
        assert val == "facebook/sam3", (
            f"SAM3_HF_MODEL_ID = '{val}', expected 'facebook/sam3'"
        )

    @pytest.mark.parametrize(
        "yaml_path",
        [
            p
            for p in SKYPILOT_YAMLS_WITH_HF_DOWNLOAD
            if p.exists() and _extract_hf_repos_from_yaml_setup(p)
        ],
        ids=lambda p: p.name,
    )
    def test_skypilot_yaml_uses_correct_sam3_repo(
        self, yaml_path: Path
    ) -> None:
        """Every SkyPilot YAML with HF calls must use 'facebook/sam3'."""
        repos = _extract_hf_repos_from_yaml_setup(yaml_path)
        sam3_repos = [r for r in repos if "sam3" in r.lower()]
        for repo in sam3_repos:
            assert repo == "facebook/sam3", (
                f"{yaml_path.name} uses '{repo}' for SAM3, "
                f"expected 'facebook/sam3'. "
                f"This was likely a merge conflict regression — "
                f"see PR #940 fix and commit 993768f1 revert."
            )

    def test_all_skypilot_yamls_match_source_constant(self) -> None:
        """Cross-file consistency: all YAML SAM3 repos == SAM3_HF_MODEL_ID."""
        source_id = _extract_hf_constant(
            ADAPTERS_DIR / "sam3_backbone.py", "SAM3_HF_MODEL_ID"
        )
        assert source_id is not None

        mismatches = []
        for yaml_path in SKYPILOT_YAMLS_WITH_HF_DOWNLOAD:
            if not yaml_path.exists():
                continue
            repos = _extract_hf_repos_from_yaml_setup(yaml_path)
            sam3_repos = [r for r in repos if "sam3" in r.lower()]
            for repo in sam3_repos:
                if repo != source_id:
                    mismatches.append(f"{yaml_path.name}: '{repo}'")

        assert not mismatches, (
            f"SAM3 HF repo mismatch vs source '{source_id}': "
            + ", ".join(mismatches)
        )


# ---------------------------------------------------------------------------
# VesselFM HF repo consistency
# ---------------------------------------------------------------------------


class TestVesselFmHfRepoConsistency:
    """VesselFM HF repo name must be identical across source and SkyPilot YAMLs."""

    def test_vesselfm_source_constant_exists(self) -> None:
        """VESSELFM_HF_REPO must be defined in vesselfm.py."""
        val = _extract_hf_constant(
            ADAPTERS_DIR / "vesselfm.py", "VESSELFM_HF_REPO"
        )
        assert val is not None, "VESSELFM_HF_REPO not found in vesselfm.py"
        assert val == "bwittmann/vesselFM", (
            f"VESSELFM_HF_REPO = '{val}', expected 'bwittmann/vesselFM'"
        )

    @pytest.mark.parametrize(
        "yaml_path",
        [
            p
            for p in SKYPILOT_YAMLS_WITH_HF_DOWNLOAD
            if p.exists()
            and any(
                "vessel" in r.lower()
                for r in _extract_hf_repos_from_yaml_setup(p)
            )
        ],
        ids=lambda p: p.name,
    )
    def test_skypilot_yaml_uses_correct_vesselfm_repo(
        self, yaml_path: Path
    ) -> None:
        """Every SkyPilot YAML with VesselFM calls must use 'bwittmann/vesselFM'."""
        repos = _extract_hf_repos_from_yaml_setup(yaml_path)
        vfm_repos = [r for r in repos if "vessel" in r.lower()]
        for repo in vfm_repos:
            assert repo == "bwittmann/vesselFM", (
                f"{yaml_path.name} uses '{repo}' for VesselFM, "
                f"expected 'bwittmann/vesselFM'"
            )


# ---------------------------------------------------------------------------
# Model config profile cross-check
# ---------------------------------------------------------------------------


class TestModelProfileHfRepoConsistency:
    """Model profile YAML hf_repo must match source constants."""

    @pytest.mark.parametrize(
        "profile_name,expected_repo",
        [
            ("sam3_vanilla", "facebook/sam3"),
            ("sam3_topolora", "facebook/sam3"),
            ("sam3_hybrid", "facebook/sam3"),
        ],
    )
    def test_model_profile_hf_repo(
        self, profile_name: str, expected_repo: str
    ) -> None:
        """Model profile YAML must declare the correct hf_repo."""
        profile_path = Path(f"configs/model_profiles/{profile_name}.yaml")
        if not profile_path.exists():
            pytest.skip(f"Profile {profile_name}.yaml not found")
        cfg = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        actual = (
            cfg.get("weights", {}).get("hf_repo")
            or cfg.get("model", {}).get("hf_repo")
            or cfg.get("hf_repo")
        )
        assert actual == expected_repo, (
            f"{profile_name}.yaml hf_repo='{actual}', expected '{expected_repo}'"
        )
