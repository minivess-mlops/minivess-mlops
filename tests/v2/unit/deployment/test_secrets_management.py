"""Tests for SOPS+age secrets management scaffolding — T-03.1 / issues #550, #538.

Verifies:
- .sops.yaml exists with creation_rules for .env files and age key groups
- scripts/setup_dev.sh exists, references SOPS, validates required env vars
- scripts/_rotate_credentials.py exists, uses str.split (not regex), uses secrets module
- .env.example documents SOPS usage

Rule #16: No regex — yaml.safe_load(), str.split(), pathlib.Path, str.partition().
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent.parent
SOPS_YAML = ROOT / ".sops.yaml"
SETUP_DEV = ROOT / "scripts" / "setup_dev.sh"
ROTATE_CREDS = ROOT / "scripts" / "_rotate_credentials.py"
ENV_EXAMPLE = ROOT / ".env.example"


def test_sops_yaml_exists() -> None:
    assert SOPS_YAML.exists(), (
        ".sops.yaml not found. Create SOPS config with creation_rules for .env files."
    )


def test_sops_yaml_has_creation_rules() -> None:
    config = yaml.safe_load(SOPS_YAML.read_text(encoding="utf-8"))
    assert "creation_rules" in config, (
        ".sops.yaml missing 'creation_rules' key. "
        "Add at least one rule targeting .env files."
    )
    rules = config["creation_rules"]
    assert isinstance(rules, list) and len(rules) >= 1, (
        ".sops.yaml creation_rules must be a non-empty list."
    )


def test_sops_yaml_has_env_rule_with_age_key_group() -> None:
    config = yaml.safe_load(SOPS_YAML.read_text(encoding="utf-8"))
    rules = config.get("creation_rules", [])
    env_rules = [r for r in rules if ".env" in r.get("path_regex", "")]
    assert env_rules, (
        ".sops.yaml has no creation_rule targeting .env files. "
        "Add: path_regex: '\\.env(\\.enc)?$'"
    )
    for rule in env_rules:
        has_age = "age" in rule or "key_groups" in rule
        assert has_age, (
            f"SOPS rule for .env has no 'age' or 'key_groups': {rule}. "
            "Add age public key placeholder."
        )


def test_setup_dev_sh_exists() -> None:
    assert SETUP_DEV.exists(), (
        "scripts/setup_dev.sh not found. "
        "Create setup script that decrypts .env.enc or validates existing .env."
    )


def test_setup_dev_sh_references_sops() -> None:
    content = SETUP_DEV.read_text(encoding="utf-8")
    has_sops = "sops" in content.lower() or "decrypt" in content.lower()
    assert has_sops, "scripts/setup_dev.sh must reference SOPS for .env decryption."


def test_setup_dev_sh_validates_required_env_vars() -> None:
    content = SETUP_DEV.read_text(encoding="utf-8")
    assert "MODEL_CACHE_HOST_PATH" in content, (
        "scripts/setup_dev.sh must check MODEL_CACHE_HOST_PATH is set. "
        "This is a required var with no fallback (machine-specific path)."
    )


def test_rotate_credentials_exists() -> None:
    assert ROTATE_CREDS.exists(), (
        "scripts/_rotate_credentials.py not found. "
        "Create Python credential rotation script."
    )


def test_rotate_credentials_uses_no_regex() -> None:
    """Rule #16: regex banned for structured data (.env key=value parsing)."""
    content = ROTATE_CREDS.read_text(encoding="utf-8")
    assert "import re" not in content, (
        "Rule #16 violation: scripts/_rotate_credentials.py must not use 'import re'. "
        "Parse .env with str.partition('=') or str.split('=', 1)."
    )


def test_rotate_credentials_uses_str_split_for_parsing() -> None:
    content = ROTATE_CREDS.read_text(encoding="utf-8")
    has_split_or_partition = ".split(" in content or ".partition(" in content
    assert has_split_or_partition, (
        "scripts/_rotate_credentials.py must use str.split() or str.partition() "
        "to parse .env lines — not sed or regex (Rule #16)."
    )


def test_rotate_credentials_uses_secrets_module() -> None:
    content = ROTATE_CREDS.read_text(encoding="utf-8")
    assert "import secrets" in content, (
        "scripts/_rotate_credentials.py must use 'import secrets' "
        "for cryptographically secure credential generation."
    )


def test_env_example_documents_sops_team_setup() -> None:
    content = ENV_EXAMPLE.read_text(encoding="utf-8")
    has_sops_ref = "sops" in content.lower() or "setup_dev" in content.lower()
    assert has_sops_ref, (
        ".env.example must document SOPS team setup workflow. "
        "Add a comment block referencing scripts/setup_dev.sh."
    )
