"""L3 Pulumi IaC validation tests (#623).

Validates Pulumi stacks using plain string checks and YAML/INI parsing.
No cloud credentials required — runs in staging.

UpCloud stack archived 2026-03-16: __main__.py moved to deployment/archived/upcloud/pulumi/
Active GCP stack: deployment/pulumi/gcp/__main__.py
Shared Pulumi.yaml/pyproject.toml: deployment/pulumi/ (still present)

Approach: read raw file content and check for required substrings.
Matches existing pattern in test_dockerfile_mlflow.py (reviewer consensus).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PULUMI_DIR = Path("deployment/pulumi")
# UpCloud __main__.py archived 2026-03-16 — still testable in archived location
PULUMI_UPCLOUD_DIR = Path("deployment/archived/upcloud/pulumi")


@pytest.mark.pulumi
class TestPulumiYamlConfig:
    """Validate Pulumi.yaml configuration."""

    def test_pulumi_yaml_parseable(self) -> None:
        """Pulumi.yaml is valid YAML with required fields."""
        config = yaml.safe_load(
            (PULUMI_DIR / "Pulumi.yaml").read_text(encoding="utf-8")
        )
        assert config["name"] == "minivess-mlflow"
        assert config["runtime"]["name"] == "python"

    def test_runtime_uses_uv_toolchain(self) -> None:
        """Runtime options specify toolchain: uv (not pip/venv)."""
        config = yaml.safe_load(
            (PULUMI_DIR / "Pulumi.yaml").read_text(encoding="utf-8")
        )
        assert config["runtime"]["options"]["toolchain"] == "uv"

    def test_config_keys_have_descriptions(self) -> None:
        """All config keys have description fields."""
        config = yaml.safe_load(
            (PULUMI_DIR / "Pulumi.yaml").read_text(encoding="utf-8")
        )
        for key, val in config.get("config", {}).items():
            if isinstance(val, dict):
                assert "description" in val or "default" in val, (
                    f"{key} needs description"
                )

    def test_mlflow_admin_password_is_secret(self) -> None:
        """mlflow_admin_password config is marked secret: true."""
        config = yaml.safe_load(
            (PULUMI_DIR / "Pulumi.yaml").read_text(encoding="utf-8")
        )
        pw_config = config["config"]["minivess-mlflow:mlflow_admin_password"]
        assert pw_config.get("secret") is True

    def test_pyproject_uses_uv(self) -> None:
        """pyproject.toml exists (uv toolchain) with required deps."""
        # UpCloud pyproject.toml still in deployment/pulumi/ (shared Pulumi root)
        pyproject = PULUMI_DIR / "pyproject.toml"
        assert pyproject.exists(), "Pulumi stack must use pyproject.toml (uv)"
        content = pyproject.read_text(encoding="utf-8")
        assert "pulumi" in content
        assert "pulumi-upcloud" in content  # archived stack deps still in pyproject
        assert "pulumi-command" in content


@pytest.mark.pulumi
class TestPulumiMainModuleTemplates:
    """Validate __main__.py templates using plain string checks.

    Approach: read raw file content, check for required substrings.
    Matches existing pattern in test_dockerfile_mlflow.py.
    No ast.parse() needed (reviewer consensus: over-engineered for f-strings).
    """

    @pytest.fixture()
    def main_content(self) -> str:
        return (PULUMI_UPCLOUD_DIR / "__main__.py").read_text(encoding="utf-8")

    def test_docker_compose_has_required_env_vars(self, main_content: str) -> None:
        """Deploy template includes all required env vars."""
        required = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "MLFLOW_S3_ENDPOINT_URL",
            "MLFLOW_FLASK_SERVER_SECRET_KEY",
            "MLFLOW_AUTH_CONFIG_PATH",
        ]
        for var in required:
            assert var in main_content, f"Missing env var: {var}"

    def test_basic_auth_ini_has_database_uri(self, main_content: str) -> None:
        """basic_auth.ini template includes database_uri field."""
        assert "database_uri" in main_content

    def test_basic_auth_ini_no_authorization_function(self, main_content: str) -> None:
        """basic_auth.ini does NOT include authorization_function.

        Broken in MLflow v2.20.3 — see #618.
        """
        # Extract the AUTHEOF section and check it doesn't contain the bad key
        auth_start = main_content.find("basic_auth.ini")
        auth_end = main_content.find("AUTHEOF", auth_start + 1)
        auth_section = main_content[auth_start:auth_end]
        assert "authorization_function" not in auth_section

    def test_postgres_public_access_enabled(self, main_content: str) -> None:
        """ManagedDatabasePostgresql has public_access: True."""
        assert '"public_access": True' in main_content
        assert '"ip_filters"' in main_content

    def test_uri_scheme_replacement(self, main_content: str) -> None:
        """postgres:// -> postgresql:// replacement is present."""
        assert '.replace("postgres://", "postgresql://", 1)' in main_content

    def test_dockerfile_required_packages(self, main_content: str) -> None:
        """Custom Dockerfile installs psycopg2-binary, boto3, flask-wtf."""
        for pkg in ["psycopg2-binary", "boto3", "flask-wtf"]:
            assert pkg in main_content, f"Missing pip package: {pkg}"

    def test_server_template_ubuntu_2404(self, main_content: str) -> None:
        """Server template uses Ubuntu 24.04 LTS."""
        assert "Ubuntu Server 24.04" in main_content

    def test_mlflow_version_pinned(self, main_content: str) -> None:
        """MLflow image version is pinned (not :latest)."""
        assert "mlflow:v" in main_content
        assert "mlflow:latest" not in main_content

    def test_healthcheck_configured(self, main_content: str) -> None:
        """Docker Compose template includes healthcheck."""
        assert "healthcheck" in main_content
        assert "/health" in main_content

    def test_no_hardcoded_passwords(self, main_content: str) -> None:
        """No hardcoded passwords in the template — all via Pulumi config."""
        # The f-string {args[4]} is the password placeholder — no literals
        assert "password123" not in main_content
        assert "changeme" not in main_content


@pytest.mark.pulumi
class TestPulumiResourceTypes:
    """Validate expected resource types in source code.

    Uses string checks on __main__.py — no Pulumi SDK required.
    """

    def test_required_resource_types_in_source(self) -> None:
        """Source code references all required UpCloud resource types."""
        content = (PULUMI_UPCLOUD_DIR / "__main__.py").read_text(encoding="utf-8")
        required_types = [
            "ManagedDatabasePostgresql",
            "ManagedObjectStorage",
            "ManagedObjectStorageBucket",
            "ManagedObjectStorageUser",
            "ManagedObjectStorageUserAccessKey",
            "Server",
        ]
        for rtype in required_types:
            assert rtype in content, f"Missing resource type: {rtype}"

    def test_remote_command_provisioning(self) -> None:
        """Source code uses pulumi-command for remote provisioning."""
        content = (PULUMI_UPCLOUD_DIR / "__main__.py").read_text(encoding="utf-8")
        assert "command.remote.Command" in content
        assert "install-docker" in content
        assert "deploy-mlflow" in content

    def test_required_outputs_exported(self) -> None:
        """Stack exports required output values."""
        content = (PULUMI_UPCLOUD_DIR / "__main__.py").read_text(encoding="utf-8")
        required_exports = [
            "server_ip",
            "mlflow_url",
            "mlflow_username",
            "postgres_host",
            "s3_endpoint",
            "s3_bucket",
            "ssh_command",
        ]
        for export_name in required_exports:
            # Handle both single-line and multi-line export calls
            assert (
                f'pulumi.export("{export_name}"' in content
                or "pulumi.export(\n" in content
                and f'"{export_name}"' in content
            ), f"Missing export: {export_name}"

    def test_depends_on_ordering(self) -> None:
        """Remote commands have explicit depends_on for ordering."""
        content = (PULUMI_UPCLOUD_DIR / "__main__.py").read_text(encoding="utf-8")
        assert "depends_on=[server]" in content
        assert "depends_on=[provision_docker]" in content


@pytest.mark.pulumi
class TestPulumiDvcBucket:
    """Validate DVC data bucket in Pulumi stack (#630, T0.1)."""

    @pytest.fixture()
    def main_content(self) -> str:
        return (PULUMI_UPCLOUD_DIR / "__main__.py").read_text(encoding="utf-8")

    def test_dvc_data_bucket_exists(self, main_content: str) -> None:
        """Pulumi stack creates a 'minivess-dvc-data' bucket."""
        assert "minivess-dvc-data" in main_content

    def test_dvc_bucket_exported(self, main_content: str) -> None:
        """Stack exports dvc_bucket output."""
        assert (
            'pulumi.export("dvc_bucket"' in main_content
            or '"dvc_bucket"' in main_content
            and "pulumi.export" in main_content
        )

    def test_dvc_s3_endpoint_exported(self, main_content: str) -> None:
        """Stack exports dvc_s3_endpoint output."""
        assert (
            'pulumi.export("dvc_s3_endpoint"' in main_content
            or '"dvc_s3_endpoint"' in main_content
            and "pulumi.export" in main_content
        )
