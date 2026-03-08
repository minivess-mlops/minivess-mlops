"""Meta-tests: verify test tier configuration exists and is correct.

Tests that the staging vs prod test tier split infrastructure is in place:
- pytest-staging.ini exists and contains expected configuration
- run_staging_tests.sh and run_prod_tests.sh exist and are executable
- pr_readiness_check.sh uses the staging tier by default
"""

from __future__ import annotations

import stat
from configparser import ConfigParser
from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]
STAGING_INI = REPO_ROOT / "pytest-staging.ini"
STAGING_SCRIPT = REPO_ROOT / "scripts" / "run_staging_tests.sh"
PROD_SCRIPT = REPO_ROOT / "scripts" / "run_prod_tests.sh"
PR_CHECK_SCRIPT = REPO_ROOT / "scripts" / "pr_readiness_check.sh"


class TestStagingConfig:
    def test_staging_config_exists(self) -> None:
        assert STAGING_INI.exists(), f"pytest-staging.ini not found at {STAGING_INI}"

    def test_staging_config_is_valid_ini(self) -> None:
        parser = ConfigParser()
        parser.read(STAGING_INI, encoding="utf-8")
        assert parser.has_section("pytest"), (
            "pytest-staging.ini missing [pytest] section"
        )

    def test_staging_excludes_slow(self) -> None:
        content = STAGING_INI.read_text(encoding="utf-8")
        assert "not slow" in content, "Staging config must exclude slow tests"

    def test_staging_excludes_integration(self) -> None:
        content = STAGING_INI.read_text(encoding="utf-8")
        assert "not integration" in content, (
            "Staging config must exclude integration tests"
        )

    def test_staging_excludes_e2e(self) -> None:
        content = STAGING_INI.read_text(encoding="utf-8")
        assert "not e2e" in content, "Staging config must exclude e2e tests"

    def test_staging_excludes_gpu(self) -> None:
        content = STAGING_INI.read_text(encoding="utf-8")
        assert "not gpu" in content, "Staging config must exclude gpu tests"

    def test_staging_has_testpaths(self) -> None:
        parser = ConfigParser()
        parser.read(STAGING_INI, encoding="utf-8")
        testpaths = parser.get("pytest", "testpaths", fallback="")
        assert testpaths, "Staging config must define testpaths"

    def test_staging_targets_unit_dirs(self) -> None:
        parser = ConfigParser()
        parser.read(STAGING_INI, encoding="utf-8")
        testpaths = parser.get("pytest", "testpaths", fallback="")
        assert "unit" in testpaths, "Staging config must target unit test directories"


class TestTierScripts:
    def test_staging_scripts_exist(self) -> None:
        assert STAGING_SCRIPT.exists(), (
            f"run_staging_tests.sh not found at {STAGING_SCRIPT}"
        )
        assert PROD_SCRIPT.exists(), f"run_prod_tests.sh not found at {PROD_SCRIPT}"

    def test_staging_script_is_executable(self) -> None:
        mode = STAGING_SCRIPT.stat().st_mode
        assert mode & stat.S_IXUSR, "run_staging_tests.sh must be executable"

    def test_prod_script_is_executable(self) -> None:
        mode = PROD_SCRIPT.stat().st_mode
        assert mode & stat.S_IXUSR, "run_prod_tests.sh must be executable"

    def test_staging_script_uses_staging_ini(self) -> None:
        content = STAGING_SCRIPT.read_text(encoding="utf-8")
        assert "pytest-staging.ini" in content, (
            "run_staging_tests.sh must reference pytest-staging.ini"
        )

    def test_prod_script_runs_all_tests(self) -> None:
        content = PROD_SCRIPT.read_text(encoding="utf-8")
        assert "tests/" in content, (
            "run_prod_tests.sh must target the full tests/ directory"
        )


class TestPrReadinessCheck:
    def test_pr_check_uses_staging_by_default(self) -> None:
        content = PR_CHECK_SCRIPT.read_text(encoding="utf-8")
        assert "pytest-staging.ini" in content, (
            "pr_readiness_check.sh must use staging tier by default"
        )

    def test_pr_check_supports_full_flag(self) -> None:
        content = PR_CHECK_SCRIPT.read_text(encoding="utf-8")
        assert "--full" in content, (
            "pr_readiness_check.sh must support --full flag for prod tier"
        )
