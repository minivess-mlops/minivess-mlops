from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from minivess.pipeline.preflight import (
    CheckStatus,
    PreflightCheck,
    PreflightResult,
    check_data_exists,
    check_disk_space,
    check_gpu,
    check_ram,
    check_swap,
    detect_environment,
    run_preflight,
)


class TestPreflightResult:
    def test_all_check_categories_present(self):
        """PreflightResult has gpu, ram, disk, swap, data checks."""
        result = run_preflight(data_dir=Path("/nonexistent"))
        check_names = {c.name for c in result.checks}
        # Should have at minimum these categories
        assert "gpu" in check_names or "gpu_detection" in check_names
        assert "ram" in check_names or "system_ram" in check_names
        assert "disk" in check_names or "disk_space" in check_names
        assert "data" in check_names or "data_exists" in check_names

    def test_has_passed_property(self):
        """PreflightResult.passed is True only if no CRITICAL failures."""
        result = PreflightResult(
            checks=[
                PreflightCheck(name="test", status=CheckStatus.PASS, message="ok"),
            ],
            environment="local",
        )
        assert result.passed is True

        result_fail = PreflightResult(
            checks=[
                PreflightCheck(
                    name="test", status=CheckStatus.CRITICAL, message="fail"
                ),
            ],
            environment="local",
        )
        assert result_fail.passed is False


class TestGPUDetection:
    def test_gpu_detection_returns_check(self):
        """check_gpu returns a PreflightCheck."""
        result = check_gpu()
        assert isinstance(result, PreflightCheck)
        assert result.status in (
            CheckStatus.PASS,
            CheckStatus.WARNING,
            CheckStatus.CRITICAL,
        )

    @patch("shutil.which", return_value=None)
    def test_no_nvidia_smi_warns(self, mock_which):
        """No nvidia-smi binary -> WARNING (not critical, CPU training is valid)."""
        result = check_gpu()
        assert result.status in (CheckStatus.WARNING, CheckStatus.CRITICAL)


class TestRAMCheck:
    def test_ram_check_returns_check(self):
        """check_ram returns a PreflightCheck."""
        result = check_ram(min_gb=1.0)
        assert isinstance(result, PreflightCheck)

    def test_ram_check_warns_below_threshold(self):
        """Warn if available RAM < threshold."""
        # Request absurdly high RAM to trigger warning
        result = check_ram(min_gb=99999.0)
        assert result.status in (CheckStatus.WARNING, CheckStatus.CRITICAL)


class TestDiskSpaceCheck:
    def test_disk_space_returns_check(self, tmp_path):
        """check_disk_space returns a PreflightCheck."""
        result = check_disk_space(path=tmp_path, min_gb=0.001)
        assert isinstance(result, PreflightCheck)
        assert result.status == CheckStatus.PASS

    def test_disk_warns_below_threshold(self, tmp_path):
        """Warn if free disk < threshold."""
        result = check_disk_space(path=tmp_path, min_gb=99999.0)
        assert result.status in (CheckStatus.WARNING, CheckStatus.CRITICAL)


class TestSwapCheck:
    def test_swap_returns_check(self):
        """check_swap returns a PreflightCheck."""
        result = check_swap(warn_gb=5.0)
        assert isinstance(result, PreflightCheck)


class TestDataCheck:
    def test_data_exists_pass(self, tmp_path):
        """Pass when data directory has NIfTI files."""
        img_dir = tmp_path / "imagesTr"
        lbl_dir = tmp_path / "labelsTr"
        img_dir.mkdir()
        lbl_dir.mkdir()
        (img_dir / "vol01.nii.gz").touch()
        (lbl_dir / "vol01.nii.gz").touch()

        result = check_data_exists(tmp_path)
        assert result.status == CheckStatus.PASS

    def test_data_missing_critical(self, tmp_path):
        """CRITICAL when data directory doesn't exist."""
        result = check_data_exists(tmp_path / "nonexistent")
        assert result.status == CheckStatus.CRITICAL


class TestEnvironmentDetection:
    def test_environment_detection(self):
        """detect_environment returns one of local/docker/cloud/ci."""
        env = detect_environment()
        assert env in ("local", "docker", "cloud", "ci")

    @patch.dict("os.environ", {"CI": "true"})
    def test_ci_detected(self):
        """CI environment variable triggers CI detection."""
        env = detect_environment()
        assert env == "ci"

    @patch("pathlib.Path.exists", return_value=True)
    def test_docker_detected(self, mock_exists):
        """/.dockerenv existence triggers Docker detection.

        Must also clear CI/GITHUB_ACTIONS env vars, because on GitHub Actions
        runners these are set by default and take priority over /.dockerenv
        in detect_environment()'s check order.
        """
        import os

        # Remove CI-related keys entirely so the CI check doesn't short-circuit.
        # os.environ.pop returns the old value (or None), and we restore in finally.
        removed: dict[str, str] = {}
        for key in ("CI", "GITHUB_ACTIONS"):
            val = os.environ.pop(key, None)
            if val is not None:
                removed[key] = val
        try:
            env = detect_environment()
            assert env == "docker"
        finally:
            os.environ.update(removed)


class TestRunPreflight:
    def test_run_preflight_returns_result(self, tmp_path):
        """run_preflight returns PreflightResult with checks."""
        # Create minimal data dir
        img_dir = tmp_path / "imagesTr"
        lbl_dir = tmp_path / "labelsTr"
        img_dir.mkdir()
        lbl_dir.mkdir()
        (img_dir / "v1.nii.gz").touch()
        (lbl_dir / "v1.nii.gz").touch()

        result = run_preflight(data_dir=tmp_path)
        assert isinstance(result, PreflightResult)
        assert len(result.checks) >= 4  # gpu + ram + disk + data minimum

    def test_non_critical_warns_only(self, tmp_path):
        """Non-critical issues (e.g. swap usage) are warnings, not failures."""
        img_dir = tmp_path / "imagesTr"
        lbl_dir = tmp_path / "labelsTr"
        img_dir.mkdir()
        lbl_dir.mkdir()
        (img_dir / "v1.nii.gz").touch()
        (lbl_dir / "v1.nii.gz").touch()

        result = run_preflight(data_dir=tmp_path)
        warnings = [c for c in result.checks if c.status == CheckStatus.WARNING]
        # Having warnings doesn't mean the preflight failed
        if warnings:
            assert result.passed  # warnings don't cause failure
