"""Tests for auto-resume config fingerprinting (T-3.1).

Tests compute_config_fingerprint() determinism and find_completed_config()
with mocked MLflow backend.
"""

from __future__ import annotations

from minivess.pipeline.resume_discovery import (
    compute_config_fingerprint,
    find_completed_config,
    load_fold_result_from_mlflow,
)


class TestConfigFingerprint:
    def test_deterministic(self) -> None:
        """Same inputs produce identical fingerprints."""
        fp1 = compute_config_fingerprint("dice_ce", "dynunet", 0, 100, 2)
        fp2 = compute_config_fingerprint("dice_ce", "dynunet", 0, 100, 2)
        assert fp1 == fp2

    def test_different_loss_different_fingerprint(self) -> None:
        """Different loss names produce different fingerprints."""
        fp1 = compute_config_fingerprint("dice_ce", "dynunet", 0, 100, 2)
        fp2 = compute_config_fingerprint("cbdice_cldice", "dynunet", 0, 100, 2)
        assert fp1 != fp2

    def test_different_fold_different_fingerprint(self) -> None:
        """Different fold IDs produce different fingerprints."""
        fp1 = compute_config_fingerprint("dice_ce", "dynunet", 0, 100, 2)
        fp2 = compute_config_fingerprint("dice_ce", "dynunet", 1, 100, 2)
        assert fp1 != fp2

    def test_with_patch_size(self) -> None:
        """Patch size is included in fingerprint."""
        fp1 = compute_config_fingerprint("dice_ce", "dynunet", 0, 100, 2, (64, 64, 16))
        fp2 = compute_config_fingerprint("dice_ce", "dynunet", 0, 100, 2, (64, 64, 3))
        assert fp1 != fp2

    def test_fingerprint_is_hex_string(self) -> None:
        """Fingerprint is a 16-char hex string."""
        fp = compute_config_fingerprint("dice_ce", "dynunet", 0, 100, 2)
        assert len(fp) == 16
        int(fp, 16)  # Should not raise


class TestFindCompletedConfig:
    def test_returns_none_when_no_experiment(self) -> None:
        """Returns None when experiment doesn't exist."""
        result = find_completed_config(
            tracking_uri="/nonexistent",
            experiment_name="nonexistent",
            config_fingerprint="abc123",
        )
        assert result is None


class TestLoadFoldResult:
    def test_returns_resume_failed_on_error(self) -> None:
        """Returns resume_failed status when MLflow run can't be loaded."""
        result = load_fold_result_from_mlflow(
            tracking_uri="/nonexistent",
            run_id="fake-run-id",
        )
        assert result["status"] == "resume_failed"
        assert result["run_id"] == "fake-run-id"
