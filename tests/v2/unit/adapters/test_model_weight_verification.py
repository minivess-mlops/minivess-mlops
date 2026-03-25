"""Tests for SHA-256 hash pinning of HuggingFace model weight downloads.

TDD Task T3: Verify that model weight integrity is enforced via SHA-256 hashes.
- VesselFM: file-level SHA-256 hash (VESSELFM_WEIGHT_SHA256 must not be None)
- SAM3: HuggingFace revision pin (weights.hf_repo field in model profiles)
- Generic: verify_weight_sha256() helper for download verification

Uses compute_checkpoint_sha256() from checkpoint_integrity module.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


class TestWeightHashVerification:
    """Generic SHA-256 weight verification logic."""

    def test_verify_sha256_matches(self, tmp_path: Path) -> None:
        """Create file, compute hash, verify with matching hash returns True."""
        from minivess.pipeline.checkpoint_integrity import compute_checkpoint_sha256
        from minivess.pipeline.weight_verification import verify_weight_sha256

        # Create a test file with known content
        test_file = tmp_path / "model.pt"
        test_file.write_bytes(b"fake model weights for testing")

        # Compute the real hash
        expected_hash = compute_checkpoint_sha256(test_file)

        # Verify should return True when hash matches
        result = verify_weight_sha256(test_file, expected_sha256=expected_hash)
        assert result is True

    def test_verify_sha256_mismatch_raises(self, tmp_path: Path) -> None:
        """Wrong hash raises ValueError."""
        from minivess.pipeline.weight_verification import verify_weight_sha256

        test_file = tmp_path / "model.pt"
        test_file.write_bytes(b"fake model weights for testing")

        wrong_hash = "0" * 64  # 64-char hex string, almost certainly wrong

        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            verify_weight_sha256(test_file, expected_sha256=wrong_hash)

    def test_verify_sha256_missing_file_raises(self) -> None:
        """Nonexistent path raises FileNotFoundError."""
        from minivess.pipeline.weight_verification import verify_weight_sha256

        nonexistent = Path("/tmp/does_not_exist_abc123.pt")
        with pytest.raises(FileNotFoundError):
            verify_weight_sha256(nonexistent, expected_sha256="a" * 64)

    def test_verify_sha256_none_returns_computed(self, tmp_path: Path) -> None:
        """None expected hash returns computed hash string (first-download mode)."""
        from minivess.pipeline.checkpoint_integrity import compute_checkpoint_sha256
        from minivess.pipeline.weight_verification import verify_weight_sha256

        test_file = tmp_path / "model.pt"
        test_file.write_bytes(b"first download content")

        # When expected_sha256 is None, function should return the computed hash
        result = verify_weight_sha256(test_file, expected_sha256=None)
        expected = compute_checkpoint_sha256(test_file)
        assert result == expected
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex digest length


class TestVesselFMWeightHash:
    """VesselFM weight hash must be pinned (not None)."""

    def test_vesselfm_weight_sha256_is_not_none(self) -> None:
        """VESSELFM_WEIGHT_SHA256 must NOT be None (disabled = security gap)."""
        from minivess.adapters.vesselfm import VESSELFM_WEIGHT_SHA256

        assert VESSELFM_WEIGHT_SHA256 is not None, (
            "VESSELFM_WEIGHT_SHA256 is None — weight verification is disabled. "
            "Set to a placeholder string or the actual SHA-256 hash."
        )

    def test_vesselfm_weight_sha256_is_string(self) -> None:
        """VESSELFM_WEIGHT_SHA256 must be a non-empty string."""
        from minivess.adapters.vesselfm import VESSELFM_WEIGHT_SHA256

        assert isinstance(VESSELFM_WEIGHT_SHA256, str)
        assert len(VESSELFM_WEIGHT_SHA256) > 0


class TestSAM3ModelRevisionPinned:
    """SAM3 model profiles must have weights.hf_repo field for revision pinning."""

    @staticmethod
    def _sam3_profile_paths() -> list[Path]:
        """Return all sam3_*.yaml profile paths."""
        profiles_dir = Path("configs/model_profiles")
        profiles = sorted(profiles_dir.glob("sam3_*.yaml"))
        assert len(profiles) > 0, "No sam3_*.yaml profiles found"
        return profiles

    def test_sam3_profiles_have_hf_repo(self) -> None:
        """All sam3_*.yaml must have weights.hf_repo field."""
        for profile_path in self._sam3_profile_paths():
            with open(profile_path, encoding="utf-8") as f:
                profile = yaml.safe_load(f)

            assert "weights" in profile, (
                f"{profile_path.name} missing 'weights' section"
            )
            assert "hf_repo" in profile["weights"], (
                f"{profile_path.name} missing 'weights.hf_repo' field"
            )
            assert isinstance(profile["weights"]["hf_repo"], str), (
                f"{profile_path.name}: weights.hf_repo must be a string"
            )
            assert len(profile["weights"]["hf_repo"]) > 0, (
                f"{profile_path.name}: weights.hf_repo must not be empty"
            )

    def test_sam3_profiles_have_hf_revision(self) -> None:
        """All sam3_*.yaml must have weights.hf_revision field."""
        for profile_path in self._sam3_profile_paths():
            with open(profile_path, encoding="utf-8") as f:
                profile = yaml.safe_load(f)

            assert "weights" in profile, (
                f"{profile_path.name} missing 'weights' section"
            )
            assert "hf_revision" in profile["weights"], (
                f"{profile_path.name} missing 'weights.hf_revision' field"
            )
            # hf_revision can be a commit SHA or branch name — must be non-empty string
            revision = profile["weights"]["hf_revision"]
            assert isinstance(revision, str), (
                f"{profile_path.name}: weights.hf_revision must be a string"
            )
            assert len(revision) > 0, (
                f"{profile_path.name}: weights.hf_revision must not be empty"
            )


class TestVerifyScriptExists:
    """The verify_model_weights.py script must exist for manual verification."""

    def test_verify_script_exists(self) -> None:
        """scripts/verify_model_weights.py must exist."""
        assert Path("scripts/verify_model_weights.py").exists(), (
            "scripts/verify_model_weights.py does not exist — "
            "create it for manual weight verification"
        )
