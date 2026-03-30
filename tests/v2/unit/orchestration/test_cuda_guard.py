"""Tests for CUDA availability guard (Phase 1, Tasks 1.1 + 1.3).

Mirrors docker_guard.py pattern: fail-fast when CUDA unavailable,
with MINIVESS_ALLOW_CPU=1 escape hatch for pytest only.
Also tests CUDA version mismatch detection.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ── Task 1.1: require_cuda_context() ──────────────────────────────────


class TestRequireCudaContextImportable:
    def test_function_exists(self) -> None:
        from minivess.orchestration.cuda_guard import require_cuda_context

        assert callable(require_cuda_context)


class TestRequireCudaContextRaises:
    """Must raise RuntimeError when CUDA unavailable."""

    @patch("minivess.orchestration.cuda_guard.torch")
    def test_raises_when_cuda_unavailable(self, mock_torch: MagicMock) -> None:
        from minivess.orchestration.cuda_guard import require_cuda_context

        mock_torch.cuda.is_available.return_value = False
        mock_torch.version.cuda = "12.8"

        with pytest.raises(RuntimeError, match="CUDA not available"):
            require_cuda_context("train")

    @patch("minivess.orchestration.cuda_guard.torch")
    def test_error_message_contains_diagnostics(self, mock_torch: MagicMock) -> None:
        from minivess.orchestration.cuda_guard import require_cuda_context

        mock_torch.cuda.is_available.return_value = False
        mock_torch.version.cuda = "13.0"

        with pytest.raises(RuntimeError, match="13.0"):
            require_cuda_context("train")

    @patch("minivess.orchestration.cuda_guard.torch")
    def test_error_message_contains_flow_name(self, mock_torch: MagicMock) -> None:
        from minivess.orchestration.cuda_guard import require_cuda_context

        mock_torch.cuda.is_available.return_value = False
        mock_torch.version.cuda = "12.8"

        with pytest.raises(RuntimeError, match="train"):
            require_cuda_context("train")


class TestRequireCudaContextPasses:
    """Must be a no-op when CUDA IS available."""

    @patch("minivess.orchestration.cuda_guard.torch")
    def test_noop_when_cuda_available(self, mock_torch: MagicMock) -> None:
        from minivess.orchestration.cuda_guard import require_cuda_context

        mock_torch.cuda.is_available.return_value = True
        # Should not raise
        require_cuda_context("train")


class TestRequireCudaContextEscapeHatch:
    """MINIVESS_ALLOW_CPU=1 bypasses the guard (pytest only)."""

    @patch("minivess.orchestration.cuda_guard.torch")
    def test_allow_cpu_bypasses_guard(self, mock_torch: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        from minivess.orchestration.cuda_guard import require_cuda_context

        mock_torch.cuda.is_available.return_value = False
        monkeypatch.setenv("MINIVESS_ALLOW_CPU", "1")
        # Should not raise
        require_cuda_context("train")

    @patch("minivess.orchestration.cuda_guard.torch")
    def test_allow_cpu_not_set_still_raises(self, mock_torch: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        from minivess.orchestration.cuda_guard import require_cuda_context

        mock_torch.cuda.is_available.return_value = False
        mock_torch.version.cuda = "12.8"
        monkeypatch.delenv("MINIVESS_ALLOW_CPU", raising=False)

        with pytest.raises(RuntimeError):
            require_cuda_context("train")


# ── Task 1.3: detect_cuda_version_mismatch() ─────────────────────────


class TestCudaVersionMismatchDetection:
    """Advisory detection of CUDA version mismatch."""

    def test_function_exists(self) -> None:
        from minivess.orchestration.cuda_guard import detect_cuda_version_mismatch

        assert callable(detect_cuda_version_mismatch)

    @patch("minivess.orchestration.cuda_guard.torch")
    def test_returns_result_dataclass(self, mock_torch: MagicMock) -> None:
        from minivess.orchestration.cuda_guard import detect_cuda_version_mismatch

        mock_torch.version.cuda = "12.8"
        result = detect_cuda_version_mismatch()
        assert hasattr(result, "mismatch_detected")
        assert hasattr(result, "pytorch_cuda_version")

    @patch("minivess.orchestration.cuda_guard.torch")
    def test_no_mismatch_when_compatible(self, mock_torch: MagicMock) -> None:
        from minivess.orchestration.cuda_guard import detect_cuda_version_mismatch

        mock_torch.version.cuda = "12.8"
        mock_torch.cuda.is_available.return_value = True
        result = detect_cuda_version_mismatch()
        assert result.mismatch_detected is False

    @patch("minivess.orchestration.cuda_guard.torch")
    def test_mismatch_when_cuda_unavailable(self, mock_torch: MagicMock) -> None:
        from minivess.orchestration.cuda_guard import detect_cuda_version_mismatch

        mock_torch.version.cuda = "13.0"
        mock_torch.cuda.is_available.return_value = False
        result = detect_cuda_version_mismatch()
        assert result.mismatch_detected is True
        assert result.pytorch_cuda_version == "13.0"
