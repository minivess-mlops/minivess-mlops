"""Tests for configurable gate enforcement (T1).

Covers enforce_gate(), get_gate_severity(), GateAction enum,
and DataQualityError exception.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


class TestGateAction:
    """GateAction enum has expected members."""

    def test_gate_action_has_pass(self) -> None:
        from minivess.validation.enforcement import GateAction

        assert GateAction.PASS.value == "pass"

    def test_gate_action_has_warn(self) -> None:
        from minivess.validation.enforcement import GateAction

        assert GateAction.WARN.value == "warn"

    def test_gate_action_has_log(self) -> None:
        from minivess.validation.enforcement import GateAction

        assert GateAction.LOG.value == "log"

    def test_gate_action_has_halt(self) -> None:
        from minivess.validation.enforcement import GateAction

        assert GateAction.HALT.value == "halt"

    def test_gate_action_has_skip(self) -> None:
        from minivess.validation.enforcement import GateAction

        assert GateAction.SKIP.value == "skip"


class TestDataQualityError:
    """DataQualityError is a RuntimeError subclass."""

    def test_data_quality_error_is_runtime_error(self) -> None:
        from minivess.validation.enforcement import DataQualityError

        assert issubclass(DataQualityError, RuntimeError)

    def test_data_quality_error_message(self) -> None:
        from minivess.validation.enforcement import DataQualityError

        err = DataQualityError("pandera gate failed")
        assert "pandera" in str(err)


class TestEnforceGate:
    """enforce_gate() dispatches on severity level."""

    def test_enforce_gate_error_severity_failed_raises(self) -> None:
        """T1-R1: severity='error' + failed gate raises DataQualityError."""
        import pytest

        from minivess.validation.enforcement import (
            DataQualityError,
            enforce_gate,
        )
        from minivess.validation.gates import GateResult

        result = GateResult(passed=False, errors=["bad data"])
        with pytest.raises(DataQualityError):
            enforce_gate("pandera", result, severity="error")

    def test_enforce_gate_error_severity_passed_no_raise(self) -> None:
        """T1-R2: severity='error' + passed gate does NOT raise."""
        from minivess.validation.enforcement import GateAction, enforce_gate
        from minivess.validation.gates import GateResult

        result = GateResult(passed=True)
        action = enforce_gate("pandera", result, severity="error")
        assert action == GateAction.PASS

    def test_enforce_gate_warning_severity_returns_warn(self) -> None:
        """T1-R3: severity='warning' + failed gate returns GateAction.WARN."""
        from minivess.validation.enforcement import GateAction, enforce_gate
        from minivess.validation.gates import GateResult

        result = GateResult(passed=False, errors=["mild issue"])
        action = enforce_gate("ge", result, severity="warning")
        assert action == GateAction.WARN

    def test_enforce_gate_info_severity_returns_log(self) -> None:
        """T1-R4: severity='info' + failed gate returns GateAction.LOG."""
        from minivess.validation.enforcement import GateAction, enforce_gate
        from minivess.validation.gates import GateResult

        result = GateResult(passed=False, errors=["info-level issue"])
        action = enforce_gate("deepchecks", result, severity="info")
        assert action == GateAction.LOG

    def test_enforce_gate_reads_dynaconf_severity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """T1-R5: enforce_gate reads severity from Dynaconf when not overridden."""
        from minivess.validation.enforcement import GateAction, enforce_gate
        from minivess.validation.gates import GateResult

        # Monkeypatch get_gate_severity to return 'info'
        monkeypatch.setattr(
            "minivess.validation.enforcement.get_gate_severity",
            lambda gate_name: "info",
        )
        result = GateResult(passed=False, errors=["test"])
        action = enforce_gate("any_gate", result)
        assert action == GateAction.LOG

    def test_enforce_gate_skip_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """T1-R6: MINIVESS_SKIP_QUALITY_GATE=1 bypasses all gates."""
        from minivess.validation.enforcement import GateAction, enforce_gate
        from minivess.validation.gates import GateResult

        monkeypatch.setenv("MINIVESS_SKIP_QUALITY_GATE", "1")
        result = GateResult(passed=False, errors=["should be skipped"])
        action = enforce_gate("pandera", result, severity="error")
        assert action == GateAction.SKIP


class TestGetGateSeverity:
    """get_gate_severity() reads from Dynaconf settings."""

    def test_get_gate_severity_returns_configured(self) -> None:
        """T1-R7: Returns configured severity from settings.toml."""
        from minivess.validation.enforcement import get_gate_severity

        # pandera is configured as 'error' in settings.toml
        severity = get_gate_severity("pandera")
        assert severity == "error"

    def test_get_gate_severity_fallback_unknown(self) -> None:
        """T1-R8: Falls back to 'warning' for unknown gate names."""
        from minivess.validation.enforcement import get_gate_severity

        severity = get_gate_severity("nonexistent_gate_xyz")
        assert severity == "warning"
