"""Tests for NVIDIA Container Toolkit version — CVE-2025-23266 mitigation.

CVE-2025-23266 (CVSS 9.0 Critical, NVIDIAScape): container-to-host escape.
Fixed in Container Toolkit v1.17.8+.

Rule #16: No regex. Parse version with str.split().
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent.parent.parent

_SAFE_MIN = (1, 17, 8)


def _ctk_installed() -> bool:
    try:
        subprocess.run(
            ["nvidia-ctk", "--version"],
            capture_output=True,
            check=False,
            timeout=5,
        )
        return True
    except FileNotFoundError:
        return False


_skip_no_ctk = pytest.mark.skipif(
    not _ctk_installed(), reason="nvidia-ctk not installed"
)


@_skip_no_ctk
def test_nvidia_ctk_version_is_safe() -> None:
    """nvidia-ctk version must be >= 1.17.8 (CVE-2025-23266 fix)."""
    result = subprocess.run(
        ["nvidia-ctk", "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    # Output: "NVIDIA Container Toolkit CLI version 1.17.8"
    # Use str.split() — Rule #16 bans regex
    output = result.stdout.strip() or result.stderr.strip()
    parts = output.split()
    # Find the token that looks like a version (contains dots and digits)
    version_str = None
    for part in parts:
        if part.count(".") >= 1 and all(c.isdigit() or c == "." for c in part):
            version_str = part
            break

    assert version_str is not None, (
        f"Could not find version string in nvidia-ctk output: {output!r}"
    )

    version_parts = version_str.split(".")
    version_tuple = tuple(int(p) for p in version_parts[:3])

    assert version_tuple >= _SAFE_MIN, (
        f"nvidia-ctk {version_str} is BELOW safe minimum {'.'.join(str(x) for x in _SAFE_MIN)}. "
        f"CVE-2025-23266 (CVSS 9.0) is unpatched! "
        f"Fix: sudo apt-get install --only-upgrade nvidia-container-toolkit"
    )


def _parse_ctk_config_hook_disabled(cfg: dict) -> bool:
    """Check if CUDA compat hook is disabled in CTK config.

    Extracted for testability without requiring /etc/nvidia-container-toolkit.
    """
    return bool(
        cfg.get("nvidia-container-cli", {}).get("disable-cuda-compat-lib-hook", False)
    )


_CTK_CONFIG_PATHS = [
    Path("/etc/nvidia-container-toolkit/config.toml"),
    Path("/etc/nvidia-container-runtime/config.toml"),
]


def _find_ctk_config() -> Path | None:
    """Find CTK config file on this machine."""
    for p in _CTK_CONFIG_PATHS:
        if p.exists():
            return p
    return None


def test_cuda_compat_hook_not_disabled_by_config() -> None:
    """If CTK config exists on this machine, compat hook must not be disabled."""
    config_path = _find_ctk_config()
    if config_path is None:
        pytest.skip("CTK config.toml not found at any known path")

    with config_path.open("rb") as f:
        cfg = tomllib.load(f)

    assert not _parse_ctk_config_hook_disabled(cfg), (
        f"disable-cuda-compat-lib-hook = true in {config_path}. "
        "This disables the compatibility hook — set to false or remove the key."
    )


def test_hook_check_logic_detects_disabled() -> None:
    """Parser must detect when cuda-compat-lib-hook is disabled."""
    cfg_disabled = {"nvidia-container-cli": {"disable-cuda-compat-lib-hook": True}}
    assert _parse_ctk_config_hook_disabled(cfg_disabled) is True


def test_hook_check_logic_accepts_enabled() -> None:
    """Parser must accept when hook is not disabled (default or explicit)."""
    cfg_enabled = {"nvidia-container-cli": {"disable-cuda-compat-lib-hook": False}}
    assert _parse_ctk_config_hook_disabled(cfg_enabled) is False


def test_hook_check_logic_accepts_missing_key() -> None:
    """Missing key means hook is enabled (default behavior)."""
    cfg_minimal: dict[str, dict[str, bool]] = {}
    assert _parse_ctk_config_hook_disabled(cfg_minimal) is False

    cfg_no_key: dict[str, dict[str, bool]] = {"nvidia-container-cli": {}}
    assert _parse_ctk_config_hook_disabled(cfg_no_key) is False
