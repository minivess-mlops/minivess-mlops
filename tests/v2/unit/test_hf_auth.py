"""Tests for minivess.utils.hf_auth — HuggingFace token loading utilities."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from minivess.utils.hf_auth import (
    get_hf_token,
    hf_token_status,
    load_dotenv_if_present,
    require_hf_token,
)

# ── load_dotenv_if_present ────────────────────────────────────────────────────


class TestLoadDotenvIfPresent:
    def test_returns_false_when_file_missing(self, tmp_path: Path) -> None:
        result = load_dotenv_if_present(tmp_path / "nonexistent.env")
        assert result is False

    def test_returns_true_when_file_exists(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR=hello\n", encoding="utf-8")
        result = load_dotenv_if_present(env_file)
        assert result is True

    def test_loads_key_value_pairs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("MY_SECRET=abc123\n", encoding="utf-8")
        monkeypatch.delenv("MY_SECRET", raising=False)
        load_dotenv_if_present(env_file)
        assert os.environ.get("MY_SECRET") == "abc123"

    def test_strips_quotes_from_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text('QUOTED="hf_token_value"\n', encoding="utf-8")
        monkeypatch.delenv("QUOTED", raising=False)
        load_dotenv_if_present(env_file)
        assert os.environ.get("QUOTED") == "hf_token_value"

    def test_skips_comments_and_blank_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# comment\n\nVALID_KEY=yes\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("VALID_KEY", raising=False)
        load_dotenv_if_present(env_file)
        assert os.environ.get("VALID_KEY") == "yes"

    def test_does_not_override_existing_env_vars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING=from_file\n", encoding="utf-8")
        monkeypatch.setenv("EXISTING", "from_env")
        load_dotenv_if_present(env_file)
        assert os.environ["EXISTING"] == "from_env"  # env wins

    def test_idempotent_when_called_twice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("IDEM_KEY=value\n", encoding="utf-8")
        monkeypatch.delenv("IDEM_KEY", raising=False)
        load_dotenv_if_present(env_file)
        load_dotenv_if_present(env_file)  # second call is no-op
        assert os.environ.get("IDEM_KEY") == "value"


# ── get_hf_token ──────────────────────────────────────────────────────────────


class TestGetHfToken:
    def test_returns_none_when_nothing_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        # Point cache path to a non-existent file
        import minivess.utils.hf_auth as hf_auth

        monkeypatch.setattr(hf_auth, "_HF_TOKEN_CACHE", tmp_path / "token")
        assert get_hf_token() is None

    def test_reads_HF_TOKEN_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_test_token")
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        assert get_hf_token() == "hf_test_token"

    def test_reads_legacy_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_legacy_token")
        assert get_hf_token() == "hf_legacy_token"

    def test_HF_TOKEN_takes_priority_over_legacy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_primary")
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_secondary")
        assert get_hf_token() == "hf_primary"

    def test_reads_cached_token_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        cache = tmp_path / "token"
        cache.write_text("hf_cached_token\n", encoding="utf-8")
        import minivess.utils.hf_auth as hf_auth

        monkeypatch.setattr(hf_auth, "_HF_TOKEN_CACHE", cache)
        assert get_hf_token() == "hf_cached_token"


# ── require_hf_token ──────────────────────────────────────────────────────────


class TestRequireHfToken:
    def test_raises_runtime_error_when_no_token(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        import minivess.utils.hf_auth as hf_auth

        monkeypatch.setattr(hf_auth, "_HF_TOKEN_CACHE", tmp_path / "token")
        with pytest.raises(RuntimeError, match="HuggingFace token required"):
            require_hf_token("facebook/sam3")

    def test_returns_token_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_valid")
        result = require_hf_token("facebook/sam3")
        assert result == "hf_valid"


# ── hf_token_status ───────────────────────────────────────────────────────────


class TestHfTokenStatus:
    def test_unauthenticated_status(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        import minivess.utils.hf_auth as hf_auth

        monkeypatch.setattr(hf_auth, "_HF_TOKEN_CACHE", tmp_path / "token")
        status = hf_token_status()
        assert status["authenticated"] is False
        assert status["source"] == "none"

    def test_authenticated_via_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_abcdefgh")
        status = hf_token_status()
        assert status["authenticated"] is True
        assert "HF_TOKEN" in status["source"]
        assert isinstance(status["token_prefix"], str)
        assert status["token_prefix"].startswith("hf_abcd")

    def test_authenticated_via_cache(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        cache = tmp_path / "token"
        cache.write_text("hf_from_cache", encoding="utf-8")
        import minivess.utils.hf_auth as hf_auth

        monkeypatch.setattr(hf_auth, "_HF_TOKEN_CACHE", cache)
        status = hf_token_status()
        assert status["authenticated"] is True
        assert "cache" in status["source"]
