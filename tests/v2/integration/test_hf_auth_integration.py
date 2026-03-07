"""Integration tests for HuggingFace token validity and SAM3 access.

These tests make real network calls to the HuggingFace Hub.
They are skipped automatically when HF_TOKEN is not set (CI without secrets,
fresh developer machines).

Run explicitly:
    uv run pytest tests/v2/integration/test_hf_auth_integration.py -v
"""

from __future__ import annotations

import pytest

from minivess.utils.hf_auth import (
    get_hf_token,
    hf_token_status,
    load_dotenv_if_present,
    validate_model_access,
)

# Load .env so local dev works without manually exporting HF_TOKEN.
load_dotenv_if_present(".env")

_HF_TOKEN_AVAILABLE = get_hf_token() is not None

requires_hf_token = pytest.mark.skipif(
    not _HF_TOKEN_AVAILABLE,
    reason="HF_TOKEN not set — set in .env or environment to run HF integration tests",
)

SAM3_MODEL_ID = "facebook/sam3"


@requires_hf_token
@pytest.mark.integration
class TestHfTokenValidity:
    """Verify the token is valid and identifies a real HF user."""

    def test_whoami_returns_username(self) -> None:
        from huggingface_hub import whoami

        me = whoami()
        assert isinstance(me, dict)
        assert "name" in me
        assert len(me["name"]) > 0, "whoami() returned empty username"

    def test_token_status_shows_authenticated(self) -> None:
        status = hf_token_status()
        assert status["authenticated"] is True, (
            f"Expected authenticated=True, got: {status}"
        )
        assert status["source"] != "none"
        assert isinstance(status["token_prefix"], str)
        assert status["token_prefix"].startswith("hf_")


@requires_hf_token
@pytest.mark.integration
class TestSam3Access:
    """Verify the token grants access to the gated facebook/sam3 model."""

    def test_sam3_model_is_accessible(self) -> None:
        result = validate_model_access(SAM3_MODEL_ID)
        assert result["accessible"] is True, (
            f"SAM3 not accessible. Error: {result.get('error')}\n"
            f"Request access at: https://huggingface.co/{SAM3_MODEL_ID}"
        )

    def test_sam3_is_gated_model(self) -> None:
        """Confirm the model is still gated (guard against HF changing it)."""
        from huggingface_hub import model_info

        info = model_info(SAM3_MODEL_ID)
        # facebook/sam3 is a manually-gated model — verify this assumption holds
        assert info.gated, (
            f"Expected {SAM3_MODEL_ID} to be gated but gated={info.gated!r}. "
            "If Meta removed gating, update this test."
        )

    def test_sam3_model_id_is_correct(self) -> None:
        """Verify the model ID resolves to the correct repo."""
        from huggingface_hub import model_info

        info = model_info(SAM3_MODEL_ID)
        assert info.id == SAM3_MODEL_ID

    def test_sam3_transformers_class_loadable(self) -> None:
        """Verify Sam3Model class is importable from transformers."""
        from transformers import Sam3Model  # noqa: F401 — import is the test

    def test_sam3_config_downloadable(self) -> None:
        """Download only the config (tiny file) to confirm gated access works."""
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(SAM3_MODEL_ID)
        assert config is not None
        # SAM3 is a vision model — verify it has image-related config
        assert hasattr(config, "model_type"), "Config missing model_type"
