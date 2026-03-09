"""HuggingFace authentication utilities.

Provides zero-config token loading for local dev (.env file) and
containerized environments (environment variable), without requiring
python-dotenv or manual CLI login in automated pipelines.

Priority order (mirrors huggingface_hub 1.x behaviour):
  1. ``HF_TOKEN`` environment variable  ← set this in Docker/RunPod
  2. ``~/.cache/huggingface/token``     ← written by ``hf auth login``
  3. ``HUGGING_FACE_HUB_TOKEN`` env var ← legacy name, still supported

For local dev, call ``load_dotenv_if_present()`` early in your script
(before importing transformers) to load ``.env`` → ``HF_TOKEN`` is then
picked up by huggingface_hub automatically.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
_HF_TOKEN_CACHE = Path.home() / ".cache" / "huggingface" / "token"

_MISSING_TOKEN_INSTRUCTIONS = """\
════════════════════════════════════════════════════════════════
 HF_TOKEN NOT SET — required for gated models (SAM3, VesselFM)
════════════════════════════════════════════════════════════════

 PRIMARY FIX — Add to .env file at repo root (ALWAYS preferred):
   echo 'HF_TOKEN=hf_YOUR_TOKEN' >> .env

   In Docker Compose V2, .env is loaded from the compose file's
   directory by default. run_debug.sh passes --env-file .env explicitly
   so the repo-root .env is always used. If you see this error while
   using run_debug.sh, check that .env exists at the repo root.

   NEVER use `export HF_TOKEN=...` (shell env vars are ephemeral and
   not the authoritative source for this project).

 ALTERNATIVE — huggingface-cli login (cached in ~/.cache/hf/token):
   uv run huggingface-cli login

 Get your token: https://huggingface.co/settings/tokens
 Request SAM3 access: https://huggingface.co/facebook/sam3
════════════════════════════════════════════════════════════════\
"""


def load_dotenv_if_present(path: str | Path = ".env") -> bool:
    """Load a ``.env`` file into ``os.environ`` without python-dotenv.

    Skips keys that are already set (environment always wins over .env).
    Safe to call multiple times — idempotent.

    Parameters
    ----------
    path:
        Path to the ``.env`` file. Relative paths are resolved from
        the current working directory.

    Returns
    -------
    bool
        ``True`` if the file was found and loaded, ``False`` otherwise.
    """
    env_file = Path(path)
    if not env_file.exists():
        return False

    loaded = 0
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1

    if loaded:
        logger.debug("Loaded %d var(s) from %s", loaded, env_file)
    return True


def get_hf_token() -> str | None:
    """Return the active HuggingFace token or ``None`` if not set.

    Checks env vars first (Docker/RunPod), then the cached token file
    (written by ``hf auth login``).
    """
    for var in _HF_TOKEN_ENV_VARS:
        token = os.environ.get(var, "").strip()
        if token:
            return token

    if _HF_TOKEN_CACHE.exists():
        token = _HF_TOKEN_CACHE.read_text(encoding="utf-8").strip()
        if token:
            return token

    return None


def require_hf_token(model_id: str = "a gated HuggingFace model") -> str:
    """Return the HF token or raise a clear ``RuntimeError`` with fix steps.

    Parameters
    ----------
    model_id:
        Human-readable model name for the error message.

    Raises
    ------
    RuntimeError
        When no token is found in environment or cache.
    """
    token = get_hf_token()
    if token:
        return token

    logger.error(_MISSING_TOKEN_INSTRUCTIONS)
    msg = (
        f"HuggingFace token required to download {model_id!r}. "
        "See instructions logged above (ERROR level)."
    )
    raise RuntimeError(msg)


def validate_model_access(model_id: str) -> dict[str, str | bool]:
    """Check that the current token can access a specific HF model.

    Makes a real API call — use in preflight checks and integration tests,
    not in unit tests.

    Parameters
    ----------
    model_id:
        HuggingFace model ID, e.g. ``"facebook/sam3"``.

    Returns
    -------
    dict with keys:
        ``accessible`` (bool), ``gated`` (bool | str), ``error`` (str)

    Raises
    ------
    RuntimeError
        If no token is set (delegates to ``require_hf_token``).
    """
    require_hf_token(model_id)

    try:
        from huggingface_hub import model_info
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

        info = model_info(model_id)
        return {
            "accessible": True,
            "gated": info.gated or False,
            "error": "",
        }
    except GatedRepoError:
        return {
            "accessible": False,
            "gated": True,
            "error": (
                f"Access denied to {model_id!r}. "
                f"Request access at https://huggingface.co/{model_id}"
            ),
        }
    except RepositoryNotFoundError:
        return {
            "accessible": False,
            "gated": False,
            "error": f"Model {model_id!r} not found on HuggingFace Hub.",
        }
    except Exception as exc:  # network errors, etc.
        return {
            "accessible": False,
            "gated": False,
            "error": str(exc),
        }


def hf_token_status() -> dict[str, bool | str]:
    """Return a dict describing auth status — useful for preflight checks.

    Returns
    -------
    dict with keys:
        ``authenticated`` (bool), ``source`` (str), ``token_prefix`` (str)
    """
    for var in _HF_TOKEN_ENV_VARS:
        token = os.environ.get(var, "").strip()
        if token:
            return {
                "authenticated": True,
                "source": f"env:{var}",
                "token_prefix": token[:7] + "…",
            }

    if _HF_TOKEN_CACHE.exists():
        token = _HF_TOKEN_CACHE.read_text(encoding="utf-8").strip()
        if token:
            return {
                "authenticated": True,
                "source": "cache:~/.cache/huggingface/token",
                "token_prefix": token[:7] + "…",
            }

    return {"authenticated": False, "source": "none", "token_prefix": ""}
