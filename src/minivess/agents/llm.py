"""LiteLLM provider abstraction for model flexibility."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"


def call_llm(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> str:
    """Call an LLM via LiteLLM and return the text response.

    Parameters
    ----------
    prompt:
        The user prompt to send.
    model:
        LiteLLM model identifier (e.g., "anthropic:claude-sonnet-4-6",
        "ollama:qwen2.5-coder:14b").
    temperature:
        Sampling temperature.

    Returns
    -------
    Text content from the LLM response.
    """
    import litellm

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return str(response.choices[0].message.content)


def call_llm_structured(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Call an LLM and parse the response as JSON.

    Parameters
    ----------
    prompt:
        The user prompt (should request JSON output).
    model:
        LiteLLM model identifier.
    temperature:
        Sampling temperature.

    Returns
    -------
    Parsed JSON dict from the LLM response.
    """
    text = call_llm(prompt, model=model, temperature=temperature)
    return dict(json.loads(text))
