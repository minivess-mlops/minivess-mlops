"""Centralized configuration defaults for the MinIVess pipeline.

All magic constants that were scattered across modules are collected here.
Source modules re-export these values for backwards compatibility.
"""

from __future__ import annotations

# MLflow tracking URI (observability/tracking.py)
DEFAULT_TRACKING_URI: str = "mlruns"

# BentoML model tag (serving/bento_service.py)
BENTO_MODEL_TAG: str = "minivess-segmentor"

# Default training batch size (data/loader.py)
DEFAULT_BATCH_SIZE: int = 2

# Default LLM model identifier (agents/llm.py)
DEFAULT_LLM_MODEL: str = "anthropic:claude-sonnet-4-6"
