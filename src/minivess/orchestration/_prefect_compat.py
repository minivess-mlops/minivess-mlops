"""Prefect compatibility layer for MinIVess MLOps.

Provides @task and @flow decorators plus get_run_logger that work whether
Prefect is installed or not. Set PREFECT_DISABLED=1 to force no-op mode
(useful for CI environments that should not require a Prefect server).

Usage::

    from minivess.orchestration import flow, task, get_run_logger


    @task(name="train-epoch", retries=2)
    def train_one_epoch(model, loader):
        logger = get_run_logger()
        logger.info("Training epoch...")
        ...


    @flow(name="training-pipeline")
    def train_pipeline():
        train_one_epoch(model, loader)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

_DISABLED = os.environ.get("PREFECT_DISABLED", "0") == "1"

PREFECT_AVAILABLE: bool = False

if not _DISABLED:
    try:
        from prefect import flow as _prefect_flow
        from prefect import get_run_logger as _prefect_get_run_logger
        from prefect import task as _prefect_task

        PREFECT_AVAILABLE = True
    except ImportError:
        PREFECT_AVAILABLE = False

if PREFECT_AVAILABLE:
    task = _prefect_task
    flow = _prefect_flow
    get_run_logger = _prefect_get_run_logger
else:
    # ---------------------------------------------------------------------------
    # No-op replacements for use when Prefect is not installed or is disabled.
    # Both @decorator and @decorator(kwarg=value) calling styles are supported.
    # ---------------------------------------------------------------------------

    def _noop_decorator_factory(*args: Any, **kwargs: Any) -> Any:
        """Return a no-op decorator that leaves the wrapped function unchanged.

        Handles two calling patterns:
        - ``@task`` — called with the function as the first positional argument
        - ``@task(name="x", retries=3)`` — called with keyword arguments first,
          returning a decorator that then receives the function.
        """
        if len(args) == 1 and callable(args[0]) and not kwargs:
            # Pattern: @task (bare decorator — function passed directly)
            return args[0]

        # Pattern: @task(...) — return a decorator that wraps the function
        def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn

        return _decorator

    task = _noop_decorator_factory
    flow = _noop_decorator_factory

    def get_run_logger() -> logging.Logger:
        """Return a stdlib logger as a stand-in for Prefect's run logger."""
        return logging.getLogger("minivess.orchestration")
