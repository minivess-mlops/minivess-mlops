from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "real_data: requires real MiniVess dataset (not run in CI)"
    )

# Suppress warnings that occur during import of third-party libraries.
# These must be set before the libraries are imported, so pytest's
# filterwarnings config (which applies after collection) is too late.
warnings.filterwarnings(
    "ignore",
    message=".*deprecated.*",
    category=DeprecationWarning,
    module="pyparsing.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module="MetricsReloaded.*",
)
