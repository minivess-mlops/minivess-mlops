from __future__ import annotations

import warnings

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
