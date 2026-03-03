"""MONAI Deploy SDK compatibility layer.

Provides graceful import handling for the optional monai-deploy-app-sdk
package. When the SDK is not installed, duck-typed classes in
monai_deploy_app.py still work for testing and development.

Install with: ``uv add monai-deploy-app-sdk``
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MONAI_DEPLOY_AVAILABLE: bool = False

try:
    import monai.deploy  # noqa: F401

    MONAI_DEPLOY_AVAILABLE = True
except ImportError:
    MONAI_DEPLOY_AVAILABLE = False


def warn_if_monai_deploy_missing() -> None:
    """Emit a warning if MONAI Deploy SDK is not installed.

    Only warns once per process via the logger.
    """
    if not MONAI_DEPLOY_AVAILABLE:
        logger.warning(
            "monai-deploy-app-sdk is not installed. MONAI Deploy MAP "
            "packaging will use duck-typed classes (sufficient for "
            "testing but not for clinical deployment). Install with: "
            "uv add monai-deploy-app-sdk"
        )
