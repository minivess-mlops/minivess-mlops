"""MinIVess MONAI Label App — champion pre-segmentation for 3D Slicer.

This MONAI Label app provides the BentoML champion model as a pre-segmentation
source for interactive annotation in 3D Slicer. When an annotator opens a volume,
clicking "Get" triggers inference via the deployed BentoML champion endpoint.

The app registers a single InferTask (ChampionInfer) that bridges the MONAI Label
API to the BentoML REST endpoint. If BentoML is unavailable, ChampionInfer returns
a zero mask so the annotator can still segment from scratch.

Usage:
    # Start via docker compose:
    docker compose -f deployment/docker-compose.flows.yml up monai-label

    # Or manually:
    monailabel start_server --app /app/monailabel_app --studies /data/studies
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def get_app_config() -> dict[str, str]:
    """Return MONAI Label app configuration.

    Reads BentoML URL from environment (injected by docker-compose).
    """
    return {
        "bentoml_url": os.environ.get(
            "BENTOML_URL",
            f"http://minivess-bento:{os.environ.get('BENTOML_PORT', '3333')}",
        ),
        "studies_dir": os.environ.get("STUDIES_DIR", "/data/studies"),
    }


# MONAI Label app entry point.
# monailabel loads this module and calls init_app() if defined,
# or reads the config from get_app_config().
# The actual InferTask registration happens when MONAI Label
# instantiates the app — see deployment/monailabel/README.md
# for the full MONAI Label app lifecycle.
