"""MONAI Label InferTask wrapping BentoML champion model.

This module is loaded by MONAI Label server and registered as the
``champion`` inference task. When 3D Slicer requests pre-segmentation,
MONAI Label delegates to this InferTask which calls the BentoML endpoint.

The actual inference logic lives in ``src/minivess/annotation/champion_infer.py``.
This file is the thin MONAI Label adapter that bridges the MONAI Label API
(studies dir, InferTask interface) to the MinIVess inference module.

See deployment/monailabel/README.md for the full architecture.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# NOTE: This is a MONAI Label app adapter file, NOT a src/ module.
# It runs inside the monai-label Docker container. The actual inference
# logic is in src/minivess/annotation/champion_infer.py which is
# installed in the container via the minivess-base image.
#
# When MONAI Label loads this app:
# 1. It reads main.py for config (BentoML URL, studies dir)
# 2. It discovers infer tasks in lib/infers/
# 3. This file provides the ChampionInfer wrapper
#
# For now, this is a reference implementation. Full MONAI Label
# InferTask integration requires monailabel SDK which is installed
# in the Dockerfile.monailabel container.
