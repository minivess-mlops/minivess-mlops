"""Deployment artifact generation for the deploy flow.

Generates bentofile.yaml, docker-compose.yaml, and DEPLOY_README.md
from champion model metadata. Templates are string constants (no
Jinja2 dependency).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from minivess.pipeline.deploy_champion_discovery import ChampionModel

logger = logging.getLogger(__name__)


def generate_bentofile(
    service_name: str,
    models: list[str],
    output_dir: Path,
) -> Path:
    """Generate a bentofile.yaml for BentoML build.

    Parameters
    ----------
    service_name:
        BentoML service name.
    models:
        List of BentoML model tags to include.
    output_dir:
        Directory to write the file.

    Returns
    -------
    Path to the generated bentofile.yaml.
    """
    models_section = "\n".join(f'  - "{m}"' for m in models)
    content = f"""\
service: "service:OnnxSegmentationService"
description: "MinIVess 3D vessel segmentation — {service_name}"

labels:
  project: minivess-mlops
  generated: "{datetime.now(UTC).strftime("%Y-%m-%d")}"

include:
  - "src/minivess/serving/*.py"
  - "src/minivess/serving/onnx_inference.py"

python:
  packages:
    - onnxruntime
    - numpy
    - pydantic

models:
{models_section}
"""
    path = output_dir / "bentofile.yaml"
    path.write_text(content, encoding="utf-8")
    logger.info("Generated bentofile.yaml: %s", path)
    return path


def generate_docker_compose(
    services: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Generate a docker-compose.yaml for deployment.

    Parameters
    ----------
    services:
        List of service dicts with name, port, model_tag keys.
    output_dir:
        Directory to write the file.

    Returns
    -------
    Path to the generated docker-compose.yaml.
    """
    service_blocks = []
    for svc in services:
        name = svc["name"]
        port = svc["port"]
        model_tag = svc.get("model_tag", "latest")
        block = f"""\
  {name}:
    image: minivess-{name}:latest
    ports:
      - "{port}:3000"
    environment:
      - BENTO_MODEL_TAG={model_tag}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        limits:
          memory: 4G"""
        service_blocks.append(block)

    services_yaml = "\n".join(service_blocks)
    content = f"""\
version: "3.8"

services:
{services_yaml}
"""
    path = output_dir / "docker-compose.yaml"
    path.write_text(content, encoding="utf-8")
    logger.info("Generated docker-compose.yaml: %s", path)
    return path


def generate_deployment_readme(
    champions: list[ChampionModel],
    output_dir: Path,
) -> Path:
    """Generate a deployment README with champion model details.

    Parameters
    ----------
    champions:
        List of champion models to document.
    output_dir:
        Directory to write the file.

    Returns
    -------
    Path to the generated DEPLOY_README.md.
    """
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# MinIVess Deployment",
        "",
        f"**Generated:** {now}",
        "",
        "## Champion Models",
        "",
        "| Category | Run ID | Metrics |",
        "|----------|--------|---------|",
    ]

    for champion in champions:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in champion.metrics.items())
        lines.append(f"| {champion.category} | {champion.run_id} | {metrics_str} |")

    lines.extend(
        [
            "",
            "## Quick Start",
            "",
            "```bash",
            "# Build and serve with BentoML",
            "bentoml build -f bentofile.yaml",
            "bentoml serve .",
            "",
            "# Or use Docker Compose",
            "docker compose up -d",
            "```",
            "",
            "## Health Check",
            "",
            "```bash",
            "curl http://localhost:3000/health",
            "```",
            "",
        ]
    )

    content = "\n".join(lines)
    path = output_dir / "DEPLOY_README.md"
    path.write_text(content, encoding="utf-8")
    logger.info("Generated DEPLOY_README.md: %s", path)
    return path
