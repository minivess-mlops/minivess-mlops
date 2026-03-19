"""Entity extraction from paper abstracts using Pydantic AI structured output.

Extracts model families, loss functions, metrics, datasets, and techniques
from biomedical segmentation paper abstracts. Uses Pydantic AI Agent with
structured output (NOT regex — Rule #16).
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from minivess.agents.config import load_agent_config

logger = logging.getLogger(__name__)


class ExtractedEntities(BaseModel):
    """Entities extracted from a paper abstract.

    These are ML/biomedical entities relevant to the MinIVess knowledge graph.
    """

    model_families: list[str] = Field(
        default_factory=list,
        description="Model architectures mentioned (e.g., DynUNet, SAM3, SegResNet, U-Net)",
    )
    losses: list[str] = Field(
        default_factory=list,
        description="Loss functions mentioned (e.g., clDice, Dice+CE, Focal Tversky)",
    )
    metrics: list[str] = Field(
        default_factory=list,
        description="Evaluation metrics mentioned (e.g., DSC, clDice, Betti Error, HD95)",
    )
    datasets: list[str] = Field(
        default_factory=list,
        description="Datasets mentioned (e.g., MiniVess, BraTS, DRIVE)",
    )
    techniques: list[str] = Field(
        default_factory=list,
        description="Key techniques mentioned (e.g., topology-preserving, attention, self-supervised)",
    )


_EXTRACTION_SYSTEM_PROMPT = """\
You are an expert biomedical image analysis researcher. Extract structured entities
from the given paper abstract. Focus on:

1. **Model families**: Neural network architectures (e.g., U-Net, DynUNet, SAM3,
   SegResNet, UNETR, SwinUNETR, Mamba, VesselFM, nnU-Net)
2. **Loss functions**: Training objectives (e.g., Dice loss, clDice, cross-entropy,
   Focal Tversky, boundary loss)
3. **Metrics**: Evaluation metrics (e.g., Dice similarity coefficient, clDice,
   Hausdorff distance, Betti error, sensitivity, specificity)
4. **Datasets**: Named datasets used in experiments
5. **Techniques**: Key methodological innovations or approaches

Be precise and use canonical names where possible. If an entity is ambiguous,
include both the specific and general form.
"""


def build_extraction_agent(model: str | None = None) -> Agent[None, ExtractedEntities]:
    """Build a Pydantic AI Agent for entity extraction.

    Parameters
    ----------
    model:
        Override model identifier. Uses AgentConfig default if None.

    Returns
    -------
    Pydantic AI Agent that returns ExtractedEntities.
    """
    config = load_agent_config()
    model_name = model or config.model

    agent: Agent[None, ExtractedEntities] = Agent(
        model_name,
        output_type=ExtractedEntities,
        name="kg-entity-extractor",
        system_prompt=_EXTRACTION_SYSTEM_PROMPT,
    )

    return agent


def extract_entities(
    abstract: str,
    *,
    agent: Agent[None, ExtractedEntities] | None = None,
    model: str | None = None,
) -> ExtractedEntities:
    """Extract entities from a paper abstract using LLM structured output.

    Parameters
    ----------
    abstract:
        The paper abstract text to extract entities from.
    agent:
        Optional pre-built agent. If None, builds a new one.
    model:
        Override model identifier (only used if agent is None).

    Returns
    -------
    ExtractedEntities with structured fields.
    """
    if agent is None:
        agent = build_extraction_agent(model=model)

    result = agent.run_sync(
        f"Extract entities from this abstract:\n\n{abstract}",
    )
    return result.output
