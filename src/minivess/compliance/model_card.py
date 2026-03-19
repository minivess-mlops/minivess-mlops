"""Model Card generation for HuggingFace Model Hub and FDA traceability.

Implements Mitchell et al. (2019) "Model Cards for Model Reporting" with
HuggingFace-compatible YAML front matter generation.

References:
  - Issue #821: FDA audit trail + ModelCard
  - Mitchell et al. (2019). "Model Cards for Model Reporting." FAT*
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class ModelCard:
    """Model Card for documentation and transparency (Mitchell et al., 2019).

    Generates HuggingFace-compatible YAML metadata and Markdown documentation
    for model transparency and FDA IEC 62304 traceability.
    """

    model_name: str
    model_version: str
    model_type: str = "3D Segmentation"
    description: str = ""
    intended_use: str = "Research use only - biomedical vessel segmentation"
    limitations: str = ""
    training_data: str = ""
    evaluation_data: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    ethical_considerations: str = (
        "Not intended for clinical diagnostic use without regulatory approval."
    )
    caveats: str = ""
    authors: list[str] = field(default_factory=list)
    license: str = "cc-by-nc-4.0"
    pipeline_tag: str = "image-segmentation"
    library_name: str = "monai"
    tags: list[str] = field(
        default_factory=lambda: ["medical-imaging", "segmentation", "3d"]
    )

    def to_yaml(self) -> str:
        """Generate HuggingFace-compatible YAML metadata.

        Returns valid YAML that can be used as front matter in a
        HuggingFace Model Hub README.md file.

        Returns
        -------
        str
            YAML string with HuggingFace model card metadata.
        """
        metadata: dict[str, Any] = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "pipeline_tag": self.pipeline_tag,
            "library_name": self.library_name,
            "license": self.license,
            "tags": self.tags,
        }

        if self.metrics:
            metadata["metrics"] = [
                {"name": name, "type": name, "value": value}
                for name, value in self.metrics.items()
            ]

        if self.authors:
            metadata["authors"] = self.authors

        result: str = yaml.dump(
            metadata,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        return result

    def to_huggingface_readme(self) -> str:
        """Generate a complete HuggingFace README with YAML front matter.

        Returns
        -------
        str
            Complete README.md content with YAML front matter and Markdown body.
        """
        yaml_section = self.to_yaml()
        markdown_body = self.to_markdown()
        return f"---\n{yaml_section}---\n{markdown_body}\n"

    def to_markdown(self) -> str:
        """Generate Markdown documentation body.

        Returns
        -------
        str
            Markdown string with model card sections.
        """
        sections = [
            f"# Model Card: {self.model_name} v{self.model_version}",
            f"\n## Model Details\n- **Type:** {self.model_type}"
            f"\n- **Description:** {self.description}",
            f"\n## Intended Use\n{self.intended_use}",
            f"\n## Training Data\n{self.training_data}",
            f"\n## Evaluation Data\n{self.evaluation_data}",
        ]
        if self.metrics:
            metrics_lines = "\n".join(
                f"- **{k}:** {v:.4f}" for k, v in self.metrics.items()
            )
            sections.append(f"\n## Metrics\n{metrics_lines}")
        if self.limitations:
            sections.append(f"\n## Limitations\n{self.limitations}")
        sections.append(f"\n## Ethical Considerations\n{self.ethical_considerations}")
        if self.authors:
            sections.append(f"\n## Authors\n{', '.join(self.authors)}")
        return "\n".join(sections)

    @classmethod
    def from_mlflow_run_dict(
        cls,
        run_dict: dict[str, Any],
        *,
        version: str = "0.1",
    ) -> ModelCard:
        """Construct a ModelCard from an MLflow run metadata dictionary.

        Parameters
        ----------
        run_dict:
            Dictionary with 'params', 'metrics', and optionally 'tags' keys.
            Mirrors the structure of ``mlflow.get_run().data.to_dictionary()``.
        version:
            Model version string.

        Returns
        -------
        ModelCard
            Populated model card.
        """
        params = run_dict.get("params", {})
        metrics_raw = run_dict.get("metrics", {})
        tags = run_dict.get("tags", {})

        model_family = params.get("model_family", "unknown")
        loss_function = params.get("loss_function", "")
        max_epochs = params.get("max_epochs", "")

        # Filter to val/ metrics for the card
        card_metrics: dict[str, float] = {}
        for key, value in metrics_raw.items():
            if key.startswith("val/"):
                # Use the short name (e.g., "cldice" from "val/cldice")
                short_name = key.split("/", 1)[1] if "/" in key else key
                card_metrics[short_name] = float(value)

        description_parts = [f"Model family: {model_family}"]
        if loss_function:
            description_parts.append(f"Loss: {loss_function}")
        if max_epochs:
            description_parts.append(f"Epochs: {max_epochs}")

        run_name = tags.get("mlflow.runName", model_family)

        return cls(
            model_name=model_family,
            model_version=version,
            description=". ".join(description_parts),
            metrics=card_metrics,
            training_data=f"Run: {run_name}",
        )
