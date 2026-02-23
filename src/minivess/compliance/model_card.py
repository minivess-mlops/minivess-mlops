from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelCard:
    """Model Card for documentation and transparency (Mitchell et al., 2019)."""

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

    def to_markdown(self) -> str:
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
