"""Tests for ModelCard HuggingFace YAML generation.

PR-2 T2.3: ModelCard must generate valid HuggingFace-format YAML front
matter (metadata block) suitable for Model Hub upload.

References:
  - Issue #821: FDA audit trail + ModelCard
  - docs/planning/pre-full-gcp-housekeeping-and-qa.xml PR id="2" T2.3
  - Mitchell et al. (2019) "Model Cards for Model Reporting"
"""

from __future__ import annotations

import yaml


class TestModelCardToYaml:
    """ModelCard must generate valid HuggingFace YAML front matter."""

    def test_model_card_has_to_yaml_method(self) -> None:
        from minivess.compliance.model_card import ModelCard

        card = ModelCard(model_name="DynUNet", model_version="1.0")
        assert hasattr(card, "to_yaml"), "ModelCard must have to_yaml() method"

    def test_model_card_yaml_is_valid(self) -> None:
        from minivess.compliance.model_card import ModelCard

        card = ModelCard(
            model_name="DynUNet",
            model_version="1.0",
            model_type="3D Segmentation",
            description="Vessel segmentation model",
            training_data="MiniVess (70 volumes)",
            evaluation_data="MiniVess 3-fold CV",
            metrics={"cldice": 0.812, "masd": 3.45},
            authors=["Teikari et al."],
        )
        yaml_str = card.to_yaml()
        # Must be parseable YAML
        parsed = yaml.safe_load(yaml_str)
        assert isinstance(parsed, dict)

    def test_model_card_yaml_has_required_hf_fields(self) -> None:
        """HuggingFace model card YAML requires certain top-level keys."""
        from minivess.compliance.model_card import ModelCard

        card = ModelCard(
            model_name="DynUNet",
            model_version="1.0",
            model_type="3D Segmentation",
            description="Vessel segmentation model",
            training_data="MiniVess (70 volumes)",
            evaluation_data="MiniVess 3-fold CV",
            metrics={"cldice": 0.812, "masd": 3.45},
            authors=["Teikari et al."],
        )
        yaml_str = card.to_yaml()
        parsed = yaml.safe_load(yaml_str)

        # HuggingFace model card YAML metadata fields
        assert "model_name" in parsed or "model-name" in parsed
        assert "model_type" in parsed or "pipeline_tag" in parsed
        assert "metrics" in parsed
        assert "license" in parsed

    def test_model_card_yaml_metrics_are_list_of_dicts(self) -> None:
        """HuggingFace format uses list of dicts for metrics."""
        from minivess.compliance.model_card import ModelCard

        card = ModelCard(
            model_name="DynUNet",
            model_version="1.0",
            metrics={"cldice": 0.812, "masd": 3.45},
        )
        yaml_str = card.to_yaml()
        parsed = yaml.safe_load(yaml_str)

        metrics = parsed["metrics"]
        assert isinstance(metrics, list)
        assert all(isinstance(m, dict) for m in metrics)
        # Each metric dict should have "name" and "value"
        metric_names = {m["name"] for m in metrics}
        assert "cldice" in metric_names
        assert "masd" in metric_names

    def test_model_card_yaml_license_field(self) -> None:
        from minivess.compliance.model_card import ModelCard

        card = ModelCard(model_name="DynUNet", model_version="1.0")
        yaml_str = card.to_yaml()
        parsed = yaml.safe_load(yaml_str)
        # License must be present and non-empty
        assert "license" in parsed
        assert parsed["license"]

    def test_model_card_to_huggingface_readme(self) -> None:
        """Full HuggingFace README with YAML front matter + Markdown body."""
        from minivess.compliance.model_card import ModelCard

        card = ModelCard(
            model_name="DynUNet",
            model_version="1.0",
            model_type="3D Segmentation",
            description="Vessel segmentation model",
            metrics={"cldice": 0.812},
        )
        readme = card.to_huggingface_readme()
        # Must start with YAML front matter
        assert readme.startswith("---\n")
        # Must have closing YAML delimiter
        parts = readme.split("---\n")
        # parts[0] is empty (before first ---), parts[1] is YAML, rest is Markdown
        assert len(parts) >= 3
        # YAML section must be valid
        yaml_section = parts[1]
        parsed = yaml.safe_load(yaml_section)
        assert isinstance(parsed, dict)
        # Markdown body must follow
        markdown_body = "---\n".join(parts[2:])
        assert "Model Card" in markdown_body or "DynUNet" in markdown_body


class TestModelCardFromMlflowRun:
    """ModelCard can be populated from MLflow run metadata."""

    def test_from_mlflow_dict(self) -> None:
        """ModelCard.from_mlflow_run_dict() constructs card from run dict."""
        from minivess.compliance.model_card import ModelCard

        run_dict = {
            "params": {
                "model_family": "dynunet",
                "loss_function": "cbdice_cldice",
                "max_epochs": "50",
            },
            "metrics": {
                "val/cldice": 0.812,
                "val/masd": 3.45,
                "val/dice": 0.891,
            },
            "tags": {
                "mlflow.runName": "dynunet_cbdice_fold0",
            },
        }
        card = ModelCard.from_mlflow_run_dict(run_dict, version="1.0")
        assert card.model_name == "dynunet"
        assert card.model_version == "1.0"
        assert "cldice" in card.metrics or "val/cldice" in card.metrics
