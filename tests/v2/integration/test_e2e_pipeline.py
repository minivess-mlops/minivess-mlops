"""End-to-end integration test exercising the full pipeline.

Data loading -> Training -> Evaluation -> Ensemble -> Calibration
-> Model Card -> Audit Trail.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from minivess.adapters.segresnet import SegResNetAdapter
from minivess.compliance.audit import AuditTrail
from minivess.compliance.model_card import ModelCard
from minivess.config.models import (
    EnsembleConfig,
    EnsembleStrategy,
    ModelConfig,
    ModelFamily,
    TrainingConfig,
)
from minivess.ensemble.calibration import (
    CalibrationResult,
    expected_calibration_error,
    temperature_scale,
)
from minivess.ensemble.strategies import EnsemblePredictor
from minivess.pipeline.metrics import SegmentationMetrics
from minivess.pipeline.trainer import SegmentationTrainer

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline_e2e(
    tmp_path: Path,
    model_config: ModelConfig,
    segresnet_adapter: SegResNetAdapter,
    synthetic_batch: dict[str, torch.Tensor],
    synthetic_loader: list[dict[str, torch.Tensor]],
) -> None:
    """Run the full pipeline: data -> train -> eval -> ensemble -> compliance."""

    # ------------------------------------------------------------------
    # Step 1: Verify synthetic data shapes
    # ------------------------------------------------------------------
    images = synthetic_batch["image"]
    labels = synthetic_batch["label"]
    assert images.shape == (2, 1, 16, 16, 16)
    assert labels.shape == (2, 1, 16, 16, 16)

    # ------------------------------------------------------------------
    # Step 2: Instantiate SegResNet adapter & verify forward pass
    # ------------------------------------------------------------------
    model = segresnet_adapter
    model.eval()
    with torch.no_grad():
        output = model(images)
    assert output.prediction.shape == (2, 2, 16, 16, 16)
    assert output.logits.shape == (2, 2, 16, 16, 16)
    # Probabilities should sum to ~1 along channel dim
    prob_sums = output.prediction.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)

    # ------------------------------------------------------------------
    # Step 3: Train for 2 epochs using SegmentationTrainer
    # ------------------------------------------------------------------
    training_config = TrainingConfig(
        max_epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        mixed_precision=False,  # CPU
        early_stopping_patience=5,
        warmup_epochs=1,
    )
    checkpoint_dir = tmp_path / "checkpoints"
    trainer = SegmentationTrainer(model, training_config, device="cpu")

    summary = trainer.fit(
        synthetic_loader,
        synthetic_loader,  # reuse as val for simplicity
        checkpoint_dir=checkpoint_dir,
    )

    assert "best_val_loss" in summary
    assert "final_epoch" in summary
    assert "history" in summary
    assert summary["final_epoch"] == 2
    assert len(summary["history"]["train_loss"]) == 2
    assert len(summary["history"]["val_loss"]) == 2
    # Loss values should be finite and positive
    for loss_val in summary["history"]["train_loss"]:
        assert loss_val > 0.0
        assert np.isfinite(loss_val)
    # Best checkpoint should have been saved for the primary metric (val_loss)
    # The refactored trainer saves best_<metric_name>.pth (not best_model.pth)
    primary_metric = training_config.checkpoint.primary_metric
    safe_name = primary_metric.replace("/", "_")
    assert (checkpoint_dir / f"best_{safe_name}.pth").exists()

    # ------------------------------------------------------------------
    # Step 4: Compute metrics using SegmentationMetrics
    # ------------------------------------------------------------------
    seg_metrics = SegmentationMetrics(num_classes=2, device="cpu")
    model.eval()
    with torch.no_grad():
        eval_output = model(images)

    # Squeeze channel from labels: (B, 1, D, H, W) -> (B, D, H, W)
    seg_metrics.update(eval_output.prediction, labels.squeeze(1))
    metric_result = seg_metrics.compute()

    assert "dice" in metric_result.values
    dice_score = metric_result.values["dice"]
    assert 0.0 <= dice_score <= 1.0
    assert "f1_foreground" in metric_result.values
    f1_score = metric_result.values["f1_foreground"]
    assert 0.0 <= f1_score <= 1.0

    # ------------------------------------------------------------------
    # Step 5: Ensemble prediction (mean strategy)
    # ------------------------------------------------------------------
    # Create a second model to form a 2-member ensemble
    model_config_2 = ModelConfig(
        family=ModelFamily.MONAI_SEGRESNET,
        name="ensemble-member-2",
        in_channels=1,
        out_channels=2,
    )
    model_2 = SegResNetAdapter(model_config_2)

    ensemble_config = EnsembleConfig(
        strategy=EnsembleStrategy.MEAN,
        num_members=2,
    )
    ensemble = EnsemblePredictor(
        models=[model, model_2],
        config=ensemble_config,
    )

    ensemble_pred = ensemble.predict(images)
    assert ensemble_pred.shape == (2, 2, 16, 16, 16)
    # Ensemble prediction should still be valid probabilities
    ens_sums = ensemble_pred.sum(dim=1)
    assert torch.allclose(ens_sums, torch.ones_like(ens_sums), atol=1e-5)

    # ------------------------------------------------------------------
    # Step 6: Calibration check
    # ------------------------------------------------------------------
    # Flatten predictions to (N, C) for calibration analysis
    # Use argmax class confidence as "confidence" and compare to ground truth
    pred_classes = ensemble_pred.argmax(dim=1)  # (B, D, H, W)
    max_probs = ensemble_pred.max(dim=1).values  # (B, D, H, W)

    flat_confidences = max_probs.cpu().numpy().flatten().astype(np.float64)
    flat_pred_classes = pred_classes.cpu().numpy().flatten()
    flat_true_classes = labels.squeeze(1).cpu().numpy().flatten()
    flat_accuracies = (flat_pred_classes == flat_true_classes).astype(np.float64)

    ece, mce = expected_calibration_error(
        flat_confidences,
        flat_accuracies,
        n_bins=10,
    )
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= mce <= 1.0

    # Temperature scaling on logits
    # Get logits from the first model for calibration demo
    with torch.no_grad():
        logits_output = model(images)
    # Reshape logits to (N, C) for temperature_scale
    logits_np = (
        logits_output.logits.permute(0, 2, 3, 4, 1)
        .reshape(-1, 2)
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    calibrated_probs = temperature_scale(logits_np, temperature=1.5)
    assert calibrated_probs.shape == logits_np.shape
    # Calibrated probs should sum to ~1 per sample
    row_sums = calibrated_probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    calibration_result = CalibrationResult(
        ece=ece,
        mce=mce,
        calibrated_probs=calibrated_probs,
    )
    assert calibration_result.ece == ece
    assert calibration_result.mce == mce

    # ------------------------------------------------------------------
    # Step 7: Generate a model card
    # ------------------------------------------------------------------
    card = ModelCard(
        model_name="SegResNet-E2E",
        model_version="0.1.0",
        model_type="3D Segmentation",
        description="End-to-end integration test model",
        intended_use="Research use only - biomedical vessel segmentation",
        training_data="Synthetic 16x16x16 3D volumes (integration test)",
        evaluation_data="Synthetic 16x16x16 3D volumes (integration test)",
        metrics=metric_result.to_dict(),
        limitations="Trained on synthetic data only, not for clinical use.",
        authors=["Integration Test Suite"],
    )
    markdown = card.to_markdown()
    assert "SegResNet-E2E" in markdown
    assert "v0.1.0" in markdown
    assert "dice" in markdown.lower()

    # Write model card to disk and verify
    card_path = tmp_path / "model_card.md"
    card_path.write_text(markdown, encoding="utf-8")
    assert card_path.exists()
    reloaded = card_path.read_text(encoding="utf-8")
    assert "SegResNet-E2E" in reloaded

    # ------------------------------------------------------------------
    # Step 8: Create audit trail entries
    # ------------------------------------------------------------------
    audit = AuditTrail()

    # Log data access
    data_entry = audit.log_data_access(
        dataset_name="synthetic-e2e",
        file_paths=["vol_001.nii.gz", "vol_002.nii.gz"],
        actor="integration-test",
    )
    assert data_entry.event_type == "DATA_ACCESS"
    assert data_entry.data_hash is not None

    # Log model training
    train_entry = audit.log_model_training(
        model_name="SegResNet-E2E",
        config=training_config.model_dump(),
        actor="integration-test",
    )
    assert train_entry.event_type == "MODEL_TRAINING"

    # Log test evaluation
    eval_entry = audit.log_test_evaluation(
        model_name="SegResNet-E2E",
        metrics=metric_result.to_dict(),
        actor="integration-test",
    )
    assert eval_entry.event_type == "TEST_EVALUATION"

    # Log deployment
    deploy_entry = audit.log_model_deployment(
        model_name="SegResNet-E2E",
        version="0.1.0",
        actor="integration-test",
    )
    assert deploy_entry.event_type == "MODEL_DEPLOYMENT"

    # Verify all entries are recorded
    assert len(audit.entries) == 4

    # Save and reload audit trail
    audit_path = tmp_path / "audit" / "trail.json"
    audit.save(audit_path)
    assert audit_path.exists()

    loaded_audit = AuditTrail.load(audit_path)
    assert len(loaded_audit.entries) == 4
    assert loaded_audit.entries[0].event_type == "DATA_ACCESS"
    assert loaded_audit.entries[1].event_type == "MODEL_TRAINING"
    assert loaded_audit.entries[2].event_type == "TEST_EVALUATION"
    assert loaded_audit.entries[3].event_type == "MODEL_DEPLOYMENT"

    # Verify JSON structure is well-formed
    raw_json = json.loads(audit_path.read_text(encoding="utf-8"))
    assert isinstance(raw_json, list)
    assert len(raw_json) == 4
    for entry in raw_json:
        assert "timestamp" in entry
        assert "event_type" in entry
        assert "actor" in entry
        assert entry["actor"] == "integration-test"
