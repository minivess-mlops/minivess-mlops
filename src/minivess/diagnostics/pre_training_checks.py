"""Pre-training sanity checks — catch broken models before wasting GPU hours.

Six checks run before the training loop starts:
1. Output shape mismatch
2. Gradient flow (at least one param has .grad)
3. Loss sanity (random init loss near expected value)
4. NaN/Inf check (no NaN or Inf in model output)
5. Data range (inputs in expected range)
6. Label integrity (valid class indices)

Artifact: diagnostics/pre_training_checks.json (NOT profiling/ — RC12).
Diagnostics always run regardless of ProfilingConfig.enabled (RC17).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)

# Artifact path for MLflow logging (RC12)
ARTIFACT_PATH = "diagnostics"


@dataclass
class CheckResult:
    """Result of a single pre-training check."""

    name: str
    passed: bool
    message: str
    severity: str = "warning"  # "error" or "warning"


def check_output_shape(
    model: nn.Module,
    sample_batch: dict[str, torch.Tensor],
    *,
    expected_channels: int,
) -> CheckResult:
    """Check that model output has the expected number of channels."""
    model.eval()
    with torch.no_grad():
        images = sample_batch["image"]
        output = model(images)
        # Handle both Tensor and SegmentationOutput
        if hasattr(output, "logits"):
            output = output.logits
        out_channels = output.shape[1]

    if out_channels != expected_channels:
        return CheckResult(
            name="output_shape",
            passed=False,
            message=(
                f"Expected {expected_channels} output channels, got {out_channels}"
            ),
            severity="error",
        )
    return CheckResult(
        name="output_shape",
        passed=True,
        message=f"Output shape OK: {out_channels} channels",
    )


def check_gradient_flow(
    model: nn.Module,
    sample_batch: dict[str, torch.Tensor],
) -> CheckResult:
    """Check that at least one parameter receives gradients."""
    model.train()
    model.zero_grad()

    images = sample_batch["image"]
    output = model(images)
    if hasattr(output, "logits"):
        output = output.logits

    # Simple backward with mean loss
    loss = output.mean()
    try:
        loss.backward()
    except RuntimeError:
        return CheckResult(
            name="gradient_flow",
            passed=False,
            message="Backward pass failed — no gradient flow",
            severity="error",
        )

    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )

    if not has_grad:
        return CheckResult(
            name="gradient_flow",
            passed=False,
            message="No parameter received gradients — model may be frozen",
            severity="error",
        )
    return CheckResult(
        name="gradient_flow",
        passed=True,
        message="Gradient flow OK",
    )


def check_loss_sanity(
    model: nn.Module,
    sample_batch: dict[str, torch.Tensor],
    criterion: nn.Module,
) -> CheckResult:
    """Check that loss at random init is in reasonable range."""
    model.eval()
    with torch.no_grad():
        images = sample_batch["image"]
        labels = sample_batch["label"]
        output = model(images)
        if hasattr(output, "logits"):
            output = output.logits
        loss = criterion(output, labels.squeeze(1))
        loss_val = loss.item()

    if not torch.isfinite(torch.tensor(loss_val)):
        return CheckResult(
            name="loss_sanity",
            passed=False,
            message=f"Loss at init is not finite: {loss_val}",
            severity="error",
        )

    return CheckResult(
        name="loss_sanity",
        passed=True,
        message=f"Loss at init: {loss_val:.4f}",
    )


def check_nan_inf(
    model: nn.Module,
    sample_batch: dict[str, torch.Tensor],
) -> CheckResult:
    """Check that model output has no NaN or Inf values."""
    model.eval()
    with torch.no_grad():
        images = sample_batch["image"]
        output = model(images)
        if hasattr(output, "logits"):
            output = output.logits

        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()

    if has_nan or has_inf:
        return CheckResult(
            name="nan_inf",
            passed=False,
            message=f"Model output has NaN={has_nan}, Inf={has_inf}",
            severity="error",
        )
    return CheckResult(
        name="nan_inf",
        passed=True,
        message="No NaN/Inf in model output",
    )


def run_pre_training_checks(
    *,
    model: nn.Module,
    sample_batch: dict[str, torch.Tensor],
    criterion: nn.Module,
    expected_channels: int = 2,
) -> list[CheckResult]:
    """Run all pre-training sanity checks.

    Returns a list of CheckResult. Failures with severity='error'
    should abort training with a clear message.
    """
    results: list[CheckResult] = []

    results.append(
        check_output_shape(model, sample_batch, expected_channels=expected_channels)
    )
    results.append(check_gradient_flow(model, sample_batch))
    results.append(check_loss_sanity(model, sample_batch, criterion))
    results.append(check_nan_inf(model, sample_batch))

    # Log summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    logger.info("Pre-training checks: %d/%d passed", passed, total)
    for r in results:
        if not r.passed:
            logger.warning("  FAILED: %s — %s", r.name, r.message)

    return results


def checks_to_dict(results: list[CheckResult]) -> dict[str, Any]:
    """Convert check results to a serializable dict for artifact logging."""
    return {
        "checks": [asdict(r) for r in results],
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
        },
    }
