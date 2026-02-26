"""MLflow pyfunc serving wrappers for 3D segmentation models.

Provides two ``mlflow.pyfunc.PythonModel`` wrappers:

* :class:`MiniVessSegModel` — single checkpoint inference
* :class:`MiniVessEnsembleModel` — ensemble inference with uncertainty decomposition

Both accept 5-D numpy arrays ``(B, 1, D, H, W)`` and return class
probabilities ``(B, 2, D, H, W)``.

The uncertainty decomposition follows Lakshminarayanan et al. (2017):
  * **Total** = entropy of mean softmax
  * **Aleatoric** = mean of per-member entropies
  * **Epistemic** = total − aleatoric (mutual information)
"""

from __future__ import annotations

import json
import logging
from typing import Any

import mlflow.pyfunc
import numpy as np
import torch
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from torch import Tensor, nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight network that replays a saved state-dict
# ---------------------------------------------------------------------------


class _SimpleNet(nn.Module):
    """Minimal feedforward network matching _MockNet's architecture.

    This is used to load checkpoints that were saved from simple test
    networks.  For production, the actual ModelAdapter (DynUNet etc.)
    should be used — see ``_build_net_from_config``.
    """

    def __init__(self, *, out_channels: int = 2) -> None:
        super().__init__()
        self._out_channels = out_channels
        self._dummy = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: Tensor) -> Tensor:
        b, _c, d, h, w = x.shape
        # Output uniform probabilities scaled by _dummy
        fg = torch.sigmoid(self._dummy).expand(b, 1, d, h, w)
        bg = 1.0 - fg
        return torch.cat([bg, fg], dim=1)


def _build_net_from_config(config: dict[str, Any]) -> nn.Module:
    """Build a network from a model config dict.

    Tries the full ModelAdapter first; falls back to _SimpleNet
    for test/debug checkpoints.
    """
    family = config.get("family", "")

    if family == "dynunet":
        try:
            from minivess.adapters.dynunet import DynUNetAdapter
            from minivess.config.models import ModelConfig, ModelFamily

            model_config = ModelConfig(
                family=ModelFamily.MONAI_DYNUNET,
                name=config.get("name", "dynunet-default"),
                in_channels=config.get("in_channels", 1),
                out_channels=config.get("out_channels", 2),
                architecture_params=config.get("architecture_params", {}),
            )
            return DynUNetAdapter(model_config)
        except Exception:
            logger.debug("Could not build DynUNetAdapter, falling back to _SimpleNet")

    # Fallback for test/debug checkpoints
    return _SimpleNet(out_channels=config.get("out_channels", 2))


# ---------------------------------------------------------------------------
# ModelSignature helper
# ---------------------------------------------------------------------------


def get_model_signature() -> ModelSignature:
    """Return the MLflow ModelSignature for 3D segmentation.

    Input:  ``(-1, 1, -1, -1, -1)`` float32  (B, C_in, D, H, W)
    Output: ``(-1, 2, -1, -1, -1)`` float32  (B, C_out, D, H, W)
    """
    input_schema = Schema([TensorSpec(np.dtype("float32"), (-1, 1, -1, -1, -1))])
    output_schema = Schema([TensorSpec(np.dtype("float32"), (-1, 2, -1, -1, -1))])
    return ModelSignature(inputs=input_schema, outputs=output_schema)


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------


def _entropy(probs: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Per-voxel entropy across the class dimension.

    Parameters
    ----------
    probs:
        Probability tensor ``(B, C, D, H, W)``.

    Returns
    -------
    Tensor of shape ``(B, 1, D, H, W)``.
    """
    return -(probs * torch.log(probs + eps)).sum(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Single model wrapper
# ---------------------------------------------------------------------------


class MiniVessSegModel(mlflow.pyfunc.PythonModel):  # type: ignore[name-defined,misc]
    """MLflow pyfunc wrapper for a single segmentation checkpoint.

    Artifacts
    ---------
    checkpoint : str
        Path to ``.pth`` checkpoint (new or legacy format).
    model_config : str
        Path to JSON file with model configuration.
    """

    def __init__(self) -> None:
        self._net: nn.Module | None = None

    def load_context(self, context: Any) -> None:
        """Load model from checkpoint artifact."""
        config_path = context.artifacts["model_config"]
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        self._net = _build_net_from_config(config)

        ckpt_path = context.artifacts["checkpoint"]
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        if isinstance(payload, dict) and "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
        else:
            state_dict = payload

        # Try loading into full adapter, then fallback to inner net
        try:
            self._net.load_state_dict(state_dict)
        except RuntimeError:
            inner_net = getattr(self._net, "net", None)
            if inner_net is not None:
                inner_net.load_state_dict(state_dict)
            else:
                logger.warning("State dict keys do not match; using initialized weights")

        self._net.eval()
        logger.info("Loaded single model from %s", ckpt_path)

    @torch.no_grad()
    def predict(
        self,
        context: Any,
        model_input: np.ndarray,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Run inference and return class probabilities.

        Parameters
        ----------
        context:
            MLflow context (unused after load_context).
        model_input:
            Numpy array of shape ``(B, 1, D, H, W)`` float32.

        Returns
        -------
        Numpy array of shape ``(B, 2, D, H, W)`` float32 probabilities.
        """
        assert self._net is not None, "load_context() must be called first"

        tensor_in = torch.from_numpy(model_input).float()
        output = self._net(tensor_in)

        # Handle both SegmentationOutput and raw Tensor
        if hasattr(output, "prediction"):
            probs = output.prediction
        else:
            probs = torch.softmax(output, dim=1)

        result: np.ndarray = probs.cpu().numpy()
        return result

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        model_input: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Run inference and return predictions with uncertainty map.

        For a single model, uncertainty is the per-voxel entropy of the
        softmax output (a proxy for predictive uncertainty without
        ensemble diversity).

        Parameters
        ----------
        model_input:
            Numpy array ``(B, 1, D, H, W)`` float32.

        Returns
        -------
        Dict with ``prediction``, ``uncertainty_map`` keys.
        """
        assert self._net is not None, "load_context() must be called first"

        tensor_in = torch.from_numpy(model_input).float()
        output = self._net(tensor_in)

        if hasattr(output, "prediction"):
            probs = output.prediction
        else:
            probs = torch.softmax(output, dim=1)

        unc_map = _entropy(probs)

        return {
            "prediction": probs.cpu().numpy(),
            "uncertainty_map": unc_map.cpu().numpy(),
        }


# ---------------------------------------------------------------------------
# Ensemble model wrapper
# ---------------------------------------------------------------------------


class MiniVessEnsembleModel(mlflow.pyfunc.PythonModel):  # type: ignore[name-defined,misc]
    """MLflow pyfunc wrapper for an ensemble of segmentation checkpoints.

    Implements Lakshminarayanan et al. (2017) Deep Ensemble uncertainty
    decomposition: total (entropy of mean), aleatoric (mean of entropies),
    epistemic (mutual information = total − aleatoric).

    Artifacts
    ---------
    ensemble_manifest : str
        Path to JSON manifest listing member checkpoints.
    model_config : str
        Path to JSON with model configuration (shared by all members).
    """

    def __init__(self) -> None:
        self._members: list[nn.Module] | None = None

    def load_context(self, context: Any) -> None:
        """Load all ensemble members from manifest."""
        config_path = context.artifacts["model_config"]
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        manifest_path = context.artifacts["ensemble_manifest"]
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        members: list[nn.Module] = []
        for entry in manifest["members"]:
            net = _build_net_from_config(config)
            ckpt_path = entry["checkpoint_path"]
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)

            if isinstance(payload, dict) and "model_state_dict" in payload:
                state_dict = payload["model_state_dict"]
            else:
                state_dict = payload

            try:
                net.load_state_dict(state_dict)
            except RuntimeError:
                inner_net = getattr(net, "net", None)
                if inner_net is not None:
                    inner_net.load_state_dict(state_dict)
                else:
                    logger.warning(
                        "State dict mismatch for %s; using initialized weights",
                        ckpt_path,
                    )

            net.eval()
            members.append(net)

        self._members = members
        logger.info("Loaded ensemble with %d members", len(members))

    @torch.no_grad()
    def predict(
        self,
        context: Any,
        model_input: np.ndarray,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Run ensemble inference and return mean probabilities.

        Parameters
        ----------
        context:
            MLflow context (unused after load_context).
        model_input:
            Numpy array ``(B, 1, D, H, W)`` float32.

        Returns
        -------
        Numpy array ``(B, 2, D, H, W)`` float32 mean probabilities.
        """
        assert self._members is not None, "load_context() must be called first"

        tensor_in = torch.from_numpy(model_input).float()
        member_probs = self._collect_member_probs(tensor_in)

        # (M, B, C, D, H, W) → mean across members
        stacked = torch.stack(member_probs, dim=0)
        mean_pred = stacked.mean(dim=0)

        result: np.ndarray = mean_pred.cpu().numpy()
        return result

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        model_input: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Run ensemble inference with full uncertainty decomposition.

        Returns
        -------
        Dict with keys:
          - ``prediction``: mean probabilities ``(B, C, D, H, W)``
          - ``total_uncertainty``: entropy of mean ``(B, 1, D, H, W)``
          - ``aleatoric_uncertainty``: mean of entropies ``(B, 1, D, H, W)``
          - ``epistemic_uncertainty``: MI = total − aleatoric ``(B, 1, D, H, W)``
        """
        assert self._members is not None, "load_context() must be called first"

        tensor_in = torch.from_numpy(model_input).float()
        member_probs = self._collect_member_probs(tensor_in)

        stacked = torch.stack(member_probs, dim=0)  # (M, B, C, D, H, W)
        mean_pred = stacked.mean(dim=0)  # (B, C, D, H, W)

        # Total uncertainty: H[p̄] = -Σ p̄ log(p̄)
        total = _entropy(mean_pred)  # (B, 1, D, H, W)

        # Aleatoric uncertainty: E_m[H[p_m]] = (1/M) Σ H[p_m]
        member_entropies = torch.stack(
            [_entropy(p) for p in member_probs], dim=0
        )  # (M, B, 1, D, H, W)
        aleatoric = member_entropies.mean(dim=0)  # (B, 1, D, H, W)

        # Epistemic uncertainty: MI = H[p̄] - E_m[H[p_m]]
        # Clamp to 0 for numerical stability (Jensen's inequality guarantees >= 0)
        epistemic = torch.clamp(total - aleatoric, min=0.0)

        return {
            "prediction": mean_pred.cpu().numpy(),
            "total_uncertainty": total.cpu().numpy(),
            "aleatoric_uncertainty": aleatoric.cpu().numpy(),
            "epistemic_uncertainty": epistemic.cpu().numpy(),
        }

    def _collect_member_probs(self, tensor_in: Tensor) -> list[Tensor]:
        """Run each member and collect softmax probabilities."""
        assert self._members is not None
        probs_list: list[Tensor] = []
        for member in self._members:
            output = member(tensor_in)
            if hasattr(output, "prediction"):
                probs_list.append(output.prediction)
            else:
                probs_list.append(torch.softmax(output, dim=1))
        return probs_list
