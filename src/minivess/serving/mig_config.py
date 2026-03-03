"""NVIDIA MIG (Multi-Instance GPU) device management.

Provides detection of MIG-partitioned GPUs and model-to-instance
assignment for multi-model inference serving.

MIG allows a single GPU (A100, A30, H100) to be partitioned into
up to 7 isolated instances, each with dedicated memory and compute.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MIGConfig:
    """Configuration for MIG-based multi-model serving.

    Attributes
    ----------
    gpu_index:
        Physical GPU index.
    instances:
        List of MIG instance dicts with ``instance_id`` and ``model_name``.
    """

    gpu_index: int = 0
    instances: list[dict[str, Any]] = field(default_factory=list)


def detect_mig_devices() -> list[dict[str, Any]]:
    """Detect available MIG device instances.

    Returns
    -------
    List of dicts with ``gpu_index``, ``instance_id``, ``memory_gb``.
    Empty list if MIG is not available or nvidia-ml-py is not installed.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        mig_devices: list[dict[str, Any]] = []
        for gpu_idx in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
            try:
                # Check if MIG mode is enabled
                current_mode, _pending_mode = pynvml.nvmlDeviceGetMigMode(handle)
                if current_mode != pynvml.NVML_DEVICE_MIG_ENABLE:
                    continue

                # Enumerate MIG instances
                max_count = 7  # A100 supports up to 7 instances
                mig_count = pynvml.nvmlDeviceGetMaxMigDeviceCount(handle)
                for inst_idx in range(min(mig_count, max_count)):
                    try:
                        mig_handle = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(
                            handle, inst_idx
                        )
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(mig_handle)
                        mig_devices.append(
                            {
                                "gpu_index": gpu_idx,
                                "instance_id": inst_idx,
                                "memory_gb": round(mem_info.total / (1024**3), 1),
                            }
                        )
                    except pynvml.NVMLError:
                        continue
            except pynvml.NVMLError:
                # MIG not supported on this GPU
                continue

        pynvml.nvmlShutdown()
        return mig_devices

    except (ImportError, Exception):
        logger.debug("MIG detection unavailable (nvidia-ml-py not installed or no GPU)")
        return []


def assign_models_to_instances(
    models: list[str],
    instances: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign models to MIG instances using round-robin.

    Parameters
    ----------
    models:
        List of model names to assign.
    instances:
        List of MIG instance dicts with ``instance_id`` and ``memory_gb``.

    Returns
    -------
    List of assignment dicts with ``model_name``, ``instance_id``, ``memory_gb``.
    """
    if not models:
        return []

    assignments: list[dict[str, Any]] = []
    for i, model_name in enumerate(models):
        if instances:
            instance = instances[i % len(instances)]
            assignments.append(
                {
                    "model_name": model_name,
                    "instance_id": instance["instance_id"],
                    "memory_gb": instance.get("memory_gb", 0),
                }
            )
        else:
            # No MIG instances — assign to default GPU
            assignments.append(
                {
                    "model_name": model_name,
                    "instance_id": 0,
                    "memory_gb": 0,
                }
            )

    return assignments
