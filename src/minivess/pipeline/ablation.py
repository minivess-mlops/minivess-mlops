"""DynUNet width × loss ablation grid configuration.

Generates systematic experiment configurations for studying the effect
of encoder width and topology-aware losses on vessel segmentation.
"""

from __future__ import annotations

DYNUNET_WIDTH_PRESETS: dict[str, list[int]] = {
    "full": [32, 64, 128, 256],
    "half": [16, 32, 64, 128],
    "quarter": [8, 16, 32, 64],
}


def build_ablation_grid(
    widths: list[str],
    losses: list[str],
) -> list[dict[str, object]]:
    """Generate all width × loss experiment configurations.

    Parameters
    ----------
    widths:
        Width preset names (keys of ``DYNUNET_WIDTH_PRESETS``).
    losses:
        Loss function identifiers accepted by ``build_loss_function``.

    Returns
    -------
    list[dict[str, object]]
        Each entry has ``width_name``, ``filters``, ``loss_name``,
        and ``experiment_name``.
    """
    grid: list[dict[str, object]] = []
    for width_name in widths:
        filters = DYNUNET_WIDTH_PRESETS[width_name]
        for loss_name in losses:
            grid.append({
                "width_name": width_name,
                "filters": filters,
                "loss_name": loss_name,
                "experiment_name": f"dynunet-{width_name}-{loss_name}",
            })
    return grid
