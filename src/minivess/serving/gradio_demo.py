"""Gradio demo app for interactive MinIVess segmentation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def load_nifti_volume(file_path: str) -> NDArray[np.float32]:
    """Load a NIfTI file and return as float32 numpy array.

    Parameters
    ----------
    file_path:
        Path to a .nii or .nii.gz file.

    Returns
    -------
    3D numpy array (D, H, W) as float32.
    """
    import nibabel as nib

    nii = nib.load(file_path)
    data = np.asarray(nii.dataobj, dtype=np.float32)
    return data


def extract_slice(
    volume: NDArray[np.float32],
    axis: int = 2,
    index: int = 0,
) -> NDArray[np.float32]:
    """Extract a 2D slice from a 3D volume.

    Parameters
    ----------
    volume:
        3D array of shape (D, H, W).
    axis:
        Axis to slice along (0=sagittal, 1=coronal, 2=axial).
    index:
        Slice index (clamped to valid range).

    Returns
    -------
    2D slice as float32.
    """
    # Clamp index to valid range
    max_idx = volume.shape[axis] - 1
    index = max(0, min(index, max_idx))
    return np.take(volume, index, axis=axis).astype(np.float32)


def build_demo(
    model_path: Path | None = None,
    *,
    share: bool = False,
) -> Any:
    """Create and return a Gradio demo interface.

    Parameters
    ----------
    model_path:
        Path to ONNX model. If None, uses a dummy predictor.
    share:
        Whether to create a public Gradio link.

    Returns
    -------
    Gradio Blocks interface.
    """
    import gradio as gr

    def predict_slice(
        volume_slice: np.ndarray | None,
    ) -> tuple[np.ndarray | None, str]:
        """Process a single 2D slice (for demo purposes)."""
        if volume_slice is None:
            return None, "No input provided"

        # Normalize to [0, 1]
        if volume_slice.max() > 1.0:
            volume_slice = volume_slice.astype(np.float32) / 255.0

        # Convert to grayscale if RGB
        if volume_slice.ndim == 3:
            volume_slice = np.mean(volume_slice, axis=-1)

        # Simple threshold segmentation as demo fallback
        if model_path is None:
            threshold = 0.5
            mask = (volume_slice > threshold).astype(np.float32)
            info = "Demo mode (threshold segmentation, no model loaded)"
        else:
            from minivess.serving.onnx_inference import OnnxSegmentationInference

            engine = OnnxSegmentationInference(model_path)
            # Pad to (1, 1, 1, H, W) for the ONNX model
            h, w = volume_slice.shape
            input_tensor = volume_slice.reshape(1, 1, 1, h, w).astype(np.float32)
            result = engine.predict(input_tensor)
            mask = result["segmentation"][0, 0].astype(np.float32)
            info = f"ONNX model prediction (shape: {mask.shape})"

        return mask, info

    with gr.Blocks(title="MinIVess Segmentation Demo") as demo:
        gr.Markdown("# MinIVess 3D Vessel Segmentation")
        gr.Markdown("Upload a 2D slice to see segmentation results.")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Input Slice",
                    type="numpy",
                )
                submit_btn = gr.Button("Segment", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Segmentation Mask")
                info_text = gr.Textbox(label="Info", interactive=False)

        submit_btn.click(
            fn=predict_slice,
            inputs=[input_image],
            outputs=[output_image, info_text],
        )

    return demo


def main() -> None:
    """Launch the Gradio demo."""
    parser = argparse.ArgumentParser(description="MinIVess Gradio Demo")
    parser.add_argument("--model", type=Path, default=None, help="ONNX model path")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    args = parser.parse_args()

    demo = build_demo(model_path=args.model, share=args.share)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
