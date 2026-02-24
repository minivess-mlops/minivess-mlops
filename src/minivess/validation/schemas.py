from __future__ import annotations

import pandera as pa
from pandera.typing import (
    Series,  # noqa: TC002 — runtime requirement for Pandera DataFrameModel
)


class NiftiMetadataSchema(pa.DataFrameModel):
    """Schema for NIfTI file metadata DataFrame.

    Validates dataset-level metadata extracted from NIfTI headers
    at DVC pipeline boundaries.
    """

    file_path: Series[str] = pa.Field(nullable=False)
    shape_x: Series[int] = pa.Field(ge=1, le=2048)
    shape_y: Series[int] = pa.Field(ge=1, le=2048)
    shape_z: Series[int] = pa.Field(ge=1, le=512)
    voxel_spacing_x: Series[float] = pa.Field(gt=0.0, le=10.0)
    voxel_spacing_y: Series[float] = pa.Field(gt=0.0, le=10.0)
    voxel_spacing_z: Series[float] = pa.Field(gt=0.0, le=50.0)
    intensity_min: Series[float] = pa.Field()
    intensity_max: Series[float] = pa.Field()
    num_foreground_voxels: Series[int] = pa.Field(ge=0)
    has_valid_affine: Series[bool] = pa.Field()

    class Config:
        strict = True
        coerce = True

    @pa.check("intensity_max", name="intensity_range_valid")
    @classmethod
    def intensity_max_gt_min(cls, series: Series[float]) -> Series[bool]:
        """Intensity max must be greater than min (checked at row level via dataframe check)."""
        return series >= 0  # Basic check; cross-column validated below

    @pa.dataframe_check
    @classmethod
    def intensity_range_valid(cls, df: pa.typing.DataFrame) -> Series[bool]:
        """Intensity max must exceed intensity min for each sample."""
        return df["intensity_max"] > df["intensity_min"]

    @pa.dataframe_check
    @classmethod
    def affine_all_valid(cls, df: pa.typing.DataFrame) -> Series[bool]:
        """All samples must have valid affine matrices."""
        return df["has_valid_affine"]


class TrainingMetricsSchema(pa.DataFrameModel):
    """Schema for training metrics DataFrame logged to MLflow.

    Validates the tabular metrics DataFrame produced after each training run.
    Great Expectations handles batch quality checks on top of this.
    """

    run_id: Series[str] = pa.Field(nullable=False)
    epoch: Series[int] = pa.Field(ge=0)
    fold: Series[int] = pa.Field(ge=0)
    train_loss: Series[float] = pa.Field(ge=0.0)
    val_loss: Series[float] = pa.Field(ge=0.0)
    val_dice: Series[float] = pa.Field(ge=0.0, le=1.0)
    val_cldice: Series[float] = pa.Field(ge=0.0, le=1.0)
    val_nsd: Series[float] = pa.Field(ge=0.0, le=1.0)
    learning_rate: Series[float] = pa.Field(gt=0.0)

    class Config:
        strict = False  # Allow additional metric columns
        coerce = True


class AnnotationQualitySchema(pa.DataFrameModel):
    """Schema for annotation quality metrics from Label Studio exports."""

    sample_id: Series[str] = pa.Field(nullable=False)
    annotator_id: Series[str] = pa.Field(nullable=False)
    num_connected_components: Series[int] = pa.Field(ge=0)
    foreground_ratio: Series[float] = pa.Field(ge=0.0, le=1.0)
    has_boundary_touching: Series[bool] = pa.Field()
    inter_annotator_dice: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)

    class Config:
        strict = False
        coerce = True
