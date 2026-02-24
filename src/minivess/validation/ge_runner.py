"""Great Expectations batch validation runner.

Executes GE expectation suite configs (from expectations.py) against
pandas DataFrames using an ephemeral context. Returns GateResult for
pipeline integration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import great_expectations as gx

from minivess.validation.expectations import (
    build_nifti_metadata_suite,
    build_training_metrics_suite,
)
from minivess.validation.gates import GateResult

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# Mapping from suite config expectation_type to GE expectation classes
_EXPECTATION_MAP: dict[str, type] = {
    "expect_column_values_to_not_be_null": gx.expectations.ExpectColumnValuesToNotBeNull,
    "expect_column_values_to_be_between": gx.expectations.ExpectColumnValuesToBeBetween,
    "expect_column_values_to_be_unique": gx.expectations.ExpectColumnValuesToBeUnique,
    "expect_column_values_to_be_increasing": gx.expectations.ExpectColumnValuesToBeIncreasing,
    "expect_column_values_to_be_true": gx.expectations.ExpectColumnValuesToBeInSet,
    "expect_column_pair_values_a_to_be_greater_than_b": gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB,
    "expect_table_row_count_to_be_between": gx.expectations.ExpectTableRowCountToBeBetween,
}


def _build_ge_suite(
    context: gx.data_context.EphemeralDataContext,
    suite_config: dict[str, Any],
) -> gx.ExpectationSuite:
    """Convert a suite config dict to a GE ExpectationSuite.

    Parameters
    ----------
    context:
        GE ephemeral context.
    suite_config:
        Dict with ``expectation_suite_name`` and ``expectations`` list.

    Returns
    -------
    Registered ExpectationSuite.
    """
    suite_name = suite_config["expectation_suite_name"]
    suite = context.suites.add(gx.ExpectationSuite(name=suite_name))

    for exp_config in suite_config["expectations"]:
        exp_type = exp_config["expectation_type"]
        kwargs = dict(exp_config.get("kwargs", {}))

        # Handle boolean column validation (expect_column_values_to_be_true)
        if exp_type == "expect_column_values_to_be_true":
            kwargs["value_set"] = [True]

        exp_class = _EXPECTATION_MAP.get(exp_type)
        if exp_class is None:
            logger.warning("Unknown expectation type: %s — skipping", exp_type)
            continue

        suite.add_expectation(exp_class(**kwargs))

    return suite


def run_expectation_suite(
    df: pd.DataFrame,
    suite_config: dict[str, Any],
) -> GateResult:
    """Execute a GE expectation suite against a DataFrame.

    Parameters
    ----------
    df:
        DataFrame to validate.
    suite_config:
        Dict with ``expectation_suite_name`` and ``expectations`` list
        (as returned by ``build_*_suite()`` functions).

    Returns
    -------
    GateResult with pass/fail, errors, and statistics.
    """
    context = gx.get_context()

    # Set up data source and batch
    suite_name = suite_config["expectation_suite_name"]
    ds_name = f"ds_{suite_name}"
    data_source = context.data_sources.add_pandas(ds_name)
    data_asset = data_source.add_dataframe_asset(name=f"asset_{suite_name}")
    batch_def = data_asset.add_batch_definition_whole_dataframe(f"batch_{suite_name}")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    # Build and run suite
    suite = _build_ge_suite(context, suite_config)
    validation_result = batch.validate(suite)

    # Extract errors from failed expectations
    errors: list[str] = []
    for result in validation_result.results:
        if not result.success:
            exp_type = result.expectation_config.type
            exp_kwargs = result.expectation_config.kwargs
            errors.append(f"{exp_type}: {exp_kwargs}")

    # Build statistics
    n_evaluated = len(validation_result.results)
    n_successful = sum(1 for r in validation_result.results if r.success)

    return GateResult(
        passed=validation_result.success,
        errors=errors,
        statistics={
            "evaluated_expectations": n_evaluated,
            "successful_expectations": n_successful,
            "unsuccessful_expectations": n_evaluated - n_successful,
        },
    )


def validate_nifti_batch(df: pd.DataFrame) -> GateResult:
    """Run GE nifti_metadata_suite against a NIfTI metadata DataFrame.

    Parameters
    ----------
    df:
        DataFrame with NIfTI metadata columns.

    Returns
    -------
    GateResult.
    """
    suite_config = build_nifti_metadata_suite()
    return run_expectation_suite(df, suite_config)


def validate_metrics_batch(df: pd.DataFrame) -> GateResult:
    """Run GE training_metrics_suite against a metrics DataFrame.

    Parameters
    ----------
    df:
        DataFrame with training metrics columns.

    Returns
    -------
    GateResult.
    """
    suite_config = build_training_metrics_suite()
    return run_expectation_suite(df, suite_config)
