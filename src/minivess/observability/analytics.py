from __future__ import annotations

import logging
from typing import Any

import duckdb
import pandas as pd
from mlflow.tracking import MlflowClient

from minivess.observability.tracking import resolve_tracking_uri

logger = logging.getLogger(__name__)


class RunAnalytics:
    """In-process SQL analytics over MLflow experiment runs using DuckDB."""

    def __init__(self, *, tracking_uri: str | None = None) -> None:
        resolved_uri = resolve_tracking_uri(tracking_uri=tracking_uri)
        self.client = MlflowClient(tracking_uri=resolved_uri)
        self.conn = duckdb.connect(":memory:")

    def load_experiment_runs(self, experiment_name: str) -> pd.DataFrame:
        """Load all runs from an MLflow experiment into a DataFrame.

        Returns DataFrame with columns: run_id, run_name, status,
        and all logged params/metrics.
        """
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            msg = f"Experiment '{experiment_name}' not found"
            raise ValueError(msg)

        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
        )

        records = []
        for run in runs:
            record: dict[str, Any] = {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
            }
            record.update({f"param_{k}": v for k, v in run.data.params.items()})
            record.update({f"metric_{k}": v for k, v in run.data.metrics.items()})
            records.append(record)

        df = pd.DataFrame(records)
        logger.info("Loaded %d runs from experiment '%s'", len(df), experiment_name)
        return df

    def query(self, sql: str, **kwargs: Any) -> pd.DataFrame:
        """Execute a SQL query against loaded DataFrames.

        DataFrames registered via ``register_dataframe`` or passed as kwargs
        can be referenced by name in the SQL query.

        Example::

            analytics.register_dataframe("runs", runs_df)
            result = analytics.query("SELECT * FROM runs WHERE metric_val_dice > 0.8")
        """
        for name, value in kwargs.items():
            if isinstance(value, pd.DataFrame):
                self.conn.register(name, value)

        return self.conn.execute(sql).fetchdf()

    def register_dataframe(self, name: str, df: pd.DataFrame) -> None:
        """Register a DataFrame as a named table for SQL queries."""
        self.conn.register(name, df)

    def cross_fold_summary(self, runs_df: pd.DataFrame) -> pd.DataFrame:
        """Compute cross-fold summary statistics.

        Expects *runs_df* to have ``param_fold`` and metric columns.
        Returns mean and std of each metric grouped by fold.
        """
        self.register_dataframe("runs", runs_df)

        metric_cols = [c for c in runs_df.columns if c.startswith("metric_")]
        if not metric_cols:
            return pd.DataFrame()

        agg_parts = []
        for col in metric_cols:
            name = col.removeprefix("metric_")
            agg_parts.append(f'AVG(CAST("{col}" AS DOUBLE)) AS "{name}_mean"')
            agg_parts.append(f'STDDEV(CAST("{col}" AS DOUBLE)) AS "{name}_std"')

        agg_sql = ", ".join(agg_parts)
        sql = f"""
            SELECT param_fold AS fold, {agg_sql}
            FROM runs
            WHERE param_fold IS NOT NULL
            GROUP BY param_fold
            ORDER BY param_fold
        """
        return self.conn.execute(sql).fetchdf()

    def top_models(
        self,
        runs_df: pd.DataFrame,
        *,
        metric: str = "metric_val_dice",
        n: int = 5,
    ) -> pd.DataFrame:
        """Select top-N models by a given metric."""
        self.register_dataframe("runs", runs_df)
        # Include param_fold only if it exists in the DataFrame
        fold_col = ", param_fold" if "param_fold" in runs_df.columns else ""
        sql = f"""
            SELECT run_id, run_name{fold_col}, "{metric}"
            FROM runs
            WHERE "{metric}" IS NOT NULL
            ORDER BY "{metric}" DESC
            LIMIT {n}
        """
        return self.conn.execute(sql).fetchdf()

    def cost_by_model_family(self, runs_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate cost metrics by model family.

        Returns DataFrame with mean/std of cost_total_usd and effective_gpu_rate
        grouped by param_model_family.

        Returns empty DataFrame if input is empty.
        """
        if runs_df.empty:
            return pd.DataFrame()

        self.register_dataframe("cost_runs", runs_df)
        sql = """
            SELECT
                param_model_family AS model_family,
                AVG(CAST(metric_cost_total_usd AS DOUBLE)) AS total_cost_usd_mean,
                STDDEV(CAST(metric_cost_total_usd AS DOUBLE)) AS total_cost_usd_std,
                AVG(CAST(metric_cost_effective_gpu_rate AS DOUBLE)) AS effective_rate_mean
            FROM cost_runs
            WHERE param_model_family IS NOT NULL
            GROUP BY param_model_family
            ORDER BY total_cost_usd_mean DESC
        """
        return self.conn.execute(sql).fetchdf()

    def cost_trends(self, runs_df: pd.DataFrame) -> pd.DataFrame:
        """Compute cumulative cost trend over time.

        Returns DataFrame ordered by start_time with cumulative_cost_usd column.
        """
        if runs_df.empty:
            return pd.DataFrame()

        self.register_dataframe("trend_runs", runs_df)
        sql = """
            SELECT
                run_id,
                start_time,
                metric_cost_total_usd,
                SUM(CAST(metric_cost_total_usd AS DOUBLE))
                    OVER (ORDER BY start_time) AS cumulative_cost_usd
            FROM trend_runs
            WHERE metric_cost_total_usd IS NOT NULL
            ORDER BY start_time
        """
        return self.conn.execute(sql).fetchdf()

    def break_even_analysis(self, runs_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate break-even and amortization metrics by model family.

        Returns DataFrame with avg_break_even_epochs and avg_epochs_to_amortize
        grouped by param_model_family.
        """
        if runs_df.empty:
            return pd.DataFrame()

        self.register_dataframe("be_runs", runs_df)
        sql = """
            SELECT
                param_model_family AS model_family,
                AVG(CAST(metric_cost_break_even_epochs AS DOUBLE)) AS avg_break_even_epochs,
                AVG(CAST(metric_cost_epochs_to_amortize_setup AS DOUBLE)) AS avg_epochs_to_amortize
            FROM be_runs
            WHERE param_model_family IS NOT NULL
            GROUP BY param_model_family
            ORDER BY avg_break_even_epochs
        """
        return self.conn.execute(sql).fetchdf()

    def close(self) -> None:
        """Close the DuckDB connection."""
        self.conn.close()
