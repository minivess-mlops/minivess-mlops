-- DuckDB Skills state file for MinIVess MLOps biostatistics
-- Attach the biostatistics database after a flow run
-- Usage: /duckdb-skills:attach-db outputs/biostatistics/biostatistics.duckdb

-- Example queries after attaching:
-- SELECT model_family, post_training_method, ensemble_strategy, COUNT(*)
--   FROM runs GROUP BY 1, 2, 3;
-- SELECT * FROM eval_metrics WHERE metric_name = 'cldice';
-- SELECT * FROM per_volume_metrics LIMIT 10;
