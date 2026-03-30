#!/usr/bin/env Rscript
# render_all.R — Master renderer for all publication figures and tables
#
# Reads JSON sidecars from /app/r_data/ (or R_DATA_DIR env var)
# Writes PDF/PNG figures to /app/r_output/figures/
# Writes .tex/.csv tables to /app/r_output/tables/
#
# Exit code: 0 if all renderers succeed, 1 if any fail.
# Each renderer uses tryCatch — failures are logged but don't crash others.
#
# Plan: local-dynunet-mechanics-debug-plan-biostats-finish.xml v3.0

library(jsonlite)

# Resolve I/O directories from environment or defaults
r_data_dir <- Sys.getenv("R_DATA_DIR", "/app/r_data")
r_output_dir <- Sys.getenv("R_OUTPUT_DIR", "/app/r_output")

fig_dir <- file.path(r_output_dir, "figures")
tab_dir <- file.path(r_output_dir, "tables")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== Vascadia R/ggplot2 Figure Renderer ===\n")
cat(sprintf("Input:  %s\n", r_data_dir))
cat(sprintf("Output: %s\n", r_output_dir))

# Track success/failure
results <- list()

# Helper: source a script with error handling
run_renderer <- function(script_name) {
  script_path <- file.path(dirname(sys.frame(1)$ofile %||% "r_scripts"), script_name)
  if (!file.exists(script_path)) {
    # Try relative to working directory
    script_path <- file.path("r_scripts", script_name)
  }
  cat(sprintf("\n--- Rendering: %s ---\n", script_name))
  tryCatch({
    source(script_path, local = new.env(parent = globalenv()))
    cat(sprintf("  OK: %s completed\n", script_name))
    results[[script_name]] <<- "OK"
  }, error = function(e) {
    cat(sprintf("  FAILED: %s — %s\n", script_name, conditionMessage(e)))
    results[[script_name]] <<- paste("FAILED:", conditionMessage(e))
  })
}

# Render 10 figures
figure_scripts <- c(
  "fig_distribution.R",
  "fig_consort.R",
  "fig_effect_size_heatmap.R",
  "fig_forest_plot.R",
  "fig_interaction.R",
  "fig_specification_curve.R",
  "fig_generalization_gap.R",
  "fig_cd_diagram.R",
  "fig_variance_lollipop.R",
  "fig_calibration.R"
)

# Render 7 tables
table_scripts <- c(
  "tab_comparison.R",
  "tab_effect_sizes.R",
  "tab_variance.R",
  "tab_rankings.R",
  "tab_anova.R",
  "tab_power.R",
  "tab_calibration.R"
)

for (s in c(figure_scripts, table_scripts)) {
  run_renderer(s)
}

# Summary
cat("\n=== Render Summary ===\n")
n_ok <- sum(results == "OK")
n_fail <- length(results) - n_ok
cat(sprintf("Succeeded: %d / %d\n", n_ok, length(results)))
if (n_fail > 0) {
  cat("Failures:\n")
  for (name in names(results)) {
    if (results[[name]] != "OK") {
      cat(sprintf("  - %s: %s\n", name, results[[name]]))
    }
  }
  quit(status = 1)
} else {
  cat("All renderers completed successfully.\n")
}
