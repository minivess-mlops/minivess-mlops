#!/usr/bin/env Rscript
# tab_anova.R — 2-way ANOVA table: Factor, F-stat, p-value, eta-squared, omega-squared
# Input: anova_results.json
# Output: tables/tab_anova.tex, tables/tab_anova.csv

suppressPackageStartupMessages({
  library(jsonlite)
  library(knitr)
  library(kableExtra)
})

r_data_dir <- Sys.getenv("R_DATA_DIR", "/app/r_data")
r_output_dir <- Sys.getenv("R_OUTPUT_DIR", "/app/r_output")
tab_dir <- file.path(r_output_dir, "tables")
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)

json_path <- file.path(r_data_dir, "anova_results.json")
if (!file.exists(json_path)) {
  cat("tab_anova: anova_results.json not found — empty table\n")
  empty_df <- data.frame(Metric = character(), Factor = character(),
                         F_stat = character(), p = character(),
                         eta_sq = character(), omega_sq = character())
  write.csv(empty_df, file.path(tab_dir, "tab_anova.csv"), row.names = FALSE)
  writeLines("% No ANOVA results available", file.path(tab_dir, "tab_anova.tex"))
  quit(save = "no")
}

dat <- fromJSON(json_path, flatten = TRUE)
df <- as.data.frame(dat)

# Detect columns
f_col <- intersect(names(df), c("f_stat", "F", "f_value", "statistic"))[1]
p_col <- intersect(names(df), c("p_value", "p", "pvalue", "p_adj"))[1]
eta_col <- intersect(names(df), c("eta_squared", "eta_sq", "partial_eta_sq"))[1]
omega_col <- intersect(names(df), c("omega_squared", "omega_sq"))[1]
factor_col <- intersect(names(df), c("factor", "term", "source"))[1]
metric_col <- intersect(names(df), c("metric", "dependent_var"))[1]

out <- data.frame(stringsAsFactors = FALSE)
if (!is.na(metric_col)) out$Metric <- df[[metric_col]]
if (!is.na(factor_col)) out$Factor <- df[[factor_col]]
out[["F"]] <- if (!is.na(f_col)) sprintf("%.2f", df[[f_col]]) else "---"
out$p <- if (!is.na(p_col)) sprintf("%.4f", df[[p_col]]) else "---"
out[["$\\eta^2$"]] <- if (!is.na(eta_col)) sprintf("%.4f", df[[eta_col]]) else "---"
out[["$\\omega^2$"]] <- if (!is.na(omega_col)) sprintf("%.4f", df[[omega_col]]) else "---"

# Bold significant rows
if (!is.na(p_col)) {
  sig <- df[[p_col]] < 0.05
  if (!is.na(f_col)) {
    out[["F"]][sig] <- paste0("\\textbf{", out[["F"]][sig], "}")
  }
  out$p[sig] <- paste0("\\textbf{", out$p[sig], "}")
}

write.csv(out, file.path(tab_dir, "tab_anova.csv"), row.names = FALSE)

tex <- kable(out, format = "latex", booktabs = TRUE, escape = FALSE,
             caption = "Two-way ANOVA: loss function $\\times$ auxiliary calibration") %>%
  kable_styling(latex_options = c("hold_position"), font_size = 7)
writeLines(tex, file.path(tab_dir, "tab_anova.tex"))
cat("tab_anova: done\n")
