#!/usr/bin/env Rscript
# tab_comparison.R — Main comparison table: metric, condition pair, effect size, CI, p-value
# Input: pairwise_results.json
# Output: tables/tab_comparison.tex, tables/tab_comparison.csv

suppressPackageStartupMessages({
  library(jsonlite)
  library(knitr)
  library(kableExtra)
})

r_data_dir <- Sys.getenv("R_DATA_DIR", "/app/r_data")
r_output_dir <- Sys.getenv("R_OUTPUT_DIR", "/app/r_output")
tab_dir <- file.path(r_output_dir, "tables")
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)

json_path <- file.path(r_data_dir, "pairwise_results.json")
if (!file.exists(json_path)) {
  cat("tab_comparison: pairwise_results.json not found — empty table\n")
  empty_df <- data.frame(Metric = character(), Pair = character(),
                         Effect = numeric(), CI = character(), p = numeric())
  write.csv(empty_df, file.path(tab_dir, "tab_comparison.csv"), row.names = FALSE)
  writeLines("% No pairwise results available", file.path(tab_dir, "tab_comparison.tex"))
  quit(save = "no")
}

dat <- fromJSON(json_path, flatten = TRUE)
df <- as.data.frame(dat)

# Detect columns
es_col <- intersect(names(df), c("cliffs_delta", "cliff_delta", "effect_size", "cohens_d"))[1]
ci_lo <- intersect(names(df), c("ci_lower", "bca_ci_lower", "ci_lo"))[1]
ci_hi <- intersect(names(df), c("ci_upper", "bca_ci_upper", "ci_hi"))[1]
p_col <- intersect(names(df), c("p_value", "p_adj", "pvalue", "p"))[1]

if (is.na(es_col)) {
  cat("tab_comparison: no effect size column — empty table\n")
  writeLines("% No effect size column found", file.path(tab_dir, "tab_comparison.tex"))
  quit(save = "no")
}

# Build display table
out <- data.frame(
  Metric = df$metric,
  Pair = paste(df$condition_a, "vs", df$condition_b),
  Effect = sprintf("%.3f", df[[es_col]]),
  stringsAsFactors = FALSE
)

if (!is.na(ci_lo) && !is.na(ci_hi)) {
  out$CI <- sprintf("[%.3f, %.3f]", df[[ci_lo]], df[[ci_hi]])
} else {
  out$CI <- "---"
}

if (!is.na(p_col)) {
  out$p <- sprintf("%.4f", df[[p_col]])
  # Bold if significant (p < 0.05)
  sig <- df[[p_col]] < 0.05
  out$Effect[sig] <- paste0("\\textbf{", out$Effect[sig], "}")
} else {
  out$p <- "---"
}

write.csv(out, file.path(tab_dir, "tab_comparison.csv"), row.names = FALSE)

tex <- kable(out, format = "latex", booktabs = TRUE, escape = FALSE,
             caption = "Pairwise comparison of conditions") %>%
  kable_styling(latex_options = c("hold_position"), font_size = 7)
writeLines(tex, file.path(tab_dir, "tab_comparison.tex"))
cat("tab_comparison: done\n")
