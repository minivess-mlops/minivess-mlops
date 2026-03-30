#!/usr/bin/env Rscript
# tab_variance.R — ICC, Friedman stat/p, power caveat per metric
# Input: variance_decomposition.json
# Output: tables/tab_variance.tex, tables/tab_variance.csv

suppressPackageStartupMessages({
  library(jsonlite)
  library(knitr)
  library(kableExtra)
})

r_data_dir <- Sys.getenv("R_DATA_DIR", "/app/r_data")
r_output_dir <- Sys.getenv("R_OUTPUT_DIR", "/app/r_output")
tab_dir <- file.path(r_output_dir, "tables")
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)

json_path <- file.path(r_data_dir, "variance_decomposition.json")
if (!file.exists(json_path)) {
  cat("tab_variance: variance_decomposition.json not found — empty table\n")
  empty_df <- data.frame(Metric = character(), ICC = character(),
                         Friedman_stat = character(), Friedman_p = character())
  write.csv(empty_df, file.path(tab_dir, "tab_variance.csv"), row.names = FALSE)
  writeLines("% No variance decomposition data available",
             file.path(tab_dir, "tab_variance.tex"))
  quit(save = "no")
}

dat <- fromJSON(json_path, flatten = TRUE)
df <- as.data.frame(dat)

icc_col <- intersect(names(df), c("icc", "ICC", "icc_value"))[1]
fstat_col <- intersect(names(df), c("friedman_stat", "friedman_chi2", "chi2"))[1]
fp_col <- intersect(names(df), c("friedman_p", "friedman_pvalue", "p_value"))[1]
power_col <- intersect(names(df), c("power", "achieved_power", "power_caveat"))[1]

out <- data.frame(Metric = df$metric, stringsAsFactors = FALSE)

out$ICC <- if (!is.na(icc_col)) sprintf("%.4f", df[[icc_col]]) else "---"
out[["Friedman \u03c7\u00b2"]] <- if (!is.na(fstat_col)) sprintf("%.2f", df[[fstat_col]]) else "---"
out[["Friedman p"]] <- if (!is.na(fp_col)) sprintf("%.4f", df[[fp_col]]) else "---"

if (!is.na(power_col)) {
  if (is.numeric(df[[power_col]])) {
    out$Power <- sprintf("%.2f", df[[power_col]])
  } else {
    out$Power <- as.character(df[[power_col]])
  }
} else {
  out$Power <- "---"
}

write.csv(out, file.path(tab_dir, "tab_variance.csv"), row.names = FALSE)

tex <- kable(out, format = "latex", booktabs = TRUE, escape = FALSE,
             caption = "Variance decomposition: ICC and Friedman test per metric") %>%
  kable_styling(latex_options = c("hold_position"), font_size = 7) %>%
  footnote(general = "Power values are post-hoc estimates; interpret with caution.",
           general_title = "Note:", footnote_as_chunk = TRUE)
writeLines(tex, file.path(tab_dir, "tab_variance.tex"))
cat("tab_variance: done\n")
