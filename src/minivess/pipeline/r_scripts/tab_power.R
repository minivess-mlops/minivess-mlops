#!/usr/bin/env Rscript
# tab_power.R — Achieved power per effect size, recommended folds
# Input: diagnostics.json
# Output: tables/tab_power.tex, tables/tab_power.csv

suppressPackageStartupMessages({
  library(jsonlite)
  library(knitr)
  library(kableExtra)
})

r_data_dir <- Sys.getenv("R_DATA_DIR", "/app/r_data")
r_output_dir <- Sys.getenv("R_OUTPUT_DIR", "/app/r_output")
tab_dir <- file.path(r_output_dir, "tables")
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)

json_path <- file.path(r_data_dir, "diagnostics.json")
if (!file.exists(json_path)) {
  cat("tab_power: diagnostics.json not found — empty table\n")
  empty_df <- data.frame(Metric = character(), Effect_Size = character(),
                         Achieved_Power = character(), Recommended_Folds = character())
  write.csv(empty_df, file.path(tab_dir, "tab_power.csv"), row.names = FALSE)
  writeLines("% No diagnostics data available", file.path(tab_dir, "tab_power.tex"))
  quit(save = "no")
}

dat <- fromJSON(json_path, flatten = TRUE)
df <- as.data.frame(dat)

# Detect columns
es_col <- intersect(names(df), c("effect_size", "observed_effect", "cohens_d"))[1]
power_col <- intersect(names(df), c("achieved_power", "power", "post_hoc_power"))[1]
folds_col <- intersect(names(df), c("recommended_folds", "rec_folds", "n_folds_needed"))[1]
metric_col <- intersect(names(df), c("metric", "test_name"))[1]

out <- data.frame(stringsAsFactors = FALSE)
if (!is.na(metric_col)) out$Metric <- df[[metric_col]]
out[["Effect Size"]] <- if (!is.na(es_col)) sprintf("%.3f", df[[es_col]]) else "---"
out[["Achieved Power"]] <- if (!is.na(power_col)) sprintf("%.3f", df[[power_col]]) else "---"
out[["Rec. Folds"]] <- if (!is.na(folds_col)) as.character(df[[folds_col]]) else "---"

# Flag underpowered (power < 0.80)
if (!is.na(power_col)) {
  underpowered <- df[[power_col]] < 0.80
  out[["Achieved Power"]][underpowered] <- paste0(
    out[["Achieved Power"]][underpowered], "*")
}

write.csv(out, file.path(tab_dir, "tab_power.csv"), row.names = FALSE)

tex <- kable(out, format = "latex", booktabs = TRUE, escape = FALSE,
             caption = "Statistical power analysis per metric") %>%
  kable_styling(latex_options = c("hold_position"), font_size = 7) %>%
  footnote(symbol = "Underpowered (achieved power < 0.80)",
           footnote_as_chunk = TRUE)
writeLines(tex, file.path(tab_dir, "tab_power.tex"))
cat("tab_power: done\n")
