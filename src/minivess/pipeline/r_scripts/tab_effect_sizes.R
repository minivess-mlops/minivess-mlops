#!/usr/bin/env Rscript
# tab_effect_sizes.R — Cohen's d + Cliff's delta per pair
# Input: pairwise_results.json
# Output: tables/tab_effect_sizes.tex, tables/tab_effect_sizes.csv

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
  cat("tab_effect_sizes: pairwise_results.json not found — empty table\n")
  empty_df <- data.frame(Metric = character(), Pair = character(),
                         Cohens_d = character(), Cliffs_delta = character())
  write.csv(empty_df, file.path(tab_dir, "tab_effect_sizes.csv"), row.names = FALSE)
  writeLines("% No pairwise results available", file.path(tab_dir, "tab_effect_sizes.tex"))
  quit(save = "no")
}

dat <- fromJSON(json_path, flatten = TRUE)
df <- as.data.frame(dat)

cd_col <- intersect(names(df), c("cohens_d", "cohen_d"))[1]
cliff_col <- intersect(names(df), c("cliffs_delta", "cliff_delta"))[1]

out <- data.frame(
  Metric = df$metric,
  Pair = paste(df$condition_a, "vs", df$condition_b),
  stringsAsFactors = FALSE
)

if (!is.na(cd_col)) {
  out[["Cohen's d"]] <- sprintf("%.3f", df[[cd_col]])
} else {
  out[["Cohen's d"]] <- "---"
}

if (!is.na(cliff_col)) {
  out[["Cliff's \u03b4"]] <- sprintf("%.3f", df[[cliff_col]])
} else {
  out[["Cliff's \u03b4"]] <- "---"
}

# Interpret magnitude (Cliff's delta: small=0.147, medium=0.33, large=0.474)
if (!is.na(cliff_col)) {
  mag <- abs(df[[cliff_col]])
  out$Magnitude <- ifelse(mag >= 0.474, "Large",
                   ifelse(mag >= 0.33, "Medium",
                   ifelse(mag >= 0.147, "Small", "Negligible")))
} else {
  out$Magnitude <- "---"
}

write.csv(out, file.path(tab_dir, "tab_effect_sizes.csv"), row.names = FALSE)

tex <- kable(out, format = "latex", booktabs = TRUE, escape = FALSE,
             caption = "Effect sizes: Cohen's d and Cliff's delta per condition pair") %>%
  kable_styling(latex_options = c("hold_position"), font_size = 7)
writeLines(tex, file.path(tab_dir, "tab_effect_sizes.tex"))
cat("tab_effect_sizes: done\n")
