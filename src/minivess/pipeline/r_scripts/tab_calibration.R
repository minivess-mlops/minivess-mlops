#!/usr/bin/env Rscript
# tab_calibration.R — Mean ECE/Brier per condition
# Input: calibration_data.json
# Output: tables/tab_calibration.tex, tables/tab_calibration.csv

suppressPackageStartupMessages({
  library(jsonlite)
  library(knitr)
  library(kableExtra)
})

r_data_dir <- Sys.getenv("R_DATA_DIR", "/app/r_data")
r_output_dir <- Sys.getenv("R_OUTPUT_DIR", "/app/r_output")
tab_dir <- file.path(r_output_dir, "tables")
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)

json_path <- file.path(r_data_dir, "calibration_data.json")

make_empty <- function(note) {
  empty_df <- data.frame(Condition = character(), ECE = character(),
                         Brier = character(), Note = note)
  write.csv(empty_df, file.path(tab_dir, "tab_calibration.csv"), row.names = FALSE)
  tex <- kable(empty_df, format = "latex", booktabs = TRUE, escape = FALSE,
               caption = "Calibration metrics per condition") %>%
    kable_styling(latex_options = c("hold_position"), font_size = 7) %>%
    footnote(general = note, general_title = "Note:", footnote_as_chunk = TRUE)
  writeLines(tex, file.path(tab_dir, "tab_calibration.tex"))
}

if (!file.exists(json_path)) {
  cat("tab_calibration: calibration_data.json not found — empty table\n")
  make_empty("No calibration data available.")
  quit(save = "no")
}

dat <- fromJSON(json_path, flatten = TRUE)
df <- as.data.frame(dat)

# Check for calibration-specific columns
ece_col <- intersect(names(df), c("ece", "ECE", "mean_ece", "cal_ece"))[1]
brier_col <- intersect(names(df), c("brier", "brier_score", "mean_brier", "cal_brier"))[1]

if (is.na(ece_col) && is.na(brier_col)) {
  cat("tab_calibration: no ECE/Brier columns — empty table with note\n")
  make_empty("No calibration metrics (ECE, Brier) found in data.")
  quit(save = "no")
}

# Aggregate per condition if needed
cond_col <- intersect(names(df), c("condition", "model", "name"))[1]
if (is.na(cond_col)) {
  cat("tab_calibration: no condition column — using row indices\n")
  df$condition <- paste0("Row_", seq_len(nrow(df)))
  cond_col <- "condition"
}

out <- data.frame(Condition = df[[cond_col]], stringsAsFactors = FALSE)
out$ECE <- if (!is.na(ece_col)) sprintf("%.4f", df[[ece_col]]) else "---"
out$Brier <- if (!is.na(brier_col)) sprintf("%.4f", df[[brier_col]]) else "---"

write.csv(out, file.path(tab_dir, "tab_calibration.csv"), row.names = FALSE)

tex <- kable(out, format = "latex", booktabs = TRUE, escape = FALSE,
             caption = "Calibration metrics per condition") %>%
  kable_styling(latex_options = c("hold_position"), font_size = 7)
writeLines(tex, file.path(tab_dir, "tab_calibration.tex"))
cat("tab_calibration: done\n")
