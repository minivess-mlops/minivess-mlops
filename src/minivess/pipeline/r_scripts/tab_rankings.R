#!/usr/bin/env Rscript
# tab_rankings.R — Mean ranks + CD values per condition
# Input: rankings.json
# Output: tables/tab_rankings.tex, tables/tab_rankings.csv

suppressPackageStartupMessages({
  library(jsonlite)
  library(knitr)
  library(kableExtra)
})

r_data_dir <- Sys.getenv("R_DATA_DIR", "/app/r_data")
r_output_dir <- Sys.getenv("R_OUTPUT_DIR", "/app/r_output")
tab_dir <- file.path(r_output_dir, "tables")
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)

json_path <- file.path(r_data_dir, "rankings.json")
if (!file.exists(json_path)) {
  cat("tab_rankings: rankings.json not found — empty table\n")
  empty_df <- data.frame(Condition = character(), Mean_Rank = character(),
                         CD = character())
  write.csv(empty_df, file.path(tab_dir, "tab_rankings.csv"), row.names = FALSE)
  writeLines("% No ranking data available", file.path(tab_dir, "tab_rankings.tex"))
  quit(save = "no")
}

dat <- fromJSON(json_path, flatten = TRUE)
df <- as.data.frame(dat)

rank_col <- intersect(names(df), c("mean_rank", "avg_rank", "rank"))[1]
cd_col <- intersect(names(df), c("cd_value", "cd", "critical_difference"))[1]

out <- data.frame(Condition = df$condition, stringsAsFactors = FALSE)

if (!is.na(rank_col)) {
  out[["Mean Rank"]] <- sprintf("%.2f", df[[rank_col]])
} else {
  out[["Mean Rank"]] <- "---"
}

if (!is.na(cd_col)) {
  out$CD <- sprintf("%.3f", df[[cd_col]])
} else {
  out$CD <- "---"
}

# Sort by mean rank
if (!is.na(rank_col)) {
  out <- out[order(df[[rank_col]]), ]
}

# Add rank position
out <- cbind(data.frame(Pos = seq_len(nrow(out))), out)

write.csv(out, file.path(tab_dir, "tab_rankings.csv"), row.names = FALSE)

tex <- kable(out, format = "latex", booktabs = TRUE, escape = FALSE,
             row.names = FALSE,
             caption = "Condition rankings (Friedman/Nemenyi)") %>%
  kable_styling(latex_options = c("hold_position"), font_size = 7)
writeLines(tex, file.path(tab_dir, "tab_rankings.tex"))
cat("tab_rankings: done\n")
