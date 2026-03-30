#!/usr/bin/env Rscript
# fig_cd_diagram.R — Critical difference diagram: horizontal rank axis + CD bar
# Input: rankings.json
# Output: figures/fig_cd_diagram.pdf, figures/fig_cd_diagram.png

suppressPackageStartupMessages({
  library(ggplot2)
  library(jsonlite)
})

r_data_dir <- Sys.getenv("R_DATA_DIR", "/app/r_data")
r_output_dir <- Sys.getenv("R_OUTPUT_DIR", "/app/r_output")
fig_dir <- file.path(r_output_dir, "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# Nature Protocols single column
fig_w <- 3.5
fig_h <- 3.0
dpi <- 300

np_theme <- theme_minimal() +
  theme(
    text = element_text(family = "sans", size = 7),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank()
  )

json_path <- file.path(r_data_dir, "rankings.json")
if (!file.exists(json_path)) {
  cat("fig_cd_diagram: rankings.json not found — placeholder\n")
  p <- ggplot() +
    annotate("text", x = 0.5, y = 0.5, label = "No ranking data available",
             size = 3, family = "sans") +
    theme_void() + theme(text = element_text(family = "sans", size = 7))
  ggsave(file.path(fig_dir, "fig_cd_diagram.pdf"), p,
         width = fig_w, height = fig_h, device = cairo_pdf)
  png(file.path(fig_dir, "fig_cd_diagram.png"), width = fig_w, height = fig_h,
      units = "in", res = dpi, type = "cairo")
  print(p)
  dev.off()
  quit(save = "no")
}

dat <- fromJSON(json_path, flatten = TRUE)
df <- as.data.frame(dat)

# Expect: condition, mean_rank, cd_value (critical difference)
rank_col <- intersect(names(df), c("mean_rank", "avg_rank", "rank"))[1]
cd_col <- intersect(names(df), c("cd_value", "cd", "critical_difference"))[1]

if (is.na(rank_col)) {
  cat("fig_cd_diagram: no rank column found — skipping\n")
  quit(save = "no")
}

df$mean_rank <- df[[rank_col]]
df <- df[order(df$mean_rank), ]
df$y_pos <- seq_len(nrow(df))

# Get CD value (same for all rows or from first)
cd_val <- if (!is.na(cd_col)) df[[cd_col]][1] else NA

p <- ggplot(df, aes(x = mean_rank, y = reorder(condition, -mean_rank))) +
  geom_segment(aes(xend = min(mean_rank) - 0.2, yend = reorder(condition, -mean_rank)),
               linetype = "dotted", color = "grey70", linewidth = 0.2) +
  geom_point(size = 2, color = viridis::viridis(1)) +
  geom_text(aes(label = sprintf("%.2f", mean_rank)), hjust = -0.3,
            size = 2, family = "sans") +
  labs(x = "Mean Rank", y = NULL) +
  np_theme

# Add CD bar if available
if (!is.na(cd_val)) {
  best_rank <- min(df$mean_rank)
  p <- p + annotate("segment", x = best_rank, xend = best_rank + cd_val,
                     y = nrow(df) + 0.5, yend = nrow(df) + 0.5,
                     linewidth = 0.8, color = "red3") +
    annotate("text", x = best_rank + cd_val / 2, y = nrow(df) + 0.8,
             label = sprintf("CD = %.2f", cd_val), size = 2, family = "sans",
             color = "red3")
}

ggsave(file.path(fig_dir, "fig_cd_diagram.pdf"), p,
       width = fig_w, height = fig_h, device = cairo_pdf)
png(file.path(fig_dir, "fig_cd_diagram.png"), width = fig_w, height = fig_h,
    units = "in", res = dpi, type = "cairo")
print(p)
dev.off()
cat("fig_cd_diagram: done\n")
