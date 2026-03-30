#!/usr/bin/env Rscript
# fig_distribution.R — Raincloud/violin + jitter per condition per metric
# Input: per_volume_data.json
# Output: figures/fig_distribution.pdf, figures/fig_distribution.png

suppressPackageStartupMessages({
  library(ggplot2)
  library(jsonlite)
})

r_data_dir <- Sys.getenv("R_DATA_DIR", "/app/r_data")
r_output_dir <- Sys.getenv("R_OUTPUT_DIR", "/app/r_output")
fig_dir <- file.path(r_output_dir, "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# Nature Protocols double column: 183mm = 7.2in
fig_w <- 7.2
fig_h <- 5.0
dpi <- 300

np_theme <- theme_minimal() +
  theme(
    text = element_text(family = "sans", size = 7),
    strip.text = element_text(size = 7, face = "bold"),
    legend.position = "bottom",
    legend.key.size = unit(3, "mm"),
    panel.grid.minor = element_blank()
  )

json_path <- file.path(r_data_dir, "per_volume_data.json")
if (!file.exists(json_path)) {
  cat("fig_distribution: per_volume_data.json not found — creating placeholder\n")
  p <- ggplot() +
    annotate("text", x = 0.5, y = 0.5, label = "No per-volume data available",
             size = 3, family = "sans") +
    np_theme + theme(axis.text = element_blank(), axis.title = element_blank())
  ggsave(file.path(fig_dir, "fig_distribution.pdf"), p, width = fig_w, height = fig_h,
         device = cairo_pdf)
  png(file.path(fig_dir, "fig_distribution.png"), width = fig_w, height = fig_h,
      units = "in", res = dpi, type = "cairo")
  print(p)
  dev.off()
  quit(save = "no")
}

dat <- fromJSON(json_path, flatten = TRUE)
df <- as.data.frame(dat)

# Detect metric columns (everything except condition/volume identifiers)
id_cols <- c("condition", "volume_id", "fold", "split", "loss_function", "with_aux_calib")
metric_cols <- setdiff(names(df), id_cols)
metric_cols <- metric_cols[sapply(df[metric_cols], is.numeric)]

if (length(metric_cols) == 0) {
  cat("fig_distribution: no numeric metric columns found — skipping\n")
  quit(save = "no")
}

# Reshape to long format
df_long <- do.call(rbind, lapply(metric_cols, function(m) {
  data.frame(condition = df$condition, metric = m, value = df[[m]],
             stringsAsFactors = FALSE)
}))

# Try ggrain for raincloud, fallback to violin + jitter
has_ggrain <- requireNamespace("ggrain", quietly = TRUE)

if (has_ggrain) {
  suppressPackageStartupMessages(library(ggrain))
  p <- ggplot(df_long, aes(x = condition, y = value, fill = condition)) +
    geom_rain(alpha = 0.4) +
    facet_wrap(~metric, scales = "free_y") +
    scale_fill_viridis_d() +
    labs(x = "Condition", y = "Value") +
    np_theme + theme(axis.text.x = element_text(angle = 45, hjust = 1))
} else {
  p <- ggplot(df_long, aes(x = condition, y = value, fill = condition)) +
    geom_violin(alpha = 0.5, scale = "width") +
    geom_jitter(width = 0.15, size = 0.6, alpha = 0.6) +
    facet_wrap(~metric, scales = "free_y") +
    scale_fill_viridis_d() +
    labs(x = "Condition", y = "Value") +
    np_theme + theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

ggsave(file.path(fig_dir, "fig_distribution.pdf"), p, width = fig_w, height = fig_h,
       device = cairo_pdf)
png(file.path(fig_dir, "fig_distribution.png"), width = fig_w, height = fig_h,
    units = "in", res = dpi, type = "cairo")
print(p)
dev.off()
cat("fig_distribution: done\n")
