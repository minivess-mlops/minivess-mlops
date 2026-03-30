#!/usr/bin/env Rscript
# fig_generalization_gap.R — Train/val vs test (DeepVess) distribution comparison
# Input: per_volume_data.json
# Output: figures/fig_generalization_gap.pdf, figures/fig_generalization_gap.png

suppressPackageStartupMessages({
  library(ggplot2)
  library(jsonlite)
})

r_data_dir <- Sys.getenv("R_DATA_DIR", "/app/r_data")
r_output_dir <- Sys.getenv("R_OUTPUT_DIR", "/app/r_output")
fig_dir <- file.path(r_output_dir, "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# Nature Protocols double column
fig_w <- 7.2
fig_h <- 4.5
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
  cat("fig_generalization_gap: per_volume_data.json not found — placeholder\n")
  p <- ggplot() +
    annotate("text", x = 0.5, y = 0.5, label = "No per-volume data available",
             size = 3, family = "sans") +
    theme_void() + theme(text = element_text(family = "sans", size = 7))
  ggsave(file.path(fig_dir, "fig_generalization_gap.pdf"), p,
         width = fig_w, height = fig_h, device = cairo_pdf)
  png(file.path(fig_dir, "fig_generalization_gap.png"), width = fig_w, height = fig_h,
      units = "in", res = dpi, type = "cairo")
  print(p)
  dev.off()
  quit(save = "no")
}

df <- as.data.frame(fromJSON(json_path, flatten = TRUE))

# Need a split column distinguishing trainval vs test
if (!"split" %in% names(df)) {
  cat("fig_generalization_gap: no 'split' column — skipping\n")
  quit(save = "no")
}

id_cols <- c("condition", "volume_id", "fold", "split", "loss_function", "with_aux_calib")
metric_cols <- setdiff(names(df), id_cols)
metric_cols <- metric_cols[sapply(df[metric_cols], is.numeric)]

if (length(metric_cols) == 0) {
  cat("fig_generalization_gap: no numeric metrics — skipping\n")
  quit(save = "no")
}

# Classify split into trainval vs test
df$split_group <- ifelse(grepl("test|deepvess", df$split, ignore.case = TRUE),
                         "Test (DeepVess)", "Train/Val")

df_long <- do.call(rbind, lapply(metric_cols, function(m) {
  data.frame(split_group = df$split_group, condition = df$condition,
             metric = m, value = df[[m]], stringsAsFactors = FALSE)
}))

p <- ggplot(df_long, aes(x = split_group, y = value, fill = split_group)) +
  geom_boxplot(alpha = 0.6, outlier.size = 0.5, linewidth = 0.3) +
  geom_jitter(width = 0.15, size = 0.4, alpha = 0.5) +
  facet_wrap(~metric, scales = "free_y") +
  scale_fill_viridis_d(begin = 0.3, end = 0.8) +
  labs(x = NULL, y = "Value", fill = "Split") +
  np_theme

ggsave(file.path(fig_dir, "fig_generalization_gap.pdf"), p,
       width = fig_w, height = fig_h, device = cairo_pdf)
png(file.path(fig_dir, "fig_generalization_gap.png"), width = fig_w, height = fig_h,
    units = "in", res = dpi, type = "cairo")
print(p)
dev.off()
cat("fig_generalization_gap: done\n")
