#!/usr/bin/env Rscript
# fig_interaction.R — Interaction plot: metric mean +/- SE by loss_function x with_aux_calib
# Input: anova_results.json + per_volume_data.json
# Output: figures/fig_interaction.pdf, figures/fig_interaction.png

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
fig_h <- 3.5
dpi <- 300

np_theme <- theme_minimal() +
  theme(
    text = element_text(family = "sans", size = 7),
    legend.position = "bottom",
    legend.key.size = unit(3, "mm"),
    panel.grid.minor = element_blank()
  )

pvd_path <- file.path(r_data_dir, "per_volume_data.json")
if (!file.exists(pvd_path)) {
  cat("fig_interaction: per_volume_data.json not found — placeholder\n")
  p <- ggplot() +
    annotate("text", x = 0.5, y = 0.5, label = "No per-volume data available",
             size = 3, family = "sans") +
    theme_void() + theme(text = element_text(family = "sans", size = 7))
  ggsave(file.path(fig_dir, "fig_interaction.pdf"), p,
         width = fig_w, height = fig_h, device = cairo_pdf)
  png(file.path(fig_dir, "fig_interaction.png"), width = fig_w, height = fig_h,
      units = "in", res = dpi, type = "cairo")
  print(p)
  dev.off()
  quit(save = "no")
}

df <- as.data.frame(fromJSON(pvd_path, flatten = TRUE))

# Need loss_function and with_aux_calib columns
if (!all(c("loss_function", "with_aux_calib") %in% names(df))) {
  cat("fig_interaction: missing loss_function or with_aux_calib columns — skipping\n")
  quit(save = "no")
}

id_cols <- c("condition", "volume_id", "fold", "split", "loss_function", "with_aux_calib")
metric_cols <- setdiff(names(df), id_cols)
metric_cols <- metric_cols[sapply(df[metric_cols], is.numeric)]

if (length(metric_cols) == 0) {
  cat("fig_interaction: no numeric metric columns — skipping\n")
  quit(save = "no")
}

# Use the first metric for interaction plot (or facet if few)
target_metrics <- head(metric_cols, 4)

df_long <- do.call(rbind, lapply(target_metrics, function(m) {
  data.frame(loss_function = df$loss_function,
             with_aux_calib = as.factor(df$with_aux_calib),
             metric = m, value = df[[m]], stringsAsFactors = FALSE)
}))

# Compute mean +/- SE per group
agg <- aggregate(value ~ loss_function + with_aux_calib + metric, data = df_long,
                 FUN = function(x) c(mean = mean(x), se = sd(x) / sqrt(length(x))))
agg <- do.call(data.frame, agg)

p <- ggplot(agg, aes(x = loss_function, y = value.mean,
                      color = with_aux_calib, group = with_aux_calib)) +
  geom_point(size = 1.5, position = position_dodge(width = 0.3)) +
  geom_line(position = position_dodge(width = 0.3), linewidth = 0.4) +
  geom_errorbar(aes(ymin = value.mean - value.se, ymax = value.mean + value.se),
                width = 0.15, linewidth = 0.3, position = position_dodge(width = 0.3)) +
  facet_wrap(~metric, scales = "free_y") +
  scale_color_viridis_d(name = "Aux Calib") +
  labs(x = "Loss Function", y = "Mean \u00b1 SE") +
  np_theme + theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(fig_dir, "fig_interaction.pdf"), p,
       width = fig_w, height = fig_h, device = cairo_pdf)
png(file.path(fig_dir, "fig_interaction.png"), width = fig_w, height = fig_h,
    units = "in", res = dpi, type = "cairo")
print(p)
dev.off()
cat("fig_interaction: done\n")
