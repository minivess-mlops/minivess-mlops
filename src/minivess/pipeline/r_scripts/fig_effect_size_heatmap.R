#!/usr/bin/env Rscript
# fig_effect_size_heatmap.R — Heatmap of Cliff's delta across condition pairs
# Input: pairwise_results.json
# Output: figures/fig_effect_size_heatmap.pdf, figures/fig_effect_size_heatmap.png

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
fig_h <- 5.5
dpi <- 300

np_theme <- theme_minimal() +
  theme(
    text = element_text(family = "sans", size = 7),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.key.size = unit(3, "mm"),
    panel.grid = element_blank()
  )

json_path <- file.path(r_data_dir, "pairwise_results.json")
if (!file.exists(json_path)) {
  cat("fig_effect_size_heatmap: pairwise_results.json not found — placeholder\n")
  p <- ggplot() +
    annotate("text", x = 0.5, y = 0.5, label = "No pairwise results available",
             size = 3, family = "sans") +
    theme_void() + theme(text = element_text(family = "sans", size = 7))
  ggsave(file.path(fig_dir, "fig_effect_size_heatmap.pdf"), p,
         width = fig_w, height = fig_h, device = cairo_pdf)
  png(file.path(fig_dir, "fig_effect_size_heatmap.png"), width = fig_w, height = fig_h,
      units = "in", res = dpi, type = "cairo")
  print(p)
  dev.off()
  quit(save = "no")
}

dat <- fromJSON(json_path, flatten = TRUE)
df <- as.data.frame(dat)

# Expect columns: metric, condition_a, condition_b, cliffs_delta (or effect_size)
delta_col <- intersect(names(df), c("cliffs_delta", "cliff_delta", "effect_size"))[1]
if (is.na(delta_col)) {
  cat("fig_effect_size_heatmap: no effect size column found — skipping\n")
  quit(save = "no")
}

# Build heatmap per metric
df$effect <- df[[delta_col]]

p <- ggplot(df, aes(x = condition_a, y = condition_b, fill = effect)) +
  geom_tile(color = "white", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%.2f", effect)), size = 2, family = "sans") +
  facet_wrap(~metric, scales = "free") +
  scale_fill_viridis_c(name = "Cliff's \u03b4", option = "viridis") +
  labs(x = NULL, y = NULL) +
  np_theme

ggsave(file.path(fig_dir, "fig_effect_size_heatmap.pdf"), p,
       width = fig_w, height = fig_h, device = cairo_pdf)
png(file.path(fig_dir, "fig_effect_size_heatmap.png"), width = fig_w, height = fig_h,
    units = "in", res = dpi, type = "cairo")
print(p)
dev.off()
cat("fig_effect_size_heatmap: done\n")
