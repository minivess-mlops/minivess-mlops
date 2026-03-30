#!/usr/bin/env Rscript
# fig_forest_plot.R — Forest plot: point + CI whiskers (geom_pointrange)
# Input: pairwise_results.json
# Output: figures/fig_forest_plot.pdf, figures/fig_forest_plot.png

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
fig_h <- 5.0
dpi <- 300

np_theme <- theme_minimal() +
  theme(
    text = element_text(family = "sans", size = 7),
    legend.position = "bottom",
    legend.key.size = unit(3, "mm"),
    panel.grid.minor = element_blank()
  )

json_path <- file.path(r_data_dir, "pairwise_results.json")
if (!file.exists(json_path)) {
  cat("fig_forest_plot: pairwise_results.json not found — placeholder\n")
  p <- ggplot() +
    annotate("text", x = 0.5, y = 0.5, label = "No pairwise results available",
             size = 3, family = "sans") +
    theme_void() + theme(text = element_text(family = "sans", size = 7))
  ggsave(file.path(fig_dir, "fig_forest_plot.pdf"), p,
         width = fig_w, height = fig_h, device = cairo_pdf)
  png(file.path(fig_dir, "fig_forest_plot.png"), width = fig_w, height = fig_h,
      units = "in", res = dpi, type = "cairo")
  print(p)
  dev.off()
  quit(save = "no")
}

dat <- fromJSON(json_path, flatten = TRUE)
df <- as.data.frame(dat)

# Detect effect size and CI columns
es_col <- intersect(names(df), c("cliffs_delta", "cliff_delta", "effect_size", "cohens_d"))[1]
ci_lo_col <- intersect(names(df), c("ci_lower", "bca_ci_lower", "ci_lo"))[1]
ci_hi_col <- intersect(names(df), c("ci_upper", "bca_ci_upper", "ci_hi"))[1]

if (is.na(es_col) || is.na(ci_lo_col) || is.na(ci_hi_col)) {
  cat("fig_forest_plot: missing effect size or CI columns — skipping\n")
  quit(save = "no")
}

df$effect <- df[[es_col]]
df$ci_lo <- df[[ci_lo_col]]
df$ci_hi <- df[[ci_hi_col]]
df$pair_label <- paste(df$condition_a, "vs", df$condition_b)

p <- ggplot(df, aes(x = effect, y = pair_label, color = metric)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.3) +
  geom_pointrange(aes(xmin = ci_lo, xmax = ci_hi), size = 0.3, linewidth = 0.4) +
  facet_wrap(~metric, scales = "free_x", ncol = 1) +
  scale_color_viridis_d() +
  labs(x = "Effect size (BCa 95% CI)", y = NULL) +
  np_theme + theme(legend.position = "none")

ggsave(file.path(fig_dir, "fig_forest_plot.pdf"), p,
       width = fig_w, height = fig_h, device = cairo_pdf)
png(file.path(fig_dir, "fig_forest_plot.png"), width = fig_w, height = fig_h,
    units = "in", res = dpi, type = "cairo")
print(p)
dev.off()
cat("fig_forest_plot: done\n")
