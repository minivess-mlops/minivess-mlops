#!/usr/bin/env Rscript
# fig_specification_curve.R — Specification curve analysis plot
# Input: specification_curve.json
# Output: figures/fig_specification_curve.pdf, figures/fig_specification_curve.png

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
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    legend.key.size = unit(3, "mm")
  )

json_path <- file.path(r_data_dir, "specification_curve.json")

make_placeholder <- function(msg) {
  p <- ggplot() +
    annotate("text", x = 0.5, y = 0.5, label = msg, size = 3, family = "sans") +
    theme_void() + theme(text = element_text(family = "sans", size = 7))
  ggsave(file.path(fig_dir, "fig_specification_curve.pdf"), p,
         width = fig_w, height = fig_h, device = cairo_pdf)
  png(file.path(fig_dir, "fig_specification_curve.png"), width = fig_w, height = fig_h,
      units = "in", res = dpi, type = "cairo")
  print(p)
  dev.off()
}

if (!file.exists(json_path)) {
  cat("fig_specification_curve: JSON not found — placeholder\n")
  make_placeholder("No specification curve data")
  quit(save = "no")
}

dat <- fromJSON(json_path, flatten = TRUE)
df <- as.data.frame(dat)

if (nrow(df) == 0) {
  cat("fig_specification_curve: empty data — placeholder\n")
  make_placeholder("No specification curve data")
  quit(save = "no")
}

# Expect: specification (or condition), estimate, ci_lower, ci_upper
est_col <- intersect(names(df), c("estimate", "effect", "effect_size"))[1]
ci_lo <- intersect(names(df), c("ci_lower", "ci_lo", "bca_ci_lower"))[1]
ci_hi <- intersect(names(df), c("ci_upper", "ci_hi", "bca_ci_upper"))[1]
spec_col <- intersect(names(df), c("specification", "condition", "spec"))[1]

if (any(is.na(c(est_col, ci_lo, ci_hi, spec_col)))) {
  cat("fig_specification_curve: missing required columns — placeholder\n")
  make_placeholder("Specification curve data format not recognized")
  quit(save = "no")
}

df$est <- df[[est_col]]
df$lo <- df[[ci_lo]]
df$hi <- df[[ci_hi]]
df$spec <- df[[spec_col]]

# Sort by estimate
df <- df[order(df$est), ]
df$rank <- seq_len(nrow(df))

p <- ggplot(df, aes(x = rank, y = est)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.3) +
  geom_pointrange(aes(ymin = lo, ymax = hi), size = 0.2, linewidth = 0.3,
                  color = viridis::viridis(1)) +
  labs(x = "Specification (sorted)", y = "Effect estimate (95% CI)") +
  np_theme

ggsave(file.path(fig_dir, "fig_specification_curve.pdf"), p,
       width = fig_w, height = fig_h, device = cairo_pdf)
png(file.path(fig_dir, "fig_specification_curve.png"), width = fig_w, height = fig_h,
    units = "in", res = dpi, type = "cairo")
print(p)
dev.off()
cat("fig_specification_curve: done\n")
