#!/usr/bin/env Rscript
# fig_variance_lollipop.R — Lollipop chart: ICC per metric
# Input: variance_decomposition.json
# Output: figures/fig_variance_lollipop.pdf, figures/fig_variance_lollipop.png

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
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank()
  )

json_path <- file.path(r_data_dir, "variance_decomposition.json")
if (!file.exists(json_path)) {
  cat("fig_variance_lollipop: variance_decomposition.json not found — placeholder\n")
  p <- ggplot() +
    annotate("text", x = 0.5, y = 0.5, label = "No variance decomposition data",
             size = 3, family = "sans") +
    theme_void() + theme(text = element_text(family = "sans", size = 7))
  ggsave(file.path(fig_dir, "fig_variance_lollipop.pdf"), p,
         width = fig_w, height = fig_h, device = cairo_pdf)
  png(file.path(fig_dir, "fig_variance_lollipop.png"), width = fig_w, height = fig_h,
      units = "in", res = dpi, type = "cairo")
  print(p)
  dev.off()
  quit(save = "no")
}

dat <- fromJSON(json_path, flatten = TRUE)
df <- as.data.frame(dat)

# Expect: metric, icc (intraclass correlation coefficient)
icc_col <- intersect(names(df), c("icc", "ICC", "icc_value"))[1]
if (is.na(icc_col)) {
  cat("fig_variance_lollipop: no ICC column found — skipping\n")
  quit(save = "no")
}

df$icc_val <- df[[icc_col]]
df <- df[order(df$icc_val), ]

p <- ggplot(df, aes(x = icc_val, y = reorder(metric, icc_val))) +
  geom_segment(aes(xend = 0, yend = reorder(metric, icc_val)),
               linewidth = 0.4, color = "grey60") +
  geom_point(size = 2, color = viridis::viridis(1)) +
  geom_text(aes(label = sprintf("%.3f", icc_val)), hjust = -0.3,
            size = 2, family = "sans") +
  scale_x_continuous(limits = c(0, max(df$icc_val) * 1.15)) +
  labs(x = "ICC", y = NULL,
       title = "Variance Decomposition: ICC per Metric") +
  np_theme + theme(plot.title = element_text(size = 7, face = "bold"))

ggsave(file.path(fig_dir, "fig_variance_lollipop.pdf"), p,
       width = fig_w, height = fig_h, device = cairo_pdf)
png(file.path(fig_dir, "fig_variance_lollipop.png"), width = fig_w, height = fig_h,
    units = "in", res = dpi, type = "cairo")
print(p)
dev.off()
cat("fig_variance_lollipop: done\n")
