#!/usr/bin/env Rscript
# fig_consort.R — CONSORT-style flow diagram using ggplot2 geom_rect + geom_segment
# Input: metadata.json
# Output: figures/fig_consort.pdf, figures/fig_consort.png

suppressPackageStartupMessages({
  library(ggplot2)
  library(jsonlite)
})

r_data_dir <- Sys.getenv("R_DATA_DIR", "/app/r_data")
r_output_dir <- Sys.getenv("R_OUTPUT_DIR", "/app/r_output")
fig_dir <- file.path(r_output_dir, "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# Nature Protocols single column: 89mm = 3.5in
fig_w <- 3.5
fig_h <- 5.0
dpi <- 300

np_theme <- theme_void() +
  theme(text = element_text(family = "sans", size = 7))

json_path <- file.path(r_data_dir, "metadata.json")
if (!file.exists(json_path)) {
  cat("fig_consort: metadata.json not found — creating placeholder\n")
  p <- ggplot() +
    annotate("text", x = 0.5, y = 0.5, label = "No metadata available",
             size = 3, family = "sans") + np_theme
  ggsave(file.path(fig_dir, "fig_consort.pdf"), p, width = fig_w, height = fig_h,
         device = cairo_pdf)
  png(file.path(fig_dir, "fig_consort.png"), width = fig_w, height = fig_h,
      units = "in", res = dpi, type = "cairo")
  print(p)
  dev.off()
  quit(save = "no")
}

meta <- fromJSON(json_path)

n_total <- meta$n_volumes %||% 70
n_folds <- meta$n_folds %||% 3
n_train <- meta$n_train %||% 47
n_val <- meta$n_val %||% 23
n_conditions <- meta$n_conditions %||% NA
n_test <- meta$n_test_volumes %||% NA

# Box positions: x center, y center, label
boxes <- data.frame(
  x = c(0.5, 0.5, 0.3, 0.7, 0.5, 0.5),
  y = c(0.9, 0.72, 0.54, 0.54, 0.36, 0.18),
  label = c(
    sprintf("Total volumes\nn = %d", n_total),
    sprintf("%d-fold CV split", n_folds),
    sprintf("Train\nn = %d", n_train),
    sprintf("Val\nn = %d", n_val),
    ifelse(is.na(n_conditions), "Per-fold analysis",
           sprintf("Per-fold analysis\n%d conditions", n_conditions)),
    ifelse(is.na(n_test), "Statistical analysis",
           sprintf("External test\nn = %d", n_test))
  ),
  stringsAsFactors = FALSE
)

bw <- 0.28  # box half-width
bh <- 0.055 # box half-height

# Arrows: from box index -> to box index
arrows <- data.frame(
  from = c(1, 2, 2, 3, 4, 5),
  to   = c(2, 3, 4, 5, 5, 6)
)

p <- ggplot() +
  # Boxes
  geom_rect(data = boxes,
            aes(xmin = x - bw, xmax = x + bw, ymin = y - bh, ymax = y + bh),
            fill = "white", color = "grey30", linewidth = 0.4) +
  # Labels
  geom_text(data = boxes, aes(x = x, y = y, label = label),
            size = 2.2, family = "sans", lineheight = 0.9) +
  # Arrows
  geom_segment(data = arrows,
               aes(x = boxes$x[from], y = boxes$y[from] - bh,
                   xend = boxes$x[to], yend = boxes$y[to] + bh),
               arrow = arrow(length = unit(1.5, "mm"), type = "closed"),
               linewidth = 0.3, color = "grey40") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0.08, 1)) +
  np_theme

ggsave(file.path(fig_dir, "fig_consort.pdf"), p, width = fig_w, height = fig_h,
       device = cairo_pdf)
png(file.path(fig_dir, "fig_consort.png"), width = fig_w, height = fig_h,
    units = "in", res = dpi, type = "cairo")
print(p)
dev.off()
cat("fig_consort: done\n")
