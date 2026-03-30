#!/usr/bin/env Rscript
# fig_calibration.R — Reliability diagram: predicted vs observed per condition
# Input: calibration_data.json
# Output: figures/fig_calibration.pdf, figures/fig_calibration.png

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

make_placeholder <- function(msg) {
  p <- ggplot() +
    annotate("text", x = 0.5, y = 0.5, label = msg, size = 3, family = "sans") +
    theme_void() + theme(text = element_text(family = "sans", size = 7))
  ggsave(file.path(fig_dir, "fig_calibration.pdf"), p,
         width = fig_w, height = fig_h, device = cairo_pdf)
  png(file.path(fig_dir, "fig_calibration.png"), width = fig_w, height = fig_h,
      units = "in", res = dpi, type = "cairo")
  print(p)
  dev.off()
}

json_path <- file.path(r_data_dir, "calibration_data.json")
if (!file.exists(json_path)) {
  cat("fig_calibration: calibration_data.json not found — placeholder\n")
  make_placeholder("No calibration data available")
  quit(save = "no")
}

dat <- fromJSON(json_path, flatten = TRUE)
df <- as.data.frame(dat)

# Check for calibration-related columns
cal_cols <- grep("^cal_|predicted|observed|bin_mid", names(df), value = TRUE)
if (length(cal_cols) == 0 || nrow(df) == 0) {
  cat("fig_calibration: no calibration metrics found — placeholder\n")
  make_placeholder("No calibration metrics in data")
  quit(save = "no")
}

# Expect: condition, predicted (or bin_mid), observed (or fraction_positive)
pred_col <- intersect(names(df), c("predicted", "bin_mid", "mean_predicted"))[1]
obs_col <- intersect(names(df), c("observed", "fraction_positive", "mean_observed"))[1]

if (is.na(pred_col) || is.na(obs_col)) {
  cat("fig_calibration: missing predicted/observed columns — placeholder\n")
  make_placeholder("Calibration data format not recognized")
  quit(save = "no")
}

df$pred <- df[[pred_col]]
df$obs <- df[[obs_col]]

p <- ggplot(df, aes(x = pred, y = obs, color = condition)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed",
              color = "grey50", linewidth = 0.3) +
  geom_line(linewidth = 0.5) +
  geom_point(size = 1) +
  scale_color_viridis_d() +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(x = "Predicted probability", y = "Observed frequency",
       color = "Condition") +
  np_theme

ggsave(file.path(fig_dir, "fig_calibration.pdf"), p,
       width = fig_w, height = fig_h, device = cairo_pdf)
png(file.path(fig_dir, "fig_calibration.png"), width = fig_w, height = fig_h,
    units = "in", res = dpi, type = "cairo")
print(p)
dev.off()
cat("fig_calibration: done\n")
