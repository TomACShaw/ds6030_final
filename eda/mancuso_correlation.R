# Correlation heatmap for Gina Mancuso's elastic-net project plan

library(tidyverse)
library(here)

data_path <- here("data", "processed", "earthquake.rds")

if (!file.exists(data_path)) {
  stop(
    "Missing data/processed/earthquake.rds. ",
    "Place the two CSVs in data/raw and run shared/01_data_setup.R first."
  )
}

earthquake <- readRDS(data_path)

# Correlation is appropriate for the original numeric/count predictors and
# binary indicators. IDs and letter-coded categorical variables are excluded:
# their numeric codes or factor levels do not represent meaningful distances.
# The redundant has_secondary_use umbrella flag is also excluded.
correlation_data <- earthquake |>
  select(
    count_floors_pre_eq,
    age,
    area_percentage,
    height_percentage,
    count_families,
    starts_with("has_superstructure_"),
    starts_with("has_secondary_use_")
  )

# Spearman correlation is less sensitive than Pearson correlation to the
# unusual age value of 995 and to skewed count/percentage variables.
correlation_matrix <- cor(
  correlation_data,
  method = "spearman",
  use = "pairwise.complete.obs"
)

# Place strongly related variables near one another to make the heatmap easier
# to read. This changes display order only, not the correlations.
cluster_order <- hclust(as.dist(1 - abs(correlation_matrix)))$order
ordered_names <- colnames(correlation_matrix)[cluster_order]
correlation_matrix <- correlation_matrix[ordered_names, ordered_names]

pretty_label <- function(x) {
  x |>
    str_replace("^has_superstructure_", "structure: ") |>
    str_replace("^has_secondary_use_", "secondary: ") |>
    str_replace_all("_", " ")
}

correlation_long <- as.data.frame(as.table(correlation_matrix)) |>
  as_tibble() |>
  rename(variable_x = Var1, variable_y = Var2, correlation = Freq) |>
  mutate(
    x_index = match(variable_x, ordered_names),
    y_index = match(variable_y, ordered_names),
    variable_x = factor(variable_x, levels = ordered_names),
    variable_y = factor(variable_y, levels = rev(ordered_names))
  ) |>
  # Retain one triangle and the diagonal rather than plotting duplicates.
  filter(x_index <= y_index) |>
  mutate(
    label = if_else(
      abs(correlation) >= 0.30 | abs(correlation - 1) < 1e-12,
      sprintf("%.2f", correlation),
      ""
    )
  )

label_map <- set_names(pretty_label(ordered_names), ordered_names)

correlation_plot <- ggplot(
  correlation_long,
  aes(x = variable_x, y = variable_y, fill = correlation)
) +
  geom_tile(color = "white", linewidth = 0.2) +
  geom_text(aes(label = label), size = 2.2) +
  scale_fill_gradient2(
    low = "#2166AC",
    mid = "white",
    high = "#B2182B",
    midpoint = 0,
    limits = c(-1, 1),
    name = "Spearman\ncorrelation"
  ) +
  scale_x_discrete(labels = label_map) +
  scale_y_discrete(labels = label_map) +
  coord_fixed() +
  labs(
    title = "Correlation Among Numeric and Binary Building Predictors",
    subtitle = "Spearman correlations; values shown where |correlation| is at least 0.30",
    x = NULL,
    y = NULL,
    caption = paste(
      "Geographic IDs and encoded categorical predictors are excluded because",
      "their codes do not represent numeric distances."
    )
  ) +
  theme_minimal(base_size = 10) +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 55, hjust = 1, size = 7),
    axis.text.y = element_text(size = 7),
    plot.title = element_text(face = "bold"),
    plot.caption = element_text(hjust = 0)
  )

dir.create(here("results", "figures"), recursive = TRUE, showWarnings = FALSE)
dir.create(here("results", "metrics"), recursive = TRUE, showWarnings = FALSE)

ggsave(
  here("results", "figures", "mancuso_predictor_correlation_matrix.png"),
  correlation_plot,
  width = 12,
  height = 10,
  dpi = 300,
  bg = "white"
)

# Save each unique pair for easier interpretation and report writing.
correlation_pairs <- as.data.frame(as.table(correlation_matrix)) |>
  as_tibble() |>
  rename(variable_1 = Var1, variable_2 = Var2, correlation = Freq) |>
  mutate(
    index_1 = match(variable_1, ordered_names),
    index_2 = match(variable_2, ordered_names)
  ) |>
  filter(index_1 < index_2) |>
  mutate(abs_correlation = abs(correlation)) |>
  arrange(desc(abs_correlation)) |>
  select(variable_1, variable_2, correlation, abs_correlation)

write_csv(
  correlation_pairs,
  here("results", "metrics", "mancuso_predictor_correlations.csv")
)

print(correlation_plot)

cat("\nStrongest absolute predictor correlations:\n")
print(slice_head(correlation_pairs, n = 10))
