# Gina Mancuso: final training-only geographic elastic-net candidate
#
# Selected training configuration:
#   - moderate square-root inverse-frequency case weights
#   - fold-safe hierarchical level-2 and level-3 likelihood encoding
#   - smoothing strength 10
#   - penalty 0.003 and mixture 0.5
#
# The script reproduces five-fold out-of-fold diagnostics and saves a complete
# full-training model bundle for future prediction. The shared holdout is not
# accessed unless evaluate_holdout is deliberately changed to TRUE after group
# model selection is complete.

library(tidyverse)
library(tidymodels)
library(here)

source(here("models", "mancuso_geography_helpers.R"))

set.seed(42)


# --- LOCKED CONFIGURATION ----------------------------------------------------

evaluate_holdout <- FALSE

candidate_parameters <- list(
  weight_power = 0.5,
  smoothing_strength = 10,
  inner_encoding_folds = 5,
  penalty = 0.003,
  mixture = 0.5,
  seed = 42
)


# --- LOAD SHARED OBJECTS WITHOUT EXPOSING THE HOLDOUT -----------------------

split_path <- here("data", "processed", "split_data.RData")

if (!file.exists(split_path)) {
  stop(
    "The shared split file does not exist. Run shared/01_data_setup.R and ",
    "shared/02_preprocessing.R before this candidate."
  )
}

shared_objects <- new.env(parent = emptyenv())
load(split_path, envir = shared_objects)

required_objects <- c("folds", "train_data")
if (evaluate_holdout) {
  required_objects <- c(required_objects, "test_data")
}

missing_objects <- required_objects[!vapply(
  required_objects,
  exists,
  logical(1),
  envir = shared_objects,
  inherits = FALSE
)]

if (length(missing_objects) > 0) {
  stop(
    "The shared split file is missing required objects: ",
    paste(missing_objects, collapse = ", ")
  )
}

folds <- shared_objects$folds
train_data <- shared_objects$train_data

if (evaluate_holdout) {
  holdout_data <- shared_objects$test_data
}

rm(shared_objects)

message("Executing the locked geographic elastic-net candidate.")
message(sprintf(
  "Training rows: %s; outer CV folds: %s; holdout evaluation: %s",
  format(nrow(train_data), big.mark = ","),
  nrow(folds),
  evaluate_holdout
))


# --- MODEL SPECIFICATION -----------------------------------------------------

candidate_spec <- multinom_reg(
  penalty = candidate_parameters$penalty,
  mixture = candidate_parameters$mixture
) |>
  set_engine("glmnet", standardize = FALSE) |>
  set_mode("classification")


fit_candidate_workflow <- function(weighted_analysis_data) {
  workflow() |>
    add_recipe(make_geographic_recipe(weighted_analysis_data)) |>
    add_model(candidate_spec) |>
    add_case_weights(class_weights) |>
    fit(data = weighted_analysis_data)
}


# --- LEAKAGE-SAFE OUT-OF-FOLD DIAGNOSTICS -----------------------------------

oof_predictions <- map2_dfr(
  folds$splits,
  seq_len(nrow(folds)),
  function(split_object, fold_number) {
    fold_id <- folds$id[[fold_number]]
    message("  Fitting ", fold_id, " of ", nrow(folds))

    outer_analysis <- analysis(split_object)
    outer_assessment <- assessment(split_object)

    encoded_analysis <- crossfit_geography(
      analysis_data = outer_analysis,
      prior_n = candidate_parameters$smoothing_strength,
      v = candidate_parameters$inner_encoding_folds,
      seed = 4200 + fold_number
    )

    outer_maps <- build_geography_maps(
      reference_data = outer_analysis,
      prior_n = candidate_parameters$smoothing_strength
    )
    encoded_assessment <- apply_geography_maps(
      new_data = outer_assessment,
      maps = outer_maps
    )

    weighted_analysis <- add_power_weights(
      data = encoded_analysis,
      reference_data = encoded_analysis,
      power = candidate_parameters$weight_power
    )

    fold_fit <- fit_candidate_workflow(weighted_analysis)

    class_predictions <- predict(
      fold_fit,
      new_data = encoded_assessment,
      type = "class"
    )
    probability_predictions <- predict(
      fold_fit,
      new_data = encoded_assessment,
      type = "prob"
    )

    bind_cols(
      encoded_assessment |>
        select(building_id, damage_grade),
      class_predictions,
      probability_predictions
    ) |>
      mutate(fold = fold_id, .before = 1)
  }
)

oof_overall <- geographic_overall_metrics(oof_predictions)
oof_per_class <- geographic_class_metrics(oof_predictions)

oof_fold_metrics <- oof_predictions |>
  group_by(fold) |>
  group_modify(
    ~ geographic_overall_metrics(.x) |>
      select(.metric, .estimator, .estimate)
  ) |>
  ungroup()

oof_fold_class_metrics <- oof_predictions |>
  group_by(fold) |>
  group_modify(~ geographic_class_metrics(.x)) |>
  ungroup()

oof_confusion <- conf_mat(
  oof_predictions,
  truth = damage_grade,
  estimate = .pred_class
)

oof_confusion_long <- as.data.frame(oof_confusion$table) |>
  as_tibble() |>
  group_by(Truth) |>
  mutate(row_percent = 100 * Freq / sum(Freq)) |>
  ungroup()


# --- FULL-TRAINING FIT FOR FUTURE PREDICTION --------------------------------

# Cross-fitted encodings prevent each training building's own outcome from
# contributing to its model input. Separate maps built from all training rows
# are saved only for encoding genuinely new data later.
message("Building the full-training candidate bundle.")

full_training_encoded <- crossfit_geography(
  analysis_data = train_data,
  prior_n = candidate_parameters$smoothing_strength,
  v = candidate_parameters$inner_encoding_folds,
  seed = 9900 + candidate_parameters$seed
)

full_geography_maps <- build_geography_maps(
  reference_data = train_data,
  prior_n = candidate_parameters$smoothing_strength
)

full_training_weighted <- add_power_weights(
  data = full_training_encoded,
  reference_data = full_training_encoded,
  power = candidate_parameters$weight_power
)

full_training_fit <- fit_candidate_workflow(full_training_weighted)

class_weight_summary <- full_training_weighted |>
  as_tibble() |>
  group_by(damage_grade) |>
  summarise(
    observations = n(),
    observed_percent = 100 * n() / nrow(full_training_weighted),
    case_weight = as.numeric(first(class_weights)),
    weighted_total = sum(as.numeric(class_weights)),
    .groups = "drop"
  )

candidate_bundle <- list(
  workflow_fit = full_training_fit,
  geography_maps = full_geography_maps,
  parameters = candidate_parameters,
  outcome_levels = levels(train_data$damage_grade),
  training_rows = nrow(train_data),
  created_at = Sys.time(),
  session_info = sessionInfo()
)


# --- SAVE TRAINING-ONLY REVIEW OUTPUTS --------------------------------------

metrics_dir <- here("results", "metrics")
figures_dir <- here("results", "figures")
dir.create(metrics_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

parameter_table <- tibble(
  weight_power = candidate_parameters$weight_power,
  smoothing_strength = candidate_parameters$smoothing_strength,
  inner_encoding_folds = candidate_parameters$inner_encoding_folds,
  outer_validation_folds = nrow(folds),
  penalty = candidate_parameters$penalty,
  mixture = candidate_parameters$mixture,
  evaluate_holdout = evaluate_holdout
)

write_csv(
  parameter_table,
  file.path(metrics_dir, "elastic_net_geographic_candidate_parameters.csv")
)
write_csv(
  class_weight_summary,
  file.path(metrics_dir, "elastic_net_geographic_candidate_class_weights.csv")
)
write_csv(
  oof_overall,
  file.path(metrics_dir, "elastic_net_geographic_candidate_oof_metrics.csv")
)
write_csv(
  oof_per_class,
  file.path(metrics_dir, "elastic_net_geographic_candidate_oof_per_class.csv")
)
write_csv(
  oof_fold_metrics,
  file.path(metrics_dir, "elastic_net_geographic_candidate_fold_metrics.csv")
)
write_csv(
  oof_fold_class_metrics,
  file.path(metrics_dir, "elastic_net_geographic_candidate_fold_class_metrics.csv")
)
write_csv(
  oof_confusion_long,
  file.path(metrics_dir, "elastic_net_geographic_candidate_confusion.csv")
)

saveRDS(
  oof_predictions,
  file.path(metrics_dir, "elastic_net_geographic_candidate_oof_predictions.rds")
)
saveRDS(
  candidate_bundle,
  file.path(metrics_dir, "elastic_net_geographic_candidate_bundle.rds")
)

writeLines(
  capture.output(sessionInfo()),
  file.path(metrics_dir, "elastic_net_geographic_candidate_session_info.txt")
)


# --- OPTIONAL ONE-TIME HOLDOUT EVALUATION ----------------------------------

if (evaluate_holdout) {
  message("WARNING: evaluating the locked holdout.")

  encoded_holdout <- apply_geography_maps(
    new_data = holdout_data,
    maps = full_geography_maps
  )

  holdout_predictions <- bind_cols(
    holdout_data |>
      select(building_id, damage_grade),
    predict(full_training_fit, new_data = encoded_holdout, type = "class"),
    predict(full_training_fit, new_data = encoded_holdout, type = "prob")
  )

  holdout_overall <- geographic_overall_metrics(holdout_predictions)
  holdout_per_class <- geographic_class_metrics(holdout_predictions)

  write_csv(
    holdout_overall,
    file.path(metrics_dir, "elastic_net_geographic_candidate_holdout_metrics.csv")
  )
  write_csv(
    holdout_per_class,
    file.path(
      metrics_dir,
      "elastic_net_geographic_candidate_holdout_per_class.csv"
    )
  )
  saveRDS(
    holdout_predictions,
    file.path(
      metrics_dir,
      "elastic_net_geographic_candidate_holdout_predictions.rds"
    )
  )
}


# --- REVIEW FIGURES ----------------------------------------------------------

class_diagnostic_plot <- oof_per_class |>
  pivot_longer(
    cols = c(precision, recall, f1),
    names_to = "metric",
    values_to = "estimate"
  ) |>
  ggplot(aes(x = class, y = estimate, fill = metric)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  labs(
    title = "Geographic elastic-net candidate: class diagnostics",
    subtitle = "Leakage-safe training-only out-of-fold predictions",
    x = "Damage class",
    y = "Estimate",
    fill = "Metric"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

confusion_plot <- oof_confusion_long |>
  ggplot(aes(x = Prediction, y = Truth, fill = row_percent)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.1f%%", row_percent)), size = 4) +
  scale_fill_viridis_c(option = "C", limits = c(0, 100)) +
  labs(
    title = "Geographic elastic-net OOF confusion matrix",
    subtitle = "Percentages are calculated within each true damage class",
    x = "Predicted class",
    y = "True class",
    fill = "Row %"
  ) +
  theme_minimal(base_size = 12)

fold_qwk_plot <- oof_fold_metrics |>
  filter(.metric == "qwk") |>
  ggplot(aes(x = fold, y = .estimate, group = 1)) +
  geom_line(linewidth = 0.8, color = "#2563EB") +
  geom_point(size = 3, color = "#2563EB") +
  labs(
    title = "Geographic elastic-net QWK across validation folds",
    subtitle = "Selected penalty 0.003 and mixture 0.5",
    x = "Outer validation fold",
    y = "Quadratic weighted kappa"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  file.path(figures_dir, "elastic_net_geographic_candidate_class_metrics.png"),
  class_diagnostic_plot,
  width = 8,
  height = 5.5,
  dpi = 300
)
ggsave(
  file.path(figures_dir, "elastic_net_geographic_candidate_confusion.png"),
  confusion_plot,
  width = 7,
  height = 5.5,
  dpi = 300
)
ggsave(
  file.path(figures_dir, "elastic_net_geographic_candidate_fold_qwk.png"),
  fold_qwk_plot,
  width = 8,
  height = 5.5,
  dpi = 300
)

message("Candidate complete. Holdout evaluated: ", evaluate_holdout)
