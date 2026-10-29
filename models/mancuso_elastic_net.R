# Gina Mancuso: moderate class-weighted multinomial elastic net
#
# This is the primary elastic-net candidate for group review. It uses the
# group's shared 80/20 split and exact five-fold cross-validation memberships,
# but it does not evaluate the shared holdout unless evaluate_holdout is
# deliberately changed after the group finishes model selection.

library(tidyverse)
library(tidymodels)
library(here)

set.seed(42)

# Keep this FALSE during group review. The holdout should be evaluated once,
# only after the group agrees that training-only model selection is complete.
evaluate_holdout <- FALSE

# Square-root inverse-frequency weighting was selected in the controlled
# training-only comparison. A value of 0 would be unweighted and 1 would be
# fully balanced inverse-frequency weighting.
weight_power <- 0.5


# --- LOAD THE GROUP'S SHARED MODELING OBJECTS --------------------------------

processed_path <- here("data", "processed", "earthquake.rds")
split_path <- here("data", "processed", "split_data.RData")

if (!file.exists(processed_path)) {
  source(here("shared", "01_data_setup.R"))
}

if (!file.exists(split_path)) {
  source(here("shared", "02_preprocessing.R"))
}

load(split_path)

required_objects <- c("folds", "train_data", "test_data")
missing_objects <- required_objects[!vapply(required_objects, exists, logical(1))]

if (length(missing_objects) > 0) {
  stop(
    "The shared split file is missing required objects: ",
    paste(missing_objects, collapse = ", ")
  )
}

message("Executing the moderate-weight elastic-net candidate.")
message(sprintf(
  "Training rows: %s; locked holdout rows: %s; CV folds: %s",
  format(nrow(train_data), big.mark = ","),
  format(nrow(test_data), big.mark = ","),
  nrow(folds)
))


# --- MODERATE CLASS WEIGHTS --------------------------------------------------

# Begin with balanced inverse-frequency weights N / (K * n_class), then take
# the square root. This helps the less frequent classes without forcing every
# class to contribute exactly the same total weight.
add_power_weights <- function(data, reference_data, power = weight_power) {
  weight_lookup <- reference_data |>
    count(damage_grade, name = "class_n") |>
    mutate(
      balanced_weight = nrow(reference_data) /
        (n_distinct(reference_data$damage_grade) * class_n),
      case_weight = balanced_weight^power
    )

  weight_values <- weight_lookup$case_weight[
    match(data$damage_grade, weight_lookup$damage_grade)
  ]

  data |>
    mutate(class_weights = importance_weights(weight_values))
}

weighted_train_data <- add_power_weights(train_data, train_data)

# Calculate weights separately from each fold's analysis portion. This avoids
# using assessment-fold outcome frequencies while preserving the group's exact
# row memberships.
weighted_folds <- folds
weighted_folds$splits <- map(
  weighted_folds$splits,
  function(split_object) {
    split_object$data <- add_power_weights(
      split_object$data,
      analysis(split_object)
    )
    split_object
  }
)

memberships_match <- map2_lgl(
  folds$splits,
  weighted_folds$splits,
  ~ identical(analysis(.x)$building_id, analysis(.y)$building_id) &&
    identical(assessment(.x)$building_id, assessment(.y)$building_id)
)

if (!all(memberships_match)) {
  stop("Weight construction changed the shared fold memberships.")
}

class_weight_summary <- weighted_train_data |>
  as_tibble() |>
  group_by(damage_grade) |>
  summarise(
    observations = n(),
    observed_percent = 100 * n() / nrow(weighted_train_data),
    case_weight = as.numeric(first(class_weights)),
    weighted_total = sum(as.numeric(class_weights)),
    .groups = "drop"
  )


# --- PREPROCESSING AND MODEL -------------------------------------------------

# These steps reproduce the group's shared base recipe. The extra
# importance-weights column receives the case_weights role automatically and
# is therefore never used as a predictor.
elastic_net_recipe <- recipe(damage_grade ~ ., data = weighted_train_data) |>
  update_role(building_id, new_role = "id") |>
  step_rm(has_secondary_use, geo_level_2_id, geo_level_3_id) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors(), one_hot = FALSE) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

elastic_net_spec <- multinom_reg(
  penalty = tune(),
  mixture = tune()
) |>
  set_engine("glmnet", standardize = FALSE) |>
  set_mode("classification")

elastic_net_workflow <- workflow() |>
  add_recipe(elastic_net_recipe) |>
  add_model(elastic_net_spec) |>
  add_case_weights(class_weights)


# --- TRAINING-ONLY TUNING ----------------------------------------------------

# This expanded range contains the selected moderate-weight penalty internally,
# rather than at a grid boundary.
elastic_net_grid <- crossing(
  penalty = c(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
  mixture = c(0.25, 0.5, 0.75, 0.9, 1)
)

qwk <- metric_tweak("qwk", kap, weighting = "quadratic")
f_macro <- metric_tweak("f_macro", f_meas, estimator = "macro")

model_metrics <- metric_set(
  qwk,
  f_macro,
  bal_accuracy,
  accuracy
)

set.seed(42)
elastic_net_results <- tune_grid(
  elastic_net_workflow,
  resamples = weighted_folds,
  grid = elastic_net_grid,
  metrics = model_metrics,
  control = control_grid(
    verbose = TRUE,
    save_pred = FALSE,
    parallel_over = "resamples"
  )
)

cv_metrics <- collect_metrics(elastic_net_results)

# QWK is the primary selection metric. If multiple settings have exactly the
# same mean QWK, retain the strongest penalty and then the larger mixture.
best_parameters <- cv_metrics |>
  filter(.metric == "qwk") |>
  filter(near(mean, max(mean))) |>
  arrange(desc(penalty), desc(mixture)) |>
  slice(1) |>
  select(penalty, mixture, .config)

selected_workflow <- elastic_net_workflow |>
  finalize_workflow(best_parameters)


# --- OUT-OF-FOLD DIAGNOSTICS -------------------------------------------------

# Refit only the selected configuration on each analysis fold and save its
# assessment-fold predictions. Every row is therefore predicted by a model
# that was not trained on that row.
set.seed(42)
selected_resamples <- fit_resamples(
  selected_workflow,
  resamples = weighted_folds,
  metrics = model_metrics,
  control = control_resamples(save_pred = TRUE)
)

oof_predictions <- collect_predictions(selected_resamples)

oof_metrics <- model_metrics(
  oof_predictions,
  truth = damage_grade,
  estimate = .pred_class
)

class_metrics <- function(predictions) {
  map_dfr(
    levels(predictions$damage_grade),
    function(class_level) {
      truth_binary <- factor(
        predictions$damage_grade == class_level,
        levels = c(TRUE, FALSE)
      )
      estimate_binary <- factor(
        predictions$.pred_class == class_level,
        levels = c(TRUE, FALSE)
      )

      tibble(
        class = class_level,
        support = sum(predictions$damage_grade == class_level),
        observed_percent = 100 * support / nrow(predictions),
        predicted = sum(predictions$.pred_class == class_level),
        predicted_percent = 100 * predicted / nrow(predictions),
        precision = precision_vec(
          truth_binary,
          estimate_binary,
          event_level = "first"
        ),
        recall = recall_vec(
          truth_binary,
          estimate_binary,
          event_level = "first"
        ),
        f1 = f_meas_vec(
          truth_binary,
          estimate_binary,
          event_level = "first"
        )
      )
    }
  )
}

oof_per_class <- class_metrics(oof_predictions)
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


# --- TRAINING-ONLY COEFFICIENTS ---------------------------------------------

# Fit the selected workflow on all training rows for interpretation. This does
# not score or otherwise access the locked holdout.
selected_training_fit <- selected_workflow |>
  fit(data = weighted_train_data)

selected_penalty <- best_parameters$penalty[[1]]
coefficient_matrices <- coef(
  extract_fit_engine(selected_training_fit),
  s = selected_penalty
)

selected_coefficients <- imap_dfr(
  coefficient_matrices,
  function(coefficient_matrix, class_name) {
    tibble(
      class = class_name,
      term = rownames(coefficient_matrix),
      estimate = as.numeric(coefficient_matrix[, 1]),
      retained = estimate != 0
    )
  }
)

top_coefficients <- selected_coefficients |>
  filter(term != "(Intercept)", retained) |>
  group_by(class) |>
  slice_max(order_by = abs(estimate), n = 12, with_ties = FALSE) |>
  ungroup()


# --- SAVE TRAINING-ONLY REVIEW OUTPUTS --------------------------------------

dir.create(here("results", "metrics"), recursive = TRUE, showWarnings = FALSE)
dir.create(here("results", "figures"), recursive = TRUE, showWarnings = FALSE)

write_csv(
  class_weight_summary,
  here("results", "metrics", "elastic_net_candidate_class_weights.csv")
)
write_csv(
  cv_metrics,
  here("results", "metrics", "elastic_net_candidate_tuning_metrics.csv")
)
write_csv(
  best_parameters,
  here("results", "metrics", "elastic_net_candidate_best_parameters.csv")
)
write_csv(
  oof_metrics,
  here("results", "metrics", "elastic_net_candidate_oof_metrics.csv")
)
write_csv(
  oof_per_class,
  here("results", "metrics", "elastic_net_candidate_oof_per_class.csv")
)
write_csv(
  oof_confusion_long,
  here("results", "metrics", "elastic_net_candidate_oof_confusion.csv")
)
write_csv(
  selected_coefficients,
  here("results", "metrics", "elastic_net_candidate_coefficients.csv")
)

saveRDS(
  elastic_net_results,
  here("results", "metrics", "elastic_net_candidate_tuning_results.rds")
)
saveRDS(
  selected_training_fit,
  here("results", "metrics", "elastic_net_candidate_training_fit.rds")
)

tuning_plot <- cv_metrics |>
  filter(.metric == "qwk") |>
  mutate(mixture = factor(mixture)) |>
  ggplot(aes(x = penalty, y = mean, color = mixture, group = mixture)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.2) +
  geom_errorbar(
    aes(ymin = mean - std_err, ymax = mean + std_err),
    width = 0,
    alpha = 0.35
  ) +
  scale_x_log10() +
  labs(
    title = "Moderate-weight elastic-net tuning",
    subtitle = "Five-fold cross-validation; error bars show +/- 1 standard error",
    x = "Penalty (log scale)",
    y = "Mean quadratic weighted kappa",
    color = "Mixture"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave(
  here("results", "figures", "elastic_net_candidate_tuning.png"),
  tuning_plot,
  width = 9,
  height = 6,
  dpi = 300,
  bg = "white"
)

confusion_plot <- oof_confusion_long |>
  ggplot(aes(x = Prediction, y = Truth, fill = row_percent)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.1f%%", row_percent)), size = 4) +
  scale_fill_viridis_c(option = "C", limits = c(0, 100)) +
  labs(
    title = "Moderate-weight elastic-net confusion matrix",
    subtitle = "Pooled out-of-fold predictions; percentages are within actual class",
    x = "Predicted damage grade",
    y = "Actual damage grade",
    fill = "Row percent"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  here("results", "figures", "elastic_net_candidate_confusion.png"),
  confusion_plot,
  width = 8,
  height = 6,
  dpi = 300,
  bg = "white"
)

coefficient_plot <- top_coefficients |>
  ggplot(aes(x = reorder(term, abs(estimate)), y = estimate, fill = estimate > 0)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  facet_wrap(~ class, scales = "free_y") +
  scale_fill_manual(values = c("TRUE" = "#2C7FB8", "FALSE" = "#D95F0E")) +
  labs(
    title = "Largest retained lasso coefficients by damage class",
    subtitle = "Positive values increase and negative values decrease the class score",
    x = NULL,
    y = "Coefficient"
  ) +
  theme_minimal(base_size = 11)

ggsave(
  here("results", "figures", "elastic_net_candidate_coefficients.png"),
  coefficient_plot,
  width = 12,
  height = 8,
  dpi = 300,
  bg = "white"
)


# --- OPTIONAL HOLDOUT EVALUATION --------------------------------------------

if (evaluate_holdout) {
  stop(
    "Holdout remains locked for group review. Remove this stop only after the ",
    "group explicitly approves final holdout evaluation."
  )
}


# --- CONSOLE SUMMARY ---------------------------------------------------------

cat("\nModerate class weights:\n")
print(class_weight_summary)

cat("\nSelected parameters (highest mean CV QWK):\n")
print(best_parameters)

cat("\nPooled out-of-fold metrics:\n")
print(oof_metrics)

cat("\nPooled out-of-fold class metrics:\n")
print(oof_per_class)

cat("\nPooled out-of-fold confusion matrix:\n")
print(oof_confusion)

message("The shared holdout remained locked and was not evaluated.")
