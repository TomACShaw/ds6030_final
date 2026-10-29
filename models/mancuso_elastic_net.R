# Gina Mancuso: multinomial elastic net
#
# This script deliberately uses the group's shared split, cross-validation
# folds, preprocessing recipe, and metrics so its results can be compared
# directly with the other model families.

library(tidyverse)
library(tidymodels)
library(here)

set.seed(42)

# Keep this FALSE while refining hyperparameters. The 20% holdout should be
# evaluated only after the cross-validation results identify the final model.
evaluate_holdout <- FALSE

# --- LOAD THE GROUP'S SHARED MODELING OBJECTS --------------------------------

processed_path <- here("data", "processed", "earthquake.rds")
split_path <- here("data", "processed", "split_data.RData")

# Generated data objects are intentionally ignored by Git. Recreate them from
# the raw CSVs when they are not already available locally.
if (!file.exists(processed_path)) {
  source(here("shared", "01_data_setup.R"))
}

if (!file.exists(split_path)) {
  source(here("shared", "02_preprocessing.R"))
}

load(split_path)

required_objects <- c("base_recipe", "folds", "train_data", "test_data")
missing_objects <- required_objects[!vapply(required_objects, exists, logical(1))]

if (length(missing_objects) > 0) {
  stop(
    "The shared split file is missing required objects: ",
    paste(missing_objects, collapse = ", ")
  )
}

message("Executing elastic-net pipeline with the group's shared objects.")
message(sprintf(
  "Training rows: %s; holdout rows: %s; CV folds: %s",
  format(nrow(train_data), big.mark = ","),
  format(nrow(test_data), big.mark = ","),
  nrow(folds)
))


# --- MODEL SPECIFICATION ------------------------------------------------------

# penalty controls the total amount of coefficient shrinkage.
# mixture = 0 is ridge, mixture = 1 is lasso, and intermediate values blend
# the two. The shared recipe already normalizes all numeric predictors, so
# glmnet's additional internal standardization is disabled.
elastic_net_spec <- multinom_reg(
  penalty = tune(),
  mixture = tune()
) |>
  set_engine("glmnet", standardize = FALSE) |>
  set_mode("classification")

elastic_net_workflow <- workflow() |>
  add_recipe(base_recipe) |>
  add_model(elastic_net_spec)


# --- REFINED TUNING GRID ------------------------------------------------------

# The preliminary search selected the smallest tested penalty (1e-4) and a
# lasso-like mixture. Expand below that boundary and add more mixture values to
# determine whether the apparent optimum is stable. Everything else remains
# fixed so this remains directly comparable with the group's other models.
elastic_net_grid <- crossing(
  penalty = c(1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3),
  mixture = c(0.25, 0.5, 0.75, 0.9, 1)
)


# --- SHARED METRICS -----------------------------------------------------------

# Match the metric definitions used in the group's XGBoost workflow.
qwk <- metric_tweak("qwk", kap, weighting = "quadratic")
f_macro <- metric_tweak("f_macro", f_meas, estimator = "macro")

model_metrics <- metric_set(
  qwk,
  f_macro,
  bal_accuracy,
  accuracy
)


# --- FIVE-FOLD CROSS-VALIDATION AND TUNING -----------------------------------

set.seed(42)
elastic_net_results <- tune_grid(
  elastic_net_workflow,
  resamples = folds,
  grid = elastic_net_grid,
  metrics = model_metrics,
  control = control_grid(
    verbose = TRUE,
    save_pred = FALSE,
    parallel_over = "resamples"
  )
)

cv_metrics <- collect_metrics(elastic_net_results)

# Several very small penalties can land on the same fitted glmnet solution and
# therefore have exactly identical QWK values. Among exact ties, retain the
# largest penalty because it provides at least as much regularization without
# sacrificing observed cross-validation performance.
best_parameters <- cv_metrics |>
  filter(.metric == "qwk") |>
  filter(near(mean, max(mean))) |>
  arrange(desc(penalty), desc(mixture)) |>
  slice(1) |>
  select(penalty, mixture, .config)


# --- SAVE CROSS-VALIDATION OUTPUTS -------------------------------------------

dir.create(here("results", "metrics"), recursive = TRUE, showWarnings = FALSE)
dir.create(here("results", "figures"), recursive = TRUE, showWarnings = FALSE)

write_csv(
  cv_metrics,
  here("results", "metrics", "elastic_net_refined_cv_metrics.csv")
)
write_csv(
  best_parameters,
  here("results", "metrics", "elastic_net_refined_best_parameters.csv")
)

saveRDS(
  elastic_net_results,
  here("results", "metrics", "elastic_net_refined_tuning_results.rds")
)

elastic_net_tuning_plot <- autoplot(elastic_net_results, metric = "qwk")

ggsave(
  here("results", "figures", "elastic_net_refined_tuning.png"),
  elastic_net_tuning_plot,
  width = 9,
  height = 6,
  dpi = 300,
  bg = "white"
)


# --- OPTIONAL FINAL HOLDOUT EVALUATION ---------------------------------------

if (evaluate_holdout) {
  final_elastic_net_workflow <- elastic_net_workflow |>
    finalize_workflow(best_parameters)

  final_elastic_net_fit <- final_elastic_net_workflow |>
    fit(data = train_data)

  holdout_predictions <- augment(
    final_elastic_net_fit,
    new_data = test_data
  )

  holdout_metrics <- model_metrics(
    holdout_predictions,
    truth = damage_grade,
    estimate = .pred_class
  )

  holdout_confusion <- conf_mat(
    holdout_predictions,
    truth = damage_grade,
    estimate = .pred_class
  )

  # Compute one-vs-all metrics for each ordered damage grade.
  per_class_metrics <- map_dfr(
    levels(test_data$damage_grade),
    function(class_level) {
      truth_binary <- factor(
        holdout_predictions$damage_grade == class_level,
        levels = c(TRUE, FALSE)
      )

      estimate_binary <- factor(
        holdout_predictions$.pred_class == class_level,
        levels = c(TRUE, FALSE)
      )

      tibble(
        class = class_level,
        support = sum(holdout_predictions$damage_grade == class_level),
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

  write_csv(
    holdout_metrics,
    here("results", "metrics", "elastic_net_refined_holdout_metrics.csv")
  )
  write_csv(
    as.data.frame(holdout_confusion$table),
    here("results", "metrics", "elastic_net_refined_confusion_matrix.csv")
  )
  write_csv(
    per_class_metrics,
    here("results", "metrics", "elastic_net_refined_per_class_metrics.csv")
  )
  saveRDS(
    final_elastic_net_fit,
    here("results", "metrics", "elastic_net_refined_final_model.rds")
  )
}


# --- CONSOLE SUMMARY ----------------------------------------------------------

cat("\nBest shared-fold tuning parameters (selected by QWK):\n")
print(best_parameters)

cat("\nBest cross-validation results by QWK:\n")
print(show_best(elastic_net_results, metric = "qwk", n = 10))

if (evaluate_holdout) {
  cat("\nShared holdout metrics:\n")
  print(holdout_metrics)

  cat("\nShared holdout confusion matrix:\n")
  print(holdout_confusion)

  cat("\nShared holdout metrics by damage grade:\n")
  print(per_class_metrics)
} else {
  message(
    "Holdout evaluation skipped. Set evaluate_holdout <- TRUE only after ",
    "selecting the final elastic-net configuration."
  )
}
