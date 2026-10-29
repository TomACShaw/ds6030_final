# Gina Mancuso: multinomial elastic net
#
# This script deliberately uses the group's shared split, cross-validation
# folds, preprocessing recipe, and metrics so its results can be compared
# directly with the other model families.

library(tidyverse)
library(tidymodels)
library(here)

set.seed(42)

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


# --- INITIAL COMPARISON GRID --------------------------------------------------

# This intentionally retains the small preliminary grid for the first direct
# comparison. A later experiment can expand the range around the best region
# without changing the shared split, folds, recipe, or metrics.
elastic_net_grid <- crossing(
  penalty = 10^seq(-4, -1, length.out = 4),
  mixture = c(0, 0.5, 1)
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
    save_pred = TRUE,
    parallel_over = "resamples"
  )
)

cv_metrics <- collect_metrics(elastic_net_results)
best_parameters <- select_best(elastic_net_results, metric = "qwk")


# --- FINAL FIT ON THE SHARED 80/20 SPLIT -------------------------------------

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


# --- SAVE COMPARISON-READY OUTPUTS -------------------------------------------

dir.create(here("results", "metrics"), recursive = TRUE, showWarnings = FALSE)
dir.create(here("results", "figures"), recursive = TRUE, showWarnings = FALSE)

write_csv(
  cv_metrics,
  here("results", "metrics", "elastic_net_shared_cv_metrics.csv")
)
write_csv(
  best_parameters,
  here("results", "metrics", "elastic_net_shared_best_parameters.csv")
)
write_csv(
  holdout_metrics,
  here("results", "metrics", "elastic_net_shared_holdout_metrics.csv")
)
write_csv(
  as.data.frame(holdout_confusion$table),
  here("results", "metrics", "elastic_net_shared_confusion_matrix.csv")
)
write_csv(
  per_class_metrics,
  here("results", "metrics", "elastic_net_shared_per_class_metrics.csv")
)

saveRDS(
  elastic_net_results,
  here("results", "metrics", "elastic_net_shared_tuning_results.rds")
)
saveRDS(
  final_elastic_net_fit,
  here("results", "metrics", "elastic_net_shared_final_model.rds")
)

elastic_net_tuning_plot <- autoplot(elastic_net_results, metric = "qwk")

ggsave(
  here("results", "figures", "elastic_net_shared_tuning.png"),
  elastic_net_tuning_plot,
  width = 9,
  height = 6,
  dpi = 300,
  bg = "white"
)


# --- CONSOLE SUMMARY ----------------------------------------------------------

cat("\nBest shared-fold tuning parameters (selected by QWK):\n")
print(best_parameters)

cat("\nBest cross-validation results by QWK:\n")
print(show_best(elastic_net_results, metric = "qwk"))

cat("\nShared holdout metrics:\n")
print(holdout_metrics)

cat("\nShared holdout confusion matrix:\n")
print(holdout_confusion)

cat("\nShared holdout metrics by damage grade:\n")
print(per_class_metrics)
