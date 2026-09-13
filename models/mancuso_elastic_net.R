# Gina Mancuso: multinomial elastic-net starter
#
# The outcome has three classes, so this is multinomial classification even
# though the parsnip model is named `multinom_reg()`. Elastic net adds a
# regularization penalty to multinomial logistic regression.

library(tidyverse)
library(tidymodels)
library(here)

set.seed(42)

# TRUE gives us a meeting-ready first run. Change to FALSE for a more thorough
# search after the group agrees on the shared resampling design.
quick_run <- TRUE

data_path <- here("data", "processed", "earthquake.rds")

if (!file.exists(data_path)) {
  stop(
    "Missing data/processed/earthquake.rds. ",
    "Place the two CSVs in data/raw and run shared/01_data_setup.R first."
  )
}

earthquake <- readRDS(data_path) |>
  mutate(
    # Factor order matters for quadratic weighted kappa.
    damage_grade = factor(
      damage_grade,
      levels = c(1, 2, 3),
      labels = c("low", "medium", "high")
    ),
    # Geographic IDs are labels, not continuous measurements.
    geo_level_1_id = factor(geo_level_1_id),
    across(
      c(
        land_surface_condition,
        foundation_type,
        roof_type,
        ground_floor_type,
        other_floor_type,
        position,
        plan_configuration,
        legal_ownership_status
      ),
      as.factor
    )
  )

# Preserve one untouched holdout set for an honest final check.
earthquake_split <- initial_split(
  earthquake,
  prop = 0.80,
  strata = damage_grade
)

train_data <- training(earthquake_split)
holdout_data <- testing(earthquake_split)

# Keep the quick run inexpensive. The final run should use folds shared by all
# group members rather than model-specific folds.
fold_count <- if (quick_run) 3 else 5
cv_folds <- vfold_cv(train_data, v = fold_count, strata = damage_grade)

elastic_net_recipe <- recipe(damage_grade ~ ., data = train_data) |>
  # building_id identifies rows but should not predict damage.
  update_role(building_id, new_role = "id") |>
  # The group plan proposes retaining broad geography and dropping the two
  # highest-cardinality geographic IDs for the initial model.
  # The umbrella secondary-use flag is redundant with the detailed flags.
  step_rm(geo_level_2_id, geo_level_3_id, has_secondary_use) |>
  step_novel(all_nominal_predictors()) |>
  step_other(all_nominal_predictors(), threshold = 0.005) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors())

elastic_net_spec <- multinom_reg(
  penalty = tune(),
  mixture = tune()
) |>
  set_engine("glmnet", standardize = TRUE) |>
  set_mode("classification")

elastic_net_workflow <- workflow() |>
  add_recipe(elastic_net_recipe) |>
  add_model(elastic_net_spec)

# penalty: overall strength of coefficient shrinkage
# mixture: 0 = ridge, 1 = lasso, values between = elastic net
if (quick_run) {
  elastic_net_grid <- crossing(
    penalty = 10^seq(-4, -1, length.out = 4),
    mixture = c(0, 0.5, 1)
  )
} else {
  elastic_net_grid <- crossing(
    penalty = 10^seq(-5, 0, length.out = 10),
    mixture = seq(0, 1, by = 0.25)
  )
}

# Quadratic kappa respects that low -> medium is a smaller mistake than
# low -> high. Micro-F1 is also included because DrivenData uses it.
kap_quad <- metric_tweak("kap_quad", kap, weighting = "quadratic")
f_meas_micro <- metric_tweak("f_meas_micro", f_meas, estimator = "micro")

model_metrics <- metric_set(
  kap_quad,
  f_meas_micro,
  accuracy,
  bal_accuracy
)

set.seed(42)
elastic_net_results <- tune_grid(
  elastic_net_workflow,
  resamples = cv_folds,
  grid = elastic_net_grid,
  metrics = model_metrics,
  control = control_grid(verbose = TRUE, save_pred = TRUE)
)

cv_metrics <- collect_metrics(elastic_net_results)
best_parameters <- select_best(elastic_net_results, metric = "kap_quad")

final_elastic_net_workflow <- elastic_net_workflow |>
  finalize_workflow(best_parameters)

elastic_net_holdout_fit <- last_fit(
  final_elastic_net_workflow,
  split = earthquake_split,
  metrics = model_metrics
)

holdout_metrics <- collect_metrics(elastic_net_holdout_fit)
holdout_predictions <- collect_predictions(elastic_net_holdout_fit)
holdout_confusion <- conf_mat(
  holdout_predictions,
  truth = damage_grade,
  estimate = .pred_class
)

# Class-specific metrics make majority-class failures visible.
confusion_counts <- as.matrix(holdout_confusion$table)
true_positive <- diag(confusion_counts)
per_class_metrics <- tibble(
  damage_grade = colnames(confusion_counts),
  support = colSums(confusion_counts),
  precision = true_positive / rowSums(confusion_counts),
  recall = true_positive / colSums(confusion_counts),
  f1 = 2 * precision * recall / (precision + recall)
)

dir.create(here("results", "metrics"), recursive = TRUE, showWarnings = FALSE)

write_csv(
  cv_metrics,
  here("results", "metrics", "mancuso_elastic_net_cv_metrics.csv")
)
write_csv(
  holdout_metrics,
  here("results", "metrics", "mancuso_elastic_net_holdout_metrics.csv")
)
write_csv(
  as.data.frame(holdout_confusion$table),
  here("results", "metrics", "mancuso_elastic_net_confusion_matrix.csv")
)
write_csv(
  per_class_metrics,
  here("results", "metrics", "mancuso_elastic_net_per_class_metrics.csv")
)

cat("\nBest tuning parameters (selected by quadratic weighted kappa):\n")
print(best_parameters)

cat("\nHoldout metrics:\n")
print(holdout_metrics)

cat("\nHoldout confusion matrix:\n")
print(holdout_confusion)

cat("\nHoldout metrics by damage grade:\n")
print(per_class_metrics)
