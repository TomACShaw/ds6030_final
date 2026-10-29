# Gina Mancuso: class-weighted multinomial elastic net
#
# This is a controlled extension of mancuso_elastic_net.R. It preserves the
# group's training/holdout split, exact cross-validation fold assignments,
# preprocessing steps, tuning grid, and evaluation metrics. The only modeling
# change is the use of balanced inverse-frequency importance weights during
# fitting. The holdout set is not evaluated in this experiment.

library(tidyverse)
library(tidymodels)
library(here)

set.seed(42)


# --- LOAD THE GROUP'S SHARED MODELING OBJECTS --------------------------------

split_path <- here("data", "processed", "split_data.RData")

if (!file.exists(split_path)) {
  stop(
    "The shared split file does not exist. Run shared/01_data_setup.R and ",
    "shared/02_preprocessing.R before this experiment."
  )
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


# --- BALANCED CLASS WEIGHTS ---------------------------------------------------

# For K classes, each class receives weight N / (K * n_class). This makes the
# total training weight contributed by each class equal without duplicating or
# discarding observations. Fold-specific weights are calculated using only the
# analysis portion of each fold.
add_balanced_weights <- function(data, reference_data) {
  weight_lookup <- reference_data |>
    count(damage_grade, name = "class_n") |>
    mutate(
      weight = nrow(reference_data) /
        (n_distinct(reference_data$damage_grade) * class_n)
    )

  weight_values <- weight_lookup$weight[
    match(data$damage_grade, weight_lookup$damage_grade)
  ]

  data |>
    mutate(class_weights = importance_weights(weight_values))
}

weighted_train_data <- add_balanced_weights(train_data, train_data)

# Preserve the exact existing analysis and assessment row indices. Only attach
# weights to each split's underlying data. Assessment weights are ignored;
# tidymodels uses importance weights during model fitting only.
weighted_folds <- folds
weighted_folds$splits <- map(
  weighted_folds$splits,
  function(split_object) {
    split_object$data <- add_balanced_weights(
      split_object$data,
      analysis(split_object)
    )
    split_object
  }
)

if (!all(map2_lgl(
  folds$splits,
  weighted_folds$splits,
  ~ identical(analysis(.x)$building_id, analysis(.y)$building_id) &&
    identical(assessment(.x)$building_id, assessment(.y)$building_id)
))) {
  stop("Weighted fold construction changed the shared fold memberships.")
}

class_weight_summary <- weighted_train_data |>
  as_tibble() |>
  group_by(damage_grade) |>
  summarise(
    observations = n(),
    case_weight = as.numeric(first(class_weights)),
    weighted_total = sum(as.numeric(class_weights)),
    .groups = "drop"
  )


# --- PREPROCESSING AND MODEL --------------------------------------------------

# Reproduce the group's base_recipe steps. Creating the recipe with the
# hardhat importance-weights column automatically gives that column the
# case_weights role, so it cannot become a predictor.
weighted_recipe <- recipe(damage_grade ~ ., data = weighted_train_data) |>
  update_role(building_id, new_role = "id") |>
  step_rm(has_secondary_use, geo_level_2_id, geo_level_3_id) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors(), one_hot = FALSE) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

weighted_elastic_net_spec <- multinom_reg(
  penalty = tune(),
  mixture = tune()
) |>
  set_engine("glmnet", standardize = FALSE) |>
  set_mode("classification")

weighted_elastic_net_workflow <- workflow() |>
  add_recipe(weighted_recipe) |>
  add_model(weighted_elastic_net_spec) |>
  add_case_weights(class_weights)

elastic_net_grid <- crossing(
  penalty = c(1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3),
  mixture = c(0.25, 0.5, 0.75, 0.9, 1)
)


# --- METRICS -----------------------------------------------------------------

qwk <- metric_tweak("qwk", kap, weighting = "quadratic")
f_macro <- metric_tweak("f_macro", f_meas, estimator = "macro")

model_metrics <- metric_set(
  qwk,
  f_macro,
  bal_accuracy,
  accuracy
)


# --- WEIGHTED FIVE-FOLD TUNING ------------------------------------------------

set.seed(42)
weighted_tuning_results <- tune_grid(
  weighted_elastic_net_workflow,
  resamples = weighted_folds,
  grid = elastic_net_grid,
  metrics = model_metrics,
  control = control_grid(
    verbose = TRUE,
    save_pred = FALSE,
    parallel_over = "resamples"
  )
)

weighted_cv_metrics <- collect_metrics(weighted_tuning_results)

weighted_best_parameters <- weighted_cv_metrics |>
  filter(.metric == "qwk") |>
  filter(near(mean, max(mean))) |>
  arrange(desc(penalty), desc(mixture)) |>
  slice(1) |>
  select(penalty, mixture, .config)


# --- OUT-OF-FOLD PREDICTIONS FOR THE SELECTED WEIGHTED MODEL -----------------

selected_weighted_workflow <- weighted_elastic_net_workflow |>
  finalize_workflow(weighted_best_parameters)

set.seed(42)
weighted_selected_resamples <- fit_resamples(
  selected_weighted_workflow,
  resamples = weighted_folds,
  metrics = model_metrics,
  control = control_resamples(save_pred = TRUE)
)

weighted_oof_predictions <- collect_predictions(weighted_selected_resamples)

weighted_oof_metrics <- model_metrics(
  weighted_oof_predictions,
  truth = damage_grade,
  estimate = .pred_class
)

weighted_oof_confusion <- conf_mat(
  weighted_oof_predictions,
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
        predicted = sum(predictions$.pred_class == class_level),
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

weighted_oof_per_class <- class_metrics(weighted_oof_predictions)


# --- OPTIONAL DIRECT COMPARISON WITH SAVED UNWEIGHTED CV PREDICTIONS ---------

unweighted_tuning_path <- here(
  "results",
  "metrics",
  "elastic_net_shared_tuning_results.rds"
)

if (file.exists(unweighted_tuning_path)) {
  unweighted_tuning_results <- readRDS(unweighted_tuning_path)
  unweighted_oof_predictions <- collect_predictions(
    unweighted_tuning_results,
    parameters = tibble(penalty = 1e-4, mixture = 1)
  )

  unweighted_oof_metrics <- model_metrics(
    unweighted_oof_predictions,
    truth = damage_grade,
    estimate = .pred_class
  )
  unweighted_oof_per_class <- class_metrics(unweighted_oof_predictions)

  oof_overall_comparison <- bind_rows(
    unweighted_oof_metrics |> mutate(model = "Unweighted"),
    weighted_oof_metrics |> mutate(model = "Class weighted")
  ) |>
    select(model, everything())

  oof_class_comparison <- bind_rows(
    unweighted_oof_per_class |> mutate(model = "Unweighted"),
    weighted_oof_per_class |> mutate(model = "Class weighted")
  ) |>
    select(model, everything())
}


# --- SAVE CV-ONLY OUTPUTS -----------------------------------------------------

dir.create(here("results", "metrics"), recursive = TRUE, showWarnings = FALSE)
dir.create(here("results", "figures"), recursive = TRUE, showWarnings = FALSE)

write_csv(
  class_weight_summary,
  here("results", "metrics", "elastic_net_weighted_class_weights.csv")
)
write_csv(
  weighted_cv_metrics,
  here("results", "metrics", "elastic_net_weighted_cv_metrics.csv")
)
write_csv(
  weighted_best_parameters,
  here("results", "metrics", "elastic_net_weighted_best_parameters.csv")
)
write_csv(
  weighted_oof_metrics,
  here("results", "metrics", "elastic_net_weighted_oof_metrics.csv")
)
write_csv(
  weighted_oof_per_class,
  here("results", "metrics", "elastic_net_weighted_oof_per_class.csv")
)
write_csv(
  as.data.frame(weighted_oof_confusion$table),
  here("results", "metrics", "elastic_net_weighted_oof_confusion_matrix.csv")
)

if (exists("oof_overall_comparison")) {
  write_csv(
    oof_overall_comparison,
    here("results", "metrics", "elastic_net_weighted_vs_unweighted_oof.csv")
  )
  write_csv(
    oof_class_comparison,
    here(
      "results",
      "metrics",
      "elastic_net_weighted_vs_unweighted_oof_per_class.csv"
    )
  )
}

saveRDS(
  weighted_tuning_results,
  here("results", "metrics", "elastic_net_weighted_tuning_results.rds")
)

weighted_tuning_plot <- autoplot(weighted_tuning_results, metric = "qwk")

ggsave(
  here("results", "figures", "elastic_net_weighted_tuning.png"),
  weighted_tuning_plot,
  width = 9,
  height = 6,
  dpi = 300,
  bg = "white"
)


# --- CONSOLE SUMMARY ----------------------------------------------------------

cat("\nBalanced class weights:\n")
print(class_weight_summary)

cat("\nSelected weighted parameters (by unweighted validation QWK):\n")
print(weighted_best_parameters)

cat("\nBest weighted cross-validation configurations:\n")
print(show_best(weighted_tuning_results, metric = "qwk", n = 10))

cat("\nSelected weighted model pooled out-of-fold metrics:\n")
print(weighted_oof_metrics)

cat("\nSelected weighted model pooled out-of-fold class metrics:\n")
print(weighted_oof_per_class)

if (exists("oof_overall_comparison")) {
  cat("\nDirect out-of-fold comparison with unweighted elastic net:\n")
  print(oof_overall_comparison)

  cat("\nDirect out-of-fold class comparison:\n")
  print(oof_class_comparison)
}

message("The shared holdout set was not evaluated.")
