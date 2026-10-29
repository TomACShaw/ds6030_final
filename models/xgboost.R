library(tidymodels)
library(tidyverse)
library(future)
library(here)

set.seed(42)

processed_path <- here("data", "processed", "earthquake.rds")
split_path <- here("data", "processed", "split_data.RData")

if (!file.exists(processed_path)) {
  source(here("shared", "01_data_setup.R"))
}

if (!file.exists(split_path)) {
  source(here("shared", "02_preprocessing.R"))
}

load(split_path)

physical_cores <- parallel::detectCores(logical = FALSE)
if (is.na(physical_cores)) {
  physical_cores <- future::availableCores()
}

workers <- min(nrow(folds), max(1, physical_cores - 1))
plan(multisession, workers = workers)

# XGBoost baseline

model_xgb <- boost_tree(
  mode = "classification",
  trees = 100,
  learn_rate = 0.1,
  tree_depth = 3
) |>
  set_engine("xgboost", nthread = 1)

workflow_xgb <- workflow() |>
  add_recipe(base_recipe) |>
  add_model(model_xgb)

qwk <- metric_tweak("qwk", kap, weighting = "quadratic")
f_macro <- metric_tweak("f_macro", f_meas, estimator = "macro")
model_metrics <- metric_set(qwk, f_macro, bal_accuracy, accuracy)

control_baseline <- control_resamples(
  save_pred = TRUE,
  verbose = TRUE,
  parallel_over = "resamples"
)

cache_baseline_path <- here("results", "metrics", "xgb_baseline_resamples_v1.rds")

if (file.exists(cache_baseline_path)) {
  xgb_fit_cv <- readRDS(cache_baseline_path)
} else {
  xgb_fit_cv <- workflow_xgb |>
    fit_resamples(
      resamples = folds,
      metrics = model_metrics,
      control = control_baseline
    )

  saveRDS(xgb_fit_cv, cache_baseline_path)
}

xgb_baseline_metrics <- collect_metrics(xgb_fit_cv)

write_csv(
  xgb_baseline_metrics,
  here("results", "metrics", "xgb_baseline_metrics.csv")
)

print(xgb_baseline_metrics)

# XGBoost tuning

run_tuning <- FALSE

model_xgb_tune <- boost_tree(
  mode = "classification",
  trees = tune(),
  learn_rate = tune(),
  tree_depth = tune()
) |>
  set_engine("xgboost", nthread = 1)

workflow_xgb_tune <- workflow() |>
  add_recipe(base_recipe) |>
  add_model(model_xgb_tune)

parameters_xgb <- extract_parameter_set_dials(workflow_xgb_tune) |>
  update(
    trees = trees(c(100L, 1000L)),
    learn_rate = learn_rate(c(-3, -0.7)),
    tree_depth = tree_depth(c(2L, 8L))
  )

set.seed(42)

grid_xgb <- grid_space_filling(
  parameters_xgb,
  size = 30,
  type = "latin_hypercube"
)

control_tune <- control_grid(
  verbose = TRUE,
  parallel_over = "resamples"
)

cache_tuning_path <- here("results", "metrics", "xgb_tuning_results_v1.rds")

if (run_tuning) {
  if (file.exists(cache_tuning_path)) {
    xgb_tune_results <- readRDS(cache_tuning_path)
  } else {
    xgb_tune_results <- workflow_xgb_tune |>
      tune_grid(
        resamples = folds,
        grid = grid_xgb,
        metrics = model_metrics,
        control = control_tune
      )

    saveRDS(xgb_tune_results, cache_tuning_path)
  }

  xgb_tuning_metrics <- collect_metrics(xgb_tune_results)

  xgb_tuning_plot <- autoplot(xgb_tune_results, metric = "qwk")

  ggsave(
    here("results", "figures", "xgb_tuning_results.png"),
    xgb_tuning_plot,
    width = 10,
    height = 7,
    dpi = 300
  )

  write_csv(
    xgb_tuning_metrics,
    here("results", "metrics", "xgb_tuning_metrics.csv")
  )

  best_xgb_parameters <- select_best(xgb_tune_results, metric = "qwk")

  best_xgb_workflow <- workflow_xgb_tune |>
    finalize_workflow(best_xgb_parameters)

  xgb_tuned_metrics <- xgb_tuning_metrics |>
    semi_join(
      best_xgb_parameters,
      by = c("trees", "learn_rate", "tree_depth")
    ) |>
    select(.metric, .estimator, mean, n, std_err)

  xgb_cv_comparison <- bind_rows(
    xgb_baseline_metrics |>
      select(.metric, .estimator, mean, n, std_err) |>
      mutate(model = "Baseline"),
    xgb_tuned_metrics |>
      mutate(model = "Tuned")
  ) |>
    select(model, everything())

  write_csv(
    best_xgb_parameters,
    here("results", "metrics", "xgb_best_parameters.csv")
  )

  write_csv(
    xgb_cv_comparison,
    here("results", "metrics", "xgb_cv_comparison.csv")
  )

  plan(sequential)

  final_xgb_fit <- best_xgb_workflow |>
    fit(data = train_data)

  xgb_holdout_predictions <- augment(
    final_xgb_fit,
    new_data = test_data
  )

  xgb_holdout_metrics <- model_metrics(
    xgb_holdout_predictions,
    truth = damage_grade,
    estimate = .pred_class
  )

  xgb_holdout_confusion <- conf_mat(
    xgb_holdout_predictions,
    truth = damage_grade,
    estimate = .pred_class
  )

  xgb_holdout_confusion_table <- xgb_holdout_confusion$table |>
    as.data.frame() |>
    as_tibble()

  xgb_holdout_per_class <- map_dfr(
    levels(test_data$damage_grade),
    function(class_level) {
      truth_binary <- factor(
        xgb_holdout_predictions$damage_grade == class_level,
        levels = c(TRUE, FALSE)
      )

      estimate_binary <- factor(
        xgb_holdout_predictions$.pred_class == class_level,
        levels = c(TRUE, FALSE)
      )

      tibble(
        class = class_level,
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

  saveRDS(
    final_xgb_fit,
    here("results", "metrics", "xgb_final_model.rds")
  )

  saveRDS(
    xgb_holdout_predictions,
    here("results", "metrics", "xgb_holdout_predictions.rds")
  )

  write_csv(
    xgb_holdout_metrics,
    here("results", "metrics", "xgb_holdout_metrics.csv")
  )

  write_csv(
    xgb_holdout_confusion_table,
    here("results", "metrics", "xgb_holdout_confusion_matrix.csv")
  )

  write_csv(
    xgb_holdout_per_class,
    here("results", "metrics", "xgb_holdout_per_class_metrics.csv")
  )

  print(xgb_tuning_plot)
  print(show_best(xgb_tune_results, metric = "qwk"))
  print(best_xgb_parameters)
  print(xgb_cv_comparison)
  print(xgb_holdout_metrics)
  print(xgb_holdout_confusion)
  print(xgb_holdout_per_class)
}

plan(sequential)
