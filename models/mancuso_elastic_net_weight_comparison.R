# Gina Mancuso: elastic-net class-weight comparison
#
# Compare unweighted, moderately weighted, and fully balanced multinomial
# elastic-net models using identical training data, fold memberships,
# preprocessing, tuning grids, and validation metrics. The shared holdout is
# deliberately not evaluated anywhere in this script.

library(tidyverse)
library(tidymodels)
library(here)

set.seed(42)


# --- LOAD SHARED TRAINING OBJECTS --------------------------------------------

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


# --- EXPERIMENT SETTINGS ------------------------------------------------------

# weight_power = 0: all weights equal 1 (unweighted)
# weight_power = 0.5: square root of balanced inverse-frequency weights
# weight_power = 1: fully balanced inverse-frequency weights
weight_schemes <- tribble(
  ~weight_scheme, ~weight_power,
  "Unweighted", 0,
  "Moderate", 0.5,
  "Fully balanced", 1
)

# The fully balanced preliminary experiment selected 0.001, the upper boundary
# of its grid. Extend the range until substantially stronger regularization is
# represented, while retaining 0.0001 for comparison with the unweighted fit.
expanded_grid <- crossing(
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


# --- HELPERS -----------------------------------------------------------------

add_power_weights <- function(data, reference_data, weight_power) {
  weight_lookup <- reference_data |>
    count(damage_grade, name = "class_n") |>
    mutate(
      balanced_weight = nrow(reference_data) /
        (n_distinct(reference_data$damage_grade) * class_n),
      case_weight = balanced_weight^weight_power
    )

  weight_values <- weight_lookup$case_weight[
    match(data$damage_grade, weight_lookup$damage_grade)
  ]

  data |>
    mutate(class_weights = importance_weights(weight_values))
}

make_weighted_folds <- function(weight_power) {
  weighted_folds <- folds
  weighted_folds$splits <- map(
    weighted_folds$splits,
    function(split_object) {
      split_object$data <- add_power_weights(
        split_object$data,
        analysis(split_object),
        weight_power
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

  weighted_folds
}

make_recipe <- function(weighted_train_data) {
  recipe(damage_grade ~ ., data = weighted_train_data) |>
    update_role(building_id, new_role = "id") |>
    step_rm(has_secondary_use, geo_level_2_id, geo_level_3_id) |>
    step_novel(all_nominal_predictors()) |>
    step_dummy(all_nominal_predictors(), one_hot = FALSE) |>
    step_zv(all_predictors()) |>
    step_normalize(all_numeric_predictors())
}

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

run_weight_scheme <- function(weight_scheme, weight_power) {
  message("\nStarting weight scheme: ", weight_scheme)

  weighted_train_data <- add_power_weights(
    train_data,
    train_data,
    weight_power
  )
  weighted_folds <- make_weighted_folds(weight_power)

  weight_summary <- weighted_train_data |>
    as_tibble() |>
    group_by(damage_grade) |>
    summarise(
      observations = n(),
      case_weight = as.numeric(first(class_weights)),
      weighted_total = sum(as.numeric(class_weights)),
      .groups = "drop"
    ) |>
    mutate(
      weight_scheme = weight_scheme,
      weight_power = weight_power,
      .before = 1
    )

  model_spec <- multinom_reg(
    penalty = tune(),
    mixture = tune()
  ) |>
    set_engine("glmnet", standardize = FALSE) |>
    set_mode("classification")

  model_workflow <- workflow() |>
    add_recipe(make_recipe(weighted_train_data)) |>
    add_model(model_spec) |>
    add_case_weights(class_weights)

  set.seed(42)
  tuning_results <- tune_grid(
    model_workflow,
    resamples = weighted_folds,
    grid = expanded_grid,
    metrics = model_metrics,
    control = control_grid(
      verbose = TRUE,
      save_pred = FALSE,
      parallel_over = "resamples"
    )
  )

  tuning_metrics <- collect_metrics(tuning_results)

  best_parameters <- tuning_metrics |>
    filter(.metric == "qwk") |>
    filter(near(mean, max(mean))) |>
    arrange(desc(penalty), desc(mixture)) |>
    slice(1) |>
    select(penalty, mixture, .config)

  selected_workflow <- model_workflow |>
    finalize_workflow(best_parameters)

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
  oof_per_class <- class_metrics(oof_predictions)
  oof_confusion <- conf_mat(
    oof_predictions,
    truth = damage_grade,
    estimate = .pred_class
  )

  message("Completed weight scheme: ", weight_scheme)

  list(
    weight_summary = weight_summary,
    tuning_results = tuning_results,
    tuning_metrics = tuning_metrics,
    best_parameters = best_parameters,
    oof_metrics = oof_metrics,
    oof_per_class = oof_per_class,
    oof_confusion = oof_confusion
  )
}


# --- RUN ALL THREE CONTROLLED EXPERIMENTS ------------------------------------

comparison_results <- map2(
  weight_schemes$weight_scheme,
  weight_schemes$weight_power,
  run_weight_scheme
)
names(comparison_results) <- weight_schemes$weight_scheme


# --- COMBINE RESULTS ----------------------------------------------------------

weight_summary <- imap_dfr(
  comparison_results,
  ~ .x$weight_summary
)

tuning_metrics <- imap_dfr(
  comparison_results,
  ~ .x$tuning_metrics |> mutate(weight_scheme = .y, .before = 1)
)

best_parameters <- imap_dfr(
  comparison_results,
  ~ .x$best_parameters |> mutate(weight_scheme = .y, .before = 1)
)

oof_overall_comparison <- imap_dfr(
  comparison_results,
  ~ .x$oof_metrics |> mutate(weight_scheme = .y, .before = 1)
)

oof_class_comparison <- imap_dfr(
  comparison_results,
  ~ .x$oof_per_class |> mutate(weight_scheme = .y, .before = 1)
)

oof_confusion_comparison <- imap_dfr(
  comparison_results,
  ~ as.data.frame(.x$oof_confusion$table) |>
    mutate(weight_scheme = .y, .before = 1)
)

unweighted_overall <- oof_overall_comparison |>
  filter(weight_scheme == "Unweighted") |>
  select(.metric, unweighted_estimate = .estimate)

oof_overall_comparison <- oof_overall_comparison |>
  left_join(unweighted_overall, by = ".metric") |>
  mutate(change_from_unweighted = .estimate - unweighted_estimate)

unweighted_class <- oof_class_comparison |>
  filter(weight_scheme == "Unweighted") |>
  select(
    class,
    unweighted_precision = precision,
    unweighted_recall = recall,
    unweighted_f1 = f1
  )

oof_class_comparison <- oof_class_comparison |>
  left_join(unweighted_class, by = "class") |>
  mutate(
    precision_change = precision - unweighted_precision,
    recall_change = recall - unweighted_recall,
    f1_change = f1 - unweighted_f1
  )


# --- SAVE CV-ONLY OUTPUTS -----------------------------------------------------

dir.create(here("results", "metrics"), recursive = TRUE, showWarnings = FALSE)
dir.create(here("results", "figures"), recursive = TRUE, showWarnings = FALSE)

write_csv(
  weight_summary,
  here("results", "metrics", "elastic_net_weight_comparison_weights.csv")
)
write_csv(
  tuning_metrics,
  here("results", "metrics", "elastic_net_weight_comparison_tuning.csv")
)
write_csv(
  best_parameters,
  here("results", "metrics", "elastic_net_weight_comparison_best.csv")
)
write_csv(
  oof_overall_comparison,
  here("results", "metrics", "elastic_net_weight_comparison_overall.csv")
)
write_csv(
  oof_class_comparison,
  here("results", "metrics", "elastic_net_weight_comparison_per_class.csv")
)
write_csv(
  oof_confusion_comparison,
  here("results", "metrics", "elastic_net_weight_comparison_confusion.csv")
)

saveRDS(
  map(comparison_results, "tuning_results"),
  here("results", "metrics", "elastic_net_weight_comparison_tuning.rds")
)

comparison_plot <- tuning_metrics |>
  filter(.metric == "qwk") |>
  mutate(
    weight_scheme = factor(
      weight_scheme,
      levels = weight_schemes$weight_scheme
    ),
    mixture = factor(mixture)
  ) |>
  ggplot(aes(x = penalty, y = mean, color = mixture, group = mixture)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 2) +
  geom_errorbar(
    aes(ymin = mean - std_err, ymax = mean + std_err),
    width = 0,
    alpha = 0.35
  ) +
  scale_x_log10() +
  facet_wrap(~ weight_scheme, ncol = 1, scales = "free_y") +
  labs(
    title = "Elastic-net QWK across class-weighting schemes",
    subtitle = "Same five folds; error bars show ±1 standard error",
    x = "Penalty (log scale)",
    y = "Mean five-fold QWK",
    color = "Mixture"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave(
  here("results", "figures", "elastic_net_weight_comparison_qwk.png"),
  comparison_plot,
  width = 10,
  height = 12,
  dpi = 300,
  bg = "white"
)


# --- CONSOLE SUMMARY ----------------------------------------------------------

cat("\nClass-weight definitions:\n")
print(weight_summary)

cat("\nBest parameters by weight scheme:\n")
print(best_parameters)

cat("\nPooled out-of-fold overall comparison:\n")
print(oof_overall_comparison)

cat("\nPooled out-of-fold class comparison:\n")
print(oof_class_comparison)

message("The shared holdout set remained locked and was not evaluated.")
