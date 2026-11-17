# Gina Mancuso: training-only ordinal elastic-net sensitivity experiment
#
# This script compares the current multinomial geographic elastic-net candidate
# with a parallel cumulative-logit elastic net. The ordinal model learns one
# shared predictor direction and two ordered cut points for low < medium < high.
#
# IMPORTANT: This experiment never extracts, accesses, or evaluates the shared
# holdout. The shared .RData container is loaded into an isolated environment,
# and only the training data and resampling folds are copied out.

library(tidyverse)
library(tidymodels)
library(ordinalNet)
library(here)

source(here("models", "mancuso_geography_helpers.R"))

set.seed(42)


# --- TRAINING-ONLY CONFIGURATION --------------------------------------------

experiment_parameters <- list(
  weight_power = 0.5,
  smoothing_strength = 10,
  inner_encoding_folds = 5,
  mixture = 0.5,
  penalty_grid = c(0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1),
  seed = 42,
  outcome_order = c("low", "medium", "high")
)


# --- LOAD ONLY TRAINING OBJECTS ---------------------------------------------

split_path <- here("data", "processed", "split_data.RData")

if (!file.exists(split_path)) {
  stop(
    "The shared split file does not exist. Run shared/01_data_setup.R and ",
    "shared/02_preprocessing.R before this experiment."
  )
}

shared_objects <- new.env(parent = emptyenv())
load(split_path, envir = shared_objects)

required_objects <- c("folds", "train_data")
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
rm(shared_objects)

message("Running the training-only ordinal elastic-net experiment.")
message(sprintf(
  "Training rows: %s; outer CV folds: %s; holdout accessed: FALSE",
  format(nrow(train_data), big.mark = ","),
  nrow(folds)
))


# --- ORDINAL MODEL HELPERS ---------------------------------------------------

make_weighted_ordinal_response <- function(outcome, case_weights, levels) {
  outcome_index <- match(as.character(outcome), levels)

  if (anyNA(outcome_index)) {
    stop("The ordinal response contains an unexpected damage class.")
  }

  response <- matrix(
    0,
    nrow = length(outcome_index),
    ncol = length(levels),
    dimnames = list(NULL, levels)
  )
  response[cbind(seq_along(outcome_index), outcome_index)] <- as.numeric(
    case_weights
  )
  response
}


prepare_ordinal_matrices <- function(
  weighted_analysis,
  encoded_assessment,
  outcome_order
) {
  preprocessing <- make_geographic_recipe(weighted_analysis) |>
    prep(training = weighted_analysis, retain = TRUE)

  baked_analysis <- bake(preprocessing, new_data = NULL)
  baked_assessment <- bake(preprocessing, new_data = encoded_assessment)

  predictor_names <- summary(preprocessing) |>
    filter(role == "predictor") |>
    pull(variable)

  if (length(predictor_names) == 0) {
    stop("The ordinal preprocessing recipe produced no predictors.")
  }

  list(
    x_analysis = as.matrix(
      baked_analysis |>
        select(all_of(predictor_names))
    ),
    x_assessment = as.matrix(
      baked_assessment |>
        select(all_of(predictor_names))
    ),
    y_analysis = make_weighted_ordinal_response(
      outcome = weighted_analysis$damage_grade,
      case_weights = weighted_analysis$class_weights,
      levels = outcome_order
    )
  )
}


predict_ordinal_path <- function(
  ordinal_fit,
  x_assessment,
  assessment_data,
  fold_id,
  outcome_order
) {
  map_dfr(
    seq_along(ordinal_fit$lambdaVals),
    function(lambda_index) {
      probabilities <- predict(
        ordinal_fit,
        newx = x_assessment,
        whichLambda = lambda_index,
        type = "response"
      )

      if (ncol(probabilities) != length(outcome_order)) {
        stop("The ordinal model returned an unexpected probability matrix.")
      }

      colnames(probabilities) <- paste0(".pred_", outcome_order)
      predicted_level <- outcome_order[max.col(probabilities, ties.method = "first")]

      bind_cols(
        assessment_data |>
          select(building_id, damage_grade),
        tibble(
          .pred_class = factor(
            predicted_level,
            levels = levels(assessment_data$damage_grade)
          )
        ),
        as_tibble(probabilities)
      ) |>
        mutate(
          fold = fold_id,
          penalty = ordinal_fit$lambdaVals[[lambda_index]],
          .before = 1
        )
    }
  )
}


severe_error_summary <- function(predictions) {
  predictions |>
    summarise(
      observations = n(),
      low_to_high = sum(
        as.character(damage_grade) == "low" &
          as.character(.pred_class) == "high"
      ),
      high_to_low = sum(
        as.character(damage_grade) == "high" &
          as.character(.pred_class) == "low"
      )
    ) |>
    mutate(
      severe_errors = low_to_high + high_to_low,
      severe_error_rate = severe_errors / observations
    )
}


# --- FIT THE ORDINAL PATH IN EACH SHARED OUTER FOLD -------------------------

ordinal_predictions <- map2_dfr(
  folds$splits,
  seq_len(nrow(folds)),
  function(split_object, fold_number) {
    fold_id <- folds$id[[fold_number]]
    message("  Preparing ", fold_id, " of ", nrow(folds))

    outer_analysis <- analysis(split_object)
    outer_assessment <- assessment(split_object)

    encoded_analysis <- crossfit_geography(
      analysis_data = outer_analysis,
      prior_n = experiment_parameters$smoothing_strength,
      v = experiment_parameters$inner_encoding_folds,
      seed = 12000 + fold_number
    )

    outer_maps <- build_geography_maps(
      reference_data = outer_analysis,
      prior_n = experiment_parameters$smoothing_strength
    )
    encoded_assessment <- apply_geography_maps(
      new_data = outer_assessment,
      maps = outer_maps
    )

    weighted_analysis <- add_power_weights(
      data = encoded_analysis,
      reference_data = encoded_analysis,
      power = experiment_parameters$weight_power
    )

    ordinal_data <- prepare_ordinal_matrices(
      weighted_analysis = weighted_analysis,
      encoded_assessment = encoded_assessment,
      outcome_order = experiment_parameters$outcome_order
    )

    message(
      "  Fitting ordinal path for ", fold_id,
      " (", ncol(ordinal_data$x_analysis), " predictors)"
    )

    ordinal_fit <- ordinalNet(
      x = ordinal_data$x_analysis,
      y = ordinal_data$y_analysis,
      alpha = experiment_parameters$mixture,
      standardize = FALSE,
      family = "cumulative",
      link = "logit",
      parallelTerms = TRUE,
      nonparallelTerms = FALSE,
      lambdaVals = experiment_parameters$penalty_grid,
      keepTrainingData = FALSE
    )

    predict_ordinal_path(
      ordinal_fit = ordinal_fit,
      x_assessment = ordinal_data$x_assessment,
      assessment_data = outer_assessment,
      fold_id = fold_id,
      outcome_order = experiment_parameters$outcome_order
    )
  }
)


# --- SELECT THE PENALTY USING TRAINING-ONLY CROSS-VALIDATION ----------------

ordinal_fold_metrics <- ordinal_predictions |>
  group_by(penalty, fold) |>
  group_modify(
    ~ geographic_overall_metrics(.x) |>
      select(.metric, .estimator, .estimate)
  ) |>
  ungroup()

ordinal_qwk_summary <- ordinal_fold_metrics |>
  filter(.metric == "qwk") |>
  group_by(penalty) |>
  summarise(
    mean_qwk = mean(.estimate),
    sd_qwk = sd(.estimate),
    se_qwk = sd_qwk / sqrt(n()),
    folds = n(),
    .groups = "drop"
  ) |>
  arrange(desc(mean_qwk))

best_numeric <- ordinal_qwk_summary |>
  slice_max(mean_qwk, n = 1, with_ties = FALSE)

one_se_cutoff <- best_numeric$mean_qwk - best_numeric$se_qwk

selected_ordinal <- ordinal_qwk_summary |>
  filter(mean_qwk >= one_se_cutoff) |>
  slice_max(penalty, n = 1, with_ties = FALSE) |>
  mutate(
    selection_rule = "largest penalty within one SE of best mean QWK",
    best_mean_qwk = best_numeric$mean_qwk,
    best_penalty = best_numeric$penalty,
    one_se_cutoff = one_se_cutoff
  )

selected_penalty <- selected_ordinal$penalty[[1]]

if (selected_penalty %in% range(experiment_parameters$penalty_grid)) {
  warning(
    "The selected ordinal penalty lies on the search boundary. Interpret ",
    "the experiment as sensitivity evidence rather than final tuning."
  )
}

selected_predictions <- ordinal_predictions |>
  filter(near(penalty, selected_penalty))

ordinal_overall <- geographic_overall_metrics(selected_predictions)
ordinal_per_class <- geographic_class_metrics(selected_predictions)
ordinal_severe <- severe_error_summary(selected_predictions)

ordinal_confusion <- conf_mat(
  selected_predictions,
  truth = damage_grade,
  estimate = .pred_class
)

ordinal_confusion_long <- as.data.frame(ordinal_confusion$table) |>
  as_tibble() |>
  group_by(Truth) |>
  mutate(row_percent = 100 * Freq / sum(Freq)) |>
  ungroup()


# --- COMPARE WITH THE LOCKED MULTINOMIAL CANDIDATE --------------------------

metrics_dir <- here("results", "metrics")
figures_dir <- here("results", "figures")
dir.create(metrics_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

candidate_overall_path <- file.path(
  metrics_dir,
  "elastic_net_geographic_candidate_oof_metrics.csv"
)
candidate_class_path <- file.path(
  metrics_dir,
  "elastic_net_geographic_candidate_oof_per_class.csv"
)
candidate_confusion_path <- file.path(
  metrics_dir,
  "elastic_net_geographic_candidate_confusion.csv"
)
candidate_fold_path <- file.path(
  metrics_dir,
  "elastic_net_geographic_candidate_fold_metrics.csv"
)

comparison_files <- c(
  candidate_overall_path,
  candidate_class_path,
  candidate_confusion_path,
  candidate_fold_path
)

if (!all(file.exists(comparison_files))) {
  stop(
    "Run models/mancuso_geographic_elastic_net_candidate.R before the ordinal ",
    "comparison so its training-only metrics are available."
  )
}

candidate_overall <- read_csv(candidate_overall_path, show_col_types = FALSE)
candidate_per_class <- read_csv(candidate_class_path, show_col_types = FALSE)
candidate_confusion <- read_csv(candidate_confusion_path, show_col_types = FALSE)
candidate_fold_metrics <- read_csv(candidate_fold_path, show_col_types = FALSE)

candidate_severe <- candidate_confusion |>
  summarise(
    observations = sum(Freq),
    low_to_high = sum(Freq[Truth == "low" & Prediction == "high"]),
    high_to_low = sum(Freq[Truth == "high" & Prediction == "low"])
  ) |>
  mutate(
    severe_errors = low_to_high + high_to_low,
    severe_error_rate = severe_errors / observations
  )

overall_comparison <- bind_rows(
  candidate_overall |>
    mutate(model = "Multinomial candidate"),
  ordinal_overall |>
    mutate(model = "Ordinal sensitivity")
) |>
  select(model, everything())

class_comparison <- bind_rows(
  candidate_per_class |>
    mutate(model = "Multinomial candidate"),
  ordinal_per_class |>
    mutate(model = "Ordinal sensitivity")
) |>
  select(model, everything())

severe_comparison <- bind_rows(
  candidate_severe |>
    mutate(model = "Multinomial candidate"),
  ordinal_severe |>
    mutate(model = "Ordinal sensitivity")
) |>
  select(model, everything())

fold_qwk_comparison <- candidate_fold_metrics |>
  filter(.metric == "qwk") |>
  select(fold, multinomial_qwk = .estimate) |>
  inner_join(
    ordinal_fold_metrics |>
      filter(.metric == "qwk", near(penalty, selected_penalty)) |>
      select(fold, ordinal_qwk = .estimate),
    by = "fold"
  ) |>
  mutate(qwk_difference = ordinal_qwk - multinomial_qwk)

paired_qwk_summary <- fold_qwk_comparison |>
  summarise(
    mean_multinomial_qwk = mean(multinomial_qwk),
    mean_ordinal_qwk = mean(ordinal_qwk),
    mean_qwk_difference = mean(qwk_difference),
    folds_improved = sum(qwk_difference > 0),
    folds_total = n()
  )


# --- SAVE TRAINING-ONLY OUTPUTS ---------------------------------------------

parameter_table <- tibble(
  model_family = "parallel cumulative logit",
  weight_power = experiment_parameters$weight_power,
  smoothing_strength = experiment_parameters$smoothing_strength,
  inner_encoding_folds = experiment_parameters$inner_encoding_folds,
  outer_validation_folds = nrow(folds),
  mixture = experiment_parameters$mixture,
  selected_penalty = selected_penalty,
  holdout_accessed = FALSE,
  holdout_evaluated = FALSE
)

write_csv(
  parameter_table,
  file.path(metrics_dir, "elastic_net_ordinal_parameters.csv")
)
write_csv(
  ordinal_qwk_summary,
  file.path(metrics_dir, "elastic_net_ordinal_tuning_qwk.csv")
)
write_csv(
  selected_ordinal,
  file.path(metrics_dir, "elastic_net_ordinal_selected.csv")
)
write_csv(
  ordinal_fold_metrics,
  file.path(metrics_dir, "elastic_net_ordinal_fold_metrics.csv")
)
write_csv(
  ordinal_overall,
  file.path(metrics_dir, "elastic_net_ordinal_oof_metrics.csv")
)
write_csv(
  ordinal_per_class,
  file.path(metrics_dir, "elastic_net_ordinal_oof_per_class.csv")
)
write_csv(
  ordinal_confusion_long,
  file.path(metrics_dir, "elastic_net_ordinal_confusion.csv")
)
write_csv(
  overall_comparison,
  file.path(metrics_dir, "elastic_net_ordinal_model_comparison.csv")
)
write_csv(
  class_comparison,
  file.path(metrics_dir, "elastic_net_ordinal_class_comparison.csv")
)
write_csv(
  severe_comparison,
  file.path(metrics_dir, "elastic_net_ordinal_severe_error_comparison.csv")
)
write_csv(
  fold_qwk_comparison,
  file.path(metrics_dir, "elastic_net_ordinal_fold_qwk_comparison.csv")
)
write_csv(
  paired_qwk_summary,
  file.path(metrics_dir, "elastic_net_ordinal_paired_qwk_summary.csv")
)

saveRDS(
  selected_predictions,
  file.path(metrics_dir, "elastic_net_ordinal_oof_predictions.rds")
)


# --- REVIEW FIGURES ----------------------------------------------------------

tuning_plot <- ordinal_qwk_summary |>
  ggplot(aes(x = penalty, y = mean_qwk)) +
  geom_errorbar(
    aes(ymin = mean_qwk - se_qwk, ymax = mean_qwk + se_qwk),
    width = 0
  ) +
  geom_line(color = "#7C3AED") +
  geom_point(size = 3, color = "#7C3AED") +
  geom_vline(
    xintercept = selected_penalty,
    linetype = "dashed",
    color = "#DC2626"
  ) +
  scale_x_log10() +
  labs(
    title = "Ordinal elastic-net penalty sensitivity",
    subtitle = "Five-fold training-only CV; error bars show +/- 1 SE",
    x = "Penalty (log scale)",
    y = "Mean quadratic weighted kappa"
  ) +
  theme_minimal(base_size = 12)

recall_plot <- class_comparison |>
  ggplot(aes(x = class, y = recall, fill = model)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Class recall: multinomial versus ordinal elastic net",
    subtitle = "Leakage-safe pooled out-of-fold training predictions",
    x = "True damage class",
    y = "Recall",
    fill = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

severe_plot <- severe_comparison |>
  ggplot(aes(x = model, y = severe_error_rate, fill = model)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  scale_y_continuous(labels = scales::label_percent(accuracy = 0.01)) +
  labs(
    title = "Severe ordinal errors",
    subtitle = "Low predicted high or high predicted low",
    x = NULL,
    y = "Share of all validation predictions"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  file.path(figures_dir, "elastic_net_ordinal_tuning_qwk.png"),
  tuning_plot,
  width = 8,
  height = 5.5,
  dpi = 300
)
ggsave(
  file.path(figures_dir, "elastic_net_ordinal_recall_comparison.png"),
  recall_plot,
  width = 8,
  height = 5.5,
  dpi = 300
)
ggsave(
  file.path(figures_dir, "elastic_net_ordinal_severe_errors.png"),
  severe_plot,
  width = 7,
  height = 5.5,
  dpi = 300
)

message("Ordinal experiment complete. Holdout accessed/evaluated: FALSE/FALSE")
message("Selected ordinal penalty: ", selected_penalty)
message(
  "Pooled ordinal QWK: ",
  round(ordinal_overall$.estimate[ordinal_overall$.metric == "qwk"], 6)
)
