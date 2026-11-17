library(tidymodels)
library(tidyverse)
library(finetune) 
library(furrr)
library(here)

plan(multisession, workers = parallel::detectCores(logical = FALSE))
options(future.globals.maxSize = 3000 * 1024^2)


# Settings

tune_prop <- 0.1        # share of training data used for tuning
final_fit_prop <- 1     # share of training data used for the final fit
use_cache <- TRUE       # cache fitted results as .rds files in results/metrics
cache_dir <- here("results", "metrics")
output_dir <- here("results", "metrics", "svm")


# Helpers

cached <- function(file, expr) {
  path <- file.path(cache_dir, file)
  if (use_cache && file.exists(path)) return(readRDS(path))
  result <- expr
  if (use_cache) saveRDS(result, path)
  result
}

# kernlab cannot combine class.weights with prob.model = TRUE.
# Weighted probabilities are made from the unweighted probability

reweight_probs <- function(preds, weights) {
  p <- sweep(as.matrix(preds[prob_cols]), 2, weights[lv], `*`)
  p <- p / rowSums(p)
  preds[prob_cols] <- as.data.frame(p)
  preds$.pred_class <- factor(lv[max.col(p, ties.method = "first")],
    levels = lv)
  preds
}

# kernlab bug: when its Platt line search stops early it prints
# "line search fails A B ..." and stores a vector instead of list(A, B) for
# that class pair, which breaks predict().

fit_with_platt_repair <- function(wf, data) {
  log <- capture.output(fitted <- fit(wf, data))
  pm <- extract_fit_engine(fitted)@prob.model
  broken <- which(!map_lgl(pm, is.list))
  if (length(broken) == 0) return(fitted)

  fails <- str_match_all(paste(log, collapse = " "),
    "line search fails\\s+(\\S+)\\s+(\\S+)")[[1]]
  stopifnot(nrow(fails) == length(broken))
  for (k in seq_along(broken)) {
    pm[[broken[k]]] <- list(A = as.numeric(fails[k, 2]),
      B = as.numeric(fails[k, 3]))
  }
  fitted$fit$fit$fit@prob.model <- pm
  message("Repaired Platt parameters for class pair(s): ",
    paste(broken, collapse = ", "))
  fitted
}

# Precision, recall and F1 for each class (one class vs the rest).
per_class_table <- function(preds) {
  map_dfr(lv, \(cl) {
    truth <- factor(preds$damage_grade == cl, levels = c(TRUE, FALSE))
    est <- factor(preds$.pred_class == cl, levels = c(TRUE, FALSE))
    tibble(class = cl,
      precision = precision_vec(truth, est),
      recall = recall_vec(truth, est),
      f1 = f_meas_vec(truth, est))
  })
}


# Data

if (!file.exists(here("data", "processed", "earthquake.rds"))) {
  source(here("shared", "01_data_setup.R"))
}
split_path <- here("data", "processed", "split_data.RData")
if (!file.exists(split_path)) {
  source(here("shared", "02_preprocessing.R"))
}
load(split_path)

lv <- levels(train_data$damage_grade)
prob_cols <- paste0(".pred_", lv)

set.seed(1)
tune_data <- training(initial_split(train_data, prop = tune_prop,
  strata = damage_grade))
set.seed(1)
resamples <- vfold_cv(tune_data, v = 5, strata = damage_grade)

# Class weights by inverse class frequency.
class_counts <- count(train_data, damage_grade)
class_weights <- setNames(
  nrow(train_data) / (nrow(class_counts) * class_counts$n),
  as.character(class_counts$damage_grade))

# Metrics
qwk <- metric_tweak("qwk", kap, weighting = "quadratic")
f_macro <- metric_tweak("f_macro", f_meas, estimator = "macro")
f_micro <- metric_tweak("f_micro", f_meas, estimator = "micro")
report_metrics <- metric_set(qwk, f_macro, f_micro, bal_accuracy, accuracy,
  roc_auc)


# Models
# Tuning runs without the probability model (prob.model = FALSE), models are selected by QWK, which only needs
# class predictions.

make_svm_wf <- function(kernel, weighted, prob = FALSE) {
  spec <- switch(kernel,
    rbf = svm_rbf(mode = "classification", rbf_sigma = tune(), cost = tune()),
    linear = svm_linear(mode = "classification", cost = tune()),
    poly = svm_poly(mode = "classification", cost = tune(), degree = 2,
      scale_factor = tune()))
  engine_args <- list(prob.model = prob, cache = 1000, tol = 0.01)
  if (weighted) engine_args$class.weights <- class_weights
  spec <- do.call(set_engine, c(list(spec, "kernlab"), engine_args))
  workflow() |> add_model(spec) |> add_recipe(base_recipe)
}

make_params <- function(kernel, wf) {
  params <- extract_parameter_set_dials(wf)
  switch(kernel,
    rbf = params |>
      update(rbf_sigma = rbf_sigma(c(-4, -1)), cost = cost(c(-2, 3))),
    linear = params |> update(cost = cost(c(-2, 3))),
    poly = params |> update(cost = cost(c(-2, 3))))
}

grid_size <- c(rbf = 20, linear = 10, poly = 20)


# Tuning
# Racing evaluates a in parallel and drops candidates that
# are significantly worse after the first 3 folds.

model_names <- c("rbf_none", "rbf_weighted", "linear_none",
  "linear_weighted", "poly_none", "poly_weighted")

tune_results <- list()
for (m in model_names) {
  kernel <- str_remove(m, "_.*")
  wf <- make_svm_wf(kernel, weighted = str_detect(m, "weighted"))
  set.seed(1)
  tune_results[[m]] <- cached(paste0("svm_race_", m, "_v4.rds"),
    tune_race_anova(wf, resamples = resamples, grid = grid_size[[kernel]],
      param_info = make_params(kernel, wf), metrics = metric_set(qwk),
      control = control_race(save_pred = TRUE,
        parallel_over = "everything")))
}

# Best CV QWK of each model; the highest is selected.
qwk_by_model <- imap_dfr(tune_results, \(res, m)
  show_best(res, metric = "qwk", n = 1) |> mutate(model = m)) |>
  arrange(desc(mean))
qwk_by_model |> select(model, mean, std_err)

best_model <- qwk_by_model$model[1]
best_kernel <- str_remove(best_model, "_.*")
best_weighted <- str_detect(best_model, "weighted")
best_parameter <- select_best(tune_results[[best_model]], metric = "qwk")
param_tag <- best_parameter |>
  select(-.config) |>
  unlist() |>
  signif(3) |>
  paste(collapse = "_")


# Selected model: CV with probabilities
# Refitted with the probability model (always unweighted, see
# reweight_probs()) for ROC AUC. The raw predictions are the unweighted
# baseline; the reweighted ones are the selected model.

best_prob_wf <- make_svm_wf(best_kernel, weighted = FALSE, prob = TRUE) |>
  finalize_workflow(best_parameter)

svm_cv <- cached(
  paste0("svm_cv_best_prob_", best_kernel, "_", param_tag, "_v5.rds"),
  fit_resamples(best_prob_wf, resamples, metrics = report_metrics,
    control = control_resamples(save_pred = TRUE)))

baseline_preds <- collect_predictions(svm_cv)
cv_preds <- if (best_weighted) {
  reweight_probs(baseline_preds, class_weights)
} else {
  baseline_preds
}


# Final model and holdout predictions

set.seed(1)
final_data <- if (final_fit_prop < 1) {
  training(initial_split(train_data, prop = final_fit_prop,
    strata = damage_grade))
} else {
  train_data
}

final_fit <- cached(
  paste0("svm_final_model_", best_kernel, "_", param_tag, "_",
    final_fit_prop, "_v6.rds"),
  fit_with_platt_repair(best_prob_wf, final_data))

# Prediction cost grows with the number of support vectors, so the holdout
# set is predicted in parallel chunks.
holdout_preds <- test_data |>
  split(ceiling(seq_len(nrow(test_data)) / 5000)) |>
  future_map(\(d) augment(final_fit, new_data = d),
    .options = furrr_options(packages = c("tidymodels", "kernlab"))) |>
  bind_rows()
plan(sequential)

if (best_weighted) {
  holdout_preds <- reweight_probs(holdout_preds, class_weights)
}


# Saved outputs

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

summarise_cv <- function(preds, label) {
  preds |>
    group_by(id) |>
    report_metrics(truth = damage_grade, all_of(prob_cols),
      estimate = .pred_class) |>
    group_by(.metric) |>
    summarize(mean = mean(.estimate),
      std_err = sd(.estimate) / sqrt(n()), .groups = "drop") |>
    mutate(model = label, .before = 1)
}

selected_label <- paste0(
  if (best_weighted) "Class-weighted " else "Unweighted ",
  toupper(best_kernel), " SVM (",
  best_parameter |>
    select(-.config) |>
    imap_chr(\(v, p) paste(p, signif(v, 3))) |>
    paste(collapse = ", "),
  ")")

outputs <- list(
  cv_metrics = bind_rows(
    summarise_cv(baseline_preds, "Unweighted baseline"),
    summarise_cv(cv_preds, selected_label)),
  cv_per_class_metrics = per_class_table(cv_preds),
  best_parameters = best_parameter,
  holdout_metrics = report_metrics(holdout_preds, truth = damage_grade,
    all_of(prob_cols), estimate = .pred_class),
  per_class_metrics = per_class_table(holdout_preds))

iwalk(outputs, \(table, name) {
  print(table)
  write_csv(table, file.path(output_dir, paste0(name, ".csv")))
})

report_metrics         
summarise_cv(cv_preds, "check") 
read_csv(file.path(output_dir, "cv_metrics.csv"))