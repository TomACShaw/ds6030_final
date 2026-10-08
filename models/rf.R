# tidymodels and data management
library(tidyverse)
library(tidymodels)
library(here)

# parallelization
library(future)
library(doFuture)
library(future.callr) # More robust backend for Positron

# 1. Environment & Parallel Backend Setup ---------------------------------------
set.seed(42)
load(here("data", "processed", "split_data.RData"))

# Increase the export memory limit to 2 GB to handle the large dataset
options(future.globals.maxSize = 2000 * 1024^2)

# Dynamically detect usable cores (leaving one for OS processes)
total_cores <- max(1, parallel::detectCores(logical = FALSE) - 1)
registerDoFuture()

# We have 5 folds, so we never need more than 5 parallel workers during tuning
cv_workers <- min(5, total_cores)
worker_threads <- max(1, floor(total_cores / cv_workers))

# Use callr instead of multisession to bypass socket connection errors
plan(callr, workers = cv_workers)
message(paste("Active parallel workers:", nbrOfWorkers()))
message(sprintf("Assigned %d threads per worker.", worker_threads))


# Model Specification -------------------------------------------------------

# Define class weights based on their inverse frequencies
rf_class_weights <- train_data |>
  count(damage_grade) |>
  mutate(weight = sum(n) / n) |> 
  arrange(damage_grade) |> 
  pull(weight)

# Initialize the model
base_rf_spec <- rand_forest(
  mtry  = tune(),
  trees = 100,    
  min_n = tune()
) |> set_mode("classification")

# Inject the weights so workers receive the raw numbers
rf_spec <- rlang::inject(
  set_engine(
    base_rf_spec,
    "ranger",
    num.threads   = worker_threads,  # Dynamically allocated based on hardware
    importance    = "impurity",
    class.weights = !!rf_class_weights 
  )
)

# 4. Tidymodels Workflow -------------------------------------------------------
rf_wf <- workflow() |>
  add_recipe(base_recipe) |>
  add_model(rf_spec)

# 5. Parameter Set & Ranges ----------------------------------------------------
rf_params <- extract_parameter_set_dials(rf_wf) |>
  update(
    mtry  = mtry(range = c(5, 85)),
    min_n = min_n(range = c(5, 50))
  )

# 6. Metrics & Execution Controls ----------------------------------------------
qwk <- metric_tweak("qwk", kap, weighting = "quadratic")
f_micro <- metric_tweak("f_micro", f_meas, estimator = "micro")

custom_metrics <- metric_set(qwk, f_micro, accuracy)

# ctrl_grid <- control_grid(
#   save_pred     = TRUE,
#   parallel_over = "everything"
# )

ctrl_bayes <- control_bayes(
  save_pred     = TRUE,
  verbose       = TRUE,
  no_improve    = 10,
  seed          = 42,
  parallel_over = "resamples"
)

# 7. Space-Filling Grid Search -------------------------------------------------
# message("Executing Space-Filling Grid Search...")
# rf_grid_results <- tune_grid(
#   rf_wf,
#   resamples = folds,
#   grid      = grid_space_filling(rf_params, size = 15),
#   metrics   = custom_metrics,
#   control   = ctrl_grid
# )

# saveRDS(rf_grid_results, here("results", "metrics", "rf_grid_results.rds"))

# 8. Bayesian Optimization -----------------------------------------------------
message("Executing Bayesian Optimization...")
rf_bayes_results <- tune_bayes(
  rf_wf,
  resamples  = folds,
  param_info = rf_params,
  initial    = 5,
  iter       = 10,
  metrics    = custom_metrics,
  control    = ctrl_bayes
)

saveRDS(rf_bayes_results, here("results", "metrics", "rf_bayes_results.rds"))

# 9. Final Fit -----------------------------------------------------------------
# Clean up parallelization to maximize final training efficiency
plan(sequential)
registerDoSEQ()

best_params <- select_best(rf_bayes_results, metric = "qwk")

# Update model to 1000 trees for final fit
final_rf_spec <- rlang::inject(
  rf_spec |> 
    update(trees = 1000) |> 
    set_engine("ranger",
      num.threads   = total_cores,  # allocate all available cores for training
      importance    = "impurity",
      class.weights = !!rf_class_weights)
)

# Finalize workflow with final model specification and tuned parameters
final_rf_wf <- rf_wf |> 
  update_model(final_rf_spec) |>
  finalize_workflow(best_params)

# Manually train on full training data and evaluate on test data
final_fit_model <- fit(final_rf_wf, data = train_data)
final_preds <- augment(final_fit_model, new_data = test_data)
final_metrics   <- custom_metrics(final_preds, truth = damage_grade, estimate = .pred_class)

saveRDS(final_fit_model, here("results", "metrics", "final_rf_model.rds"))
saveRDS(final_metrics, here("results", "metrics", "final_rf_metrics.rds"))

message("Random Forest tuning pipeline complete.")