set.seed(42)  # global random seed
library(tidymodels)
library(here)


# --- PREPROCESSING ------------------------------------------------------------

# Load the full (joined) dataset
data <- readRDS(here("data", "processed", "earthquake.rds"))

# Convert outcome to a labeled categorical factor
data <- data |>
  mutate(
    damage_grade = factor(
      damage_grade, 
      levels = c(3, 2, 1), 
      labels = c("high", "medium", "low"),
      ordered = TRUE
    )
  )

# Convert other categoricals to factors
data <- data |>
  mutate(across(c(where(is_character), geo_level_1_id), as.factor))


# --- SPLITS -------------------------------------------------------------------

eq_split <- initial_split(data, prop = 0.80, strata = damage_grade)
train_data <- training(eq_split)
test_data  <- testing(eq_split)

# 5-fold stratified CV
folds <- vfold_cv(train_data, v = 5, strata = damage_grade)


# --- RECIPE -------------------------------------------------------------------

base_recipe <- recipe(damage_grade ~ ., data = train_data) |>
  update_role(building_id, new_role = "id") |>
  step_rm(has_secondary_use, geo_level_2_id, geo_level_3_id) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors(), one_hot = FALSE) |>  # no one_hot prevents multicollinearity
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())


# --- SAVE OBJECTS -------------------------------------------------------------

# Save to "data/processed" to ensure identical data loading in each model script
save(
  base_recipe,
  folds,
  train_data,
  test_data,
  file = "data/processed/split_data.RData"
)