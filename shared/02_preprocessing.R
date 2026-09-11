data <- readRDS(here("data", "processed", "earthquake.rds"))

# Convert outcome to a labeled categorical factor
data <- data |>
  mutate(
    damage_grade = factor(
      damage_grade, 
      levels = c(3, 2, 1), 
      labels = c("high", "medium", "low")
    )
  )

# Convert other categoricals to factors
cat_cols <- c(
  "land_surface_condition",
  "foundation_type",
  "roof_type",
  "ground_floor_type",
  "other_floor_type",
  "position",
  "plan_configuration",
  "legal_ownership_status"
)

data <- data |> mutate(across(all_of(cat_cols), as.factor))