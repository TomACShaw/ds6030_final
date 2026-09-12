library(tidyverse)
library(tidymodels)
library(here)

# --- LOAD AND JOIN CSVs -------------------------------------------------------

# Read both CSV files
values_raw <- read_csv(here("data", "raw", "train_values.csv"),
  show_col_types = FALSE)

labels_raw <- read_csv(here("data", "raw", "train_labels.csv"),
  show_col_types = FALSE)

# Join as tibble
data <- values_raw |>
  inner_join(labels_raw, by = "building_id", relationship = "one-to-one") |>
  as_tibble()

# Throws an error if any rows are dropped
stopifnot(nrow(data) == nrow(values_raw))

# Save to "data/processed" for further processing and modeling
saveRDS(data, here("data", "processed", "earthquake.rds"))

#test comments