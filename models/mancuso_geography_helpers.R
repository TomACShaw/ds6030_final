# Reusable helpers for Gina Mancuso's geographic elastic-net candidate.
#
# These functions implement moderate case weighting, hierarchical geographic
# likelihood encoding, cross-fitting, model preprocessing, and diagnostics.
# All outcome-based geographic features must be constructed from an explicit
# reference data set so that assessment or holdout outcomes are never used.

add_power_weights <- function(data, reference_data, power) {
  weight_lookup <- reference_data |>
    count(damage_grade, name = "class_n") |>
    mutate(
      balanced_weight = nrow(reference_data) /
        (n_distinct(reference_data$damage_grade) * class_n),
      case_weight = balanced_weight^power
    )

  weight_values <- weight_lookup$case_weight[
    match(data$damage_grade, weight_lookup$damage_grade)
  ]

  data |>
    mutate(class_weights = importance_weights(weight_values))
}


class_count_table <- function(data, group_variables) {
  data |>
    mutate(.damage_class = as.character(damage_grade)) |>
    count(across(all_of(group_variables)), .damage_class) |>
    pivot_wider(
      names_from = .damage_class,
      values_from = n,
      names_prefix = "count_",
      values_fill = 0
    ) |>
    mutate(
      count_low = coalesce(count_low, 0L),
      count_medium = coalesce(count_medium, 0L),
      count_high = coalesce(count_high, 0L),
      observations = count_low + count_medium + count_high
    )
}


build_geography_maps <- function(reference_data, prior_n) {
  global_probabilities <- tibble(
    probability_low = mean(as.character(reference_data$damage_grade) == "low"),
    probability_medium = mean(
      as.character(reference_data$damage_grade) == "medium"
    ),
    probability_high = mean(
      as.character(reference_data$damage_grade) == "high"
    )
  )

  level_1_map <- class_count_table(reference_data, "geo_level_1_id") |>
    mutate(
      probability_low = (
        count_low + prior_n * global_probabilities$probability_low
      ) / (observations + prior_n),
      probability_medium = (
        count_medium + prior_n * global_probabilities$probability_medium
      ) / (observations + prior_n),
      probability_high = (
        count_high + prior_n * global_probabilities$probability_high
      ) / (observations + prior_n)
    ) |>
    select(geo_level_1_id, starts_with("probability_"))

  level_2_parent <- reference_data |>
    distinct(geo_level_2_id, geo_level_1_id)

  if (anyDuplicated(level_2_parent$geo_level_2_id)) {
    stop("A level-2 ID maps to multiple level-1 parents.")
  }

  level_2_map <- class_count_table(reference_data, "geo_level_2_id") |>
    left_join(level_2_parent, by = "geo_level_2_id") |>
    left_join(
      level_1_map |>
        rename(
          parent_probability_low = probability_low,
          parent_probability_medium = probability_medium,
          parent_probability_high = probability_high
        ),
      by = "geo_level_1_id"
    ) |>
    mutate(
      probability_low = (
        count_low + prior_n * parent_probability_low
      ) / (observations + prior_n),
      probability_medium = (
        count_medium + prior_n * parent_probability_medium
      ) / (observations + prior_n),
      probability_high = (
        count_high + prior_n * parent_probability_high
      ) / (observations + prior_n)
    ) |>
    select(
      geo_level_2_id,
      geo_level_1_id,
      starts_with("probability_")
    )

  level_3_parent <- reference_data |>
    distinct(geo_level_3_id, geo_level_2_id, geo_level_1_id)

  if (anyDuplicated(level_3_parent$geo_level_3_id)) {
    stop("A level-3 ID maps to multiple geographic parents.")
  }

  level_3_map <- class_count_table(reference_data, "geo_level_3_id") |>
    left_join(level_3_parent, by = "geo_level_3_id") |>
    left_join(
      level_2_map |>
        select(geo_level_2_id, starts_with("probability_")) |>
        rename(
          parent_probability_low = probability_low,
          parent_probability_medium = probability_medium,
          parent_probability_high = probability_high
        ),
      by = "geo_level_2_id"
    ) |>
    mutate(
      probability_low = (
        count_low + prior_n * parent_probability_low
      ) / (observations + prior_n),
      probability_medium = (
        count_medium + prior_n * parent_probability_medium
      ) / (observations + prior_n),
      probability_high = (
        count_high + prior_n * parent_probability_high
      ) / (observations + prior_n)
    ) |>
    select(
      geo_level_3_id,
      geo_level_2_id,
      geo_level_1_id,
      starts_with("probability_")
    )

  list(
    global = global_probabilities,
    level_1 = level_1_map,
    level_2 = level_2_map,
    level_3 = level_3_map,
    prior_n = prior_n
  )
}


apply_geography_maps <- function(new_data, maps) {
  new_data |>
    left_join(
      maps$level_1 |>
        rename(
          geo1_probability_low = probability_low,
          geo1_probability_medium = probability_medium,
          geo1_probability_high = probability_high
        ),
      by = "geo_level_1_id"
    ) |>
    mutate(
      geo1_probability_low = coalesce(
        geo1_probability_low,
        maps$global$probability_low
      ),
      geo1_probability_medium = coalesce(
        geo1_probability_medium,
        maps$global$probability_medium
      ),
      geo1_probability_high = coalesce(
        geo1_probability_high,
        maps$global$probability_high
      )
    ) |>
    left_join(
      maps$level_2 |>
        select(geo_level_2_id, starts_with("probability_")) |>
        rename(
          geo2_probability_low = probability_low,
          geo2_probability_medium = probability_medium,
          geo2_probability_high = probability_high
        ),
      by = "geo_level_2_id"
    ) |>
    mutate(
      geo2_probability_low = coalesce(
        geo2_probability_low,
        geo1_probability_low
      ),
      geo2_probability_medium = coalesce(
        geo2_probability_medium,
        geo1_probability_medium
      ),
      geo2_probability_high = coalesce(
        geo2_probability_high,
        geo1_probability_high
      )
    ) |>
    left_join(
      maps$level_3 |>
        select(geo_level_3_id, starts_with("probability_")) |>
        rename(
          geo3_probability_low = probability_low,
          geo3_probability_medium = probability_medium,
          geo3_probability_high = probability_high
        ),
      by = "geo_level_3_id"
    ) |>
    mutate(
      geo3_probability_low = coalesce(
        geo3_probability_low,
        geo2_probability_low
      ),
      geo3_probability_medium = coalesce(
        geo3_probability_medium,
        geo2_probability_medium
      ),
      geo3_probability_high = coalesce(
        geo3_probability_high,
        geo2_probability_high
      )
    ) |>
    # Medium probability is implied by low + medium + high = 1, so retaining
    # only low and high avoids exact linear dependence.
    select(
      -starts_with("geo1_probability_"),
      -geo2_probability_medium,
      -geo3_probability_medium
    )
}


crossfit_geography <- function(
  analysis_data,
  prior_n,
  v,
  seed
) {
  indexed_analysis <- analysis_data |>
    mutate(.geography_row_id = row_number())

  set.seed(seed)
  encoding_folds <- vfold_cv(
    indexed_analysis,
    v = v,
    strata = damage_grade
  )

  crossfit_features <- map_dfr(
    encoding_folds$splits,
    function(encoding_split) {
      encoding_reference <- analysis(encoding_split)
      encoding_assessment <- assessment(encoding_split)
      encoding_maps <- build_geography_maps(
        encoding_reference,
        prior_n = prior_n
      )

      apply_geography_maps(encoding_assessment, encoding_maps) |>
        select(
          .geography_row_id,
          matches("^geo[23]_probability_(low|high)$")
        )
    }
  )

  if (
    nrow(crossfit_features) != nrow(indexed_analysis) ||
      anyDuplicated(crossfit_features$.geography_row_id)
  ) {
    stop("Cross-fitting did not return exactly one row per analysis building.")
  }

  encoded_analysis <- indexed_analysis |>
    left_join(crossfit_features, by = ".geography_row_id") |>
    select(-.geography_row_id)

  encoded_columns <- c(
    "geo2_probability_low",
    "geo2_probability_high",
    "geo3_probability_low",
    "geo3_probability_high"
  )

  if (anyNA(encoded_analysis[encoded_columns])) {
    stop("Cross-fitted geographic features contain missing values.")
  }

  encoded_analysis
}


make_geographic_recipe <- function(weighted_analysis_data) {
  recipe(damage_grade ~ ., data = weighted_analysis_data) |>
    update_role(building_id, new_role = "id") |>
    step_rm(has_secondary_use, geo_level_2_id, geo_level_3_id) |>
    step_novel(all_nominal_predictors()) |>
    step_dummy(all_nominal_predictors(), one_hot = FALSE) |>
    step_zv(all_predictors()) |>
    step_normalize(all_numeric_predictors())
}


geographic_class_metrics <- function(predictions) {
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


geographic_overall_metrics <- function(predictions) {
  qwk <- metric_tweak("qwk", kap, weighting = "quadratic")
  f_macro <- metric_tweak("f_macro", f_meas, estimator = "macro")
  metric_functions <- metric_set(qwk, f_macro, bal_accuracy, accuracy)
  per_class <- geographic_class_metrics(predictions)

  metric_functions(
    predictions,
    truth = damage_grade,
    estimate = .pred_class
  ) |>
    bind_rows(
      tibble(
        .metric = "macro_recall",
        .estimator = "macro",
        .estimate = mean(per_class$recall)
      )
    )
}
