# Gina Mancuso — Elastic-Net Model

## Purpose

This branch contains Gina's preliminary penalized multinomial logistic
regression model for predicting earthquake damage grade (`low`, `medium`, or
`high`). The model uses the elastic-net family implemented by `glmnet` through
Tidymodels.

Elastic net differs from the group's unpenalized multinomial logistic-regression
baseline because it shrinks coefficients to reduce overfitting. Two parameters
are tuned:

- `penalty` controls the overall amount of coefficient shrinkage.
- `mixture` controls the kind of shrinkage: 0 is ridge, 1 is lasso, and values
  between 0 and 1 blend the two.

## Current workflow

The starter analysis:

1. uses an 80/20 split stratified by damage grade;
2. uses three-fold stratified cross-validation for a quick preliminary screen;
3. treats `geo_level_1_id` and the encoded character predictors as categorical;
4. excludes `building_id`, the high-cardinality `geo_level_2_id` and
   `geo_level_3_id`, and the redundant `has_secondary_use` umbrella flag;
5. groups rare categorical levels, dummy-encodes categorical predictors, and
   removes zero-variance predictors;
6. tunes `penalty` and `mixture` and selects the candidate with the highest
   cross-validated quadratic weighted kappa; and
7. evaluates the selected workflow once on the untouched holdout set.

## Preliminary results

The best candidate in the quick grid used `penalty = 0.0001` and `mixture = 1`
(the lasso endpoint of the elastic-net family).

| Holdout metric | Estimate |
|---|---:|
| Quadratic weighted kappa | 0.479 |
| Micro-F1 | 0.670 |
| Accuracy | 0.670 |
| Balanced accuracy | 0.668 |

| Damage grade | Precision | Recall | F1 |
|---|---:|---:|---:|
| Low | 0.582 | 0.343 | 0.432 |
| Medium | 0.676 | 0.813 | 0.738 |
| High | 0.671 | 0.518 | 0.585 |

The model recognizes medium damage most often. Low-damage recall is the main
weakness, and high-damage recall is moderate. These are preliminary results,
not the final group comparison.

## Files

- Model: `models/mancuso_elastic_net.R`
- Cross-validation metrics: `results/metrics/mancuso_elastic_net_cv_metrics.csv`
- Holdout metrics: `results/metrics/mancuso_elastic_net_holdout_metrics.csv`
- Per-class metrics: `results/metrics/mancuso_elastic_net_per_class_metrics.csv`
- Confusion matrix: `results/metrics/mancuso_elastic_net_confusion_matrix.csv`

## How to run

From the repository root:

```r
source("shared/01_data_setup.R")
source("models/mancuso_elastic_net.R")
```

The raw CSV files must be available in `data/raw/`. They are intentionally not
tracked by Git.

## Before the final model comparison

- Replace the model-specific folds with the group's shared folds.
- Run the fuller tuning grid by setting `quick_run <- FALSE`.
- Because the current winner is on the edge of the quick penalty grid, verify
  smaller penalties rather than treating `0.0001` as final.
- Compare alternative treatments of the geographic IDs.
- Investigate whether class weighting or a different decision rule improves
  low- and high-damage recall.

## AI assistance disclosure

OpenAI Codex was used as a tutoring and coding aid to explain elastic net,
structure the initial Tidymodels workflow, debug the implementation, and verify
the preliminary outputs. Gina Mancuso is responsible for reviewing,
understanding, and adapting the analysis for the group submission.
