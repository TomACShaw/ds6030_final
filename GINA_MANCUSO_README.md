# Gina Mancuso — Elastic-Net Model

## Purpose

This branch contains Gina's penalized multinomial logistic regression model for
predicting earthquake damage grade (`low`, `medium`, or `high`). The model uses
the elastic-net family implemented by `glmnet` through Tidymodels and now uses
the group's shared split, folds, preprocessing recipe, and metrics.

Elastic net differs from the group's unpenalized multinomial logistic-regression
baseline because it shrinks coefficients to reduce overfitting. Two parameters
are tuned:

- `penalty` controls the overall amount of coefficient shrinkage.
- `mixture` controls the kind of shrinkage: 0 is ridge, 1 is lasso, and values
  between 0 and 1 blend the two.

## Current workflow

The current shared-pipeline analysis:

1. uses an 80/20 split stratified by damage grade;
2. uses the group's five-fold stratified cross-validation object;
3. uses the same `base_recipe` used by the other group models;
4. treats `geo_level_1_id` and encoded character predictors as categorical;
5. excludes `building_id`, the high-cardinality `geo_level_2_id` and
   `geo_level_3_id`, and the redundant `has_secondary_use` umbrella flag;
6. dummy-encodes categorical predictors, removes zero-variance predictors, and
   normalizes numeric predictors;
7. tunes `penalty` and `mixture` and selects the candidate with the highest
   cross-validated quadratic weighted kappa (choosing the strongest penalty
   when candidates have exactly tied QWK values); and
8. leaves the shared 20% holdout untouched during tuning. Holdout evaluation
   is controlled by `evaluate_holdout`, which remains `FALSE` until the final
   elastic-net configuration is selected.

## Preliminary results

The best candidate in the initial 12-combination grid used
`penalty = 0.0001` and `mixture = 1` (the lasso endpoint of the elastic-net
family). Its mean five-fold cross-validated results were:

| Cross-validation metric | Estimate |
|---|---:|
| Quadratic weighted kappa | 0.481 |
| Macro-F1 | 0.590 |
| Accuracy | 0.670 |
| Balanced accuracy | 0.670 |

| Holdout metric | Estimate |
|---|---:|
| Quadratic weighted kappa | 0.478 |
| Macro-F1 | 0.585 |
| Accuracy | 0.670 |
| Balanced accuracy | 0.668 |

| Damage grade | Precision | Recall | F1 |
|---|---:|---:|---:|
| Low | 0.582 | 0.342 | 0.431 |
| Medium | 0.676 | 0.814 | 0.739 |
| High | 0.672 | 0.518 | 0.585 |

The model recognizes medium damage most often. Low-damage recall is the main
weakness, and high-damage recall is moderate. The results now use the same
validation objects and preprocessing recipe as the other group models, making
the cross-validation estimates directly comparable.

## Refined tuning results

The expanded grid evaluated 35 combinations: seven penalties from `0.000001`
through `0.001` and five mixture values from `0.25` through `1.0`. The highest
mean cross-validated QWK was `0.480969` at mixture `1.0`. Penalties from
`0.000001` through `0.0001` produced exactly the same QWK at that mixture, so
the script retains `penalty = 0.0001`, the strongest penalty on the tied
plateau. This is not evidence that `0.0001` is meaningfully better than the
smaller tied penalties; it is a parsimonious tie-breaking choice.

The refined run did not evaluate the holdout set. At the retained setting, the
mean five-fold estimates remain QWK `0.481`, macro-F1 `0.590`, accuracy `0.670`,
and balanced accuracy `0.670`. The tuning refinement therefore confirms that
the preliminary result is stable rather than producing a meaningful gain.

## Files

- Model: `models/mancuso_elastic_net.R`
- Correlation-matrix code: `eda/mancuso_correlation.R`
- Correlation-matrix figure: `results/figures/mancuso_predictor_correlation_matrix.png`
- Predictor-correlation table: `results/metrics/mancuso_predictor_correlations.csv`
- Shared-fold tuning plot: `results/figures/elastic_net_shared_tuning.png`
- Cross-validation metrics: `results/metrics/elastic_net_shared_cv_metrics.csv`
- Best parameters: `results/metrics/elastic_net_shared_best_parameters.csv`
- Holdout metrics: `results/metrics/elastic_net_shared_holdout_metrics.csv`
- Per-class metrics: `results/metrics/elastic_net_shared_per_class_metrics.csv`
- Confusion matrix: `results/metrics/elastic_net_shared_confusion_matrix.csv`
- Refined tuning plot: `results/figures/elastic_net_refined_tuning.png`
- Refined cross-validation metrics: `results/metrics/elastic_net_refined_cv_metrics.csv`
- Refined selected parameters: `results/metrics/elastic_net_refined_best_parameters.csv`

## How to run

From the repository root:

```r
source("models/mancuso_elastic_net.R")
```

The raw CSV files must be available in `data/raw/`. If the generated processed
objects are missing, the model script runs the group's shared setup and
preprocessing scripts before loading `split_data.RData`.

## Before the final model comparison

- Confirm the group's final model-selection rules and compare each model using
  the same cross-validation metrics before unlocking the holdout.
- Compare alternative treatments of the geographic IDs.
- Investigate whether class weighting or a different decision rule improves
  low- and high-damage recall.

## AI assistance disclosure

OpenAI Codex was used as a tutoring aid to explain elastic net,
structure the initial Tidymodels workflow, debug the implementation, and verify
the preliminary outputs. Gina Mancuso is responsible for reviewing,
understanding, and adapting the analysis for the group submission.
