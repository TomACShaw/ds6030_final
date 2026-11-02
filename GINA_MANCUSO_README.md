# Gina Mancuso — Weighted Multinomial Regression Review

## Current candidate for group review

This branch contains a moderately class-weighted multinomial logistic
regression model for predicting earthquake damage grade (`high`, `medium`, or
`low`). The model was fit with the elastic-net implementation in `glmnet`, but
the selected penalty is so weak that its predictions are effectively those of
an almost-unpenalized weighted multinomial regression. It uses the group's
shared 80/20 split, exact five-fold cross-validation memberships,
preprocessing decisions, and evaluation metrics.

The current training-only candidate uses:

- square-root inverse-frequency class weights (`weight_power = 0.5`);
- `penalty = 0.0003` and `mixture = 1`, selected by the maximum mean CV QWK
  rule from a broad plateau of statistically indistinguishable settings; and
- quadratic weighted kappa (QWK) as the primary tuning metric.

The tuning curves are nearly flat for penalties from `0.0001` through
`0.001`. Differences among those settings are smaller than the approximately
`0.0016` standard error across folds. Therefore, `penalty = 0.0003` and
`mixture = 1` should not be interpreted as a uniquely superior lasso
configuration. A one-standard-error selection rule will be evaluated as a
training-only follow-up before the candidate is finalized.

The model has **not** been evaluated on the shared holdout. The script contains
an explicit stop if holdout evaluation is enabled during group review.

## Why moderate class weights were selected

The unweighted model predicted medium damage too often and had weak recall for
low damage. Three weighting schemes were compared using the same five folds,
preprocessing, tuning grid, and unweighted validation metrics:

| Weight scheme | High | Medium | Low |
|---|---:|---:|---:|
| Unweighted | 1.000 | 1.000 | 1.000 |
| Moderate | 0.999 | 0.765 | 1.856 |
| Fully balanced | 0.997 | 0.586 | 3.445 |

The moderate weights are the square root of fully balanced inverse-frequency
weights. They increase the influence of the less frequent low-damage class
without forcing all three classes to contribute equal total training weight.

| Weight scheme | Penalty | Mixture | QWK | Macro-F1 | Macro recall | Yardstick `bal_accuracy` | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unweighted | 0.0001 | 1.00 | 0.481 | 0.590 | 0.563 | 0.670 | 0.670 |
| **Moderate** | **0.0003** | **1.00** | **0.529** | **0.617** | 0.629 | 0.711 | 0.656 |
| Fully balanced | 0.0300 | 0.25 | 0.512 | 0.576 | **0.657** | **0.722** | 0.599 |

Among the three tested powers, moderate weighting produced the highest QWK and
macro-F1 while sacrificing 1.4 percentage points of accuracy relative to the
unweighted model. Fully balanced weighting produced the highest macro recall
but overpredicted low and high damage, reducing macro-F1 and accuracy. This
experiment supports moderate weighting over no weighting for this fitted
model; it does **not** establish that a square-root weight power is optimal.
Only three powers were tested, and threshold adjustment was not evaluated as
an alternative explanation for the improvement.

`yardstick::bal_accuracy()` is reported separately from macro recall. For a
multiclass outcome, yardstick averages the one-versus-all balanced accuracy for
each class, where each class's value is `(sensitivity + specificity) / 2`.
Macro recall is the direct average of the three class recalls.

Hyperparameter selection used the mean QWK calculated separately across the
five folds. The comparison table reports QWK after pooling all out-of-fold
predictions. QWK is nonlinear, so those two summaries need not be identical;
the selection rule is based on the mean fold-level value.

## Current out-of-fold results

These estimates pool predictions from the five assessment folds. Every row was
predicted by a model that was not trained on that row.

| Metric | Estimate |
|---|---:|
| Quadratic weighted kappa | 0.529 |
| Macro-F1 | 0.617 |
| Macro recall | 0.629 |
| Yardstick multiclass `bal_accuracy` | 0.711 |
| Accuracy | 0.656 |

| Damage grade | Precision | Recall | F1 | Observed share | Predicted share |
|---|---:|---:|---:|---:|---:|
| High | 0.623 | 0.626 | 0.624 | 33.4% | 33.6% |
| Medium | 0.711 | 0.687 | 0.699 | 56.9% | 55.0% |
| Low | 0.489 | 0.575 | 0.528 | 9.7% | 11.4% |

The predicted shares are close to the observed training shares, but this is not
proof that they match the unknown population distribution. The more important
evidence is the simultaneous improvement in QWK, macro-F1, and low- and
high-damage recall under cross-validation.

## Out-of-fold confusion matrix

Rows below are the actual damage class and percentages sum to 100% within each
row.

| Actual class | Predicted high | Predicted medium | Predicted low |
|---|---:|---:|---:|
| High | 62.6% | 36.0% | 1.4% |
| Medium | 21.9% | 68.7% | 9.4% |
| Low | 2.4% | 40.1% | 57.5% |

Most remaining errors are between adjacent classes. Extreme high-versus-low
errors remain uncommon, which is one reason QWK is useful for this ordered
outcome.

## Preprocessing and validation

The primary script:

1. loads the group's shared stratified 80/20 split;
2. preserves the group's exact five stratified cross-validation folds;
3. calculates class weights separately from each analysis fold to avoid using
   the corresponding assessment-fold outcome frequencies;
4. treats `building_id` as an identifier rather than a predictor;
5. removes `has_secondary_use`, `geo_level_2_id`, and `geo_level_3_id`;
6. retains `geo_level_1_id` as a categorical predictor;
7. handles novel categorical levels, creates dummy variables, removes
   zero-variance predictors, and normalizes numeric predictors;
8. tunes seven penalties and five mixture values using mean five-fold QWK; and
9. refits the selected setting across the same folds to create out-of-fold
   metrics, a confusion matrix, and class-specific metrics.

## Coefficient interpretation limitations

The saved coefficient table describes the fit at one weakly penalized setting
on a broad performance plateau. Equally competitive penalty and mixture
settings can retain different coefficients, so the zero-versus-nonzero pattern
should not be treated as stable feature selection. In multinomial `glmnet`,
each class receives coefficients but only differences among class scores are
identified; the relatively sparse medium-class coefficient vector can partly
reflect that parameterization rather than a substantive property of medium
damage.

Numeric predictors and dummy variables were standardized before fitting, so a
dummy-variable coefficient is expressed per standard deviation rather than as
an intuitive 0-to-1 category change. The coefficient figure remains useful for
exploratory model inspection, but not for causal or definitive feature
importance claims. The categorical codes are intentionally obfuscated and
must not be assigned unsupported physical meanings.

## Important training-only follow-ups

- The shared recipe drops `geo_level_2_id` and `geo_level_3_id`. These
  high-cardinality geographic variables may contain substantial predictive
  signal and will be investigated using fold-safe encoding without changing
  the group recipe unilaterally.
- The multinomial model treats the damage classes as nominal even though QWK
  treats them as ordered. An ordinal model or expected-grade cutpoint approach
  will be evaluated as a separate training-only comparison.
- A one-standard-error selection rule and additional class-weight powers remain
  candidate sensitivity analyses. No conclusion from these follow-ups will use
  the holdout.

## Holdout status

The moderate-weight candidate and class-weight comparison did not score the
shared holdout. However, an earlier unweighted prototype was evaluated on the
holdout before the weighting experiment, and its low-class recall and tendency
to predict medium damage were reviewed. Those observations contributed to the
decision to investigate class weighting. The holdout therefore cannot be
described as completely unseen during model development. The moderate
configuration itself was selected using cross-validation, but any later
holdout result must be reported as evaluation on a previously viewed holdout.

## Files for review

- Primary model: `models/mancuso_elastic_net.R`
- Controlled weight comparison: `models/mancuso_elastic_net_weight_comparison.R`
- Weight-comparison figure: `results/figures/elastic_net_weight_comparison_qwk.png`
- QWK tuning figure: `results/figures/elastic_net_candidate_tuning.png`
- Out-of-fold confusion figure: `results/figures/elastic_net_candidate_confusion.png`
- Coefficient figure: `results/figures/elastic_net_candidate_coefficients.png`
- Selected parameters: `results/metrics/elastic_net_candidate_best_parameters.csv`
- Class weights: `results/metrics/elastic_net_candidate_class_weights.csv`
- Overall out-of-fold metrics: `results/metrics/elastic_net_candidate_oof_metrics.csv`
- Class-specific metrics: `results/metrics/elastic_net_candidate_oof_per_class.csv`
- Confusion matrix data: `results/metrics/elastic_net_candidate_oof_confusion.csv`
- All selected-model coefficients: `results/metrics/elastic_net_candidate_coefficients.csv`
- Full tuning results: `results/metrics/elastic_net_candidate_tuning_metrics.csv`
- Weight-comparison parameters: `results/metrics/elastic_net_weight_comparison_best.csv`
- Weight-comparison overall metrics: `results/metrics/elastic_net_weight_comparison_overall.csv`
- Weight-comparison class metrics: `results/metrics/elastic_net_weight_comparison_per_class.csv`
- Weight-comparison confusion matrices: `results/metrics/elastic_net_weight_comparison_confusion.csv`
- Weight definitions: `results/metrics/elastic_net_weight_comparison_weights.csv`
- Weight-comparison tuning results: `results/metrics/elastic_net_weight_comparison_tuning.csv`

## Questions for group review

1. Are all model families using the same split, folds, preprocessing decisions,
   outcome ordering, and metric definitions?
2. Does the group agree that QWK should remain the primary selection metric,
   with macro-F1, macro recall, yardstick multiclass balanced accuracy, and
   accuracy as supporting evidence?
3. Should the group test fold-safe encoding of `geo_level_2_id` and
   `geo_level_3_id` in every model family?
4. Are any additional training-only experiments necessary before freezing the
   elastic-net candidate?
5. Given the earlier exploratory unweighted holdout evaluation, how should the
   group document and conduct the final evaluation?

## How to run

From the repository root:

```r
source("models/mancuso_elastic_net.R")
```

The raw CSV files must be available in `data/raw/`. If the generated processed
objects are missing, the script runs the group's shared data setup and
preprocessing scripts before loading the shared split.

## AI assistance disclosure

OpenAI Codex was used as a tutoring aid to explain elastic net, structure and
debug the Tidymodels workflow, verify the outputs, and improve documentation.
Gina Mancuso is responsible for reviewing, understanding, and adapting the
analysis for the group submission.
