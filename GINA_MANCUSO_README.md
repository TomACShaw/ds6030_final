# Gina Mancuso — Elastic-Net Model

## Current candidate for group review

This branch contains a moderate class-weighted multinomial elastic-net model
for predicting ordered earthquake damage grade (`high`, `medium`, or `low`).
It uses the group's shared 80/20 split, exact five-fold cross-validation
memberships, preprocessing decisions, and evaluation metrics so its
cross-validation results can be compared directly with the other model
families.

The current training-only candidate uses:

- square-root inverse-frequency class weights (`weight_power = 0.5`);
- `penalty = 0.0003`;
- `mixture = 1`, making the selected elastic-net endpoint a lasso model; and
- quadratic weighted kappa (QWK) as the primary tuning metric.

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

| Weight scheme | Penalty | Mixture | QWK | Macro-F1 | Balanced accuracy | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Unweighted | 0.0001 | 1.00 | 0.481 | 0.590 | 0.670 | 0.670 |
| **Moderate** | **0.0003** | **1.00** | **0.529** | **0.617** | 0.711 | 0.656 |
| Fully balanced | 0.0300 | 0.25 | 0.512 | 0.576 | **0.722** | 0.599 |

The moderate scheme produced the highest QWK and macro-F1. It improved class
balance while sacrificing only 1.4 percentage points of overall accuracy. The
fully balanced model obtained the highest average recall but overpredicted low
and high damage, which reduced macro-F1 and accuracy.

## Current out-of-fold results

These estimates pool predictions from the five assessment folds. Every row was
predicted by a model that was not trained on that row.

| Metric | Estimate |
|---|---:|
| Quadratic weighted kappa | 0.529 |
| Macro-F1 | 0.617 |
| Balanced accuracy | 0.711 |
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

## Feature selection and interpretation

At `mixture = 1`, lasso can set coefficients exactly to zero. Excluding the
three class intercepts, the selected moderate-weight model retained:

| Damage grade | Retained coefficients | Zero coefficients |
|---|---:|---:|
| High | 62 | 24 |
| Medium | 33 | 53 |
| Low | 71 | 15 |

Geographic indicators are prominent for the high- and low-damage class scores,
while the medium class uses fewer and generally smaller coefficients. The
categorical letter codes in this competition are intentionally obfuscated, so
the analysis should describe associations with the encoded categories rather
than assigning unsupported physical meanings to them. Coefficients are also
associations, not causal effects.

## Holdout status

The moderate-weight candidate and the class-weight comparison have not scored
the shared holdout. An earlier unweighted prototype did produce preliminary
holdout metrics before the current candidate was selected. Therefore, the
holdout has not remained completely unseen over the entire project, although
the moderate model was selected from cross-validation results only. The group
should review this history and agree on how to describe and conduct final
evaluation before enabling any additional holdout run.

## Files for review

- Primary model: `models/mancuso_elastic_net.R`
- Controlled weight comparison: `models/mancuso_elastic_net_weight_comparison.R`
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

## Questions for group review

1. Are all model families using the same split, folds, preprocessing decisions,
   outcome ordering, and metric definitions?
2. Does the group agree that QWK should remain the primary selection metric,
   with macro-F1, balanced accuracy, accuracy, and class recall as supporting
   evidence?
3. Does the group agree that the moderate-weight model provides a more useful
   tradeoff than either the unweighted or fully balanced elastic net?
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
