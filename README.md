# ds6030_final
A repository for the Fa26 UVA DS6030 Group 6 final project, predicting the level of damage to buildings caused by the 2015 Gorkha earthquake in Nepal. Group members are Tom Shaw, Dan Clark, Gina Mancuso, Alexander Chang, and Kristofer Miller.

## Gina Mancuso: Geographic Elastic-Net Candidate

The final training-only elastic-net candidate is implemented in
`models/mancuso_geographic_elastic_net_candidate.R`, with reusable geographic
encoding functions in `models/mancuso_geography_helpers.R`.

The candidate uses:

- moderate square-root inverse-frequency class weights;
- smoothed likelihood encodings for geographic levels 2 and 3, with geographic
  level 1 retained by the shared preprocessing pipeline;
- cross-fitted geographic encodings so a training row's outcome does not
  contribute to its own encoded predictors;
- smoothing strength 10;
- multinomial elastic-net regression with penalty 0.003 and mixture 0.50; and
- five-fold cross-validation on the training set.

### Training-only validation results

| Metric | Estimate |
|---|---:|
| Quadratic weighted kappa | 0.6125 |
| Accuracy | 0.7249 |
| Macro F1 | 0.6857 |
| Macro recall | 0.6984 |

| Damage class | Recall |
|---|---:|
| Low | 0.6536 |
| Medium | 0.7662 |
| High | 0.6752 |

These are pooled out-of-fold training estimates, not holdout results. The
candidate script keeps `evaluate_holdout <- FALSE`; the shared holdout remains
locked until the group agrees on final model selection and a common evaluation
procedure.

### Ordinal sensitivity analysis

Because the response is ordered (`low < medium < high`), an additional
training-only sensitivity analysis tested a parallel cumulative-logit elastic
net. This model uses one shared damage-risk direction and two ordered class
boundaries. It was fit with the same five validation folds, geographic
encodings, class-weighting approach, and mixture of 0.50 as the multinomial
candidate. The ordinal penalty was selected by the one-standard-error rule.

The ordinal model reduced severe low-to-high or high-to-low errors from 1,524
to 901, a 40.9% reduction. However, it also moved more low-damage buildings
into the medium class. Low-damage recall decreased from 0.6536 to 0.5896,
pooled QWK decreased from 0.6125 to 0.6019, and the ordinal model had lower QWK
in all five validation folds. Accuracy, macro F1, and macro recall also
decreased.

This experiment shows that explicitly enforcing the outcome order can prevent
some extreme classification errors, but the single shared ordinal effect is
too restrictive for the overall prediction task. The geographic multinomial
elastic net therefore remains the final candidate because it provides better
overall agreement and more balanced class-specific performance. The ordinal
experiment is retained in `models/mancuso_ordinal_elastic_net_experiment.R` as
evidence that an alternative outcome formulation was evaluated before final
model selection.

### Reproducing the candidate

First run the shared data setup and preprocessing scripts to create
`data/processed/split_data.RData`. From the project root, run:

```r
source("models/mancuso_geographic_elastic_net_candidate.R")
```

The script saves compact metric tables under `results/metrics/`, figures under
`results/figures/`, and local `.rds` prediction artifacts. The `.rds` artifacts
are reproducible and intentionally excluded from Git because they are large
binary files.

The ordinal sensitivity script additionally requires the `ordinalNet` package:

```r
install.packages("ordinalNet")
source("models/mancuso_ordinal_elastic_net_experiment.R")
```


## Project File Structure

```
└── ds6030_final/
    ├── ds6030_final.Rproj
    ├── README.md
    ├── data/
    │   ├── raw/
    │   │   ├── train_values.csv
    │   │   └── train_labels.csv
    │   └── processed/
    │       └── earthquake.rds
    ├── eda/
    │   ├── shaw_eda.qmd
    │   ├── clark_eda.qmd
    │   ├── mancuso_eda.qmd
    │   ├── chang_eda.qmd
    │   └── miller_eda.qmd
    ├── shared/
    │   ├── 01_data_setup.R
    │   ├── 02_preprocessing.R
    │   └── 03_cv_metrics.R
    ├── models/
    │   ├── model1.R
    │   ├── model2.R
    │   ├── model3.R
    │   ├── model4.R
    │   └── model5.R
    └── results/
        ├── metrics/
        ├── figures/
        └── reports/
```
