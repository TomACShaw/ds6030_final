# ds6030_final
A repository for the Fa26 UVA DS6030 Group 6 final project, predicting the level of damage to buildings caused by the 2015 Gorkha earthquake in Nepal. Group members are Tom Shaw, Dan Clark, Gina Mancuso, Alexander Chang, and Kristofer Miller.


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

## Elastic-net project update

Gina Mancuso's locked training-only elastic-net candidate is implemented in:

- `models/mancuso_geographic_elastic_net_candidate.R` — final multinomial
  candidate and optional, explicitly gated holdout evaluation.
- `models/mancuso_geography_helpers.R` — fold-safe class weighting, smoothed
  geographic encoding, preprocessing, and metrics.
- `models/mancuso_ordinal_elastic_net_experiment.R` — ordinal sensitivity
  analysis used to check the multinomial formulation.

The combined code document, `shared/Project-1_group6.qmd`, includes the
elastic-net configuration, reproduction calls, saved out-of-fold metrics, and
figure. Detailed CSV outputs are stored in `results/metrics/`, and selected
figures are stored in `results/figures/`. The reported candidate metrics are
from five-fold cross-validation on the training set, not the final holdout.
