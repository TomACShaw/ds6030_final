# ds6030_final
A repository for the Fa26 UVA DS6030 Group 6 final project, predicting the level of damage to buildings caused by the 2015 Gorkha earthquake in Nepal. Group members are Tom Shaw, Dan Clark, Gina Mancuso, Alexander Chang, and Kristofer Miller.


## Project File Structure

```
├── .github/
│   └── pull_request_template.md    # PR submission checklist and reviewer guidelines

├── data/
│   ├── raw/                        # Immutable raw datasets (never edited, git-ignored)
│   └── processed/                  # Cleaned tables and shared resamples (.rds/.parquet)

├── R/                              # Reusable helper functions and custom modules
│   ├── utils_data.R                # Custom ingestion, formatting, and sanity checkers
│   ├── utils_recipes.R             # Preprocessing recipe blueprints
│   └── utils_eval.R                # Custom scoring, ROC plots, and calibration routines

├── scripts/                        # Numbered executable pipeline stages
│   ├── 01_data_cleaning.R          # Ingests data/raw, writes data/processed
│   ├── 02_eda.R                    # Generates standalone exploratory data figures
│   ├── 03_resampling_setup.R       # Establishes seed, train/test split, and CV folds
│   └── models/                     # Individual model tuning workflows
│       ├── tune_elastic_net.R      # Regularized regression
│       ├── tune_random_forest.R    # Random forest via ranger
│       ├── tune_xgboost.R          # Boosted trees via xgboost
│       ├── tune_svm.R              # Support vector machines via kernlab
│       └── 04_compare_models.R     # Aggregates tuning results across model families

├── results/                        # Tracked performance artifacts
│   ├── figures/                    # Exported high-resolution plots (PNG/PDF)
│   └── metrics/                    # Light CSV summaries of CV scores and test evals

├── reports/                        # Final Quarto manuscript
│   ├── final_report.qmd            # Master document compiling all sections
│   ├── references.bib              # BibTeX citation sources
│   ├── _quarto.yml                 # Quarto rendering configurations
│   └── sections/                   # Modular text and code sub-documents
│       ├── 01_introduction.qmd
│       ├── 02_eda.qmd
│       ├── 03_methodology.qmd
│       ├── 04_tuning_results.qmd
│       └── 05_discussion.qmd

├── .gitignore                      # Enforces exclusion of data, caches, and binaries
├── ds6030_final.Rproj              # Root indicator for RStudio / Positron
└── README.md                       # Repository overview and execution guide
```
