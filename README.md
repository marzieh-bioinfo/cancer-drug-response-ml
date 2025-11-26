# Cancer Drug Response ML

This project is about **predicting how cancer cells respond to drugs** using **pharmacogenomics data** and **machine learning**.

## Goals

- Use a real public dataset (drug response + molecular descriptors, with the option to extend to gene expression).
- Clean and explore the data (EDA).
- Train machine learning models to predict drug sensitivity.
- Evaluate the models and understand which features are most informative.

## Tools

- Python
- pandas, numpy, scikit-learn
- matplotlib / seaborn

## Data

Public pharmacogenomics data from Kaggle:  
**“Genomics of Drug Sensitivity in Cancer (GDSC)”** – downloaded as a zip and used locally  
(drug response and molecular features for many cancer cell lines and drugs).

This project demonstrates my skills in **bioinformatics, data analysis, and machine learning** on real biological data.

---

## Project Workflow

Using 242,035 drug–cell line pairs from the GDSC summary table, I organise the analysis as a small, reproducible pipeline:

- `script/1_explore_data.py`  
  Exploratory data analysis (EDA) and summary of the dataset.  
  Output: `results/eda/eda_summary.txt`.

- `script/2_prepare_dataset.py`  
  Builds a clean modelling table, selecting relevant columns and handling missing values.  
  Output: `results/processed/model_data.csv`.

- `script/3_train_baseline_model.py`  
  Trains baseline models (ridge regression and random forest) to predict **LN_IC50**  
  from simple identifier features and evaluates performance.  
  Outputs: metrics and plots in `results/models/`.

---

## Baseline experiment: non-leaky features

In this baseline, I deliberately use only simple identifier features (`DRUG_ID`, `COSMIC_ID`) to predict **LN_IC50**, and exclude AUC and Z-score to avoid target leakage from the dose–response curve.

**Test performance:**

| Model         |  RMSE |  MAE |   R² |
|---------------|------:|-----:|-----:|
| Ridge         |  2.72 | 2.08 | 0.03 |
| Random forest |  1.61 | 1.20 | 0.66 |

The linear ridge model explains almost none of the variance (R² ≈ 0.03), while the non-linear
random forest still captures about 66% of the variance in **LN_IC50** (R² ≈ 0.66). This indicates that:

1. drug response is strongly structured across drug–cell line combinations, and  
2. meaningful signal is present even when not using direct dose–response summaries such as AUC or Z-score.

Generated metrics and prediction plots are stored in `results/models/`.
