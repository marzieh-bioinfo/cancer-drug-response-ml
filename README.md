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

## Baseline Experiment (Summary)

Using ~242,000 drug–cell line pairs from the GDSC summary table, I trained baseline models to predict **LN_IC50** from simple features (`COSMIC_ID`, `DRUG_ID`, `AUC`, `Z_SCORE`).  
A ridge regression model explained ~64% of the variance (R² ≈ 0.64), while a random forest achieved **R² ≈ 0.99** and **RMSE ≈ 0.30**, indicating that non-linear models can almost fully capture the relationship between these summary descriptors and drug sensitivity.

---

## Current Results

Using 242,035 drug–cell line pairs from the GDSC summary table, I trained baseline models
to predict drug sensitivity (**LN_IC50**) from simple features (`COSMIC_ID`, `DRUG_ID`, `AUC`, `Z_SCORE`).

**Pipeline:**

- `script/1_explore_data.py`  
  Exploratory data analysis (EDA) and summary of the dataset.  
  Output: `results/eda/eda_summary.txt`.

- `script/2_prepare_dataset.py`  
  Builds a clean modelling table.  
  Output: `results/processed/model_data.csv`.

- `script/3_train_baseline_model.py`  
  Trains baseline models (ridge regression and random forest) and evaluates performance.  
  Outputs: metrics and plots in `results/models/`.

**Baseline performance (test set):**

| Model         | RMSE |  MAE |   R² |
|---------------|-----:|-----:|-----:|
| Ridge         | 1.67 | 1.15 | 0.64 |
| Random forest | 0.31 | 0.11 | 0.99 |

The random forest model explains ~99% of the variance in **LN_IC50**, showing that
even simple summary features contain strong signal about drug response.  
Generated plots and metrics are stored in `results/models/`.
