# 🌊 FloodSense: Regional Flood Risk Assessment & Prediction

**Academic Case Study 74: Flood Risk Analysis**  
*Predicting Regional Flood Probability and Risk Tiers from Hydro-Environmental and Infrastructure Indicators*

---

## 📌 Project Overview

**FloodSense** is an academic machine learning decision-support system designed to assist regional planning authorities and disaster response teams. It addresses two core tasks:
1. **Primary Task (Regression):** Continuous prediction of regional flood probability (`FloodProbability`, range $\approx 0.285$ to $0.725$).
2. **Derived Task (3-Class Risk Tier Classification):** Quantile-based classification into actionable municipal tiers: **Low**, **Medium**, and **High Risk**.

### Key Statistical Results (Unseen 20,000-Sample Test Set)
- **Coefficient of Determination ($R^2$):** **`0.87092`** (explains 87.1% of flood likelihood variance)
- **Root Mean Squared Error (RMSE):** **`0.01845`** (sub-2% error standard deviation)
- **Mean Absolute Error (MAE):** **`0.01441`**
- **High-Risk Class Recall:** **`90.51%`** (**0** true high-risk regions misclassified as Low Risk)
- **Macro-Averaged F1-Score:** **`0.73299`**

---

## 📂 Project Directory Structure

```text
flood-risk/
├── data/
│   ├── raw/                 # Original CSVs (train.csv, flood.csv, test.csv)
│   └── processed/           # 100k verified representative sample (sample_100k.csv)
├── notebooks/
│   ├── 01_eda.ipynb         # Full Exploratory Data Analysis & visual observations
│   └── 02_modeling.ipynb    # Model experiments, CV comparisons, and scaling benchmarks
├── src/
│   ├── preprocess.py        # Splitting, pipeline creation, and domain feature engineering
│   ├── train.py             # 5-fold CV training across 13 models, tuning, and serialization
│   └── evaluate.py          # Holdout test evaluation, confusion matrix, and permutation importance
├── models/
│   ├── final_model.joblib   # Self-contained trained pipeline with internal feature engineering
│   ├── tier_cutoffs.json    # Quantile thresholds (Low <= 0.48 < Medium <= 0.525 < High)
│   └── metrics.json         # Exact test regression, classification, and error statistics
├── reports/
│   ├── figures/             # Diagnostic plots (PNG format)
│   └── results/             # model_comparison.csv (13 experimental configurations)
├── app/
│   └── streamlit_app.py     # Interactive decision-support Streamlit web application
├── docs/
│   ├── notes.md             # Complete project running notes and report reference
│   └── viva_prep.md         # 15 academic viva voce questions and technical answers
├── requirements.txt         # Pinned Python package dependencies
└── README.md                # Project documentation and execution instructions
```

---

## 🚀 Step-by-Step Reproduction Guide

Follow this exact sequential order to reproduce all results from scratch:

### 1. Environment Setup & Dependency Installation
Create and activate an isolated Python 3.12 virtual environment, then install dependencies:
```bash
# Navigate to the project directory
cd flood-risk

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate       # On macOS / Linux
# venv\Scripts\activate        # On Windows

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Exploratory Data Analysis (EDA)
Inspect the interactive Jupyter Notebook or view generated plots in `reports/figures/`:
```bash
# Launch Jupyter Notebook
jupyter notebook notebooks/01_eda.ipynb
```
*Visualizations produced:* `target_distribution.png`, `indicator_distributions.png`, `correlation_heatmap.png`, `top_correlations.png`, `mean_target_by_score.png`, and `outliers_boxplot.png`.

### 3. Model Training & Cross-Validation
Run the comprehensive training script to evaluate all 13 model configurations with 5-fold cross-validation and save the final pipeline:
```bash
PYTHONPATH=. python src/train.py
```
*Outputs generated:* `reports/results/model_comparison.csv`, `models/final_model.joblib`, and `models/tier_cutoffs.json`.

### 4. Holdout Evaluation & Diagnostics
Execute the diagnostic evaluation on the unseen 20,000-sample test set:
```bash
PYTHONPATH=. python src/evaluate.py
```
*Outputs generated:* `models/metrics.json`, `confusion_matrix.png`, `predicted_vs_actual.png`, `residuals_vs_predicted.png`, `residual_histogram.png`, and `permutation_importance.png`.

### 5. Launch the Interactive Decision Support Dashboard
Run the Streamlit web application:
```bash
PYTHONPATH=. streamlit run app/streamlit_app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser to interact with the dashboard:
- **Tab 1 (Regional Risk Assessment):** Adjust the 20 environmental and governance sliders to obtain real-time flood probability and color-coded risk tier recommendations.
- **Tab 2 (Model Performance & Audit):** Review KPI metrics, the model comparison table, and diagnostic plots.

---

## 📊 Summary of Experimental Model Comparison

| Model | Category | Scaled | Features | CV $R^2$ Mean | Test $R^2$ | Test RMSE | Test MAE |
|---|---|---|---|---|---|---|---|
| **HistGradientBoosting (20 + 3 Aggregates)** | **Feature Engineering** | **False** | **23 (Eng.)** | **0.86503** | **0.87092** | **0.01845** | **0.01441** |
| MLP Neural Net (Scaled, 32-16 layers) | Neural Network | True | 20 (Orig.) | 0.84457 | 0.85215 | 0.01974 | 0.01554 |
| Linear Regression (Scaled, 20 Feat) | Baseline | True | 20 (Orig.) | 0.84421 | 0.84958 | 0.01991 | 0.01570 |
| Ridge Regression (Scaled, $\alpha=1.0$) | Regularization | True | 20 (Orig.) | 0.84421 | 0.84958 | 0.01991 | 0.01570 |
| Tuned Ridge ($\alpha=0.207$) | Hyperparameter Tuning | True | 20 (Orig.) | 0.84421 | 0.84958 | 0.01991 | 0.01570 |
| Tuned HistGradientBoosting | Hyperparameter Tuning | False | 20 (Orig.) | 0.79760 | 0.80573 | 0.02263 | 0.01809 |
| HistGradientBoosting (Default, 20 Feat) | Ensemble Non-Linear | False | 20 (Orig.) | 0.75598 | 0.76422 | 0.02493 | 0.02039 |
| Random Forest (50 Trees, max_depth=10) | Ensemble Non-Linear | False | 20 (Orig.) | 0.43933 | 0.43646 | 0.03854 | 0.03150 |

---

## 📖 Report & Viva Voce References
- **Project Notes & Report Reference:** [docs/notes.md](docs/notes.md) contains structured sections matching academic report standards: problem definition, dataset inventory, EDA, preprocessing, model development, evaluation, limitations, and dashboard design.
- **Viva Voce Defense Guide:** [docs/viva_prep.md](docs/viva_prep.md) contains 15 curated oral examination questions covering data leakage prevention, tree-vs-linear behavior, quantile cutoffs, and real-world hydrology extensions.
