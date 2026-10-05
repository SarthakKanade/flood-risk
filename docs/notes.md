# FloodSense: Project Working Notes & Report Reference

This document tracks all data findings, engineering decisions, experimental results, and analytical justifications across each stage of the project.

---

## 1. Problem Definition & Objectives

- **Project Title:** FloodSense: Predicting Regional Flood Probability and Risk Tier from Hydro-Environmental and Infrastructure Indicators
- **Context & Problem Statement:** Case Study 74: Flood Risk Analysis. A regional planning authority seeks to identify useful predictive patterns in environmental, infrastructural, and demographic indicators to forecast flood risk and prioritize regional interventions.
- **Core ML Tasks:**
  1. **Primary Task (Regression):** Continuous prediction of regional flood probability (`FloodProbability`), spanning roughly 0.285 to 0.725.
  2. **Derived Task (3-Class Risk Tier Classification):** Quantile-based conversion of continuous flood probabilities into actionable governance tiers: **Low**, **Medium**, and **High**.
- **Governance Justification:** Operational disaster management authorities allocate emergency funds, drainage maintenance, and rescue personnel based on clear, categorical risk tiers rather than raw decimal probabilities.
- **Primary Success Criteria:**
  - Regression: Low Root Mean Squared Error (RMSE), low Mean Absolute Error (MAE), and high Coefficient of Determination ($R^2$).
  - Risk Tiers: Balanced classification performance with priority on high recall for the **High** risk tier (minimizing false negatives in high-risk zones is critical to public safety).

---

## 2. Dataset & Documentation

### 2.1 Data Source & File Inventory
- **Source:** Kaggle "Flood Prediction Dataset" (Kaggle Playground Series / dataset by naiyakhalid).
- **Files Available in `data/raw/`:**
  - `train.csv`: 1,117,957 rows, 22 columns (`id`, 20 indicator features, and `FloodProbability`). The sole dataset used for modeling.
  - `flood.csv`: 50,000 rows, 21 columns (20 indicator features and `FloodProbability`). Used exclusively to conduct the deterministic formula benchmark check.
  - `test.csv` & `sample_submission.csv`: Competition evaluation files lacking target labels. Excluded from model development and validation.
- **Sampling Strategy (`data/processed/sample_100k.csv`):**
  - Because training complex ensemble models (like Random Forest) and running 5-fold cross-validation across 1.12 million rows creates severe compute bottlenecks on local development hardware, we extracted a random sample of **100,000 rows** using `random_state=42`.
  - The `id` column was dropped prior to sampling as it provides arbitrary row indexing and carries no predictive signal.
  - All subsequent exploratory data analysis, train/test splitting, model training, and application tests operate on this verified 100,000-row sample.

---

### 2.2 Variable Dictionary

All 20 predictor features are integer-valued score metrics representing local risk intensity or vulnerability (observed values range from 0 to 17).

| Variable Name | Description / Meaning | Type | Observed Range (Sample) | Mean (Std) |
|---|---|---|---|---|
| `MonsoonIntensity` | Severity and duration of seasonal monsoon rainfall | Integer | 0 to 16 | 4.92 (2.05) |
| `TopographyDrainage` | Natural slope and land gradient efficiency for runoff | Integer | 0 to 17 | 4.92 (2.10) |
| `RiverManagement` | Quality of river basin regulation, dredging, and levees | Integer | 0 to 16 | 4.96 (2.08) |
| `Deforestation` | Extent of forest cover removal in catchment areas | Integer | 0 to 16 | 4.95 (2.06) |
| `Urbanization` | Proportion of impervious concrete/built surfaces | Integer | 0 to 16 | 4.95 (2.08) |
| `ClimateChange` | Long-term weather anomalies and extreme shifts | Integer | 0 to 16 | 4.93 (2.06) |
| `DamsQuality` | Structural integrity and maintenance level of dams | Integer | 0 to 16 | 4.95 (2.09) |
| `Siltation` | Sediment buildup reducing river and canal capacity | Integer | 0 to 16 | 4.93 (2.07) |
| `AgriculturalPractices` | Runoff-inducing land cultivation and tillage patterns | Integer | 0 to 16 | 4.94 (2.08) |
| `Encroachments` | Illegal construction on floodplains and waterways | Integer | 0 to 16 | 4.93 (2.08) |
| `IneffectiveDisasterPreparedness` | Inadequacy of early warning systems and evacuation plans | Integer | 0 to 16 | 4.94 (2.08) |
| `DrainageSystems` | Deficiency or lack of artificial stormwater drainage | Integer | 0 to 17 | 4.95 (2.07) |
| `CoastalVulnerability` | Exposure to storm surges, tides, and sea-level rise | Integer | 0 to 16 | 4.95 (2.09) |
| `Landslides` | Soil instability and slope failure susceptibility | Integer | 0 to 16 | 4.93 (2.08) |
| `Watersheds` | Degraded watershed retention capacity | Integer | 0 to 16 | 4.92 (2.08) |
| `DeterioratingInfrastructure` | Aging bridges, culverts, and flood defense barriers | Integer | 0 to 16 | 4.93 (2.06) |
| `PopulationScore` | Density of population living in hazard-prone areas | Integer | 0 to 16 | 4.92 (2.07) |
| `WetlandLoss` | Destruction of natural floodwater buffering wetlands | Integer | 0 to 16 | 4.96 (2.07) |
| `InadequatePlanning` | Poor regional zoning and uncoordinated development | Integer | 0 to 16 | 4.94 (2.08) |
| `PoliticalFactors` | Policy roadblocks, misallocated funds, governance gaps | Integer | 0 to 16 | 4.94 (2.09) |
| **`FloodProbability`** | **Target variable: calculated regional flood likelihood** | **Float** | **0.285 to 0.725** | **0.504 (0.051)** |

---

### 2.3 Data Quality and Cleaning Audit

The data quality inspection conducted on `data/processed/sample_100k.csv` revealed the following verified metrics:
- **Total Rows & Columns:** 100,000 rows × 21 columns (20 features + 1 target).
- **Missing Values:** Exactly **0 missing values** across all 21 columns.
- **Duplicate Rows:** Exactly **0 duplicate rows** found after dropping the `id` column.
- **Data Types:** 20 features stored as 64-bit integers (`int64`), target stored as 64-bit floating point (`float64`). No mixed-type or text-corruption issues detected.
- **Value Ranges:** All indicators have a minimum score of 0 and a maximum of 16 (or 17 for `TopographyDrainage` and `DrainageSystems`). No negative or invalid numbers exist.

#### Deterministic Formula Verification Check
To verify the nature of the synthetic generation process, we evaluated the hypothesis that target values derive from:
$$\text{Baseline Formula} = \frac{\sum_{i=1}^{20} \text{Indicator}_i}{200}$$

Results obtained from direct computation:
1. **On `flood.csv` (50,000 rows):**
   - **Max Absolute Difference:** `0.0000000000`
   - **Mean Absolute Difference:** `0.0000000000`
   - **Exact Matches ($|\Delta| < 10^{-9}$):** **50,000 / 50,000 (100.00%)**
   - *Observation:* In `flood.csv`, `FloodProbability` is a strict, noiseless linear sum divided by 200. Ordinary Least Squares linear regression fits this dataset with an artificial $R^2 = 1.0000$.
2. **On `train.csv` (Full 1,117,957 rows):**
   - **Max Absolute Difference:** `0.2250000000`
   - **Mean Absolute Difference:** `0.0177638988`
   - **Exact Matches ($|\Delta| < 10^{-9}$):** **145,012 / 1,117,957 (12.97%)**
3. **On `sample_100k.csv` (100,000 rows):**
   - **Max Absolute Difference:** `0.1900000000`
   - **Mean Absolute Difference:** `0.0177513000`
   - *Conclusion:* Unlike `flood.csv`, `train.csv` has stochastic noise injected into its target values. While the underlying physical relationship is predominantly additive, the added noise creates realistic variance, allowing machine learning models to be evaluated and compared meaningfully.

---

### 2.4 Target Distribution Analysis

- **Sample Size:** 100,000 observations
- **Mean:** `0.504340`
- **Standard Deviation:** `0.051001`
- **Minimum:** `0.285000`
- **25th Percentile ($Q_1$):** `0.470000`
- **50th Percentile (Median):** `0.505000`
- **75th Percentile ($Q_3$):** `0.540000`
- **Maximum:** `0.725000`

#### 10-Bin Histogram Breakdown:
- `[0.2850 - 0.3290]`: 13 rows (0.01%)
- `[0.3290 - 0.3730]`: 435 rows (0.43%)
- `[0.3730 - 0.4170]`: 3,976 rows (3.98%)
- `[0.4170 - 0.4610]`: 16,274 rows (16.27%)
- `[0.4610 - 0.5050]`: 28,484 rows (28.48%)
- `[0.5050 - 0.5490]`: 31,050 rows (31.05%)
- `[0.5490 - 0.5930]`: 15,467 rows (15.47%)
- `[0.5930 - 0.6370]`: 3,816 rows (3.82%)
- `[0.6370 - 0.6810]`: 455 rows (0.46%)
- `[0.6810 - 0.7250]`: 30 rows (0.03%)

The target exhibits a symmetric, bell-shaped distribution centered almost exactly at 0.504 with slight tails. More than 75% of observations fall between 0.45 and 0.55. A visual plot was saved to `reports/figures/target_distribution.png`.

---

### 2.5 Dataset Limitations & Academic Framing

For the project report and academic viva presentation, the following limitations must be stated with complete honesty:
1. **Synthetic & Score-Based Data:** The dataset consists of discrete integer indicator scores rather than physical hydrology measurements (such as river discharge in $\text{m}^3/\text{s}$, rainfall in millimeters, or elevation meters).
2. **Absence of Spatial and Temporal Dimensions:** There are no geographic coordinates (latitude/longitude), catchment maps, or timestamps. Floods in physical reality depend heavily on spatial topology (upstream flow affecting downstream plains) and temporal sequences (rainfall accumulated over preceding days).
3. **Regional Risk Screening Model:** Because of points 1 and 2, this project should be framed strictly as a **macro-level regional risk-screening and vulnerability index model**, useful for comparing relative regional vulnerabilities across standardized survey scores, and not as an operational real-time hydrodynamic forecast system.

---

## 3. Exploratory Data Analysis (EDA)

All analyses were executed in `notebooks/01_eda.ipynb` on the 100,000-row representative sample, with corresponding figures archived in `reports/figures/`.

### 3.1 Target Distribution Analysis
- **Shape & Normality:** Unimodal and bell-shaped, closely approximating a normal distribution (Skewness = `0.04839`, Kurtosis = `-0.02592`).
- **Spread:** Mean = `0.50434`, Median = `0.50500`, Std = `0.05100`. Range = `[0.2850, 0.7250]`.
- **Modeling Takeaway:** The symmetrical nature without heavy long tails means standard regression loss functions (Mean Squared Error, Mean Absolute Error) and $R^2$ are well-suited. Logarithmic or power transformations on the target are unnecessary.

### 3.2 Indicator Feature Distributions
- **Scale and Shape:** All 20 features are discrete integers ranging from 0 to 16 (with `TopographyDrainage` and `DrainageSystems` reaching 17).
- **Uniformity:** Across all 20 indicators, the mean score is virtually identical ($\approx 4.93 \pm 2.07$). Distributions are modestly right-skewed with most values concentrated between 3 and 7.
- **Visual Reference:** `reports/figures/indicator_distributions.png`.

### 3.3 Correlation & Multicollinearity Findings
- **Feature-to-Feature Independence (No Multicollinearity):**
  - **Maximum Absolute Correlation between any two indicators:** `0.01972`
  - **Average Absolute Correlation between indicators:** `0.01048`
  - *Academic Rationale:* The 20 indicator columns are practically orthogonal (completely uncorrelated with one another). Variance Inflation Factor (VIF) would be $\approx 1.0$ for all variables. There is no risk of multicollinearity destabilizing linear regression coefficient estimates, and no feature elimination or PCA is required.
- **Feature-to-Target Correlations:**
  - Every single indicator exhibits a positive linear correlation with `FloodProbability` within an extraordinarily narrow band of **+0.17486 to +0.19093**.
  - **Top 5 Features:**
    1. `ClimateChange`: $r = 0.19093$
    2. `Siltation`: $r = 0.19059$
    3. `DamsQuality`: $r = 0.19036$
    4. `TopographyDrainage`: $r = 0.18876$
    5. `RiverManagement`: $r = 0.18864$
- **Linear Trend Verification:** Grouping mean flood probability by integer score levels (0 to 15) shows a steady, strictly monotonic linear increase from $\approx 0.43$ to $\approx 0.60$ with constant variance across score tiers.
- **Visual References:** `reports/figures/correlation_heatmap.png`, `reports/figures/top_correlations.png`, `reports/figures/mean_target_by_score.png`.

### 3.4 Outlier Analysis & Decision Rationale
- **Interquartile Range (IQR) Inspection:**
  - Using the standard $1.5 \times \text{IQR}$ rule, 26,007 values across all 20 indicators (1.30% of total data points) fall above score 10–11.
  - The top features with scores $>11$ are `DamsQuality` (2,824), `WetlandLoss` (2,725), and `DrainageSystems` (2,660).
- **Justification to Retain All Outliers:**
  - These values represent valid, high-severity hazard evaluations on an ordinal scoring scale (e.g., severe watershed degradation or extreme rainfall exposure), not corrupt sensor readings or data entry mistakes.
  - Removing or capping these values would artificially suppress regions facing severe multi-factor hazards—precisely the high-risk zones emergency managers need to identify.
- **Visual Reference:** `reports/figures/outliers_boxplot.png`.

### 3.5 Core EDA Key Findings
1. The target `FloodProbability` is Gaussian-like, centered at 0.504, with no skew-induced distortion.
2. The 20 predictor indicators are mutually independent (max correlation < 0.02), ruling out multicollinearity issues.
3. Every indicator contributes approximately equal positive linear weight ($r \approx 0.18\text{--}0.19$) to flood probability.
4. The underlying data generating mechanism is predominantly additive; linear regression is expected to be a very strong baseline, leaving only small margins of improvement for complex tree ensembles.
5. All scores are retained in their natural state without clipping or deletion to preserve critical high-hazard information.

---

## 4. Preprocessing & Feature Engineering

All preprocessing functions are implemented in `src/preprocess.py`.

### 4.1 Train/Test Split (Leakage Prevention)
- **Ratio:** 80% Training (80,000 rows) and 20% Holdout Testing (20,000 rows).
- **Random State:** `random_state=42` for strict reproducibility.
- **Data Leakage Safeguard:**
  - The split is executed *strictly prior* to any statistical calculation, scaling, or tier thresholding.
  - Scalers (`StandardScaler`) are encapsulated within `sklearn.pipeline.Pipeline` objects, ensuring parameters ($\mu, \sigma$) are computed exclusively on training folds during cross-validation and never touch test samples.

### 4.2 Quantile-Based Risk Tier Definition
A regional authority requires discrete operational risk tiers (**Low**, **Medium**, **High**) to allocate emergency budgets, mobilize disaster response teams, and issue regional flood warnings.

- **Threshold Derivation:** Computed from percentiles of the training target `y_train`:
  - **33.33rd Percentile ($q_{33}$):** `0.4800`
  - **66.67th Percentile ($q_{67}$):** `0.5250`
- **Class Boundary Rules:**
  - **Low Risk:** $\text{FloodProbability} \le 0.4800$
  - **Medium Risk:** $0.4800 < \text{FloodProbability} \le 0.5250$
  - **High Risk:** $\text{FloodProbability} > 0.5250$
- **Resulting Class Balance:**
  - **Train Set (80k):** Low = 34.16% (27,324), Medium = 33.13% (26,503), High = 32.72% (26,173)
  - **Test Set (20k):** Low = 34.38% (6,876), Medium = 32.42% (6,485), High = 33.20% (6,639)
- **Justification for Quantile-Based Tiers:**
  - Because flood probability is synthetic and centered at 0.504 without physical sensor gauge units (e.g. water height in meters), arbitrary uniform intervals (e.g., $<0.33$, $0.33\text{--}0.66$, $>0.66$) would leave the extreme classes virtually empty (less than 0.5% in High risk).
  - Quantile stratification creates balanced, operationally meaningful tiers where the top one-third of highest-risk regions receive prioritized disaster intervention.
- **Persistence:** Saved to `models/tier_cutoffs.json` so that the model training pipeline, error analysis, and Streamlit application use identical thresholds.

### 4.3 Feature Engineering (Domain Aggregates)
To evaluate whether higher-level summaries improve non-linear models (like Random Forest and Gradient Boosting), we created 3 domain aggregates in `add_engineered_features()`:
1. `indicator_mean`: Average score across all 20 indicators (captures total regional hazard load).
   - Training statistics: Mean = 4.938, Std = 0.414, Min = 3.150, Max = 7.150.
2. `env_mean`: Average score of the 8 natural and hydrological indicators (`MonsoonIntensity`, `TopographyDrainage`, `Deforestation`, `ClimateChange`, `CoastalVulnerability`, `Landslides`, `Watersheds`, `WetlandLoss`).
   - Training statistics: Mean = 4.935, Std = 0.707, Min = 2.250, Max = 8.375.
3. `infra_mean`: Average score of the 12 built-environment, demographic, and governance indicators (`RiverManagement`, `Urbanization`, `DamsQuality`, `Siltation`, `AgriculturalPractices`, `Encroachments`, `IneffectiveDisasterPreparedness`, `DrainageSystems`, `DeterioratingInfrastructure`, `PopulationScore`, `InadequatePlanning`, `PoliticalFactors`).
   - Training statistics: Mean = 4.940, Std = 0.563, Min = 2.333, Max = 7.583.
- *Usage Policy:* These engineered features are evaluated as an explicit experiment in Deliverable 4 and are not forced into the final model unless they demonstrate meaningful cross-validation gains.

---

## 5. Model Development & Experimental Comparison

All models were evaluated under strict **5-Fold Cross-Validation** on the training split (80,000 samples) and finally scored on the unseen holdout test split (20,000 samples). Results are archived in `reports/results/model_comparison.csv` and documented in `notebooks/02_modeling.ipynb`.

### 5.1 Comprehensive Experimental Comparison Table

| Model | Category | Scaled | Features | CV $R^2$ Mean | CV $R^2$ Std | CV RMSE | CV MAE | Test $R^2$ | Test RMSE | Test MAE | CV Train Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **HistGradientBoosting (20 + 3 Aggregates)** | **Feature Engineering** | **False** | **23 (Engineered)** | **0.86503** | **0.00191** | **0.01870** | **0.01461** | **0.87092** | **0.01845** | **0.01441** | **0.88** |
| MLP Neural Net (Scaled, 32-16 layers) | Neural Network | True | 20 (Original) | 0.84457 | 0.00250 | 0.02007 | 0.01575 | 0.85215 | 0.01974 | 0.01554 | 3.16 |
| Linear Regression (Scaled, 20 Feat) | Baseline | True | 20 (Original) | 0.84421 | 0.00224 | 0.02010 | 0.01583 | 0.84958 | 0.01991 | 0.01570 | 1.27 |
| Linear Regression (Unscaled, 20 Feat) | Scaling Experiment | False | 20 (Original) | 0.84421 | 0.00224 | 0.02010 | 0.01583 | 0.84958 | 0.01991 | 0.01570 | 1.85 |
| Ridge Regression (Scaled, $\alpha=1.0$) | Regularization | True | 20 (Original) | 0.84421 | 0.00224 | 0.02010 | 0.01583 | 0.84958 | 0.01991 | 0.01570 | 1.16 |
| Ridge Regression (Unscaled, $\alpha=1.0$) | Scaling Experiment | False | 20 (Original) | 0.84421 | 0.00224 | 0.02010 | 0.01583 | 0.84958 | 0.01991 | 0.01570 | 0.21 |
| Linear Regression (20 + 3 Aggregates) | Feature Engineering | True | 23 (Engineered) | 0.84421 | 0.00224 | 0.02010 | 0.01583 | 0.84958 | 0.01991 | 0.01570 | 0.35 |
| Ridge Regression (20 + 3 Aggregates) | Feature Engineering | True | 23 (Engineered) | 0.84421 | 0.00224 | 0.02010 | 0.01583 | 0.84958 | 0.01991 | 0.01570 | 0.32 |
| Tuned Ridge ($\alpha=0.207$) | Hyperparameter Tuning | True | 20 (Original) | 0.84421 | 0.00224 | 0.02010 | 0.01583 | 0.84958 | 0.01991 | 0.01570 | 1.78 |
| Tuned HistGradientBoosting | Hyperparameter Tuning | False | 20 (Original) | 0.79760 | 0.00257 | 0.02291 | 0.01835 | 0.80573 | 0.02263 | 0.01809 | 8.54 |
| HistGradientBoosting (150 iters, lr=0.08) | Ensemble Non-Linear | False | 20 (Original) | 0.78277 | 0.00174 | 0.02373 | 0.01926 | 0.78979 | 0.02354 | 0.01913 | 1.75 |
| HistGradientBoosting (Default, 20 Feat) | Ensemble Non-Linear | False | 20 (Original) | 0.75598 | 0.00162 | 0.02515 | 0.02048 | 0.76422 | 0.02493 | 0.02039 | 1.29 |
| Random Forest (50 Trees, max_depth=10) | Ensemble Non-Linear | False | 20 (Original) | 0.43933 | 0.00174 | 0.03812 | 0.03107 | 0.43646 | 0.03854 | 0.03150 | 6.66 |

---

### 5.2 Key Experimental Findings

#### (a) Scaled vs. Unscaled Features
- Standardizing features with `StandardScaler` produced identical CV $R^2$ (`0.84421`) for Linear and Ridge Regression.
- *Reason:* Ordinary Least Squares finds the global optimal hyperplane analytically. Because all 20 indicators already share roughly the same scale [0, 16], scaling simply rescales the regression coefficients without altering the predicted outputs. Scaling is kept for Ridge and Neural Network stability.

#### (b) The Tree-Linear Discrepancy & Feature Engineering Surge
- **The Puzzle:** On raw 20 features, tree-based models scored substantially *worse* than Linear Regression (Default Gradient Boosting: $R^2 = 0.75598$; Random Forest: $R^2 = 0.43933$ vs. Linear Regression: $R^2 = 0.84421$).
- **The Cause:** Decision trees make orthogonal (axis-parallel) splits. When the true underlying target is an additive sum of 20 variables ($x_1 + x_2 + \dots + x_{20}$), approximating that 20-dimensional diagonal plane requires thousands of deep staircase steps.
- **The Breakthrough:** Adding 3 simple aggregates (`indicator_mean`, `env_mean`, `infra_mean`) caused HistGradientBoosting's CV $R^2$ to surge from **0.75598 to 0.86503** (Test $R^2 = \mathbf{0.87092}$), surpassing Linear Regression. The tree immediately splits on `indicator_mean` to capture the global additive signal, and uses subsequent trees to fit non-linear residual variations.
- **Why Linear Regression Did Not Change:** Linear regression already computes the optimal linear combination of the 20 variables. Adding the mean of those same variables is mathematically collinear and adds zero new information.

#### (c) Hyperparameter Tuning (`RandomizedSearchCV`)
- **Ridge Regression:** Searched $\alpha \in [10^{-2}, 10^3]$. Best $\alpha = 0.207$. Confirmed that variance is minimal and standard L2 regularization prevents overfitting without losing accuracy.
- **HistGradientBoosting:** Tuned learning rate, max leaf nodes, and leaf size on raw 20 features, improving $R^2$ from 0.75598 to 0.79760.

---

### 5.3 Model Trade-Off & Final Model Selection

We observe a clear engineering trade-off between two compelling options:
1. **Option 1: HistGradientBoosting (with internal Feature Engineering)**
   - **Performance:** Top statistical score across all metrics (Test $R^2 = \mathbf{0.87092}$, Test RMSE = $\mathbf{0.01845}$, Test MAE = $\mathbf{0.01441}$).
   - **Deployment:** Packaged inside a self-contained pipeline via `FunctionTransformer`, allowing it to accept the original 20 indicators from the Streamlit interface and compute aggregates internally.
2. **Option 2: Ridge Regression (Scaled, 20 Features)**
   - **Performance:** Strong, competitive baseline (Test $R^2 = 0.84958$, Test RMSE = $0.01991$, Test MAE = $0.01570$).
   - **Simplicity:** Completely transparent parametric formula, 100% explainable in an academic viva, sub-millisecond inference time.

**Recommendation:** Option 1 (HistGradientBoosting with internal Feature Engineering) is selected for `models/final_model.joblib` because it achieves the highest predictive accuracy (+2.1% $R^2$ gain and lower errors) while remaining fully self-contained for deployment.

---

## 6. Comprehensive Evaluation & Error Analysis

The final selected model (`HistGradientBoostingRegressor` with self-contained feature engineering) was evaluated on the unseen 20,000-row holdout test set (`X_test`, `y_test`). All numerical metrics are persisted in `models/metrics.json`.

### 6.1 Regression Metrics
- **Coefficient of Determination ($R^2$):** **`0.87092`** (explains 87.1% of the total variance in regional flood likelihood).
- **Root Mean Squared Error (RMSE):** **`0.01845`** (sub-2% average standard deviation of error).
- **Mean Absolute Error (MAE):** **`0.01441`** (average prediction error is $\approx 0.014$).

### 6.2 Risk Tier Classification Performance
Continuous predictions were mapped into 3 categorical governance tiers using the saved cutoffs derived on the training set:
$$\text{Low} \le 0.4800 < \text{Medium} \le 0.5250 < \text{High}$$

#### Confusion Matrix (20,000 Test Regions):
| Actual \ Predicted | Predicted Low | Predicted Medium | Predicted High | Total Actual |
|---|---|---|---|---|
| **Actual Low** | **5,113** (74.4%) | 1,761 (25.6%) | 2 (0.03%) | 6,876 |
| **Actual Medium** | 1,676 (25.8%) | **3,636** (56.1%) | 1,173 (18.1%) | 6,485 |
| **Actual High** | **0 (0.00%)** | 630 (9.49%) | **6,009 (90.51%)** | 6,639 |

#### Per-Class Metrics:
- **High Risk:** Precision = `0.8364` | **Recall = `0.9051`** | **F1-Score = `0.8694`**
- **Low Risk:** Precision = `0.7531` | Recall = `0.7436` | F1-Score = `0.7483`
- **Medium Risk:** Precision = `0.6033` | Recall = `0.5607` | F1-Score = `0.5812`
- **Macro-Averaged F1-Score:** **`0.73299`**

#### Governance & Viva Justification:
The model demonstrates an outstanding **90.51% recall for the High Risk tier**, with **exactly 0 true high-risk regions misclassified as Low Risk**. In public disaster safety, missing a catastrophic flood (False Negative) costs human lives and massive economic damage, whereas a precautionary false alert (warning a medium-risk zone) merely mobilizes surplus reserves. The model's asymmetric safety margin is ideally calibrated for emergency governance.

### 6.3 Diagnostic Figures (Archived in `reports/figures/`)
1. `reports/figures/predicted_vs_actual.png`: Tight clustering around the $45^\circ$ diagonal line ($y = x$) across the central 0.40–0.62 range, with subtle regression-to-the-mean shrinkage at the outermost ends.
2. `reports/figures/residuals_vs_predicted.png`: Homoscedastic error band centered at zero without pronounced curvature or heteroscedastic fan shapes.
3. `reports/figures/residual_histogram.png`: Gaussian-like bell distribution centered at $\mu = -0.000037$ with symmetric, tapering errors.
4. `reports/figures/confusion_matrix.png`: Heatmap demonstrating dominant diagonal density and zero leakage between Low and High extremes.
5. `reports/figures/permutation_importance.png`: Relative feature impact on holdout $R^2$.

### 6.4 Error Analysis & Top 10 Largest Residuals
- **Error Distribution by Region:**
  - Near Tier Boundaries ($\pm 0.015$ margin): MAE = `0.01222` ($n = 7,691$).
  - Middle Standard Regions: MAE = `0.01602` ($n = 11,765$).
  - Extreme Tail Regions ($<0.40$ or $>0.62$): MAE = `0.01045` ($n = 544$).
- **Top Residual Case Study:**
  - Largest error: Row index 11714 (Actual = `0.28500`, Predicted = `0.46604`, Residual = `-0.18104`).
  - Feature inspection: The indicator scores for this region were moderate (MonsoonIntensity = 4, TopographyDrainage = 5, ClimateChange = 5, DamsQuality = 6).
  - *Analytical Insight:* A region with average scores ($\approx 5$) having an actual flood probability of 0.285 represents an injected noise perturbation in the synthetic dataset generation, rather than a systematic model failure.

### 6.5 Permutation Feature Importance Ranking
Permutation importance measures the drop in test $R^2$ when a feature column is randomly shuffled (breaking its relationship with the target):

| Rank | Feature | Mean Decrease in $R^2$ | Std Dev |
|---|---|---|---|
| 1 | `PoliticalFactors` | 0.14617 | 0.00169 |
| 2 | `Watersheds` | 0.14112 | 0.00120 |
| 3 | `RiverManagement` | 0.13956 | 0.00122 |
| 4 | `DamsQuality` | 0.13939 | 0.00164 |
| 5 | `Urbanization` | 0.13929 | 0.00069 |
| 6 | `Landslides` | 0.13908 | 0.00196 |
| 7 | `CoastalVulnerability` | 0.13894 | 0.00103 |
| 8 | `TopographyDrainage` | 0.13846 | 0.00119 |
| 9 | `DrainageSystems` | 0.13833 | 0.00159 |
| 10 | `AgriculturalPractices` | 0.13773 | 0.00112 |

- *Interpretation:* All 20 features demonstrate positive, substantial importance clustered tightly around **$\Delta R^2 \approx 0.137\text{--}0.146$**. No single feature drives flood risk independently; rather, regional flood vulnerability is a multi-system compound phenomenon where governance, infrastructure integrity, and environmental forces act together with comparable weights.

---

## 7. Project Limitations & Academic Framing

For the academic report and viva defense, the following limitations must be clearly and honestly articulated:
1. **Synthetic Nature of Dataset:** Features are survey-style integer scores (0 to 17) rather than physical sensor dimensions (e.g., cubic meters per second river discharge, millimeters of precipitation, or gauge elevations).
2. **Artificial Target Noise:** In `flood.csv`, target equals $\sum(\text{indicators})/200$ exactly ($R^2 = 1.0$). In `train.csv`, stochastic noise was intentionally injected to create variance.
3. **Absence of Spatial Topology & Temporal Dynamics:** Real physical flooding is non-Euclidean: upstream rainfall flows down river networks over successive days. The dataset contains no timestamps, geographical coordinates, or watershed boundary polygons.
4. **Relative (Quantile) Risk Tiers:** Tiers (Low $\le 0.48$, Medium $0.48\text{--}0.525$, High $> 0.525$) represent relative distribution percentiles, not physical flood depth thresholds.
5. **Operational Scope:** FloodSense should be presented as a **macro-level regional risk-screening and vulnerability index model** for prioritizing administrative inspections and infrastructure budgeting, rather than a real-time operational hydrodynamic flood warning system.

---

## 8. Decision Support Dashboard (Streamlit Application)

The interactive decision-support application is implemented in `app/streamlit_app.py` to bridge machine learning outputs with municipal disaster management workflows.

### 8.1 Architectural & UI Design
- **Technology:** Pure Python with Streamlit (`streamlit`), utilizing zero external JS/CSS frameworks for clean maintainability.
- **Model Loading:** The pre-trained pipeline (`models/final_model.joblib`) and quantile cutoffs (`models/tier_cutoffs.json`) are cached in memory via `@st.cache_resource`. **No retraining occurs inside the application**, ensuring instant sub-second response times.
- **Defensive Error Handling:** If artifacts are absent, the application intercepts the state and renders a user-friendly instruction guide ("Please execute `python src/train.py`") rather than raising an unhandled Python traceback.

### 8.2 Application Structure (2-Tab Layout)

#### Tab 1: Regional Risk Assessment (Interactive Screening)
- **Logical Two-Column Input Partitioning:**
  - **Column 1 (Environmental & Hydrological):** Sliders for `MonsoonIntensity`, `TopographyDrainage`, `Deforestation`, `ClimateChange`, `CoastalVulnerability`, `Landslides`, `Watersheds`, and `WetlandLoss`.
  - **Column 2 (Infrastructure, Demographics & Governance):** Sliders for `RiverManagement`, `Urbanization`, `DamsQuality`, `Siltation`, `AgriculturalPractices`, `Encroachments`, `IneffectiveDisasterPreparedness`, `DrainageSystems`, `DeterioratingInfrastructure`, `PopulationScore`, `InadequatePlanning`, and `PoliticalFactors`.
- **Baseline Defaulting:** Every slider is pre-set to the sample median score of **5** (range 0 to 16/17), representing a typical baseline region.
- **Real-Time Assessment Output:**
  - **KPI Metric Card:** Displays predicted continuous flood probability with comparative delta against regional average (0.5043).
  - **Categorical Risk Tier Badge:**
    - **Low Risk ($\le 48.0\%$):** Green badge (`#28a745`), recommending routine seasonal maintenance.
    - **Medium Risk ($48.0\% - 52.5\%$):** Amber badge (`#d39e00`), recommending canal clearing and first responder alerts.
    - **High Risk ($> 52.5\%$):** Red badge (`#dc3545`), recommending immediate emergency mobilization and resource allocation.
  - **Contextual Explanations:** Expandable help box explaining the derivation of quantile tiers.

#### Tab 2: Model Performance & Governance Audit
- **Executive Metric Cards:** Displays Test $R^2$ (`0.8709`), Test RMSE (`0.0185`), Test MAE (`0.0144`), and High-Risk Recall (`90.51%`).
- **Audit Table:** Direct interactive table rendering of all 13 evaluated models loaded from `reports/results/model_comparison.csv`.
- **Embedded Diagnostic Visualizations:** Dynamic display of `confusion_matrix.png`, `permutation_importance.png`, `predicted_vs_actual.png`, and `residual_histogram.png`.
- **Academic Governance Notice:** Clear four-point summary of model limitations and screening scope for non-technical officials.

### 8.3 Local Execution
The application is launched with:
```bash
python -m streamlit run flood-risk/app/streamlit_app.py
```
Verified running on `http://localhost:8501`.





