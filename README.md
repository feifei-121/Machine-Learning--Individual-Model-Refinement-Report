# Machine-Learning--Enhanced Customer Churn Prediction: An Individual Model Refinement Report
## 1. Overview

This project refines a group Random Forest churn model for a UK retail bank.  

The notebook:

- Recreates the **group baseline** model.
- Adds **12 engineered features**.
- Performs **grid search hyperparameter tuning** with **5‑fold stratified CV**.
- Evaluates improvements in **accuracy, precision, recall, F1**.
- Conducts **fairness analysis** across **gender** and **geography**.
- Quantifies **business impact** (revenue saved, costs, ROI, churn reduction).

All experiments are run with a fixed random seed to ensure reproducibility.

---

## 2. File Structure

- `Individual Refinement` – Main analysis notebook.
- `Churn_Modelling_Cleaned.csv` – Input dataset (10,000 customers, 11 original features).

No external data is used. The personal work uses the same cleaned dataset as the group project.

---

## 3. Environment & Dependencies

The notebook was developed and tested with:

- **Python:** 3.10.12  

Python packages:

- `pandas==2.0.3`
- `numpy==1.24.3`
- `scikit-learn==1.3.0`
- `matplotlib==3.7.2`
- `seaborn==0.12.2`
- `scipy==1.11.1`

Install via:

```bash
pip install pandas==2.0.3 numpy==1.24.3 scikit-learn==1.3.0 \
            matplotlib==3.7.2 seaborn==0.12.2 scipy==1.11.1
```

---

## 4. How to Run

1. **Place files**  
   Ensure the following files are in the same directory:

   - `Individual Refinement.ipynb`
   - `Churn_Modelling_Cleaned.csv`

2. **Open the notebook**  
   Start Jupyter and open `new_code.ipynb`:

   ```bash
   jupyter notebook
   ```

3. **Run all cells**  
   In Jupyter: `Cell` → `Run All`.

4. **Random seed / reproducibility**

   The following configuration is set near the top of the notebook:

   ```python
   RANDOM_SEED = 42
   np.random.seed(RANDOM_SEED)
   ```

   - Train/test split: **80/20**, stratified by `Exited`.
   - CV: **StratifiedKFold** with `n_splits = 5`, `shuffle = True`, `random_state = 42`.
   - No re‑sampling (no SMOTE/undersampling); the natural class distribution is preserved.

Running all cells once, without modification, should reproduce the metrics reported in the individual report (up to floating‑point rounding).

---

## 5. Notebook Structure (Section Map)

### 1. Setup and Configuration

- Imports libraries.
- Sets global plotting style.
- Defines global constants (e.g. `RANDOM_SEED`, CV folds, business parameters, baseline metrics from group report).

### 2. Load and Audit Data

- Loads `Churn_Modelling_Cleaned.csv` into `df`.
- Prints shape and head.
- Verifies dtypes, missing values (should be none), duplicates (should be none).
- Computes churn rate (≈ 20.37%).

### 3. Exploratory Data Analysis

- `describe()` on numerical columns.
- Computes churn rate by `Geography` and `Gender`.
- Builds multi‑panel bar charts for:
  - Churn by geography.
  - Churn by gender.
  - Churn by `NumOfProducts`.
  - Churn by `IsActiveMember`.

These plots inform the feature engineering choices.

### 4. Feature Engineering

#### 4.1 Group Features (Baseline)

- Recreates group‑level features:
  - `BalanceIsZero`
  - `Age_Balance_Interaction`
  - `AgeBin` (binned age).

#### 4.2 Individual Features (Personal Work)

Adds 12 new features, grouped as:

- **Tenure**  
  - `IsNewCustomer`, `IsLongTermCustomer`, `TenureAgeRatio`
- **Products**  
  - `HasSingleProduct`, `HasMultipleProducts`
- **Financial**  
  - `BalanceToSalaryRatio`, `IsHighBalance`, `IsLowBalance`
- **Credit**  
  - `IsGoodCredit`, `IsPoorCredit`
- **Risk**  
  - `IsHighRiskGeography`, `IsHighRiskAge`

Also constructs a **composite `RiskScore`** by weighted sum of risk flags.

Performs simple correlation analysis between new features and `Exited`:

- Stronger correlation for `RiskScore` than any single component.

### 5. Data Preparation

- One‑hot encodes `Geography`, `Gender`, and `AgeBin` (no `drop_first`).
- Drops only the target `Exited` to create `X`, `y`.
- Confirms final feature count: **33 features**.

### 6. Train–Test Split

- `train_test_split` with `test_size=0.2`, `stratify=y`, `random_state=RANDOM_SEED`.
- Confirms class distribution in train and test matches overall churn rate.

### 7. Baseline Model (Group Version)

- Trains a **RandomForestClassifier** with:

  ```python
  n_estimators=100,
  max_depth=5,
  random_state=RANDOM_SEED
  ```

- Uses full feature set consistent with baseline.

- Reports:

  - Accuracy
  - Precision
  - Recall
  - F1-score

- These values are used as the **group baseline** for comparison.

### 8. Hyperparameter Optimization (Improved Model)

- Defines grid:

  ```python
  param_grid = {
      'n_estimators': [100, 200, 300],
      'max_depth': [5, 10, 15, 20],
      'min_samples_split': [2, 5, 10],
      'min_samples_leaf': [1, 2, 4],
      'max_features': ['sqrt', 'log2']
  }
  ```

- Uses:

  ```python
  StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
  GridSearchCV(estimator=RandomForestClassifier(random_state=RANDOM_SEED),
               param_grid=param_grid,
               cv=cv_strategy,
               scoring='f1',
               n_jobs=-1,
               verbose=1)
  ```

- Optimization metric: **F1** (not accuracy), to reflect cost asymmetry.

- Prints **best parameters** and **mean CV F1**.

### 9. Improved Model Evaluation

- Retrieves `best_model = grid_search.best_estimator_`.
- Evaluates on hold‑out test set:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Constructs comparison table (**baseline vs improved**).
- Builds bar chart of metric improvements.

### 10. Confusion Matrix Analysis

- Computes confusion matrices for both baseline and improved models.
- Prints:
  - True positives, false positives, false negatives.
  - Additional churners captured by improved model.
- Plots side‑by‑side confusion matrix heatmaps.

### 11. Feature Importance

- Extracts `best_model.feature_importances_`.
- Builds table and bar chart for **Top 15 features**, including:
  - `Age`
  - `RiskScore`
  - `Age_Balance_Interaction`
  - `BalanceToSalaryRatio`
  - Product and tenure indicators, etc.

### 12. Fairness Analysis

- Builds `df_test` with original demographic variables plus:
  - `Predicted`
  - `Actual`
  - `Probability`

#### 12.1 Gender Fairness

- Splits test set by `Gender` (Male/Female).
- Computes, for each:
  - Sample size
  - Accuracy
  - Precision
  - Recall
  - F1
  - FNR (False Negative Rate)
- Reports gender disparities (e.g. accuracy difference ≈ 5.5 pp).
- Plots side‑by‑side bar chart for performance by gender.

#### 12.2 Geographic Fairness

- Splits by `Geography` (France/Germany/Spain).
- Computes same metric set.
- Reports that Germany is harder to predict (higher base churn).
- Plots tri‑bar comparison.

### 13. Business Impact Analysis

#### 13.1 Targeting Strategy

- Ranks test customers by predicted churn probability.
- Targets **top 20%** (by probability).
- Calculates:
  - Number of targeted customers.
  - Number of actual churners among targeted.
  - Capture rate (share of churners in top 20%).
  - Precision at top 20%.

#### 13.2 Financial Calibration

Uses assignment‑specified business parameters:

- `AVG_CUSTOMER_VALUE = 5000` GBP
- `INTERVENTION_COST = 150` GBP
- `INTERVENTION_SUCCESS_RATE = 0.60`

Computes on test set:

- Prevented churns.
- Revenue saved.
- Intervention costs.
- Net benefit.
- **ROI** (~11.33:1).

#### 13.3 Annual Projection

- Scales test results to full bank portfolio (10,000 customers).
- Reports:
  - Annual targeted customers.
  - Annual prevented churns.
  - Annual revenue protected.
  - Annual cost.
  - Annual net benefit.
  - Annual ROI.

Computes **churn rate reduction**:

- Current churn ≈ 20.37%.
- New churn after intervention ≈ 12.97%.
- Reduction ≈ 7.4 percentage points.

Plots:

- Annual financial bars (revenue, cost, net benefit).
- Current vs projected churn rate.

---

## 6. Notes & Limitations

- **ROC AUC is not reported** because this unit did not cover it and the module guidance discouraged using metrics not taught.
- Only **Random Forest** is fully tuned and used; other algorithms (Logistic Regression, Neural Networks, XGBoost) were tested separately and reported in the written report, but not implemented here to keep code focused and interpretable.
- No resampling is applied; instead, F1 optimization and stratification are used to handle class imbalance.
- The notebook prints multiple intermediate summaries and saves no files; all outputs are in‑memory or visual plots.
