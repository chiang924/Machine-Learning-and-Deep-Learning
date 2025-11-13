# Machine-Learning-and-Deep-Learning
Using machine learning and deep learning to process large-scale data, optimize and find the best parameters, and predict data beyond the labeled samples.

---

## Project Structure

- `Collapse_Prediction/`  
  End-to-end pipeline for binary landslide prediction using Random Forest and XGBoost and CATBoost.
- `HW2/`, `HW3/`, `HW4/`  
  Jupyter notebooks and scripts for other course assignments.
- `README.md`  
  Project description and basic usage.

The main focus of this repository is the **collapse prediction** task, which uses real labeled data to predict potential landslide events on unseen samples.

---

## Collapse Prediction Pipeline

The core script implements the following workflow:

1. **Data Loading**
   - Read `train.xlsx` and `test.xlsx` with `pandas`.
   - Use the column `崩塌` in `train.xlsx` as the binary target label (0 / 1).
   - All remaining numeric columns are treated as candidate features.

2. **Train/Validation Split and Imputation**
   - Split the training data into **80% train / 20% validation** with stratified sampling to keep the class ratio.
   - Use `sklearn.SimpleImputer(strategy="median")`:
     - Fit the imputer on the **training subset only** (avoid data leakage).
     - Apply the same transformation to validation and test sets.

3. **Feature Selection via Random Forest**
   - Train a `RandomForestClassifier` on the **imputed training data**.
   - Compute feature importances and sort them in descending order.
   - Select the **top 10 most important features** and rebuild:
     - `X_train_sel`, `X_val_sel`, `X_test_sel` using only these 10 columns.

4. **Single XGBoost Model (CPU)**
   - Define an `XGBClassifier` with:
     - `n_estimators=10000`, `max_depth=8`, `learning_rate=0.05`
     - `subsample=0.9`, `colsample_bytree=0.9`
     - `tree_method="hist"`, `n_jobs=-1`
   - Handle class imbalance with `scale_pos_weight = #neg / #pos` computed from the training set.
   - Train on `X_train_sel`, evaluate on `X_val_sel`:
     - Metrics: **ROC-AUC**, **PR-AUC**, **F1-score**.
     - Scan thresholds from 0.0 to 1.0 to find the best F1 (also keep a fixed low threshold for comparison).
     - Print classification report and confusion matrix on the validation set.

5. **5-Fold Ensemble (Blending)**
   - Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the **original training rows** but only with the selected 10 features.
   - For each fold:
     - Re-fit a new median imputer on the **fold’s training subset**.
     - Impute the fold’s validation data and the global test data.
     - Recompute `scale_pos_weight` on that fold’s train labels.
     - Train a new XGBoost model with the same hyper-parameters.
     - Evaluate ROC-AUC and PR-AUC on the fold’s validation set.
     - Predict probabilities on the test set and add them to a running sum.
   - Average the predictions from all 5 folds to obtain a **blended test probability** for each sample.

6. **Thresholding and Submission File**
   - Two options to convert probabilities to labels:
     - **Fixed threshold** (e.g., `THRESHOLD = 0.008`).
     - **Dynamic top-k threshold** to match an expected number of positive cases (optional).
   - Generate final predictions `y_test_pred` (0 / 1).
   - If the test data has an `ID` column, reuse it; otherwise create `ID = 0 ... n-1`.
   - Save results to `submission.csv`:

     ```text
     ID,Label
     0,0
     1,1
     ...
     ```

---

## How to Run

1. Prepare the input files in the project root:
   - `train.xlsx` (must contain a column named `崩塌` as the label).
   - `test.xlsx` (same feature columns as `train.xlsx`, but no label).

2. Install dependencies (example):

   ```bash
   pip install numpy pandas scikit-learn xgboost openpyxl
   
