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
   - Train on `X_trai_
