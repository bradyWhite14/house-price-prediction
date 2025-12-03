# 🏠 House Price Prediction and Feature Engineering (Ames, Iowa)

## Overview
This project applies foundational machine learning and robust data preprocessing techniques to build a predictive regression model for house sale prices using the complex Ames Housing dataset. The goal was to demonstrate proficiency in handling real-world data challenges (missing values, skewness) before training an effective model.

---

## 🛠️ Technical Stack
* **Language:** Python
* **Core Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* **Model:** Scikit-learn Linear Regression (Serving as a baseline model)

---

## 📈 Data Science Workflow & Critical Thinking

The success of the model relied entirely on extensive data preprocessing. This section details the critical steps taken to prepare the raw data for modeling.

### 1. Data Cleaning & Missing Value Imputation
* **Challenge:** The dataset contained numerous missing values (over 1,700 unique NaNs) across 43 features.
* **Solution:** Used rule-based imputation based on the Ames dataset documentation:
    * Categorical features (e.g., `Pool QC`, `Fence`) where missingness means 'None' were filled with the string **'None'**.
    * Numerical features (e.g., `Lot Frontage`) were imputed using the **median** of the corresponding neighborhood.
* **Outcome:** Achieved a training dataset with **zero missing values** for final model input.

### 2. Target Variable Transformation (Normalization)
* **Challenge:** The target variable, `SalePrice`, was severely **right-skewed** (Skewness $\approx 1.74$), violating the core assumption of normal distribution required by Linear Regression models.
* **Solution:** Applied a **log transformation** ($\text{np.log1p}$) to normalize the distribution of `SalePrice`.
* **Outcome:** Skewness was successfully reduced to approximately **$-\mathbf{0.0148}$**, resulting in a near-normal distribution and ensuring stable model training. 

### 3. Feature Engineering
* **Challenge:** Converting 43 textual (object) features into a numerical format readable by the machine learning model.
* **Solution:** Applied **One-Hot Encoding** (`pd.get_dummies`) to transform all categorical variables into binary (0 or 1) features.
* **Outcome:** The feature space was successfully expanded from 80 original features to **319 engineered features**.

---

## 🤖 Model Performance

| Model | Metric (Log Space) | Metric (Original Dollar Units) |
| :--- | :--- | :--- |
| **Linear Regression** | RMSE approx 0.15$ | RMSE $20,000 |

**Interpretation:** The model is highly accurate, predicting house prices with an average prediction error of approximately $20,000 across the entire dataset. This establishes a strong, well-documented baseline for future work.

---

## ➡️ Next Steps & Future Work

1.  **Regularization:** Implement a **Ridge or Lasso Regression** model to mitigate potential multicollinearity issues caused by the 319 features and further reduce the RMSE.
2.  **Model Ensembling:** Experiment with non-linear models like **XGBoost** or **Random Forests** for improved accuracy.
