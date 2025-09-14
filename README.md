# Capstone Project: Predicting Loan Default Risk

## Project Overview
This project aims to predict the likelihood of a loan applicant defaulting using demographic, financial, and behavioral data from Home Credit’s loan application records. The analysis supports fairer lending decisions and promotes financial inclusion by leveraging alternative data sources.

## Research Question
Can we predict the likelihood of a loan applicant defaulting using demographic, financial, and behavioral data from Home Credit’s loan application records?

## Data Source
- Dataset: [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)
- Primary files: `application_train.csv`, `application_test.csv`
- Supplementary files: `bureau.csv`, `previous_application.csv`, `installments_payments.csv`, etc.
- Column descriptions: `HomeCredit_columns_description.csv`

---

## Project Structure
```bash
AI-ML-Assignment-20.1/
├── src/
│   ├── data/
│   │   └── application_train.csv
│   │   └── application_test.csv
│   └── home_credit.ipynb
└── README.md
```

---

## Data Understanding

### Target Variable
- `TARGET`: Binary classification (1 = default, 0 = repaid)
- Highly imbalanced (~8% defaults)

### Key Insights from EDA
- **Numerical Features**:
  - `AMT_INCOME_TOTAL` and `AMT_CREDIT` show weak correlation with default.
  - `DAYS_BIRTH` (converted to age) and `DAYS_EMPLOYED` show a slight correlation with risk.
  - `CNT_CHILDREN` and `CNT_FAM_MEMBERS` are strongly correlated—potential multicollinearity.

- **Categorical Features**:
  - Higher default rates among applicants with lower education, rented housing, and single/civil marriage status.
  - Males show higher default rates than females.
  - Revolving loans have lower default rates than cash loans.

- **Missing Values**:
  - Dropped columns with >50% missing data.
  - Imputed remaining missing values using median (numerical) and mode (categorical).

---

## Data Preparation

### Cleaning & Imputation
- Dropped high-missing columns (e.g., `BASEMENTAREA_*`, `NONLIVINGAREA_*`)
- Imputed numerical features with median
- Imputed categorical features with mode

### Feature Engineering
- Converted `DAYS_BIRTH` to `AGE`
- Replaced anomalous `DAYS_EMPLOYED = 365243` with NaN
- Created:
  - `EMPLOYMENT_YEARS`
  - `INCOME_CREDIT_RATIO`
  - `CHILDREN_RATIO`

### Encoding
- Label encoded binary features (`CODE_GENDER`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`)
- One-hot encoded multi-class features (`NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`)

### Train-Test Split
- Stratified split into training and validation sets (80/20)
- SMOTE will be applied only to training data during modeling to address class imbalance

---

## Modeling: Logistic Regression (Baseline)

### Model Setup
- Algorithm: Logistic Regression
- Preprocessing: StandardScaler applied to all features
- Imbalance handled using SMOTE on training data

### Evaluation Metrics

| Metric         | Value   | Interpretation |
|----------------|---------|----------------|
| AUC-ROC        | 0.720   | Strong ability to rank defaulters higher than non-defaulters |
| Precision      | 0.397   | ~39.7% of predicted defaulters were correct |
| Recall         | 0.018   | Only ~1.8% of actual defaulters were identified |
| F1-score       | 0.034   | Low balance between precision and recall |
| Confusion Matrix | `[[56403   135] [4876    89]]` | Very conservative predictions—few defaulters flagged, but more accurate when flagged |

---

## Next Steps

### Advanced Models
- Implement tree-based models (Random Forest) to capture non-linear patterns and improve recall.

### Feature Selection
- Use feature importance and dimensionality reduction to reduce noise and improve generalization.

### Ensemble Methods
- Combine multiple models to improve robustness and predictive power.

---

## Source Code
- [Data - Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)
- [Jupyter Notebook](https://github.com/kloudingenuity/AI-ML-Assignment-20.1/blob/main/src/home_credit.ipynb)

--- 
