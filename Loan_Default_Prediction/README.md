# Loan Default Prediction 🏦

## Project Overview
This project aims to predict the likelihood of a borrower defaulting on a loan based on their financial history, demographics, and loan details. Since loan default datasets are typically highly imbalanced (most people pay their loans back), this project heavily explores different techniques for handling class imbalance, including algorithmic penalization and manual oversampling.

## Dataset
The dataset contains financial and demographic records for over 250,000 loans. 
**Key Features include:**
* **Demographics:** Age, Education, EmploymentType, MaritalStatus, HasDependents
* **Financials:** Income, CreditScore, DTIRatio (Debt-to-Income), HasMortgage
* **Loan Details:** LoanAmount, InterestRate, LoanTerm, NumCreditLines, LoanPurpose, HasCoSigner
* **Target Variable:** `Default` (1 = Defaulted, 0 = Repaid)

## Tech Stack
* **Python 3**
* **Data Manipulation:** `pandas`
* **Machine Learning:** `scikit-learn`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Statistics:** `scipy`

## Methodology
1. **Data Preprocessing:** * Handled ordinal categorical variables (`Education`, `EmploymentType`) via manual mapping.
   * Encoded nominal categorical variables using `LabelEncoder`.
   * Dropped non-predictive identifiers (`LoanID`).
2. **Exploratory Data Analysis (EDA):** Generated a correlation heatmap to check for multicollinearity among variables.
3. **Handling Class Imbalance:**
   * *Approach 1 (Algorithmic):* Used `class_weight="balanced"` inside machine learning models to penalize majority class errors.
   * *Approach 2 (Data-level):* Manually oversampled the minority class by duplicating default records to achieve a 50/50 ratio.
4. **Modeling:** Trained and evaluated Logistic Regression, Decision Trees, and Random Forests.
5. **Evaluation:** Evaluated models using Confusion Matrices, Classification Reports (Precision, Recall, F1-Score), and ROC-AUC via Stratified K-Fold Cross-Validation.

## Key Findings
* **Random Forest** achieved the highest overall accuracy (89%), but it severely overfit the training data and completely failed to identify the minority class (Recall: 0.03).
* **Logistic Regression (with class balancing)** proved to be the most reliable model for this specific business case. While overall accuracy dropped to ~67%, it successfully identified ~67% of the actual defaults (Recall), which is crucial for risk management.
* **Hyperparameter Tuning** via `GridSearchCV` on the Logistic Regression model yielded the best parameters: `{'C': 10, 'fit_intercept': True, 'penalty': 'l1'}`.
* The model achieves a mean **ROC-AUC score of ~0.74** across 5-fold cross-validation.

## How to Run
1. Clone the repository.
2. Open the Jupyter Notebook: `jupyter notebook "Loan Default Prediction.ipynb"`.
3. Ensure the `Loan_default.csv` file is in the correct path referenced in the notebook.
