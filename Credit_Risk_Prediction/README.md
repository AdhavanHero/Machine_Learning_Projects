
# 🏦 Credit Risk Prediction

## 📌 Project Overview
This project aims to predict credit risk (Loan Approval Status) using various Machine Learning algorithms. By analyzing applicant demographics, financial history, and loan details, the models classify whether a loan should be approved or not. 

The pipeline includes rigorous data preprocessing, handling class imbalances using SMOTE, hyperparameter tuning, and advanced feature selection techniques to optimize model performance.

## 🧰 Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `xgboost`
* **Imbalance Handling:** `imbalanced-learn` (SMOTE)
* **Feature Selection:** `boruta` (BorutaPy), `RFE` (Recursive Feature Elimination)

## 🛠️ Project Workflow

### 1. Data Cleaning & Preprocessing
* Handled missing values using statistical imputation (Mode for categorical variables like Gender, Married, Dependents; Mean for continuous variables like LoanAmount).
* Converted categorical text data into numerical format using `LabelEncoder`.
* Dropped unnecessary identifiers (e.g., `Loan_ID`).
* Applied `StandardScaler` to normalize feature variables, ensuring optimal performance for distance-based and gradient-descent algorithms.

### 2. Handling Data Imbalance
* Initial exploration revealed an imbalance in the target variable (`Loan_Status`).
* Applied **SMOTE (Synthetic Minority Over-sampling Technique)** on the training data to synthetically balance the approved and rejected loan classes, preventing model bias.

### 3. Model Training & Evaluation
Several classification models were trained and evaluated using standard metrics (Accuracy, Precision, Recall, F1-Score, and Confusion Matrices):
* **Logistic Regression:** ~73% Accuracy
* **Decision Tree Classifier:** ~69% Accuracy
* **Random Forest Classifier:** ~77% Accuracy
* **XGBoost Classifier:** ~78% Accuracy (Best Performing Base Model)

### 4. Hyperparameter Tuning
Optimized the XGBoost model using:
* **GridSearchCV:** Exhaustive search across parameters (max_depth, learning_rate, n_estimators, etc.).
* **RandomizedSearchCV:** Faster, randomized parameter optimization.

### 5. Feature Selection
To simplify the model and reduce noise, advanced feature selection methods were applied:
* **Boruta (via BorutaPy):** Confirmed important features such as `Married`, `ApplicantIncome`, `Credit_History`, and `Property_Area`.
* **Recursive Feature Elimination (RFE) with XGBoost:** Extracted the top 8 most influential features, yielding a streamlined model that maintained a strong ~77% accuracy.

## 🚀 Key Results
The **XGBoost Classifier** paired with **RFE-selected features** provided the best balance of simplicity and predictive power. 

**Top 8 Selected Features:**
1. `Married`
2. `Dependents`
3. `ApplicantIncome`
4. `CoapplicantIncome`
5. `LoanAmount`
6. `Loan_Amount_Term`
7. `Credit_History`
8. `Property_Area`

## 💻 How to Run
1. Clone the repository: `git clone <[your-repo-link](https://github.com/AdhavanHero/Machine_Learning_Projects/edit/main/Credit_Risk_Prediction/README.md)>`
2. Ensure you have the required libraries installed: 
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn xgboost boruta
