# Credit Card Fraud Detection using Machine Learning

This project aims to detect fraudulent credit card transactions using a machine learning classification model. The dataset used is highly imbalanced and anonymized through PCA transformation. We perform end-to-end preprocessing, model training, evaluation, and deployment steps to identify fraudulent behavior effectively.

## Project Overview

- **Dataset**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Total Transactions**: 284,807
- **Fraudulent Transactions**: 492 (0.172%)
- **Challenge**: Highly imbalanced dataset, anonymized features
- **Algorithm Used**: Random Forest Classifier
- **Evaluation Metrics**: Precision, Recall, F1-Score, Area Under Precision-Recall Curve (AUPRC)

---

## Objective

To build a machine learning model that accurately classifies transactions as **fraudulent** or **genuine**, while effectively handling class imbalance and evaluating performance using appropriate metrics.

---

## Dataset Description

- **V1 to V28**: Principal components from PCA (original features not disclosed)
- **Time**: Seconds elapsed between each transaction and the first
- **Amount**: Transaction amount
- **Class**: Target variable (0 = Genuine, 1 = Fraudulent)

---

## Workflow Summary

### 1. Data Loading and Inspection

- Load dataset using Pandas.
- Perform an initial inspection of shape, types, and class distribution.

### 2. Data Preprocessing

- Drop the 'Time' column (irrelevant for classification).
- Normalize the 'Amount' column using `StandardScaler` to scale values between -1 and 1.
- Define feature matrix `X` and target vector `y`.

### 3. Handling Class Imbalance

- Use **SMOTE (Synthetic Minority Oversampling Technique)** to balance the class distribution by synthetically generating minority class samples.

### 4. Train-Test Split

- Split the dataset into training and testing sets (70:30) while preserving class balance using stratified sampling.

### 5. Model Training

- A **Random Forest Classifier** is trained on the resampled dataset.
- Random Forest is chosen for its robustness, ability to handle high-dimensional data, and good performance on imbalanced datasets.

### 6. Model Evaluation

- Generate predictions on the test set.
- Evaluate using:
  - **Classification Report**: Precision, Recall, F1-Score
  - **Confusion Matrix**
  - **Precision-Recall Curve**
  - **Area Under Precision-Recall Curve (AUPRC)** for meaningful performance measurement on imbalanced data

### 7. Saving Model and Predictions

- Save the trained model using `joblib` as `credit_fraud_model.pkl`
- Generate predictions and fraud probabilities for all transactions
- Save:
  - All predictions: `all_predictions_with_probabilities.csv`
  - Only suspicious transactions: `fraud_cases_only.csv`

---

## Results Snapshot

- Significant improvement in recall and precision after handling class imbalance
- AUPRC is used instead of accuracy due to the skewed class distribution

---

##  Files Included

| File Name                          | Description |
|-----------------------------------|-------------|
| `fraud_detection_model.py`        | Python script version of the notebook |
| `credit_fraud_model.pkl`          | Trained Random Forest model |
| `all_predictions_with_probabilities.csv` | Full prediction results with fraud probability |
| `fraud_cases_only.csv`            | Filtered transactions marked actual or predicted fraud |
| `README.md`                       | Project documentation |

---

##  Technologies Used

- Python 3.x
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Joblib

---
