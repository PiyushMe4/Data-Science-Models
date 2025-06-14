# Credit Card Fraud Detection Using Machine Learning

This project builds a machine learning model to detect fraudulent credit card transactions from a real-world dataset. The model is trained to work with extremely imbalanced data and highlights important features, model evaluation, and suspicious transaction outputs.

---

## Project Objective

To classify transactions as **fraudulent (1)** or **genuine (0)** using a supervised machine learning model trained on anonymized PCA-transformed data.

---

## Dataset Overview

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Samples**: 284,807 transactions
- **Fraud Cases**: 492 (≈0.172%)
- **Features**:
  - `Time` – Seconds since the first transaction
  - `Amount` – Transaction amount
  - `V1` to `V28` – PCA components
  - `Class` – Target (1 = Fraud, 0 = Genuine)

---

## Project Structure

1. **Data Preprocessing**
   - Load dataset
   - Drop duplicates
   - Scale `Time` and `Amount` with `StandardScaler`

2. **Handle Class Imbalance**
   - Use `SMOTE` to oversample fraud cases
   - Split into train/test sets (80:20)

3. **Model Training**
   - Random Forest Classifier with `class_weight='balanced'`

4. **Model Evaluation**
   - Metrics: Precision, Recall, F1-Score, AUPRC
   - Confusion matrix

5. **Results and Export**
   - Save cleaned training and test predictions
   - Export top 10 suspicious transactions

6. **Visualizations**
   - Class distribution before/after SMOTE
   - Feature importances
   - Fraud probability distribution

---

## Output Files

| File | Description |
|------|-------------|
| `fraud_detection_model.ipynb` | Main Jupyter notebook |
| `top_suspicious.csv` | Top 10 suspicious transactions based on fraud probability |
| `README.md` | Project documentation |

---

## Sample Visualizations

> Below visuals are generated automatically in the notebook:

- Class Distribution Before and After SMOTE  
  ![Class Distribution]

- Top 10 Important Features  
  ![Feature Importance]

- Fraud Probability Histogram  
  ![Fraud Probability]

- Top 10 Suspicious Transactions (tabular output shown inline in notebook)
