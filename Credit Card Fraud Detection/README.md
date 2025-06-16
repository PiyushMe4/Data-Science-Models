# Credit Card Fraud Detection Project

## Overview

This project aims to detect fraudulent credit card transactions using a supervised machine learning approach. The dataset is highly imbalanced, with only 0.17% of the transactions labeled as fraud. SMOTE (Synthetic Minority Oversampling Technique) was used to address this imbalance.

## Dataset Overview

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Samples**: 284,807 transactions
- **Fraud Cases**: 492 (≈0.172%)
- **Features**:
  - `Time` – Seconds since the first transaction
  - `Amount` – Transaction amount
  - `V1` to `V28` – PCA components
  - `Class` – Target (1 = Fraud, 0 = Genuine)

 ## Tools and Technologies Used

- Python
- Jupyter Notebook
- Pandas, NumPy
- Seaborn, Matplotlib
- scikit-learn (Random Forest, Logistic Regression, SMOTE, train_test_split)
- imbalanced-learn
- joblib

## Workflow

1. **Data Loading and Preprocessing**  
   - The dataset is read and cleaned (`Time` column dropped, `Amount` scaled)
   - Feature and label separation is done

2. **Handling Imbalance using SMOTE**  
   - SMOTE is applied to generate synthetic fraud samples and balance the dataset

3. **Model Training**  
   - Random Forest Classifier is trained on the balanced data
   - The model is used to make predictions and generate fraud probabilities

4. **Evaluation and Visualization**  
   - Confusion matrix and classification report for performance analysis
   - Precision-Recall curve is plotted for visual performance
   - Bar plot of top 10 most important features used by the model

5. **Saving Output**  
   - The cleaned and resampled training and test datasets are saved as CSV files
   - A final labeled dataset with predictions and probabilities is saved
   - The trained model is saved using `joblib`

## Output

- A trained Random Forest model capable of classifying fraudulent vs genuine transactions
- CSV files:
  - `training_dataset.csv`
  - `test_dataset.csv`
  - `fraud_cases_only.csv`
  - `all_predictions_with_probabilities.csv`
  - `final_model_predictions.csv`
- Visualizations including:
  - Class distribution before/after SMOTE
  - Precision-Recall Curve
  - Top 10 feature importances
  - Confusion matrix with annotated labels

## Acknowledgement

This project is one of the three required tasks for the Data Science Internship at CodSoft.  
I have actively used the Data Science course by Krish Naik and various YouTube tutorials to guide my learning and implementation throughout this project.
