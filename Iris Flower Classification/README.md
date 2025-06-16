# Iris Flower Classification Project

## Overview

This project builds a machine learning model to classify Iris flowers into three species: Setosa, Versicolor, and Virginica. The dataset contains 150 samples with four features — sepal length, sepal width, petal length, and petal width. The objective is to train a model that accurately classifies a given flower species based on these measurements.

## Dataset

**Source:** [ Iris Flower Dataset ](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)

## Tools and Technologies Used

- Python
- Jupyter Notebook
- Pandas, NumPy
- Seaborn, Matplotlib
- scikit-learn (KNN Classifier, StandardScaler, train_test_split, evaluation metrics)

## Workflow

1. **Data Loading**  
   The Iris dataset is loaded using Pandas and cleaned by trimming column names and target labels.

2. **Data Visualization**  
   Various plots are created to understand feature distribution and inter-feature relationships:
   - Violin plot for petal length
   - Pair plot for feature comparison
   - Correlation heatmap
   - Sepal width scatter strip plot

3. **Preprocessing**  
   - The feature matrix `X` is scaled using `StandardScaler`
   - Data is split into training and testing sets

4. **Model Building**  
   - A K-Nearest Neighbors (KNN) classifier is trained on the training data
   - Predictions are made on the test data

5. **Evaluation**  
   - Accuracy, confusion matrix, and classification report are printed to evaluate performance

## Output

- Accuracy score of the model on unseen data
- Confusion matrix for class-wise error analysis
- Classification report (Precision, Recall, F1-score)

## Acknowledgement

- This project is one of the three required tasks for the Data Science Internship at CodSoft.  
- I have actively used the Data Science course by Krish Naik and various YouTube tutorials to guide my learning and implementation throughout this project.
---
