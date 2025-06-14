# Iris Flower Classification - Machine Learning Project

This project implements a machine learning model to classify Iris flowers into one of three species — Setosa, Versicolor, or Virginica — based on sepal and petal measurements. The classification is performed using the K-Nearest Neighbors (KNN) algorithm, and includes exploratory data analysis with visualizations.

---

## Dataset

**Source:**[ Iris Flower Dataset ](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)

**Features Used:**
- SepalLengthCm
- SepalWidthCm
- PetalLengthCm
- PetalWidthCm
- Species (Target)

---

## Visualizations

Four visualizations were created to understand patterns and separability among classes:
- VIZ-A: Violin plot of Petal Length across Species
- VIZ-B: Pairplot of all numerical features colored by Species
- VIZ-C: Correlation heatmap of all features
- VIZ-D: Stripplot showing Sepal Width distribution per Species

---

## Model Details

- Algorithm: K-Nearest Neighbors (K=5)
- Preprocessing: Feature scaling using StandardScaler
- Split: 80% training and 20% testing
- Metrics: Accuracy, Confusion Matrix, Precision, Recall, F1-score

---

## Results

- Accuracy: Approximately 97% on test data
- All classes were correctly classified with minimal overlap
- Clear separation observed in the feature space between species

---

## Tools Used

- Python (Jupyter Notebook)
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn

---

## Project Objective

This project is a part of my Internship in Data Science at CodSoft to practice in classification-based machine learning problems. It was built to strengthen understanding of feature relationships, exploratory data analysis, and supervised learning workflows.

---

