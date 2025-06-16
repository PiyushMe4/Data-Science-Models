# Titanic Survival Prediction – End-to-End ML Pipeline

This project implements an end-to-end machine learning pipeline to predict Titanic passenger survival using structured input, proper data handling, robust preprocessing, and visual clarity. It includes custom transformations, pipeline integration, user interactivity, and data export for full transparency.

---

## 📦 Project Workflow

1. **Data Loading**
   - Load raw Titanic dataset using `pandas`.
   - Preview and understand structure.

2. **Data Cleaning & Preprocessing**
   - Drop uninformative columns: `PassengerId`, `Name`, `Ticket`, `Cabin`
   - Handle missing values using median (Age) and mode (Embarked).
   - Feature engineering:
     - `FamilySize = SibSp + Parch + 1`
     - `IsAlone = 1 if FamilySize == 1 else 0`
   - Categorical encoding with `OneHotEncoder`
   - Scaling with `StandardScaler`

3. **Data Splitting**
   - Use `train_test_split` to divide into 80% training and 20% testing.
   - Export `train_data.csv` and `test_data.csv` after combining `X` and `y`.

4. **Model Pipeline Construction**
   - Use `ColumnTransformer` to preprocess numerical and categorical features.
   - Wrap `RandomForestClassifier` in a `Pipeline`.
   - Fit on training data.

5. **Model Evaluation**
   - Predict on test set.
   - Print:
     - Accuracy score
     - Confusion matrix
     - Classification report

6. **Feature Importance Extraction**
   - Extract feature names from preprocessor.
   - Clean names (remove prefixes like `num__`, `cat__`)
   - Plot sorted barplot with multicolors and reduced figure size for compact visualization.

7. **Visualizations**
   - Boxplot: Age vs Survival
   - Heatmap: Correlation matrix
   - Feature Importance Barplot

8. **Model & Scaler Export**
   - Save both trained model and fitted scaler as `.pkl` files using `joblib`.

9. **User Input & Prediction**
   - Accept structured input with prompt messages indicating expected value ranges.
   - Encode and scale input.
   - Predict survival with probability.
   - Print prediction in clean language.
   - Show a few entries from the training set for comparison.

---

## 🗂 Dataset Used

- **Source**: Titanic Dataset from Kaggle
- **Features**:
  - Pclass (1st, 2nd, 3rd class)
  - Sex (male/female)
  - Age
  - SibSp (siblings/spouses aboard)
  - Parch (parents/children aboard)
  - Fare (fare paid)
  - Embarked (port of embarkation: C, Q, S)
- **Engineered**:
  - FamilySize
  - IsAlone

---

## 🧠 Model Details

| Component         | Method / Tool Used                      |
|-------------------|-----------------------------------------|
| Model             | RandomForestClassifier (n_estimators=100) |
| Pipeline          | Preprocessor + Classifier               |
| Categorical       | OneHotEncoder                           |
| Numerical         | StandardScaler                          |
| Imputation        | SimpleImputer (median for age, mode for embarked) |
| Evaluation        | Accuracy, Confusion Matrix, Classification Report |

---

## 🧪 Metrics

| Metric               | Value (approx.) |
|----------------------|-----------------|
| Accuracy             | 0.79 – 0.81     |
| Precision, Recall    | Detailed in report output |
| Confusion Matrix     | 2x2 matrix displayed clearly |
| Classification Report| Shown in full |

---

## 📤 Output Details

### 1. **Model Evaluation Metrics**
- **Accuracy Score**: Printed to console (typically ~80%)
- **Confusion Matrix**: Tabular layout showing correct vs. incorrect predictions
- **Classification Report**: Precision, recall, F1-score for both survival classes

### 2. **Visualizations**
- **Correlation Heatmap**: Shows relationships among features using color intensity
- **Boxplot of Age vs Survival**: Highlights age distribution in survivors vs non-survivors
- **Feature Importance Barplot**: 
  - Horizontal multicolor bars sorted by importance
  - Clean, readable feature names
  - Reduced figure size to avoid excessive scrolling

### 3. **File Exports**
- `train_data.csv`: Training portion of the preprocessed data
- `test_data.csv`: Test portion used for evaluation
- `titanic_model.pkl`: Final trained Random Forest model with preprocessing pipeline
- `titanic_scaler.pkl`: Scaler object for future standardization (if needed separately)

---

## 🧾 Input Prompt Details

After model training, the user is asked for input with clearly stated formats:

```text
Ticket Class (1 = 1st, 2 = 2nd, 3 = 3rd):
Sex (Male/Female):
Age:
No. of Siblings/Spouses aboard:
No. of Parents/Children aboard:
Fare Paid:
Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton):

- Returns:
- Whether the passenger would likely have survived
- Confidence percentage of prediction

---

## Acknowledgement

This project is one of the three required tasks for the Data Science Internship at CodSoft.  
I have actively used the Data Science course by  and various YouTube tutorials to guide my learning and implementation throughout this project.

