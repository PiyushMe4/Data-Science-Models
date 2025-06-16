# Titanic Survival Prediction – Machine Learning Project

This project is a full end-to-end machine learning pipeline that predicts whether a passenger would survive the Titanic disaster. It includes data preprocessing, feature engineering, model training, evaluation, visualization, and interactive prediction through user input. The goal is to build a **human-readable**, practical, and reproducible ML solution.

---

## Project Workflow

1. **Load the Dataset**
   - Load the Titanic CSV file and inspect data structure.

2. **Clean the Data**
   - Drop irrelevant features like `PassengerId`, `Name`, `Ticket`, and `Cabin`.
   - Fill missing values:
     - `Age`: replaced with median.
     - `Embarked`: replaced with mode.

3. **Feature Engineering**
   - Add new features:
     - `FamilySize = SibSp + Parch + 1`
     - `IsAlone = 1 if FamilySize == 1 else 0`

4. **Encoding & Scaling**
   - Encode categorical features using `OneHotEncoder`.
   - Standardize numeric features using `StandardScaler`.

5. **Train-Test Split**
   - Split the cleaned dataset into training and testing sets.
   - Save both datasets as `train_data.csv` and `test_data.csv`.

6. **Build ML Pipeline**
   - Combine preprocessing and `RandomForestClassifier` using `Pipeline`.
   - Train the model on scaled, encoded data.

7. **Evaluate Model**
   - Print accuracy, confusion matrix, and classification report.
   - Plot feature importance, heatmap, and boxplot for visual inspection.

8. **Make Prediction (User Input)**
   - Take user input for passenger attributes.
   - Predict survival using the trained model.
   - Show the confidence and display a sample of training data for comparison.

9. **Export Model**
   - Save trained model and scaler as `.pkl` files using `joblib`.

---

## Dataset Details

**Source**: Kaggle Titanic Dataset  
**Features Used**:
- `Pclass` – Ticket class
- `Sex` – Male or Female
- `Age`
- `SibSp` – # of siblings/spouses aboard
- `Parch` – # of parents/children aboard
- `Fare` – Ticket fare
- `Embarked` – C = Cherbourg, Q = Queenstown, S = Southampton  
**Engineered**:
- `FamilySize` – Total family members aboard
- `IsAlone` – Binary indicator for solo travelers

---

## Visualizations

The project generates clear and minimal visualizations:

1. **Feature Importance Plot**  
   - Horizontal bars with color variety  
   - Features sorted by importance

2. **Age vs Survival Boxplot**  
   - Compares survival trends across age distributions

3. **Correlation Heatmap**  
   - Color intensity shows positive or negative correlation between features

---

## Model Evaluation

- **Accuracy**: ~80%  
- **Confusion Matrix**: Displays correct vs incorrect predictions  
- **Classification Report**: Shows precision, recall, F1-score for each class  

These metrics give insight into the model’s performance on unseen test data.

---

## User Input Prediction

**After training, the program prompts the user with:**
- Ticket Class (1 = 1st, 2 = 2nd, 3 = 3rd):
- Sex (Male/Female):
- Age:
- Number of Siblings/Spouses aboard:
- Number of Parents/Children aboard:
- Fare Paid:
- Port of Embarkation (C/Q/S):

---

### Output:
- Whether the passenger would have survived or not
- Prediction confidence as a percentage
- First five preprocessed training rows for comparison

---

## Files Generated

- `train_data.csv` – Preprocessed training set  
- `test_data.csv` – Preprocessed test set  
- `titanic_model.pkl` – Trained classification model  
- `titanic_scaler.pkl` – Preprocessing scaler

---

## Technologies Used

| Task              | Tool / Library        |
|-------------------|------------------------|
| Data Handling     | pandas, numpy          |
| Visualization     | seaborn, matplotlib    |
| ML Model          | RandomForestClassifier |
| Preprocessing     | sklearn Pipeline, ColumnTransformer |
| Model Export      | joblib                 |

---

## Acknowledgement

This project was developed with a strong emphasis on interpretability, user interaction, and real-world machine learning workflow. 

I sincerely acknowledge the guidance and conceptual clarity provided by educational creators such as:
- **Krish Naik** – for practical end-to-end project building strategies.
- **freeCodeCamp.org** – for comprehensive, beginner-friendly ML content.

Additionally, various blog posts, Stack Overflow discussions, and open web documentation contributed greatly to decisions made during preprocessing, modeling, and visualization.
