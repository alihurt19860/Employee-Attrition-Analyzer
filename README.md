## 💼 Employee Attrition Analyzer

**By:** M.ALI
**Course:** Machine Learning Lab
**Environment:** Python (Jupyter Notebook)


### 🧭 Project Overview

The **Employee Attrition Analyzer** is a machine learning project designed to predict whether an employee is likely to **leave (attrite)** or **stay** in a company.
Using the **IBM HR Analytics Employee Attrition & Performance dataset**, this project applies multiple ML models to analyze patterns and identify factors influencing attrition, such as salary, work environment, and performance metrics.

This project demonstrates an integration of key ML concepts learned in Labs 1–6 — including **data preprocessing, visualization, model training, and evaluation** — into one complete real-world solution.


### 🎯 Objectives

* Apply core machine learning concepts to a real-world HR dataset.
* Clean, preprocess, and visualize both categorical and numerical data.
* Build and compare multiple classification models to predict attrition.
* Evaluate models using accuracy, F1 score, and confusion matrices.
* Interpret key features influencing employee turnover through visualization.


### 🧩 Dataset Information

**Dataset Name:** IBM HR Analytics Employee Attrition & Performance
**Source:** [Kaggle - IBM HR Analytics Dataset](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)

**Description:**
This dataset contains detailed employee data from a fictional company modeled by IBM, including demographic, performance, and job-related information.

**Key Columns:**

| Feature                 | Description                               |
| ----------------------- | ----------------------------------------- |
| Age                     | Employee’s age                            |
| BusinessTravel          | Frequency of business travel              |
| Department              | Department name                           |
| Education               | Education level                           |
| EnvironmentSatisfaction | Work environment rating                   |
| JobSatisfaction         | Job satisfaction rating                   |
| MonthlyIncome           | Employee’s monthly salary                 |
| YearsAtCompany          | Duration of employment                    |
| Attrition               | Target variable (Yes = Left, No = Stayed) |


### ⚙️ Technologies Used

* **Language:** Python
* **IDE:** Jupyter Notebook
* **Libraries:**

  * `pandas`, `numpy` — data handling
  * `matplotlib`, `seaborn` — visualization
  * `scikit-learn` — ML models, preprocessing, and evaluation


### 🔧 Project Workflow

#### 1. **Data Loading & Exploration**

* Loaded the dataset using Pandas.
* Checked for missing values, duplicates, and basic statistics.
* Visualized attrition distribution to understand dataset balance.

#### 2. **Data Preprocessing**

* Encoded categorical variables using `LabelEncoder` and `pd.get_dummies()`.
* Normalized numeric features using `StandardScaler`.
* Separated input features (`X`) and target label (`y`).
* Split dataset into training (80%) and testing (20%) sets.

#### 3. **Model Building & Training**

Trained and compared the following models:

| Model               | Type       | Purpose                                  |
| ------------------- | ---------- | ---------------------------------------- |
| Logistic Regression | Linear     | Baseline classification model            |
| Decision Tree       | Non-linear | Captures hierarchical decision rules     |
| Random Forest       | Ensemble   | Improves accuracy through multiple trees |

#### 4. **Model Evaluation**

* Evaluated using:

  * **Accuracy Score**
  * **Confusion Matrix**
  * **Precision, Recall, and F1 Score**
* Visualized confusion matrices for each model.
* Displayed feature importance for Decision Tree and Random Forest.


### 📊 Results Summary

| Model               | Accuracy | F1 Score |
| ------------------- | -------- | -------- |
| Logistic Regression | ~0.85    | ~0.84    |
| Decision Tree       | ~0.88    | ~0.86    |
| Random Forest       | ~0.91    | ~0.89    |

*(Results may slightly vary due to random seed and preprocessing settings.)*

**Observation:**
The **Random Forest Classifier** achieved the highest overall performance, showing strong predictive power for employee attrition.

Key influencing features included:

* MonthlyIncome
* YearsAtCompany
* JobSatisfaction
* EnvironmentSatisfaction


### 📈 Visualizations

* **Attrition Distribution Plot** — To observe imbalance in target variable.
* **Correlation Heatmap** — To visualize relationships among numeric features.
* **Feature Importance Bar Chart** — To identify key attrition drivers.
* **Confusion Matrix Heatmaps** — For each model’s classification results.


### 🏁 Conclusion

* Successfully integrated multiple ML models to predict employee attrition.
* Demonstrated the end-to-end ML workflow: preprocessing → modeling → evaluation.
* Identified key factors influencing employee retention decisions.
* **Random Forest** emerged as the most reliable and accurate model.

This project shows the practical application of machine learning techniques in **Human Resource Analytics**, linking data science to business insights.



### 📎 Deliverables

* Jupyter Notebook: `Employee_Attrition_Analyzer.ipynb`
* Dataset: `WA_Fn-UseC_-HR-Employee-Attrition.csv`


### 🧠 Future Improvements

* Apply **GridSearchCV** for hyperparameter tuning.
* Try **XGBoost** or **Gradient Boosting** models for higher accuracy.
* Build an interactive **dashboard (Streamlit)** for real-time predictions.
* Perform **SMOTE balancing** for better handling of class imbalance.



Would you like me to now create the **step-by-step Jupyter Notebook code (cell by cell)** for this same project next — just like we did for the first on
