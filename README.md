# 🧠 Employee Attrition Predictor

A machine learning web app to predict whether an employee is likely to leave their job, based on HR and workplace-related features. Built with Scikit-learn and Streamlit.

---

## 🚀 Features

- Predict employee attrition using a trained Random Forest Classifier
- Clean preprocessing pipeline: scaling & one-hot encoding
- Interactive Streamlit web app for input & prediction
- Evaluated with F1-score and ROC-AUC metrics
- Easily deployable to Streamlit Cloud or Hugging Face Spaces

---

## 📊 Model Inputs

| Feature             | Type        | Description                                 |
|---------------------|-------------|---------------------------------------------|
| Age                | Numeric     | Age of employee                             |
| MonthlyIncome      | Numeric     | Monthly income                              |
| YearsAtCompany     | Numeric     | Years worked at the company                 |
| DistanceFromHome   | Numeric     | Distance from home to office (km)           |
| PercentSalaryHike  | Numeric     | Most recent salary increase (%)             |
| BusinessTravel     | Categorical | Frequency of business travel                |
| JobRole            | Categorical | Employee's job position                     |
| MaritalStatus      | Categorical | Marital status                              |
| OverTime           | Categorical | Whether the employee works overtime         |

---

## 🛠️ Installation

```bash
# Clone the project
git clone https://github.com/your-username/employee-attrition-predictor.git
cd employee-attrition-predictor

# Install required libraries
pip install -r requirements.txt
```
🧪 Train the Model

```bash
python main.py
```
This will:

Load & preprocess the data

Train a RandomForestClassifier with GridSearchCV

Save the best model to models/best_model.pkl

Save scaler and encoder to models/

Evaluate with classification report and ROC-AUC

🌐 Run the Web App
```bash
streamlit run app/app.py
```
Then open http://localhost:8501 in your browser.