# Stroke Risk Prediction App

This project is a machine learning-based web application that predicts the probability of stroke using patient demographic and clinical features. The app is built using Streamlit and deployed on Streamlit Cloud.

🔗 **Live App:**  
https://healthcare-ml-project-jwhqizdl2yin3uvzrhenvp.streamlit.app/

🔗 **GitHub Repository:**  
https://github.com/AleenaChanda03/healthcare-ml-project

---

## Features

- Predicts stroke risk based on user inputs
- Handles both numerical and categorical features
- Real-time prediction through an interactive UI
- End-to-end ML pipeline (preprocessing → modeling → deployment)

---

## Input Features

- Age  
- BMI  
- Average Glucose Level  
- Hypertension  
- Heart Disease  
- Gender  
- Ever Married  
- Work Type  
- Residence Type  
- Smoking Status  

---

## Machine Learning

- Models used: Random Forest / Gradient Boosting (customize if needed)
- Preprocessing:
  - One-hot encoding for categorical variables
  - Feature alignment using saved `model_columns.pkl`
  - Feature scaling using `StandardScaler`
- Evaluation:
  - Cross-validation
  - Probability-based predictions

---

## Tech Stack

- Python
- Scikit-learn
- Pandas, NumPy
- Streamlit
- Joblib
- GitHub

---

## Project Structure

├── app.py # Streamlit app
├── stroke_model.pkl # Trained ML model
├── scaler.pkl # Feature scaler
├── model_columns.pkl # Feature alignment
├── data/ # Dataset
├── notebooks/ # EDA and modeling notebooks
├── requirements.txt # Dependencies


## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/AleenaChanda03/healthcare-ml-project.git

# Navigate to the project folder
cd healthcare-ml-project

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Deployment
Deployed using Streamlit Cloud
Automatically updates when changes are pushed to GitHub



## Future Improvements
Add more advanced models (XGBoost, Neural Networks)
Improve UI/UX design
Add model explainability (e.g., SHAP values)
Integrate real-world healthcare datasets


## Author
Aleena Chanda
GitHub: https://github.com/AleenaChanda03
LinkedIn: https://www.linkedin.com/in/aleena-chanda-893137126/







