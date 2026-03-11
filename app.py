import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🧠",
    layout="centered"
)

# Load model objects
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

# Title
st.title("🧠 Stroke Risk Prediction App")

st.write("""
Enter patient health details below to estimate the probability of stroke.
""")

# -----------------------
# USER INPUTS
# -----------------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 0, 100, 50)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    glucose = st.number_input("Average Glucose Level", 50.0, 300.0, 100.0)
    hypertension = st.selectbox("Hypertension", [0, 1])


with col2:
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    married = st.selectbox("Ever Married", ["No", "Yes"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence = st.selectbox("Residence Type", ["Urban", "Rural"])
    smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])
# -----------------------
# PREDICTION
# -----------------------

if st.button("Predict Stroke Risk"):

    # Create dataframe from inputs
    input_df = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "avg_glucose_level": [glucose],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "gender": [gender],
        "ever_married": [married],
        "work_type": [work_type],
        "Residence_type": [residence],
        "smoking_status": [smoking]
    })

    # Convert categorical variables to dummy variables
    input_df = pd.get_dummies(input_df)

    # Align columns with training data
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Predict probability
    prob = model.predict_proba(input_scaled)[0][1]

    # Display result
    st.subheader(f"Stroke Risk Probability: {prob:.2f}")

    if prob > 0.4:
        st.error("High Risk")
    elif prob > 0.2:
        st.warning("Medium Risk")
    else:
        st.success("Low Risk")

# -----------------------
# MODEL INFO
# -----------------------

st.markdown("---")

st.subheader("About This Model")

st.write("""
This machine learning model predicts stroke risk based on patient health indicators.

Model features include:

• Age  
• BMI  
• Average glucose level  
• Hypertension  
• Heart disease  
• Gender  
• Marital status  
• Work type  
• Residence type  
• Smoking status  

The model was trained using the **Stroke Prediction Dataset** and deployed using **Streamlit**.
""")