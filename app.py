import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load objects
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Stroke Risk Prediction App")

st.write("Enter patient details below:")

# Inputs
age = st.slider("Age", 0, 100, 50)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
glucose = st.number_input("Average Glucose Level", 50.0, 300.0, 100.0)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])

# Predict
if st.button("Predict Stroke Risk"):

    # Create dataframe with your raw inputs
    input_df = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "avg_glucose_level": [glucose],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease]
    })

    # Add missing columns (set to 0)
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[model_columns]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader(f"Stroke Risk Probability: {prob:.2f}")

    if prob > 0.4:
        st.error("High Risk")
    elif prob > 0.2:
        st.warning("Medium Risk")
    else:
        st.success("Low Risk")