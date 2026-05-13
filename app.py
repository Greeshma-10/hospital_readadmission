import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and feature list
model = joblib.load("readmission_model.pkl")
model_features = joblib.load("model_features.pkl")

st.set_page_config(page_title="Hospital Readmission Predictor")
st.title("🏥 Hospital Readmission Predictor")
st.write("Predict whether a diabetic patient will be readmitted within 30 days.")

# ---------------- INPUTS ---------------- #

col1, col2 = st.columns(2)

with col1:
    age = st.selectbox("Age Group", [
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
    ], index=6)

    time_in_hospital = st.number_input("Time in Hospital (days)", min_value=1, max_value=14, value=3, step=1)
    num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=0, max_value=100, value=45, step=1)
    num_medications = st.number_input("Number of Medications", min_value=0, max_value=50, value=12, step=1)
    number_inpatient = st.number_input("Previous Inpatient Visits", min_value=0, max_value=20, value=0, step=1)

with col2:
    num_procedures = st.number_input("Number of Procedures", min_value=0, max_value=10, value=1, step=1)
    number_diagnoses = st.number_input("Number of Diagnoses", min_value=1, max_value=16, value=5, step=1)
    gender = st.selectbox("Gender", ["Female", "Male"])
    diabetesMed = st.selectbox("On Diabetes Medication", ["Yes", "No"])
    insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
    metformin = st.selectbox("Metformin", ["No", "Steady", "Up", "Down"])

# ---------------- PREDICTION ---------------- #

if st.button("Predict Readmission", use_container_width=True):

    # Start with zeros for all features
    input_dict = {col: 0 for col in model_features}

    # Correct encodings matching training data exactly
    user_values = {
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_medications': num_medications,
        'number_inpatient': number_inpatient,
        'num_procedures': num_procedures,
        'number_diagnoses': number_diagnoses,

        'age': {
            "[0-10)": 0, "[10-20)": 10, "[20-30)": 20, "[30-40)": 30,
            "[40-50)": 40, "[50-60)": 50, "[60-70)": 60,
            "[70-80)": 70, "[80-90)": 80, "[90-100)": 90
        }[age],

        'gender': {"Female": 0, "Male": 1}[gender],

        'insulin': {"Down": 0, "No": 1, "Steady": 2, "Up": 3}[insulin],

        'metformin': {"Down": 0, "No": 1, "Steady": 2, "Up": 3}[metformin],

        'diabetesMed': {"No": 0, "Yes": 1}[diabetesMed],
    }

    for key, val in user_values.items():
        if key in input_dict:
            input_dict[key] = val

    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Result")
    st.metric("Readmission Probability", f"{probability:.2%}")
    st.progress(float(probability))

    if prediction == 1:
        st.error("⚠️ High Risk of Readmission")
        st.write("**Factors likely contributing:** Long hospital stay, high inpatient visits, insulin changes.")
    else:
        st.success("✅ Low Risk of Readmission")
        st.write("**Patient appears stable.** Continue monitoring medications and follow-up care.")