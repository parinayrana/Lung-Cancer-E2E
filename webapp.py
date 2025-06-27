import streamlit as st
import pandas as pd
import pickle

# Load model and preprocessor
with open('artifacts\Survival_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('artifacts\preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

st.set_page_config(page_title="Lung Cancer Survival Predictor")
st.title("🩺 Lung Cancer Survival Probability Predictor")

st.markdown("Enter the patient details below to estimate survival probability:")

# Numerical Inputs
age = st.number_input("Age", min_value=0, max_value=120)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0)
cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400)
treatment_duration = st.number_input("Treatment Duration (in days)", min_value=0, max_value=2000)

# Categorical Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
stage = st.selectbox("Cancer Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])
family_history = st.selectbox("Family History", ["Yes", "No"])
smoking_status = st.selectbox("Smoking Status", ["Never Smoked", "Current Smoker", "Former Smoker", "Passive Smoker"])
treatment_type = st.selectbox("Treatment Type", ["Surgery", "Chemotherapy", "Radiation", "Combined"])

# Binary Inputs
hypertension = st.checkbox("Hypertension")
asthma = st.checkbox("Asthma")
cirrhosis = st.checkbox("Cirrhosis")
other_cancer = st.checkbox("Other Cancer History")

# Predict button
if st.button("Predict Survival"):
    # Create a single-row DataFrame from inputs
    input_df = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'cholesterol_level': [cholesterol],
        'treatment_duration': [treatment_duration],
        'gender': [gender],
        'cancer_stage': [stage],
        'family_history': [family_history],
        'smoking_status': [smoking_status],
        'treatment_type': [treatment_type],
        'hypertension': [int(hypertension)],
        'asthma': [int(asthma)],
        'cirrhosis': [int(cirrhosis)],
        'other_cancer': [int(other_cancer)]
    })

    # Preprocess and predict
    X_transformed = preprocessor.transform(input_df)
    
    prob = model.predict_proba(X_transformed)[0][1]

    st.success(f"Estimated Survival Probability: **{round(prob * 100, 2)}%**")
