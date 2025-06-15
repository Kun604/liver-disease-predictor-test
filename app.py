import streamlit as st
import joblib
import pandas as pd

# Load model and preprocessors
model = joblib.load('liver_disease_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🩺 Liver Disease Risk Prediction")

st.markdown("Please enter your medical and lifestyle information below. The system will predict the likelihood of liver disease:")

# User inputs
age = st.number_input("Age", min_value=0, max_value=120, value=40)
gender = st.selectbox("Gender", ['Female', 'Male'])
bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=22.0)
alcohol = st.number_input("Alcohol Consumption (times/week)", min_value=0, max_value=14, value=0)
smoking = st.selectbox("Do you smoke?", ['No', 'Yes'])
genetic = st.selectbox("Genetic Risk", ['Low', 'Medium', 'High'])
activity = st.number_input("Physical Activity (times/week)", min_value=0, max_value=14, value=3)
diabetes = st.selectbox("Do you have diabetes?", ['No', 'Yes'])
hypertension = st.selectbox("Do you have hypertension?", ['No', 'Yes'])
lft = st.number_input("Liver Function Test Score (0–100)", min_value=0.0, max_value=100.0, value=50.0)

# Construct input DataFrame
input_df = pd.DataFrame([{
    'Age': age,
    'Gender': gender,
    'BMI': bmi,
    'AlcoholConsumption': alcohol,
    'Smoking': smoking,
    'GeneticRisk': genetic,
    'PhysicalActivity': activity,
    'Diabetes': diabetes,
    'Hypertension': hypertension,
    'LiverFunctionTest': lft
}])

# Encode categorical variables
categorical_cols = ['Gender', 'Smoking', 'GeneticRisk', 'Diabetes', 'Hypertension']
for col in categorical_cols:
    le = label_encoders[col]
    input_df[col] = le.transform(input_df[col])

# Scale numerical variables
numerical_cols = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'LiverFunctionTest']
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

# Make prediction
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    if pred == 1:
        st.error("⚠️ Prediction: High risk of liver disease. Please consult a doctor.")
    else:
        st.success("✅ Prediction: No signs of liver disease. Keep up the healthy lifestyle!")
