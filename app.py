import streamlit as st
import pickle
import numpy as np

# Page settings
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

# Load model
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))


st.title("🩺 Diabetes Risk Prediction System")
st.markdown("### Enter your health details below")

st.divider()

# Inputs
HighBP = st.selectbox("High Blood Pressure", ["No", "Yes"])
HighChol = st.selectbox("High Cholesterol", ["No", "Yes"])
BMI = st.number_input("Body Mass Index (BMI)", 10.0, 60.0)
Smoker = st.selectbox("Smoker", ["No", "Yes"])

age_dict = {
    "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4,
    "40-44": 5, "45-49": 6, "50-54": 7, "55-59": 8,
    "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12,
    "80+": 13
}

age_range = st.selectbox("Age Range", list(age_dict.keys()))

# Convert Yes/No to 1/0
HighBP = 1 if HighBP == "Yes" else 0
HighChol = 1 if HighChol == "Yes" else 0
Smoker = 1 if Smoker == "Yes" else 0
Age = age_dict[age_range]

st.divider()

if st.button("🔍 Predict Risk"):

    features = np.array([[HighBP, HighChol, BMI, Smoker, Age]])
    features = scaler.transform(features)

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠ High Risk of Diabetes")
        st.write("Please consult a healthcare professional.")
    else:
        st.success("✅ Low Risk of Diabetes")
        st.write("Maintain a healthy lifestyle.")

st.sidebar.title("About")
st.sidebar.write("Model: Random Forest Classifier")
st.sidebar.write("Developed for Internship Project")
