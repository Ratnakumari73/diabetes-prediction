import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

df = pd.read_csv("diabetes.csv")
df["Diabetes_binary"] = df["Diabetes_012"].apply(lambda x: 0 if x == 0 else 1)
df = df.drop("Diabetes_012", axis=1)

selected_features = ["HighBP", "HighChol", "BMI", "Smoker", "Age"]
X = df[selected_features]
y = df["Diabetes_binary"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
model.fit(X, y)

st.title("🩺 Diabetes Risk Prediction System")
st.markdown("### Enter your health details below")
st.divider()

HighBP = st.selectbox("High Blood Pressure", ["No", "Yes"])
HighChol = st.selectbox("High Cholesterol", ["No", "Yes"])
BMI = st.number_input("Body Mass Index (BMI)", 10.0, 60.0)
Smoker = st.selectbox("Smoker", ["No", "Yes"])

age_dict = {"18-24":1,"25-29":2,"30-34":3,"35-39":4,"40-44":5,"45-49":6,"50-54":7,"55-59":8,"60-64":9,"65-69":10,"70-74":11,"75-79":12,"80+":13}
age_range = st.selectbox("Age Range", list(age_dict.keys()))

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
