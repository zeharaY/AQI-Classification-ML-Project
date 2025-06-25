import streamlit as st
import joblib
import numpy as np
import requests

# 🎯 Load the trained model from Google Drive
model_url = "https://drive.google.com/uc?export=download&id=10PGQwoBwrwZz55FVgxwVO2Mj0bvAFX_g"
response = requests.get(model_url)

with open("model.pkl", "wb") as f:
    f.write(response.content)

model = joblib.load("model.pkl")

# 🌿 Air Quality Prediction App
st.title("🌫️ Air Quality Classifier 🌍")
st.markdown("Use this app to predict **urban air quality** based on pollutant levels 🏙️💨")

# 🍃 Input fields (ordered as in training)
pm25 = st.number_input("PM2.5 (μg/m³) 🌁", value=0.0)
pm10 = st.number_input("PM10 (μg/m³) 🌫️", value=0.0)
no   = st.number_input("NO (μg/m³) 🧪", value=0.0)
no2  = st.number_input("NO₂ (μg/m³) 🏭", value=0.0)
nox  = st.number_input("NOx (μg/m³) 🛢️", value=0.0)
nh3  = st.number_input("NH₃ (μg/m³) 🧬", value=0.0)
co   = st.number_input("CO (mg/m³) 🚗", value=0.0)
so2  = st.number_input("SO₂ (μg/m³) 🏗️", value=0.0)
o3   = st.number_input("O₃ (μg/m³) 🌤️", value=0.0)
benzene  = st.number_input("Benzene (μg/m³) 🧴", value=0.0)
toluene  = st.number_input("Toluene (μg/m³) 🧷", value=0.0)
xylene   = st.number_input("Xylene (μg/m³) 🔬", value=0.0)

# 🔍 Predict AQI Bucket
if st.button("🚦 Predict AQI Bucket"):
    input_data = np.array([[pm25, pm10, no, no2, nox, nh3, co, so2, o3, benzene, toluene, xylene]])
    
    prediction = model.predict(input_data)

    # Optional: map encoded class to label (update this based on your LabelEncoder)
    class_labels = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    decoded = class_labels[prediction[0]] if prediction[0] < len(class_labels) else "Unknown"

    st.success(f"🏁 Predicted AQI Bucket: **{decoded}** 🏞️")
