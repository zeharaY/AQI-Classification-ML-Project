import streamlit as st
import joblib
import numpy as np
import requests


# Load the trained model
model_url = "https://drive.google.com/uc?export=download&id=10PGQwoBwrwZz55FVgxwVO2Mj0bvAFX_g"
response = requests.get(model_url)

with open("model.pkl", "wb") as f:
    f.write(response.content)

#joblib.load("aqi_model.pkl")
model = joblib.load("model.pkl")


# Title of the app
st.title("🌫️ Air Quality Classifier")

# Input fields for pollutant concentrations
pm25 = st.number_input("PM2.5", value=0.0)
pm10 = st.number_input("PM10", value=0.0)
no2 = st.number_input("NO2", value=0.0)
so2 = st.number_input("SO2", value=0.0)
co = st.number_input("CO", value=0.0)
o3 = st.number_input("O3", value=0.0)
nh3 = st.number_input("NH3", value=0.0)

if st.button("Predict AQI Bucket"):
    # Prepare input data for prediction
    # Ensure the order of features matches the training data
    input_data = np.array([[pm25, pm10, no2, so2, co, o3, nh3]])

    # Make prediction
    prediction = model.predict(input_data)

    # Display the prediction result
    # You might want to map the numerical prediction back to the original AQI_Bucket names
    # based on your LabelEncoder or a predefined mapping.
    # For now, displaying the numerical prediction.
    st.success(f"Predicted AQI Bucket (Encoded): {prediction[0]}")
