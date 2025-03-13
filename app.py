
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('crop_recommendation_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title and description
st.title("Crop Recommendation System")
st.write("Enter the soil and weather parameters to get crop recommendations")

# Input fields
nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, max_value=150.0, value=50.0)
phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, max_value=150.0, value=50.0)
potassium = st.number_input("Potassium (K)", min_value=0.0, max_value=150.0, value=50.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0)

# Prediction function
def predict_crop(n, p, k, temp, hum, ph, rain):
    input_data = np.array([[n, p, k, temp, hum, ph, rain]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return prediction[0]

# Prediction button
if st.button("Recommend Crop"):
    try:
        recommended_crop = predict_crop(nitrogen, phosphorus, potassium,
                                      temperature, humidity, ph, rainfall)
        st.success(f"Recommended Crop: {recommended_crop}")
        st.write("Note: This is a model prediction. Please consult local agricultural experts.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
