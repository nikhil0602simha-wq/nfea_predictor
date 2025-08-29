import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("gradient_boosting_model.pkl")

# Streamlit App UI
st.set_page_config(page_title="NFEA Predictor", page_icon="📈", layout="centered")

st.title("📌 NFEA Prediction App")
st.write("Enter the input parameters below to predict the NFEA value:")

# Input fields
col1, col2 = st.columns(2)

with col1:
    beta = st.number_input("β", value=0.4, format="%.3f")
    gamma = st.number_input("γ", value=80.0, format="%.3f")
    theta = st.number_input("θ (°)", value=45, step=1)

with col2:
    dc = st.number_input("dc (m)", value=0.2, format="%.3f")
    t0 = st.number_input("t0 (m)", value=0.00125, format="%.5f")

# Predict button
if st.button("🔍 Predict NFEA"):
    # Prepare input DataFrame
    input_df = pd.DataFrame([[beta, gamma, theta, dc, t0]],
                            columns=["β", "γ", "θ (°)", "dc​ (m)", "t0​ (m)"])
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Display result
    st.success(f"Predicted NFEA: {prediction[0]:.2f}")
