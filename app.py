import streamlit as st
import joblib
import pandas as pd

# Load trained model
gbr = joblib.load("gradient_boosting_model.pkl")

# Get feature names exactly as model knows them
features = gbr.feature_names_in_

st.title("🔮 NFEA Predictor")

# Input fields
user_input = []
for feature in features:
    value = st.number_input(f"Enter value for {feature}", format="%.6f")
    user_input.append(value)

# Prediction button
if st.button("Predict"):
    user_input_df = pd.DataFrame([user_input], columns=features)
    prediction = gbr.predict(user_input_df)
    st.success(f"Predicted NFEA: {prediction[0]:.2f}")
