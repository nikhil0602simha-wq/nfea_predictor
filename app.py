import streamlit as st
import pandas as pd
import joblib

# Load trained model
gbr = joblib.load("gradient_boosting_model.pkl")
features = gbr.feature_names_in_

st.title("🔮 NFEA Prediction App")

# User input fields (as text for comfortable typing)
user_input = []
for feature in features:
    val = st.text_input(f"Enter value for {feature}", "0.0")  # default 0.0
    try:
        val = float(val)
    except:
        st.warning(f"⚠️ Please enter a valid number for {feature}")
        val = 0.0
    user_input.append(val)

# Predict button
if st.button("Predict"):
    user_input_df = pd.DataFrame([user_input], columns=features)
    prediction = gbr.predict(user_input_df)
    st.success(f"Predicted NFEA: {prediction[0]:.2f}")
