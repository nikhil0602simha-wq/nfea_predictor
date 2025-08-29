import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load the trained model
# -------------------------------
with open("gradient_boosting_model.pkl", "rb") as file:
    model = pickle.load(file)

# Get the feature names used during training
try:
    model_features = model.feature_names_in_
except AttributeError:
    model_features = None

# -------------------------------
# Streamlit App UI
# -------------------------------
st.set_page_config(page_title="NFEA Predictor", page_icon="🤖", layout="centered")
st.title("📊 NFEA Prediction App")
st.write("Enter the input values below to predict the target variable.")

# -------------------------------
# Input Section
# -------------------------------
input_data = {}
if model_features is not None:
    st.subheader("Enter Values")
    for feature in model_features:
        input_data[feature] = st.number_input(f"**{feature}**", value=0.0)
else:
    st.error("Model feature names not found. Please ensure the model was trained with feature names.")

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        st.success(f"🎯 Predicted Value: **{prediction[0]:.2f}**")
    except ValueError as e:
        st.error(f"Prediction failed: {e}")
        st.info("Make sure your inputs match the model's expected features.")
