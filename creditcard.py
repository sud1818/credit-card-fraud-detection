# -----------------------------
# app.py
# -----------------------------
import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load("rf_fraud_model.pkl")

st.set_page_config(page_title="Fraud Detection", page_icon="💳")
st.title("💳 Credit Card Fraud Detection")
st.write("Enter details to predict whether a transaction is fraudulent.")

# ---------------------------
# User Inputs
# ---------------------------
transaction_amount = st.number_input(
    "Transaction Amount (₹)", 
    min_value=1.0, 
    step=1.0
)

transaction_time = st.number_input(
    "Transaction Time – seconds since midnight",
    min_value=0.0,
    max_value=86400.0,
    step=1.0
)

account_age_days = st.number_input(
    "Account Age (in days)",
    min_value=1.0,
    step=1.0
)

merchant_risk_score = st.slider(
    "Merchant Risk Score (0 – 1)",
    0.0,
    1.0,
    0.5
)

transaction_velocity = st.number_input(
    "Transaction Velocity (transactions/hour)",
    min_value=0.0,
    max_value=50.0,
    step=0.1
)

# Create DataFrame for model
input_data = pd.DataFrame({
    "Transaction_Amount": [transaction_amount],
    "Transaction_Time": [transaction_time],
    "Account_Age_Days": [account_age_days],
    "Merchant_Risk_Score": [merchant_risk_score],
    "Transaction_Velocity": [transaction_velocity]
})

# ---------------------------
# Predict
# ---------------------------
if st.button("Predict"):
    probability = model.predict_proba(input_data)[0][1]
    threshold = 0.35  # tuned threshold

    if probability > threshold:
        st.error(f"🚨 Fraud Detected! (Risk Score: {probability:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Risk Score: {probability:.2f})")
