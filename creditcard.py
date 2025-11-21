import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")

st.title("💳 Credit Card Fraud Detection App")
st.write("Enter transaction details to predict whether it is **Legitimate** or **Fraudulent**.")

# ---------------------------
# Load model safely
# ---------------------------
MODEL_PATH = "decision_tree_gini_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found! Make sure 'decision_tree_gini_model.pkl' exists in the same folder.")
    st.stop()

model = joblib.load(MODEL_PATH)

# ---------------------------
# User Inputs
# ---------------------------
transaction_amount = st.number_input(
    "Transaction Amount",
    min_value=0.01,
    value=100.0
)

transaction_time = st.number_input(
    "Transaction Time (seconds since midnight)",
    min_value=0.0,
    max_value=86400.0,
    value=36000.0
)

account_age_days = st.number_input(
    "Account Age (in days)",
    min_value=1.0,
    value=150.0
)

merchant_risk_score = st.slider(
    "Merchant Risk Score",
    0.0, 1.0, 0.5
)

transaction_velocity = st.number_input(
    "Transaction Velocity (transactions/hour)",
    min_value=0.0,
    max_value=20.0,
    value=2.0
)

# ---------------------------
# Input DataFrame
# ---------------------------
input_data = pd.DataFrame({
    "Transaction_Amount": [transaction_amount],
    "Transaction_Time": [transaction_time],
    "Account_Age_Days": [account_age_days],
    "Merchant_Risk_Score": [merchant_risk_score],
    "Transaction_Velocity": [transaction_velocity]
})

st.write("### 🔍 Input Data Preview")
st.dataframe(input_data)

# ---------------------------
# Predict
# ---------------------------
if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Legitimate Transaction")
    else:
        st.error("🚨 Fraudulent Transaction Detected!")
