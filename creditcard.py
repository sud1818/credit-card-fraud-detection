import streamlit as st
import pandas as pd
import joblib

# Load model pipeline
model = joblib.load("rf_fraud_model.pkl")

st.title("💳 Credit Card Fraud Detection (Improved)")
st.write("This app predicts whether a transaction is likely fraudulent using a Random Forest model.")

# Input UI
transaction_amount = st.number_input("Transaction Amount", min_value=1.0)
transaction_time = st.number_input("Transaction Time (sec since midnight)", min_value=0.0, max_value=86400.0)
account_age_days = st.number_input("Account Age (days)", min_value=1.0)
merchant_risk_score = st.slider("Merchant Risk Score", 0.0, 1.0, 0.5)
transaction_velocity = st.number_input("Transaction Velocity (txn/hour)", 0.0, 50.0)

# Create input DataFrame
input_data = pd.DataFrame({
    "Transaction_Amount": [transaction_amount],
    "Transaction_Time": [transaction_time],
    "Account_Age_Days": [account_age_days],
    "Merchant_Risk_Score": [merchant_risk_score],
    "Transaction_Velocity": [transaction_velocity]
})

# Prediction button
if st.button("Predict"):
    # Get probability
    fraud_probability = model.predict_proba(input_data)[0][1]

    # Adjustable threshold
    threshold = 0.35

    if fraud_probability > threshold:
        st.error(f"🚨 Fraud Likely! (Risk Score: {fraud_probability:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Risk Score: {fraud_probability:.2f})")
