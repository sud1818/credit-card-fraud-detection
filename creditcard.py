import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("decision_tree_gini_model.pkl")

st.title("💳 Credit Card Fraud Detection App")
st.write("This web app predicts whether a transaction is Legitimate or Fraudulent using a Decision Tree model.")

# Input features
transaction_amount = st.number_input("Transaction Amount", min_value=1.0)
transaction_time = st.number_input("Transaction Time (seconds since midnight)", min_value=0.0, max_value=86400.0)
account_age_days = st.number_input("Account Age (in days)", min_value=1.0)
merchant_risk_score = st.slider("Merchant Risk Score", 0.0, 1.0, 0.5)
transaction_velocity = st.number_input("Transaction Velocity (transactions/hour)", min_value=0.0, max_value=20.0)

# Create input DataFrame
input_data = pd.DataFrame({
    "Transaction_Amount": [transaction_amount],
    "Transaction_Time": [transaction_time],
    "Account_Age_Days": [account_age_days],
    "Merchant_Risk_Score": [merchant_risk_score],
    "Transaction_Velocity": [transaction_velocity]
})

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ Legitimate Transaction")
    else:
        st.error("🚨 Fraudulent Transaction Detected!")
