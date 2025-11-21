import streamlit as st
import pandas as pd
import os
import joblib  # ensure joblib is listed in requirements.txt

# Page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# Header
st.markdown(
    """
    <h1 style='text-align:center; color:#2A6AB0;'>💳 Credit Card Fraud Detection</h1>
    <p style='text-align:center;'>Enter transaction details below and the model will predict if it’s Legitimate or Fraudulent.</p>
    """,
    unsafe_allow_html=True
)

# Model load
MODEL_PATH = "decision_tree_gini_model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found: '{MODEL_PATH}'. Please add it to the app folder.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("❌ Failed to load the model. Check that the file is correct and compatible.")
    st.write("Error details:", e)
    st.stop()

# Input section
st.subheader("📌 Transaction Details")

col1, col2 = st.columns(2)

with col1:
    transaction_amount = st.number_input(
        "💰 Transaction Amount",
        min_value=0.01,
        value=100.0,
        help="Amount of the transaction"
    )
    account_age_days = st.number_input(
        "📅 Account Age (days)",
        min_value=1.0,
        value=120.0,
        help="Number of days since account creation"
    )

with col2:
    transaction_time = st.number_input(
        "⏱ Transaction Time (seconds since midnight)",
        min_value=0.0,
        max_value=86400.0,
        value=36000.0,
        help="Time of day transaction happened"
    )
    transaction_velocity = st.number_input(
        "📈 Transaction Velocity (transactions/hour)",
        min_value=0.0,
        max_value=50.0,
        value=3.0,
        help="Rate of transactions per hour for this account"
    )

merchant_risk_score = st.slider(
    "🏪 Merchant Risk Score",
    0.0, 1.0, 0.5,
    help="Risk score of the merchant (0 = safe, 1 = very risky)"
)

# Build input DataFrame
input_data = pd.DataFrame({
    "Transaction_Amount": [transaction_amount],
    "Transaction_Time": [transaction_time],
    "Account_Age_Days": [account_age_days],
    "Merchant_Risk_Score": [merchant_risk_score],
    "Transaction_Velocity": [transaction_velocity]
})

st.markdown("### 📊 Input Summary")
st.dataframe(input_data, use_container_width=True)

# Prediction button
if st.button("🔍 Detect Fraud"):
    try:
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.success("🟢 Legitimate Transaction – No suspicious activity detected.")
        else:
            st.error("🔴 Fraudulent Transaction Detected!")
    except Exception as e:
        st.error("❌ Prediction failed.")
        st.write("Error details:", e)
