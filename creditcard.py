import streamlit as st
import pandas as pd
import os

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# -------------------------------------------------
# Page Header
# -------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#2A6AB0;'>💳 Credit Card Fraud Detection</h1>
    <p style='text-align:center; font-size:18px;'>
        Enter transaction details below and the machine learning model will 
        detect whether the transaction is <b>Legitimate</b> or <b>Fraudulent</b>.
    </p>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Try importing joblib safely
# -------------------------------------------------
try:
    import joblib
except ModuleNotFoundError as e:
    st.error("❌ joblib is not installed. Install it using:  pip install joblib")
    st.stop()

# -------------------------------------------------
# Load Model Safely
# -------------------------------------------------
MODEL_PATH = "decision_tree_gini_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found.\nMake sure '{MODEL_PATH}' exists.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("❌ Failed to load model. Check model file or sklearn version.")
    st.write("Full Error:", e)
    st.stop()

# -------------------------------------------------
# Inputs
# -------------------------------------------------
st.subheader("📌 Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    transaction_amount = st.number_input(
        "💰 Transaction Amount",
        min_value=0.01,
        value=100.0
    )

    account_age_days = st.number_input(
        "📅 Account Age (days)",
        min_value=1.0,
        value=120.0
    )

with col2:
    transaction_time = st.number_input(
        "⏱ Transaction Time (sec since midnight)",
        min_value=0.0,
        max_value=86400.0,
        value=35000.0
    )

    transaction_velocity = st.number_input(
        "📈 Transaction Velocity (transactions/hour)",
        min_value=0.0,
        max_value=20.0,
        value=3.0
    )

merchant_risk_score = st.slider(
    "🏪 Merchant Risk Score",
    0.0, 1.0, 0.5
)

# -------------------------------------------------
# Create Input DataFrame
# -------------------------------------------------
input_data = pd.DataFrame({
    "Transaction_Amount": [transaction_amount],
    "Transaction_Time": [transaction_time],
    "Account_Age_Days": [account_age_days],
    "Merchant_Risk_Score": [merchant_risk_score],
    "Transaction_Velocity": [transaction_velocity]
})

st.subheader("📊 Input Summary")
st.dataframe(input_data, use_container_width=True)

# -------------------------------------------------
# Predict
# -------------------------------------------------
if st.button("🔍 Run Fraud Detection"):
    try:
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success("🟢 Legitimate Transaction – No suspicious activity detected.")
        else:
            st.error("🔴 Fraudulent Transaction Detected!")
    except Exception as e:
        st.error("❌ Prediction failed.")
        st.write("Full Error:", e)
